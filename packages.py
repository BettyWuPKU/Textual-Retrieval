import pandas as pd
import numpy as np
import re
import nltk
from nltk import pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import random
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.metrics import accuracy_score, normalized_mutual_info_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import socket
from threading import Thread
import tkinter as tk
import json

# news的种类有5种
class_num = 5
# 保留相似词5个
synonym_num = 5

"""
以下三个函数是为了用KM算法找到最优匹配
"""
# 为左节点i找可以匹配的右节点
def find_path(graph, S, T, label_left, label_right, i, visited_left, visited_right, slack_right):
    visited_left[i] = True
    for j, match_weight in enumerate(graph[i]):
        if visited_right[j]:
            continue
        gap = label_left[i] + label_right[j] - match_weight
        if gap == 0:
            visited_right[j] = True
            if j not in T or find_path(graph, S, T, label_left, label_right, T[j], visited_left, visited_right, slack_right):
                T[j] = i
                S[i] = j
                return True
        else:
            slack_right[j] = min(slack_right[j], gap)
    return False

def KM(graph, S, T):
    m = len(graph)
    label_left = []
    label_right = []
    for i in range(m):
        label_left.append(np.max(graph[i]))
        label_right.append(0)
    for i in range(m):
        slack_right = [float('inf') for _ in range(m)]
        while True:
            visited_left = [False for _ in graph]
            visited_right = [False for _ in graph]
            if find_path(graph, S, T, label_left, label_right, i, visited_left, visited_right, slack_right):
                break
            d = float('inf')
            for j, slack in enumerate(slack_right):
                # 如果点j没有被访问过(因为两顶点权重和<边需要的权重)并且牺牲更小
                if not visited_right[j] and slack < d:
                    d = slack
            for k in range(m):
                if visited_left[k]:
                    label_left[k] -= d
                if visited_right[k]:
                    label_right[k] += d
    return S, T

def calc_F_KM(y_label, y_pred):
    l = len(y_label)
    assert l == len(y_pred)
    # n[i][r]表示label为i，pred为r的数量
    n = np.zeros([class_num, class_num])
    for i in range(l):
        n[y_label[i]][y_pred[i]] += 1
    # S[i]表示初始标签i的对应pred的类别
    S, _ = KM(n, {}, {})
    # pred_label[i]表示i的对应的聚类标签
    pred_label = np.empty([class_num])
    # precision[i]表示类别i的准确率，即分到i的当中是i的个数/分到i的个数
    precision = np.zeros([class_num])    
    # recall[i]召回率，为分到i中是i的个数/原本标签是i的个数
    recall = np.zeros([class_num])
    for i in range(class_num):
        precision[i] = n[i][S[i]] / np.sum(n, axis=0)[i]
        recall[i] = n[i][S[i]] / np.sum(n, axis=1)[i]
    F = 0
    for i in range(class_num):
        F += np.sum(n, axis=1)[i]/l*(2*recall[i]*precision[i]/(precision[i]+recall[i]+1e-8))
        
    return F

def calc_purity(y_label, y_pred):
    l = len(y_label)
    assert l == len(y_pred)
    # n[i][r]表示label为i，pred为r的数量
    n = np.zeros([class_num, class_num])
    n_pred = np.zeros([class_num])
    for i in range(l):
        n[y_label[i]][y_pred[i]] += 1
    # S[i]表示初始标签i的对应pred的类别
    S, T = KM(n, {}, {})
    # n_pred[r]为第r个聚类类别中文档的个数
    n_pred = np.sum(n, axis=0)
    # p_raw[i]为类别i的纯度
    p_raw = np.zeros([class_num])
    for i in range(class_num):
        p_raw[i] = n[T[i]][i]/(n_pred[i] + 1e-8)
    # p_raw = np.sum(n, axis=1)/(n_pred+1e-8)
    return np.sum(p_raw.dot(n_pred)/l)

# 下面两个函数是 为了在socket发送list数据结构，将数据转json/从json转回来
def encode_data(data):
    return bytes((json.dumps(data)).encode('utf-8'))

def decode_data(data):
    return json.loads(bytes(data).decode('utf-8'))