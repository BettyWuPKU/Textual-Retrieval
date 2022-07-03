# 在本文件实现对数据集的预处理
# 包括文章的去标点、还原过去式、分词、去除停用词和低频词汇
# 提取词根等，构建辞典，保存为vocab.txt
from packages import *

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# 去掉特殊字符/数字，转换为小写
def rid_of_specials(sentence):
    return re.sub('[^A-Za-z]+', ' ', sentence).lower()
# 去掉停用词
def remove_sw(x):
    x = x.split(' ')
    return  ' '.join(z for z in x if z not in stop_words)
# 词形还原确定词性
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None



def generate_related_words(W, word_list):
    """
    输入TF-IDF矩阵，将每一列作为每个词的特征向量，计算两两之间的余弦相似度
    排序得到相似词，保存到synonym.txt中每行中，第一个原本的词语，剩下的是相似词
    词语之间通过','间隔开
    """
    print('Generating related words...')
    W_word = W.copy()
    # W_word = W_word.T
    word_num = len(W_word)
    correlation_word = cosine_similarity(W_word)
    word_corr = []
    with open('./synonym.txt', 'w') as f:
        for i in range(word_num):
            rank_w = np.argsort(-correlation_word[i])
            rank_w = rank_w[:synonym_num]
            word_corr.append(rank_w)
            f.write(','.join(np.array(word_list)[rank_w]) + '\n')
    print('Related words generated!')
    return word_corr

def pre_process(data):
    """
    对标题和文章内容进行预处理，步骤如下：
    1. 去除除了字母以外的特殊字符
    2. 还原词形
    3. 去除停用词

    返回处理后的data，最终的词典final_word_list，每个新闻主体的词语列表，每个新闻标题的词语列表
    """
    # 对biao内容的预处理
    # 去除停用词和词形还原的顺序
    print('Starting pre-processing...')
    data['body_processed'] = data['body'].astype(str).apply(rid_of_specials)
    data['title_processed'] = data['title'].astype(str).apply(rid_of_specials)

    before_stop = list(data['body_processed'])
    news_num = len(before_stop)
    before_stop.extend(list(data['title_processed']))
    before_stop = [i.split(' ') for i in before_stop]
    before_stop = [pos_tag(i) for i in before_stop]

    wnl = WordNetLemmatizer()
    lemmatized = []
    cnt = -1
    for s in before_stop:
        cnt += 1
        lemmatized_s = []
        for tag in s:
            wordnet_pos = get_wordnet_pos(tag[1]) or wordnet.NOUN
            lemmatized_s.append(wnl.lemmatize(tag[0], pos=wordnet_pos))
        lemmatized.append(' '.join(lemmatized_s))

    prepared_sentence = [''.join(j) for j in lemmatized]
    stopped = list(pd.Series(prepared_sentence).apply(remove_sw))
    data['body_processed'] = stopped[:news_num]
    data['title_processed'] = stopped[news_num:]

    # 生成词典
    total_words_list = []
    word_per_news = []
    for t in stopped:
        total_words_list += t.split(' ')
        word_per_news.append(t.split(' '))

    freq_dist = nltk.FreqDist(total_words_list)
    freq_list = []
    num_words = len(freq_dist.values())
    for i in range(num_words):
        freq_list.append([list(freq_dist.keys())[i],list(freq_dist.values())[i]])
    # 将词语根据频率降序排序
    freq_list = sorted(freq_list, key=lambda x:x[1], reverse=True)
    threshold = 10
    freq_list_strip = [pair for pair in freq_list if pair[1] > threshold and pair[0] != '']
    final_word_list = []
    with open('./vocab.txt', 'w') as f:
        for [word, freq] in freq_list_strip:
            f.write(word + '\n')
            final_word_list.append(word)
    print("Preprocessing complete: vocabulary generated!")

    return data, final_word_list, word_per_news[:news_num], word_per_news[news_num:]
