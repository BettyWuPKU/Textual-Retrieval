from ast import keyword
import rlcompleter
from pre_process import *

class MyServer:
    def __init__(self, host, port, filename):
        """
        初始化给定服务器运行的ip/端口，读取的data文件名，降维保留的维数
        返回结果的数量，HITS的最高迭代次数，hits的占比，关键词检索的权重
        """
        self.addr = (host, port)
        self.filename = filename
        # 特定维数 1500得到的聚类结果比较好
        self.dim = 1600
        # 返回结果top K
        self.max_reply = 20
        self.max_iter_hits = 1000
        self.type = 0
        self.hits_weight = 0.2
        self.key_words_weight = 10
    
    def start(self):
        """
        调用pre_process中的函数对文本进行预处理，计算TF-IDF，完成初步的检索
        """
        self.save_data()
        # self.load_data()
        self.serve()

    def save_data(self, filename='./data/save_data.npy'):
        """
        所有预处理，包括语料的处理，tf-idf生成、降维和相似度计算
        在debug的时候存储生成的数据
        """
        data = pd.read_csv(self.filename)
        self.data, self.word_list, self.word_per_body, self.word_per_title = pre_process(data)
        
        self.num_news = len(self.word_per_body)
        self.num_words = len(self.word_list)
        self.word_per_news = [x+y for x, y in zip(self.word_per_body, self.word_per_title)]
        self.TF, self.IDF, self.W  = self.calc_TF_IDF(self.word_list, self.word_per_news)

        self.W_reduced = self.pca_reduction(self.W, self.dim)
        self.docs_corr = cosine_similarity(self.W_reduced)

        # data_s = [self.data, self.word_list, self.word_per_body, self.word_per_title, self.W]
        # data_s = np.array(data_s)
        # np.save(filename, data_s)
        # print(f'data saved!')

    def load_data(self, filename='./data/save_data.npy'):
        """
        在debug的时候载入之前已经生成的文件的时候使用
        """
        data_l = np.load(filename, allow_pickle=True)
        [self.data, self.word_list, self.word_per_body, self.word_per_title, self.W] = list(data_l)

        self.num_news = len(self.word_per_body)
        self.num_words = len(self.word_list)
        self.word_per_news = [x+y for x, y in zip(self.word_per_body, self.word_per_title)]
        # self.TF, self.IDF, self.W  = self.calc_TF_IDF(self.word_list, self.word_per_news)
        # 对词向量进行降维

        self.W_reduced = self.pca_reduction(self.W, self.dim)
        self.docs_corr = cosine_similarity(self.W_reduced)

        print(f'data loaded!')

    def pca_reduction(self, weight, dim):
        """
        对weight降维，降到dim维
        """
        pca = PCA(n_components=dim)
        pca.fit(weight)
        pca_data = pca.transform(weight)
        return pca_data

    def handle_concurrency(self, conn):
        """
        实际每个线程处理请求时运行的程序，主要是先判断类型：如果只是查询聚类效果
        就直接返回，否则调用find_news检索文本
        结果都会发送回客户端
        """
        query = conn.recv(1024)
        query = decode_data(query)
        self.type = query[0]
        query = query[1:]
        print(f'received type: {self.type}')
        if self.type & 8:
            purity, F, NMI = self.cluster()
            result_list = [('Purity', purity), ('F', F), ('NMI', NMI)]
            conn.send(encode_data(result_list))
            print(f'cluster的评价发送完毕')

        else:
            result_type, result_list = self.find_news(len(query), query)
            result_num = 0
            if result_type == 0:
                result_num = min(self.max_reply, len(result_list))
                result_list = result_list[:result_num].tolist()
                rank_score = self.rank_score(result_list)
            else:
                rank_score = -1
                result_list = [-1]
            result_list.append(rank_score)
            conn.send(encode_data(result_list))
            print(f'"{query}"的相关的新闻发送完毕，共{result_num}条')
        conn.close()

    def serve(self):
        """
        服务器响应客户端请求的主要程序内容，通过这里的循环将工作分到线程
        调用handle_concurrency
        """
        t = self.pca_reduction(self.W.T, 600)
        self.word_corr = generate_related_words(t, self.word_list)
        self.word_corr_value = cosine_similarity(t)
        self.key_words_extraction()

        print('服务器运行中....')

        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.bind(self.addr)
        server.listen(100)

        while True:
            print('等待查询请求...')
            conn, addr = server.accept()
            # print(f'连接客户端的套接字对象为: {conn}\n 客户端的IP地址: {addr}')
            print(f'客户端的IP地址: {addr}')
            tr = Thread(target=self.handle_concurrency, args=(conn,))
            tr.start()

    def calc_TF_IDF(self, words, docs):
        """
        输入词典和文档，输出TF/IDF/W，W[d][t]表示文档d和t的相关度
        """
        print('Calculating TF-IDF...')
        num_docs = len(docs)
        num_words = len(words)
        TF = np.zeros((num_words, num_docs))
        for j in range(num_docs):
            for i in range(num_words):
                TF[i][j] = docs[j].count(words[i])
        for j in range(num_docs):
            TF[:, j] = TF[:, j] / np.log(np.sum(TF[:, j]))
        IDF = np.zeros(num_words)
        # DF[t]为含有term的文档数量
        for i in range(num_words):
            for j in range(num_docs):
                if TF[i][j]:
                    IDF[i] += 1
        IDF = np.log(num_docs/IDF)
        # Weight[t][d]表示t和d的相关度
        Weight = np.zeros((num_words, num_docs))
        for i in range(num_words):
            Weight[i] = TF[i] * IDF[i]
        # 每一行是一篇文章的向量
        Weight = Weight.T
        print('TF-IDF calculated!')
        return TF, IDF, Weight

    def find_news(self, n, q_words):
        """
        返回存在这些单词的news的下标，按照相关性(直接用查询词的one-hot
        编码和Weight相乘加和)降序排列，得到初步的结果（本来想检索每个词在文章的出现次数*权重作为分数，但是考虑到原本的tf
        就是出现的数值，进一步考虑到tf-idf还包含了表示词语“特殊性、代表性”的idf，还是用tf-idf比较好）
        可选择使用模糊匹配/HITS
        """
        query_vec = np.zeros(self.num_words)
        # blur, 对相似词赋予和原本检索关键词不同的权重，初始检索词权重为1，相似词为0.5
        if self.type & 1:
            for i in range(n):
                q_word = q_words[i]
                try:
                    idx = self.word_list.index(q_word)
                except:
                    return -1, f'word {q_word} not found in the word_list'
                query_vec[idx] = 1
                for t in self.word_corr[idx]:
                    if query_vec[t] == 0:
                        query_vec[t] = self.word_corr_value[idx][t]
        else:
            for i in range(n):
                q_word = q_words[i]
                try:
                    idx = self.word_list.index(q_word)
                except:
                    return -1, f'word {q_word} not found in the word_list'
                query_vec[idx] = 1

        query_result = self.W * query_vec
        query_result = query_result.sum(axis=1)
        # 关键词匹配
        if self.type & 4:
            w = np.max(query_result)
            t_idx = np.nonzero(query_vec)[0].tolist()
            for idx in t_idx:
                query_result = query_result + self.key_words_match(idx) * w

        nonzero_idx = np.nonzero(query_result)[0]
        # hits
        if self.type & 2:
            HITS_result = self.HITS(cosine_similarity(self.W[nonzero_idx]))
            HITS_result = np.argsort(-HITS_result)
            return 0, nonzero_idx[HITS_result]
            # query_result = HITS_res * self.hits_weight * np.max(query_result)/ np.max(HITS_res) + query_result
        sorted_idx = np.argsort(-query_result)
        sorted_idx = sorted_idx[np.in1d(sorted_idx, nonzero_idx)]
        
        return 0, sorted_idx
    
    def HITS(self, Adjoint):
        """
        采用HITS算法，根据不同文章之间的相似度优化检索，
        """
        # 邻接矩阵
        # threshold = np.max(self.docs_corr) / 2
        # threshold = 0.5
        # Adjoint = np.where(self.docs_corr < threshold, 0, 1)
        H = np.ones(Adjoint.shape[0]) / Adjoint.shape[0]
        err = 1e-2
        cnt = 0
        while (cnt < self.max_iter_hits):
            cnt += 1
            H_new = Adjoint @ (Adjoint @ H.T).T
            H_new = H_new / np.sum(H_new, axis=-1, keepdims=True)
            if abs(np.sum(H_new - H)) < err:
                break
        print(f'iterated {cnt} times')
        return H_new

    def cluster(self):
        """
        文章聚类评价文章向量的合理性，purity/F/NMI三个指标
        前两个都通过KM最大匹配得到准确的值
        这里有降维前后的对比
        """
        print('Calculating cluster scores...')
        Y_pred = KMeans(n_clusters=5, random_state=0).fit_predict(self.W_reduced)
        Y_pred2 = KMeans(n_clusters=5, random_state=0).fit_predict(self.W)
        dic_class = {'business':0, 'entertainment':1, 'politics':2, 'sport':3, 'tech':4}
        self.data['topic_num'] = self.data['topic'].apply(lambda x: dic_class[x])
        purity = calc_purity(self.data['topic_num'], Y_pred)
        purity2 = calc_purity(self.data['topic_num'], Y_pred2)
        print('             after pca              before pca')
        print(f'purity: {purity}, {purity2}')
        F = calc_F_KM(self.data['topic_num'], Y_pred)
        F2 = calc_F_KM(self.data['topic_num'], Y_pred2)
        print(f'F: {F}, {F2}')
        NMI = normalized_mutual_info_score(self.data['topic_num'], Y_pred)
        NMI2 = normalized_mutual_info_score(self.data['topic_num'], Y_pred2)
        print(f'NMI: {NMI}, {NMI2}')
        return purity, F, NMI
    
    def rank_score(self, result_sorted_idx):
        """
        对排序结果根据返回的结果中最多有多少是同一类的进行评价
        """
        if len(result_sorted_idx) == 0:
            print(f'length of result is 0')
            return
        
        result_num = min(self.max_reply, len(result_sorted_idx))
        cnt = np.zeros(5)
        dic_class = {'business':0, 'entertainment':1, 'politics':2, 'sport':3, 'tech':4}
        for i in range(result_num):
            cnt[dic_class[self.data['topic'][result_sorted_idx[i]]]] += 1
        return np.max(cnt) / result_num

    def key_words_extraction(self):
        """
        对每篇文章进行关键词提取，文章标题的内容默认就是关键词
        """
        print("Extracting key words for each passage now...")
        self.key_words_idx = []
        for i in range(self.num_news):
            idx = np.argsort(self.W[i]).tolist()[:5]
            idx.extend(np.nonzero(self.word_per_title)[0].tolist())
            self.key_words_idx.append(list(set(idx)))
        print("Extraction completed!")
        
    def key_words_match(self, idx):
        """
        对给定的检索词进行关键词匹配，返回相关关键词的文章的one-hot编码
        """
        news_related = np.zeros(self.num_news)
        for i in range(self.num_news):
            if idx in self.key_words_idx[i]:
                news_related[i] = 1
        return news_related



if __name__ == "__main__":
    filename = './data/all_news.csv'
    port = 5002
    host = '0.0.0.0'
    serve_p = MyServer(host, port, filename)
    serve_p.start()
