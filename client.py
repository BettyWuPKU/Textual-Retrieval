from packages import *


class MainPanel:
    def __init__(self, host, port):
        # 用于连接服务器
        self.addr = (host, port)
        self.filename = './data/all_news.csv'
    
    def start(self):
        """
        显示文本检索主界面
        """
        self.root = tk.Tk()
        self.root.geometry('300x300')
        self.root.title("文本检索")
        self.label = tk.Label(self.root, text="请输入检索词，用空格分隔:", font=(None, 12)).pack(pady=20)

        self.new_searchterm_entry = tk.Entry(self.root, font=(None, 12))
        self.new_searchterm_entry.pack()

        self.opt_hits = tk.BooleanVar()
        self.opt_blur = tk.BooleanVar()
        self.opt_kw = tk.BooleanVar()
        self.hits = 0
        self.blur = 0
        self.kw = 0
        self.check1 = tk.Checkbutton(self.root, text='HITS', variable=self.opt_hits, command=self.search_options)
        self.check1.pack()
        self.check2 = tk.Checkbutton(self.root, text='模糊', variable=self.opt_blur, command=self.search_options)
        self.check2.pack()
        self.check3 = tk.Checkbutton(self.root, text='关键词检索', variable=self.opt_kw, command=self.search_options)
        self.check3.pack()

        self.confirm_button = tk.Button(self.root, text="开始检索", font=(None,12), command=self.check_new_searchterm).pack(pady=10)
        self.confirm_button_cluster = tk.Button(self.root, text="获取聚类结果评价", font=(None,12), command=self.check_cluster_score).pack(pady=10)
        
        self.hint_label = tk.Label(self.root, text="", font=(None, 12))
        self.hint_label.pack()
        
        self.root.mainloop()
        
    def search_options(self):
        """
        根据多选框的选择情况确定请求种类
        """
        if self.opt_hits.get():
            self.hits = 2
        else:
            self.hits = 0
        if self.opt_blur.get():
            self.blur = 1
        else:
            self.blur = 0
        if self.opt_kw.get():
            self.kw = 4
        else:
            self.kw = 0
            
    def check_new_searchterm(self):
        """
        用户点击"确认"按钮后，检查输入是否合法
        """
        searchterm = self.new_searchterm_entry.get()
        terms = searchterm.split(' ')
        terms = [i for i in terms if i != '']
        if len(terms) == 0 or len(terms) > 3:
            self.hint_label.config(text=f"请输入1-3个检索词")
        else:
            self.search_request(terms)
    
    def check_cluster_score(self):
        """
        如果是为了查询聚类效果，则调用这个函数
        """
        client = socket.socket()
        client.connect(self.addr)
        terms = [8]
        client.send(encode_data(terms))
        print(f'sent cluster score request')
        score = client.recv(1024)
        score = decode_data(score)
        client.close()
        self.cluster_score_tk = tk.Tk()
        self.cluster_score_tk.geometry("300x300")
        self.cluster_score_tk.title("聚类评价")
        self.show_listbox(self.cluster_score_tk, score)
        
    
    def search_request(self, terms):
        """
        TODO: 请补充实现客户端与服务器端的通信
        
        1. 向服务器发送检索词
        2. 接受服务器返回的检索结果
        """
        type = 0
        type = type | self.hits | self.blur | self.kw
        # xy, x=1--hits, y=1--blur
        terms.insert(0, type)
        client = socket.socket()
        client.connect(self.addr)
        client.send(encode_data(terms))
        print(f'sent keywords: {terms}')
        idx = client.recv(102400)
        idx = decode_data(idx)
        score = idx[-1]
        idx = idx[:-1]
        client.close()
        self.data = pd.read_csv(self.filename)
        self.documents = []
        for i in idx:
            self.documents.append((self.data['title'].iloc[i], self.data['body'].iloc[i]))
        print(f'receive {len(idx)} results')


        # 这里暂且假设获得的检索结果存储在self.documents中，并
        # 且数据格式为[(title1, doc1), (title2, doc2), ...]
        # 具体形式可以自由修改（下面几个函数中的对应内容也需要改一下）
        # 展示检索结果
        if score == -1 and idx[0] == -1:
            self.hint_label.config(text=f"Wrong word to search!")
        else:
            self.hint_label.config(text=f"The score of the result (consistency): \n{score}")
            self.show_titles()
        
    def show_titles(self):
        """
        显示所有相关的文章
        
        1. 显示根据检索词搜索到的所有文章标题，使用滚动条显示（tkinter的Scrollbar控件）
        2. 点击标题，显示文章的具体内容（这里使用了 Listbox 控件的bind方法，动作为 <ListboxSelect>)
        
        """
        self.title_tk = tk.Tk()
        self.title_tk.geometry("300x300")
        self.title_tk.title("检索结果")
        self.show_listbox(self.title_tk, self.documents)
    
    def show_listbox(self, title_tk, documents):
        self.scrollbar = tk.Scrollbar(title_tk)
        self.scrollbar.pack(side='right', fill='both')
        self.listbox = tk.Listbox(title_tk, yscrollcommand=self.scrollbar.set, font=(None, 12))
        
        for doc in documents:
            self.listbox.insert("end", str(doc[0]))
        self.listbox.bind('<<ListboxSelect>>', self.show_content(documents))
        self.listbox.pack(side='left', fill='both', expand=True)
         
    def show_content(self, documents):
        """
        显示文档的具体内容
        """
        def callback(event):
            idx = event.widget.curselection()[0]
            content_tk = tk.Tk()
            content_tk.geometry("300x300")
            content_tk.title("显示全文")
            #print(self.documents[idx])
            text = tk.Text(content_tk, font=(None, 12))
            text.config(spacing1=10)  # 调整一下行间距
            text.config(spacing2=5)
            for item in documents[idx]:
                text.insert("end", str(item) + '\n')
            text["state"] = 'disabled'
            text.pack()
            
        return callback
    
if __name__ == "__main__":
    host = '127.0.0.1'
    port = 5002
    gui = MainPanel(host, port)
    gui.start()