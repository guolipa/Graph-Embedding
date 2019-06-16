import math
import time
import threading
import numpy as np
import networkx as nx
from  evaluate import evaluate_embeddings, plot_embeddings

#map node in graph to id
def nodes_map(graph):
    node2id  = {}
    id2node = []
    num = 0
    for node in graph.nodes():
        node2id[node] = num
        id2node.append(node)
        num += 1
    return node2id, id2node

def creat_alias_table(prob_dist):
    '''
    :param prob_dist: Probability distributions
    :return: prob, alias
    '''
    N = len(prob_dist)
    norm_prob = np.array(prob_dist) * N
    prob, alias = [0] * N, [0] * N
    small, large = [], []
    for i, p in enumerate(norm_prob):
        if p >= 1.0:
            large.append(i)
        else:
            small.append(i)
    while small and large:
        small_index, large_index = small.pop(), large.pop()
        prob[small_index] = norm_prob[small_index]
        alias[small_index] = large_index
        norm_prob[large_index] = norm_prob[large_index] - (1.0 - norm_prob[small_index])
        if norm_prob[large_index] >= 1.0:
            large.append(large_index)
        else:
            small.append(large_index)
    while large:
        prob[large.pop()] = 1
    while small:
        prob[small.pop()] = 1
    return prob, alias

#alias sample
def alias_sample(prob, alias):
    N = len(prob)
    i = int(np.random.random()*N)
    p = np.random.random()
    if p <= prob[i]:
        return i
    else:
        return alias[i]

def cosine_similarity(a,b):
    dot = np.dot(a,b)
    normal = np.sqrt(sum(np.power(a,2))) * np.sqrt(sum(np.power(b,2)))
    sim = dot / normal
    return sim

class LINE(object):
    def __init__(self, graph, embedding_size=10, negative_num=5, order=2, thread_nums=5, init_rho=0.025, m=10):
        self.graph = graph
        self.node_nums = graph.number_of_nodes()
        self.edge_nums = graph.number_of_edges()
        self.node2id, self.id2node = nodes_map(self.graph)
        self.edges = []

        self.embedding_size = embedding_size
        self.negative_num = negative_num
        self.order = order
        self.init_rho = init_rho
        self.rho = 0.025

        self.thread_nums = thread_nums
        self.total_samples = graph.number_of_edges() * m
        self.current_sample_count = 0

        self.node_emb = {}
        self.context_emb = {}

        self.node_prob = []
        self.node_alias = []
        self.edge_prob = []
        self.edge_alias = []

    #Initial node and context vector
    def init_embedding(self):
        for i in range(self.node_nums):
            self.node_emb[i] = (np.random.random(self.embedding_size) - 0.5) / self.embedding_size
            self.context_emb[i] = np.zeros(self.embedding_size)

    # Create an alias sampling table for nodes and edges
    def init_alias_table(self):
        power = 0.75
        # edge sample table
        self.edges = [(self.node2id[x[0]], self.node2id[x[1]]) for x in self.graph.edges()]
        total_sum = sum([self.graph[self.id2node[edge[0]]][self.id2node[edge[1]]].get('weight', 1.0) for edge in self.edges])
        edge_prob_dist = [self.graph[self.id2node[edge[0]]][self.id2node[edge[1]]].get('weight', 1.0) / total_sum for edge in self.edges]
        self.edge_prob, self.edge_alias = creat_alias_table(edge_prob_dist)
        # node sample table
        node_degree = np.zeros(self.node_nums)
        node2id = self.node2id
        for edge in self.graph.edges():
            node_degree[node2id[edge[0]]] += self.graph[edge[0]][edge[1]].get('weight', 1.0)
        total_sum = sum([math.pow(node_degree[i], power) for i in range(self.node_nums)])
        node_prob_dist = [math.pow(node_degree[i], power) / total_sum for i in range(self.node_nums)]
        self.node_prob, self.node_alias = creat_alias_table(node_prob_dist)

    def sigmoid(self, x):
        return 1.0 / (1 + np.exp(-x))

    #Parameter update
    def update_parameter(self, vec_u, vec_v, vec_error, label):
        x = np.dot(vec_u, vec_v)
        g = (label - self.sigmoid(x)) * self.rho
        vec_error += g * vec_v
        vec_v += g * vec_u
        return vec_v, vec_error

    def train_line_thread(self):
        count = 0
        last_count = 0
        while (True):
            if count > self.total_samples / self.thread_nums + 2:
                break

            if count - last_count > 10000:
                self.current_sample_count += (count - last_count)
                last_count = count
                self.rho = self.init_rho * (1 - self.current_sample_count / (self.total_samples + 1))
                if self.rho < self.init_rho * 0.0001:
                    self.rho = self.init_rho * 0.0001

            curedge = alias_sample(self.edge_prob, self.edge_alias)  # 随机取一条边
            u = self.edges[curedge][0]
            v = self.edges[curedge][1]

            vec_error = np.zeros(self.embedding_size)

            # Negative Sample
            for n in range(self.negative_num + 1):
                if n == 0:
                    target = v
                    label = 1
                else:
                    target = alias_sample(self.node_prob, self.node_alias)
                    if target == v:
                        target = alias_sample(self.node_prob, self.node_alias)
                    label = 0
                if self.order == 1:
                    self.node_emb[target], vec_error = self.update_parameter(self.node_emb[u], self.node_emb[target], vec_error, label)
                if self.order == 2:
                    self.context_emb[target], vec_error = self.update_parameter(self.node_emb[u], self.context_emb[target], vec_error, label)
            self.node_emb[u] += vec_error
            count += 1

    def train(self):
        # 创建多线程
        threads = []
        print('Model initialization......')
        # 初始化
        self.init_embedding()
        self.init_alias_table()
        print('Model initialization completed......')
        for i in range(self.thread_nums):
            thread = threading.Thread(target=self.train_line_thread())
            threads.append(thread)
        print('Start training.......')
        start = time.time()
        for i in range(self.thread_nums):
            threads[i].start()
        for i in range(self.thread_nums):
            threads[i].join()
        end = time.time()
        print('Train Finished......')
        print('Total time: %d' % (end - start))

    def get_embedding(self, c='node', normalize=True):
        '''
        c: 'node' or 'context'
        normalize: 表示是否对向量归一化
        '''
        self.embedding = {}
        self.normal_embedding = {}
        id2node = self.id2node
        # 返回节点向量还是context向量
        if c == 'node':
            embeddings = self.node_emb
        else:
            embeddings = self.context_emb
        for i, e in embeddings.items():
            self.embedding[id2node[i]] = e
        # 是否对向量归一化
        if normalize:
            for i, e in self.embedding.items():
                s = sum(np.power(e, 2))
                normal = np.sqrt(s)
                self.normal_embedding[i] = e / normal
            return self.normal_embedding
        return self.embedding

if __name__ == '__main__':
    file = r'D:\Recommender-System\Graph Embedding\Wiki_edgelist.txt'
    filename = r'D:\Recommender-System\Graph Embedding\wiki_labels.txt'
    G = nx.read_edgelist(file, create_using=nx.DiGraph(), nodetype=None,data=[('weight', int)])
    model = LINE(graph = G,embedding_size=100, negative_num=5,order=2, thread_nums = 5, init_rho = 0.025, m = 300)
    model.train()
    embeddings = model.get_embedding(normalize = False)
    evaluate_embeddings(embeddings, filename)
    plot_embeddings(embeddings, filename)