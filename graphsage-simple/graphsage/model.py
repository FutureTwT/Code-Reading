import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable

import numpy as np
import time
import random
from sklearn.metrics import f1_score
from collections import defaultdict

from graphsage.encoders import Encoder
from graphsage.aggregators import MeanAggregator

"""
Simple supervised GraphSAGE model as well as examples running the model
on the Cora and Pubmed datasets.
"""
# 有监督式GraphSAGE
class SupervisedGraphSage(nn.Module):

    def __init__(self, num_classes, enc):
        super(SupervisedGraphSage, self).__init__()
        # 最后一层函数入口，类似于递归寻找
        self.enc = enc
        # 定义交叉熵
        self.xent = nn.CrossEntropyLoss()
        # 初始化权重
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, enc.embed_dim))
        init.xavier_uniform_(self.weight)

    def forward(self, nodes):
        # k=2 的聚合过程
        embeds = self.enc(nodes)
        # 全连接层
        scores = self.weight.mm(embeds)
        return scores.t()

    # 利用xent返回损失
    def loss(self, nodes, labels):
        scores = self.forward(nodes)
        return self.xent(scores, labels.squeeze())

def load_cora():
    num_nodes = 2708
    num_feats = 1433
    # 构建特征矩阵和标签向量
    feat_data = np.zeros((num_nodes, num_feats))
    labels = np.empty((num_nodes,1), dtype=np.int64)
    node_map = {}
    # 统计所有标签
    label_map = {}
    with open("../cora/cora.content") as fp:
        for i,line in enumerate(fp):
            info = line.strip().split()
            # 除掉第一个元素和最后一个元素，左闭右开
            feat_data[i,:] = list(map(float, info[1:-1]))
            # key: node-id value: 编号
            node_map[info[0]] = i
            # key: label的名字 value: 编号
            if not info[-1] in label_map:
                # 通过长度进行编号
                label_map[info[-1]] = len(label_map)
            labels[i] = label_map[info[-1]]

    adj_lists = defaultdict(set)
    with open("../cora/cora.cites") as fp:
        for i,line in enumerate(fp):
            info = line.strip().split()
            # 将id-id转换为编号-编号
            paper1 = node_map[info[0]]
            paper2 = node_map[info[1]]
            # 无向图方式创建邻接矩阵
            adj_lists[paper1].add(paper2)
            adj_lists[paper2].add(paper1)
    return feat_data, labels, adj_lists

def run_cora():
    np.random.seed(1)
    random.seed(1)
    num_nodes = 2708
    # feat_data: ndarray 2708*1433
    # labels: ndarray 2708*1
    # adj_list: defaultdict
    feat_data, labels, adj_lists = load_cora()
    # 生成2708个嵌入，每一个嵌入1433维度
    features = nn.Embedding(2708, 1433)
    features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)
    # features.cuda()

    # forward流 下面的过程始终不会调用forward函数
    # K = 2
    # 调用MeanAggregator构造函数__init__
    agg1 = MeanAggregator(features, cuda=False)
    # 调用Encoder构造函数__init__
    enc1 = Encoder(features, 1433, 128, adj_lists, agg1, gcn=True, cuda=False)
    # t()转置，利用lambda表达式，输入是nodes，输出为enc1(nodes).t()，但是此处不会调用执行enc1的forward函数，只是把lambda匿名函数句柄传给构造函数
    agg2 = MeanAggregator(lambda nodes : enc1(nodes).t(), cuda=False)
    enc2 = Encoder(lambda nodes : enc1(nodes).t(), enc1.embed_dim, 128, adj_lists, agg2,
            base_model=enc1, gcn=True, cuda=False)
    # Encoder接受采样参数，在forward中传递给聚合层forward函数
    enc1.num_samples = 5
    enc2.num_samples = 5

    # 封装为一个整体的model
    graphsage = SupervisedGraphSage(7, enc2)
    # graphsage.cuda()
    # 生成2708的一个全排列，切分训练集，测试集和验证集
    rand_indices = np.random.permutation(num_nodes)
    test = rand_indices[:1000]
    val = rand_indices[1000:1500]
    # 将ndarray转换为list
    train = list(rand_indices[1500:])
    # 初始化优化器，并且给优化器应该优化的参数
    optimizer = torch.optim.SGD(filter(lambda p : p.requires_grad, graphsage.parameters()), lr=0.7)
    times = []
    for batch in range(1):
        # 取256大小的batch_size
        batch_nodes = train[:256]
        # 随机打乱
        random.shuffle(train)
        # 记录训练时间
        start_time = time.time()
        # 初始化梯度为0
        optimizer.zero_grad()
        # forward + loss 得到正向计算结果
        loss = graphsage.loss(batch_nodes, 
                Variable(torch.LongTensor(labels[np.array(batch_nodes)])))
        # backward 计算出梯度
        loss.backward()
        # 优化器
        optimizer.step()
        end_time = time.time()
        times.append(end_time-start_time)
        # loss是一个torch.Tensor
        print(batch, loss.item())
        # print(type(loss), loss.size())
    '''
    val_output = graphsage.forward(val) 
    print("Validation F1:", f1_score(labels[val], val_output.data.numpy().argmax(axis=1), average="micro"))
    print("Average batch time:", np.mean(times))
    '''
def load_pubmed():
    #hardcoded for simplicity...
    num_nodes = 19717
    num_feats = 500
    feat_data = np.zeros((num_nodes, num_feats))
    labels = np.empty((num_nodes, 1), dtype=np.int64)
    node_map = {}
    with open("pubmed-data/Pubmed-Diabetes.NODE.paper.tab") as fp:
        fp.readline()
        feat_map = {entry.split(":")[1]:i-1 for i,entry in enumerate(fp.readline().split("\t"))}
        for i, line in enumerate(fp):
            info = line.split("\t")
            node_map[info[0]] = i
            labels[i] = int(info[1].split("=")[1])-1
            for word_info in info[2:-1]:
                word_info = word_info.split("=")
                feat_data[i][feat_map[word_info[0]]] = float(word_info[1])
    adj_lists = defaultdict(set)
    with open("pubmed-data/Pubmed-Diabetes.DIRECTED.cites.tab") as fp:
        fp.readline()
        fp.readline()
        for line in fp:
            info = line.strip().split("\t")
            paper1 = node_map[info[1].split(":")[1]]
            paper2 = node_map[info[-1].split(":")[1]]
            adj_lists[paper1].add(paper2)
            adj_lists[paper2].add(paper1)
    return feat_data, labels, adj_lists

def run_pubmed():
    np.random.seed(1)
    random.seed(1)
    num_nodes = 19717
    feat_data, labels, adj_lists = load_pubmed()
    features = nn.Embedding(19717, 500)
    features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)
   # features.cuda()

    agg1 = MeanAggregator(features, cuda=True)
    enc1 = Encoder(features, 500, 128, adj_lists, agg1, gcn=True, cuda=False)
    agg2 = MeanAggregator(lambda nodes : enc1(nodes).t(), cuda=False)
    enc2 = Encoder(lambda nodes : enc1(nodes).t(), enc1.embed_dim, 128, adj_lists, agg2,
            base_model=enc1, gcn=True, cuda=False)
    enc1.num_samples = 10
    enc2.num_samples = 25

    graphsage = SupervisedGraphSage(3, enc2)
#    graphsage.cuda()
    rand_indices = np.random.permutation(num_nodes)
    test = rand_indices[:1000]
    val = rand_indices[1000:1500]
    train = list(rand_indices[1500:])

    optimizer = torch.optim.SGD(filter(lambda p : p.requires_grad, graphsage.parameters()), lr=0.7)
    times = []
    for batch in range(200):
        batch_nodes = train[:1024]
        random.shuffle(train)
        start_time = time.time()
        optimizer.zero_grad()
        loss = graphsage.loss(batch_nodes, 
                Variable(torch.LongTensor(labels[np.array(batch_nodes)])))
        loss.backward()
        optimizer.step()
        end_time = time.time()
        times.append(end_time-start_time)
        print(batch, loss.data[0])

    val_output = graphsage.forward(val) 
    print("Validation F1:", f1_score(labels[val], val_output.data.numpy().argmax(axis=1), average="micro"))
    print("Average batch time:", np.mean(times))

if __name__ == "__main__":
    run_cora()
