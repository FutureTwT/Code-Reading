import random
import torch
import torch.nn as nn
from torch.autograd import Variable

nodes = [2,3,5]
# (2,3) (2,9) (2,5) (3,8) (8,9)五条边
samp_neighs = [{3,5,9}, {2,8}, {2}]
unique_nodes_list = list(set.union(*samp_neighs))
unique_nodes = {n:i for i,n in enumerate(unique_nodes_list)}
column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]
row_indices = [i for i in range(len(samp_neighs)) for j in range(len(samp_neighs[i]))]
print(column_indices)
print(row_indices)
mask = Variable(torch.zeros(len(samp_neighs), len(unique_nodes)))
mask[row_indices, column_indices] = 1

num_neigh = mask.sum(1, keepdim=True)
print(num_neigh)
# 构造的是2708*1433的嵌入空间，但是这里只输出5个embedding，也就是可以直接访问部分点的嵌入向量
embed_matrix = features = nn.Embedding(2708, 1433)(torch.LongTensor(unique_nodes_list))
print(embed_matrix.size())
