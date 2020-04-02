from _collections import defaultdict
import numpy as np

adj_lists = defaultdict(set)
adj_lists[1].add(2)
adj_lists[1].add(3)
adj_lists[2].add(1)
print(adj_lists.keys())

num_nodes = 2708
rand_indices = np.random.permutation(num_nodes)
print(rand_indices)
test = rand_indices[:1000]
val = rand_indices[1000:1500]
train = list(rand_indices[1500:])
# print(type(test), train)

import torch
import torch.nn as nn
import torch.nn.functional as F
r = torch.randn(2,3)
# print(r.item()) # Error，只有单元素tensor才可以调用item函数
print(r, r.t())

class TestModel(nn.Module):
    def __init__(self):
        super(TestModel, self).__init__()

    def forward(self):
        print('forward')

test = TestModel()
# 默认调用forward函数
test() # 等价于test.forward()

import torch.nn.init as init
# 类型转换函数
weights = nn.Parameter(torch.FloatTensor([[1,2,3],[4,5,6],[7,8,9]]))
print(weights)
init.xavier_normal_(weights)
print(weights)