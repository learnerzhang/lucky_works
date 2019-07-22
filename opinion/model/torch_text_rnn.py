#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019-07-22 17:17
# @Author  : zhangzhen
# @Site    : 
# @File    : torch_text_rnn.py
# @Software: PyCharm
import torch
import torch.nn as nn
import torchvision
from torch.autograd import Variable
import torchvision.transforms as transforms

# Device configuration
from utils.path import MINIST_PATH


class TextAttBiRNN(nn.Module):
    def __init__(self, output_size, vocab_size, bidirectional, embed_dim=300, hidden_size=200, batch_size=128,
                 attention_size=300, sequence_length=50, dropout=0.5, use_cuda=False, ):
        super(TextAttBiRNN, self).__init__()
        self.batch_size = batch_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.use_cuda = use_cuda
        self.sequence_length = sequence_length

        self.lookup_table = nn.Embedding(self.vocab_size, self.embed_dim, padding_idx=0)
        self.lookup_table.weight.data.uniform_(-1., 1.)

        self.layer_size = 1
        self.lstm = nn.LSTM(self.embed_dim,
                            self.hidden_size,
                            self.layer_size,
                            dropout=self.dropout,
                            bidirectional=self.bidirectional)

        if self.bidirectional:
            self.layer_size = self.layer_size * 2
        else:
            self.layer_size = self.layer_size
        self.attention_size = attention_size

        if self.use_cuda:
            self.w_omega = Variable(torch.zeros(self.hidden_size * self.layer_size, self.attention_size).cuda())
            self.u_omega = Variable(torch.zeros(self.attention_size).cuda())
        else:
            self.w_omega = Variable(torch.zeros(self.hidden_size * self.layer_size, self.attention_size))
            self.u_omega = Variable(torch.zeros(self.attention_size))

        self.label = nn.Linear(hidden_size * self.layer_size, output_size)

    def att(self):
        pass

    def forward(self, *input):
        pass


if __name__ == '__main__':
    batch_size = 128
    # 训练数据
    train_dataset = torchvision.datasets.MNIST(root=MINIST_PATH, train=True, transform=transforms.ToTensor(),
                                               download=True)
    # 测试数据
    test_dataset = torchvision.datasets.MNIST(root=MINIST_PATH, train=False, transform=transforms.ToTensor())
    # 训练数据加载器
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    # 测试数据加载器
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
