#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019-07-22 17:17
# @Author  : zhangzhen
# @Site    : 
# @File    : torch_text_rnn.py
# @Software: PyCharm
import sys
import time
import codecs
import os
import torch
import torch.nn as nn
import torchvision
from sklearn.metrics import classification_report
from torch.autograd import Variable
import torchvision.transforms as transforms
import logging
# Device configuration
from tqdm import tqdm
import numpy as np
from utils.dl_utils import load_dict, batch_yield, dev2vec, pad_sequences
from utils.path import MODEL_PATH

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class TextAttBiRNN(nn.Module):
    def __init__(self, model_path, vocab, tag2label, bidirectional, layer_size=1, embed_dim=200, hidden_size=100,
                 learning_rate=0.001, attention_size=200, sequence_length=50, batch_size=128, epoches=20, dropout=0.5,
                 use_cuda=False, ):
        super(TextAttBiRNN, self).__init__()

        self.model_path = model_path
        self.vocab = vocab
        self.vocab_size = len(self.vocab)

        self.tag2label = tag2label
        self.output_size = len(self.tag2label)
        int2tag = {l: t for t, l in self.tag2label.items()}
        self.target_names = [int2tag[i] for i in range(self.output_size)]

        self.hidden_size = hidden_size

        self.embed_dim = embed_dim
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.use_cuda = use_cuda
        self.attention_size = attention_size
        self.sequence_length = sequence_length
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epoches = epoches

        self.lookup_table = nn.Embedding(self.vocab_size, self.embed_dim, padding_idx=0)
        self.lookup_table.weight.data.uniform_(-1., 1.)
        self.layer_size = layer_size
        self.lstm = nn.LSTM(self.embed_dim,
                            self.hidden_size,
                            self.layer_size,
                            dropout=self.dropout,
                            bidirectional=self.bidirectional)

        self.num_directions = 2 if self.bidirectional else 1

        if self.use_cuda:
            self.w_omega = Variable(torch.zeros(self.hidden_size * self.num_directions, self.attention_size).cuda())
            self.u_omega = Variable(torch.zeros(self.attention_size).cuda())
        else:
            # w (hidden_size * layer_size * num_directions, attention_size)
            # u (attention_size)
            self.w_omega = Variable(torch.zeros(self.hidden_size * self.num_directions, self.attention_size))
            self.u_omega = Variable(torch.zeros(self.attention_size))

        self.label = nn.Linear(hidden_size * self.num_directions, self.output_size)

    def set_model_path(self, model_path):
        self.model_path = model_path

    def att(self, lstm_output):
        # output: (seq_len, batch, hidden_size * num_directions)
        _batch_size = lstm_output.size()[1]

        assert lstm_output.size() == (
            self.sequence_length, _batch_size, self.hidden_size * self.num_directions), lstm_output.size()
        # (batch_size * sequence_length, hidden_size* num_directions* num_directions)
        output_reshape = torch.Tensor.reshape(lstm_output, [-1, self.hidden_size * self.num_directions])

        attn_tanh = torch.tanh(torch.mm(output_reshape, self.w_omega))
        assert attn_tanh.size() == (self.sequence_length * _batch_size, self.attention_size)

        attn_hidden_layer = torch.mm(attn_tanh, torch.Tensor.reshape(self.u_omega, [-1, 1]))
        assert attn_hidden_layer.size() == (self.sequence_length * _batch_size, 1)

        exps = torch.Tensor.reshape(torch.exp(attn_hidden_layer), [-1, self.sequence_length])
        assert exps.size() == (_batch_size, self.sequence_length)

        # softmax
        alphas = exps / torch.Tensor.reshape(torch.sum(exps, dim=1), [-1, 1])
        assert alphas.size() == (_batch_size, self.sequence_length)

        alphas_reshape = torch.Tensor.reshape(alphas, [-1, self.sequence_length, 1])
        assert alphas_reshape.size() == (_batch_size, self.sequence_length, 1)

        state = lstm_output.permute(1, 0, 2)
        assert state.size() == (_batch_size, self.sequence_length, self.hidden_size * self.num_directions)

        att_output = torch.sum(state * alphas_reshape, 1)
        assert att_output.size() == (_batch_size, self.hidden_size * self.num_directions)

        return att_output

    def forward(self, *input):
        x_input = self.lookup_table(input[0])
        _batch_size = x_input.size()[0]

        # input: (seq_len, batch, input_size)
        x_input = x_input.permute(1, 0, 2)
        if self.use_cuda:
            h_0 = Variable(torch.zeros(self.layer_size * self.num_directions, _batch_size, self.hidden_size).cuda())
            c_0 = Variable(torch.zeros(self.layer_size * self.num_directions, _batch_size, self.hidden_size).cuda())
        else:
            h_0 = Variable(torch.zeros(self.layer_size * self.num_directions, _batch_size, self.hidden_size))
            c_0 = Variable(torch.zeros(self.layer_size * self.num_directions, _batch_size, self.hidden_size))

        # h0: (num_layers * num_directions, batch, hidden_size)
        # c0: (num_layers * num_directions, batch, hidden_size)
        lstm_output, (final_hidden_state, final_cell_state) = self.lstm(x_input, (h_0, c_0))
        # output: (seq_len, batch, hidden_size * num_directions)
        # hn: (num_layers * num_directions, batch, hidden_size)
        # cn: (num_layers * num_directions, batch, hidden_size)
        # print(lstm_output.shape, final_hidden_state.shape, final_cell_state.shape)
        att_output = self.att(lstm_output)
        logits = self.label(att_output)
        return logits

    def predict(self, seqs, demo=True):
        if demo:
            input_X = dev2vec(seqs, word_dict=self.vocab, max_seq_len=self.sequence_length)
        else:
            input_X, _ = pad_sequences(seqs)

        inputs = torch.LongTensor(input_X)
        outputs = self(inputs)
        return torch.argmax(outputs, dim=1).numpy()

    def predict_prob(self, seqs, demo=True):
        if demo:
            input_X = dev2vec(seqs, word_dict=self.vocab, max_seq_len=self.sequence_length)
        else:
            input_X, _ = pad_sequences(seqs)

        inputs = torch.LongTensor(input_X)
        outputs = torch.exp(self(inputs))
        # softmax
        outputs_softmax = outputs / torch.Tensor.reshape(torch.sum(outputs, dim=1), [-1, 1])
        return outputs_softmax.detach().numpy()

    def train(self, train, dev, shuffle=True, re_train=False):
        # 定义损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        # DEV split
        dev_batches = list(
            batch_yield(dev, 5000, self.vocab, self.tag2label, max_seq_len=self.sequence_length, shuffle=shuffle))

        for epoch in range(self.epoches):
            st = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            batches = batch_yield(train, self.batch_size, self.vocab, self.tag2label, max_seq_len=self.sequence_length,
                                  shuffle=shuffle)
            num_batches = (len(train) + self.batch_size - 1) // self.batch_size
            for step, (seqs, labels) in enumerate(batches):
                sys.stdout.write(' processing: {} batch / {} batches.'.format(step + 1, num_batches) + '\r')
                # b_x, b_len_x = pad_sequences(seqs, max_sequence_length=sequence_length)
                # 前向传播
                labels = torch.LongTensor(labels)
                inputs = torch.LongTensor(seqs)
                outputs = self(inputs)
                loss = criterion(outputs, labels)

                # 反向传播和优化，注意梯度每次清零
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if step == 0 or (step + 1) % 200 == 0 or step + 1 == num_batches:
                    preds = torch.argmax(outputs, dim=1)
                    batch_acc_nums = (preds == labels).sum()
                    # print(labels.size(), preds.size(), preds == labels)
                    logger.info('{} Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Batch Acc: {:.4f}'
                                .format(st, epoch + 1, self.epoches, step + 1, num_batches, loss.item(),
                                        batch_acc_nums.numpy() / len(seqs)))
                    # print(classification_report(labels, preds, target_names=self.target_names))

            logger.info('====================== validation / test ======================')
            _step = (epoch + 1) * num_batches
            y_trues, y_preds = [], []
            tmp_loss, tmp_acc = [], []
            for dev_step, (dev_X, dev_y) in tqdm(enumerate(dev_batches)):
                dev_inputs = torch.LongTensor(dev_X)
                dev_labels = torch.LongTensor(dev_y)

                outputs = self(dev_inputs)
                test_loss = criterion(outputs, dev_labels)
                y_pred = torch.argmax(outputs, dim=1)
                test_acc = (y_pred == dev_labels).sum().numpy() / len(dev_X)

                y_trues.extend(dev_y)
                y_preds.extend(y_pred)

                tmp_loss.append(test_loss.item())
                tmp_acc.append(test_acc)

            test_loss = np.average(tmp_loss)
            test_acc = np.average(tmp_acc)

            if not os.path.exists(self.model_path):
                os.makedirs(self.model_path)
            torch.save(self.state_dict(), os.path.join(self.model_path, "model.pth"))

            # print(sum(y_trues), "|", sum(y_preds), "|", sum(tmp_loss), "|", sum(tmp_acc), "|", )
            logger.info("{} <DEV> epoch: {} | step: {} | loss:{} | acc: {} "
                        .format(st, epoch + 1, _step, test_loss, test_acc))
            print(classification_report(y_trues, y_preds, target_names=self.target_names))
