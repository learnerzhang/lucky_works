#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019-07-25 15:27
# @Author  : zhangzhen
# @Site    : 
# @File    : model.py
# @Software: PyCharm
from keras import Input
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import np_utils, plot_model
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.layers import LSTM, Dense, Embedding, Dropout, Bidirectional, BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.models import Model
from utils.dl_utils import load_dict, dev2vec
import codecs
import os

from utils.path import DATA_PATH, MODEL_PATH


class Lstm:

    def __init__(self, model_path, vocab, tag2label, input_length=128, n_units=50, output_dim=200, epochs=10,
                 batch_size=64):
        self.model_path = model_path
        self.vocab_size = len(vocab)
        self.output_dim = output_dim
        self.label_size = len(tag2label)
        self.batch_size = batch_size
        self.epochs = epochs
        self.input_length = input_length

        x_input = Input(shape=(self.input_length,))
        print(x_input)
        x_input_embed = Embedding(input_dim=self.vocab_size + 1, output_dim=self.output_dim, mask_zero=True)(x_input)
        print(x_input_embed)

        final_output = Bidirectional(LSTM(n_units, return_sequences=False))(x_input_embed)
        print(final_output)
        output_dropout = Dropout(rate=0.2)(final_output)
        print(output_dropout)

        # BatchNormalization()
        output = Dense(units=self.label_size, activation='softmax')(output_dropout)
        print(output)

        self.model = Model(input=x_input, output=output)
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        plot_model(self.model, to_file='model_lstm.png', show_shapes=True)

        self.model.summary()

    def train(self, train_X, train_y, dev_X, dev_y):
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=2, verbose=0),
            ModelCheckpoint(self.model_path, monitor='val_loss', save_best_only=True, verbose=0),
        ]
        self.model.fit(train_X, train_y,
                       validation_data=(dev_X, dev_y),
                       epochs=self.epochs,
                       batch_size=self.batch_size,
                       verbose=1,
                       callbacks=callbacks)

    def predict(self):
        pass


def read_dict():
    word2int, int2word = load_dict()
    vocab_size = len(word2int)
    word2int['_PAD_'], int2word[0] = 0, '_PAD_'
    word2int['_UNK_'], int2word[vocab_size + 1] = vocab_size, '_UNK_'
    return word2int, int2word


def read_data():
    tags = []
    sents = []
    with codecs.open(os.path.join(DATA_PATH, "emergency_train.tsv"), encoding='utf-8', mode='r') as f:
        for line in f.readlines():
            line = line.strip()
            tokens = line.split('\t')
            sents.append(tokens[1])
            tags.append(tokens[0])
    return sents, tags


if __name__ == '__main__':
    import numpy as np

    sequence_length = 65
    word2int, int2word = read_dict()
    tag2label = {'0': 0, '1': 1}
    int2tag = {l: t for t, l in tag2label.items()}

    sents, tags = read_data()
    x = dev2vec(sents, word2int, max_seq_len=sequence_length)
    y = [tag2label[t] for t in tags]

    model_path = os.path.join(MODEL_PATH, 'emergency')
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    model_path += os.sep + 'm.h5'

    x = np.array(x)
    y = np_utils.to_categorical(y)

    print(x.shape, y.shape)
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=421)
    lstm = Lstm(model_path, word2int, tag2label, input_length=sequence_length)
    lstm.train(train_x, train_y, test_x, test_y)
