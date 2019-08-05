#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019-07-25 15:27
# @Author  : zhangzhen
# @Site    : 
# @File    : keras_text_rnn.py
# @Software: PyCharm
from keras import Input
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.engine.saving import load_model
from keras.utils import plot_model
from keras.preprocessing.sequence import pad_sequences
from keras.layers import LSTM, Dense, Embedding, Dropout, Bidirectional, BatchNormalization
from keras.models import Model
import numpy as np
from utils.dl_utils import dev2vec


class Lstm:

    def __init__(self, model_path, vocab, tag2label, input_length=128, n_units=50, output_dim=200, epochs=10,
                 batch_size=64):
        self.model_path = model_path
        self.vocab = vocab
        self.vocab_size = len(self.vocab)
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

        output = BatchNormalization()(output_dropout)
        output = Dense(units=self.label_size, activation='softmax')(output)
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

    def load(self, model_path):
        self.model = load_model(model_path)

    def predict(self, seqs, demo=True):
        if demo:
            input_X = dev2vec(seqs, word_dict=self.vocab, max_seq_len=self.input_length)
        else:
            input_X, _ = pad_sequences(seqs)

        inputs = np.array(input_X)
        outputs = self.model.predict(inputs)
        return outputs
