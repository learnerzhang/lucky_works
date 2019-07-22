#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019-07-17 18:48
# @Author  : zhangzhen
# @Site    : 
# @File    : tf_text_rnn.py
# @Software: PyCharm
import sys
import time
import numpy as np
import tensorflow as tf
import logging
import os

from sklearn.metrics import classification_report

# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
from tqdm import tqdm

from utils.dl_utils import batch_yield, pad_sequences, dev2vec
from utils.path import MODEL_PATH

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class TextAttRNN:

    def __init__(self, config, model_path, vocab, tag2label, embed_size=300, hidden_size=200, batch_size=64, eopches=10,
                 lr=0.001, decay_rate=0.99, keep_rate=0.5, sequence_length=128,
                 lip_gradients=5.0, decay_steps=10000, l2_lambda=0.0001,
                 initializer=tf.random_normal_initializer(stddev=0.1)):
        self.config = config
        self.model_path = model_path

        self.vocab = vocab
        self.vocab_size = len(self.vocab)

        self.embed_size = embed_size
        self.hidden_size = hidden_size

        self.tag2label = tag2label
        self.num_classes = len(tag2label)
        int2tag = {l: t for t, l in self.tag2label.items()}
        self.target_names = [int2tag[i] for i in range(self.num_classes)]

        self.sequence_length = sequence_length

        self.eopches = eopches
        self.batch_size = batch_size
        self.keep_rate = keep_rate

        self.learning_rate = tf.Variable(lr, trainable=False, name='learning_rate')
        self.learning_rate_decay_half_op = tf.compat.v1.assign(self.learning_rate, self.learning_rate * 0.5)
        self.l2_lambda = l2_lambda

        self.global_step = tf.Variable(0, trainable=False, name="Global_Step")
        self.epoch_step = tf.Variable(0, trainable=False, name="Epoch_Step")
        self.epoch_increment = tf.compat.v1.assign(self.epoch_step, tf.add(self.epoch_step, tf.constant(1)))
        self.decay_steps, self.decay_rate = decay_steps, decay_rate

        self.add_placeholder()
        self.initializer = initializer

        self.logits = self.inference()
        self.possibility = tf.nn.sigmoid(self.logits)
        self.predictions = tf.argmax(self.logits, axis=1, name="predictions")

        self.loss_val = self.loss()
        tf.compat.v1.summary.scalar("losses", self.loss_val)

        self.opt = self.train_opt()

        # tf.argmax(self.logits, 1)-->[batch_size]
        correct_prediction = tf.equal(tf.cast(self.predictions, tf.int32), self.input_y)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="Accuracy")  # shape=()
        tf.compat.v1.summary.scalar("accuracy", self.accuracy)

    def add_placeholder(self):
        self.input_x = tf.compat.v1.placeholder(tf.int32, [None, self.sequence_length], name="input_x")  # X, 动态RNN
        self.input_y = tf.compat.v1.placeholder(tf.int32, [None, ], name="input_y")
        self.dropout_kp = tf.compat.v1.placeholder(tf.float32, name="keep_prob")

    def inference(self):
        with tf.name_scope('embedding'):
            self.Embedding = tf.compat.v1.get_variable("Embedding", shape=[self.vocab_size, self.embed_size],
                                                       initializer=self.initializer)
            self.embedded_inputs = tf.nn.embedding_lookup(self.Embedding, self.input_x)
        with tf.name_scope('biRNN'):
            # 正向
            cell_forward = tf.contrib.rnn.BasicLSTMCell(self.hidden_size)
            cell_forward = tf.contrib.rnn.DropoutWrapper(cell_forward, self.dropout_kp)
            # 反向
            cell_backward = tf.contrib.rnn.BasicLSTMCell(self.hidden_size)
            cell_backward = tf.contrib.rnn.DropoutWrapper(cell_backward, self.dropout_kp)

            output, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_forward, cell_bw=cell_backward,
                                                        inputs=self.embedded_inputs,
                                                        dtype=tf.float32)

            logger.debug("forward output: {}, backward output: {}".format(output[0].shape, output[1].shape))
            output = tf.concat(output, 2)
            logger.debug("Bi-LSTM concat output: {}".format(output.shape))

        with tf.name_scope('attention'):
            r_list = []

            att_w = tf.compat.v1.Variable(
                tf.random.truncated_normal([self.hidden_size * 2, self.embed_size], stddev=0.1), name='att_w')
            att_b = tf.compat.v1.Variable(tf.constant(0.1, shape=[self.embed_size]), name='att_b')

            att_u = tf.compat.v1.Variable(tf.random.truncated_normal([self.embed_size, 1], stddev=0.1), name='att_u')
            # x * w + b -> tmp
            for t in range(self.sequence_length):
                att_hidden = tf.tanh(tf.matmul(output[:, t, :], att_w) + tf.reshape(att_b, [1, -1]))
                att_out = tf.matmul(att_hidden, att_u)
                r_list.append(att_out)

            # total att out
            logit = tf.concat(r_list, axis=1)
            logger.info("sequences att out: {}".format(logit.shape))

            seq_weights = tf.nn.softmax(logit, name='att_softmax')
            logger.info("sequences att softmax out: {}".format(seq_weights.shape))

            # (batch_size, seq_length) -> (batch_size, seq_length, 1)
            seq_weights = tf.reshape(seq_weights, [-1, self.sequence_length, 1])

            # sum_0_29_{att_i * seq_dim_i} -> (batch_size, sum_dim)
            tmp_sum = output * seq_weights
            logger.debug("weight sum shape:{}".format(tmp_sum.shape))
            att_final_out = tf.reduce_sum(tmp_sum, 1)
            logger.debug("att shape: {}".format(seq_weights.shape))
            logger.debug("final out shape: {}".format(att_final_out.shape))

        with tf.name_scope('dropout'):
            self.out_drop = tf.nn.dropout(att_final_out, rate=1 - self.dropout_kp)

        with tf.name_scope('output'):
            w = tf.Variable(tf.random.truncated_normal([self.hidden_size * 2, self.num_classes], stddev=0.1), name='w')
            b = tf.Variable(tf.zeros([self.num_classes]), name='b')
            # (batch, hidden_size* 2) * (hidden_size, num_size)
            logits = tf.matmul(self.out_drop, w) + b
        return logits

    def loss(self):
        with tf.name_scope("loss"):
            logger.info("input-y: {}, logist: {}".format(self.input_y, self.logits))
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_y, logits=self.logits)
            logger.info("losses: {}".format(losses))
            loss = tf.reduce_mean(losses)
            logger.info("loss: {}".format(loss))

            l2_losses = tf.add_n(
                [tf.nn.l2_loss(v) for v in tf.compat.v1.trainable_variables() if 'bias' not in v.name]) * self.l2_lambda
            loss = loss + l2_losses
        return loss

    def train_opt(self):
        learning_rate = tf.compat.v1.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps,
                                                             self.decay_rate, staircase=True)
        train_op = tf.contrib.layers.optimize_loss(self.loss_val, global_step=self.global_step,
                                                   learning_rate=learning_rate, optimizer="Adam")
        return train_op

    def train(self, sess,train, dev, shuffle=True, re_train=False):
        checkpoints_path = None
        saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables())
        # DEV split
        dev_batches = \
            list(batch_yield(dev, 1000, self.vocab, self.tag2label, max_seq_len=self.sequence_length, shuffle=shuffle))
        # DEV padding
        dev_batches = [(pad_sequences(dev_seqs)[0], dev_labels) for (dev_seqs, dev_labels) in dev_batches]

        # with tf.compat.v1.Session(config=self.config) as sess:
        if not re_train:
            sess.run(tf.compat.v1.global_variables_initializer())

        self.merged = tf.compat.v1.summary.merge_all()
        train_writer = tf.compat.v1.summary.FileWriter(self.model_path + os.sep + "summaries" + os.sep + 'train',
                                                       sess.graph)
        test_writer = tf.compat.v1.summary.FileWriter(self.model_path + os.sep + "summaries" + os.sep + 'test')

        for epoch in range(self.eopches):
            num_batches = (len(train) + self.batch_size - 1) // self.batch_size
            st = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

            # 已经完成 token -> id 的转化
            batches = batch_yield(train, self.batch_size, self.vocab, self.tag2label,
                                  max_seq_len=self.sequence_length, shuffle=shuffle)

            for step, (seqs, labels) in enumerate(batches):
                b_x, b_len_x = pad_sequences(seqs, max_sequence_length=self.sequence_length)
                b_y = labels  # PADDING
                sys.stdout.write(' processing: {} batch / {} batches.'.format(step + 1, num_batches) + '\r')
                step_num = epoch * num_batches + step + 1

                summary, loss, acc, _ = sess.run([self.merged, self.loss_val, self.accuracy, self.opt],
                                                 feed_dict={self.input_x: b_x, self.input_y: b_y,
                                                            self.dropout_kp: 1 - self.keep_rate})

                train_writer.add_summary(summary, step_num)
                if step + 1 == 1 or (step + 1) % 100 == 0 or step + 1 == num_batches:
                    logger.info('{} <TRAIN> epoch: {}, step: {}, loss: {:.4}, global_step: {}, acc: {}'
                                .format(st, epoch + 1, step + 1, loss, step_num, acc))

                if step + 1 == num_batches:
                    checkpoints_path = os.path.join(self.model_path, "checkpoints")
                    if not os.path.exists(checkpoints_path):
                        os.makedirs(checkpoints_path)
                    saver.save(sess, checkpoints_path + os.sep + "model", global_step=step_num)

            # DEV
            logger.info('======================validation / test======================')
            _step = (epoch + 1) * num_batches
            y_trues, y_preds = [], []
            tmp_loss, tmp_acc = [], []
            for dev_step, (dev_X, dev_y) in tqdm(enumerate(dev_batches)):
                if dev_step == 0:
                    test_summary, test_loss, test_acc, y_pred = \
                        sess.run([self.merged, self.loss_val, self.accuracy, self.predictions],
                                 feed_dict={self.input_x: dev_X,
                                            self.input_y: dev_y,
                                            self.dropout_kp: 1.0, })
                    test_writer.add_summary(test_summary, _step)
                else:
                    test_loss, test_acc, y_pred = sess.run([self.loss_val, self.accuracy, self.predictions],
                                                           feed_dict={self.input_x: dev_X,
                                                                      self.input_y: dev_y,
                                                                      self.dropout_kp: 1.0, })
                y_trues.extend(dev_y)
                y_preds.extend(y_pred)
                tmp_loss.append(test_loss)
                tmp_acc.append(test_acc)

            logger.info("{} <DEV> epoch: {} | step: {} | loss:{} | acc: {} "
                        .format(st, epoch + 1, _step, np.average(tmp_loss), np.average(tmp_acc)))
            print(classification_report(y_trues, y_preds, target_names=self.target_names))

        logger.info("model save in {}".format(checkpoints_path))

    def predict(self, sess, seqs, demo=True):
        """预测标签"""
        """预测标签"""
        if demo:
            input_X = dev2vec(seqs, word_dict=self.vocab, max_seq_len=self.sequence_length)
        else:
            input_X, _ = pad_sequences(seqs)
        predictions = sess.run(self.predictions, feed_dict={self.input_x: input_X, self.dropout_kp: 1.0})

        return predictions

    def predict_prob(self, sess, seqs, demo=True):
        """预测概率"""
        if demo:
            input_X = dev2vec(seqs, word_dict=self.vocab, max_seq_len=self.sequence_length)
        else:
            input_X, _ = pad_sequences(seqs)
        possibility = sess.run(self.possibility, feed_dict={self.input_x: input_X, self.dropout_kp: 1.0})

        return possibility


if __name__ == '__main__':
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.2  # need ~700MB GPU memory

    model_path = MODEL_PATH + os.sep + str(int(time.time())),
    rnntext = TextAttRNN(config=config, model_path=model_path, vocab={'A': 1, 'B': 2}, tag2label={'1': 1, '0': 0})
