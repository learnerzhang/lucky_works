#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019-07-17 13:56
# @Author  : zhangzhen
# @Site    : 
# @File    : tf_text_cnn.py
# @Software: PyCharm
import sys
import time

import tensorflow as tf
import logging
import os
from sklearn.metrics import classification_report
from utils.dl_utils import variable_summaries, batch_yield, pad_sequences, dev2vec

# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class TextCNN:

    def __init__(self, config, model_path, vocab, tag2label, embed_size=300, batch_size=64, eopches=10,
                 sequence_length=50, filter_sizes=[2, 3, 4], num_filters=128, lr=0.001, decay_rate=0.99, keep_rate=0.5,
                 lip_gradients=5.0, decay_steps=10000, initializer=tf.random_normal_initializer(stddev=0.1)):

        self.config = config
        self.model_path = model_path
        # set hyperparamter
        self.tag2label = tag2label
        self.num_classes = len(tag2label)
        int2tag = {l: t for t, l in self.tag2label.items()}
        self.target_names = [int2tag[i] for i in range(self.num_classes)]

        self.batch_size = batch_size
        self.eopches = eopches
        self.sequence_length = sequence_length

        self.vocab = vocab
        self.vocab_size = len(vocab)

        self.embed_size = embed_size
        self.rate = keep_rate

        self.learning_rate = tf.Variable(lr, trainable=False, name="learning_rate")  # ADD learning_rate
        self.learning_rate_decay_half_op = tf.compat.v1.assign(self.learning_rate, self.learning_rate * decay_rate)
        self.filter_sizes = filter_sizes  # it is a list of int. e.g. [3,4,5]
        self.num_filters = num_filters
        self.initializer = initializer
        self.num_filters_total = self.num_filters * len(filter_sizes)  # how many filters totally.
        self.clip_gradients = lip_gradients

        # add placeholder (X,label)
        self.input_x = tf.compat.v1.placeholder(tf.int32, [None, self.sequence_length], name="input_x")  # X
        self.input_y = tf.compat.v1.placeholder(tf.int32, [None], name="input_y")
        self.keep_prob = tf.compat.v1.placeholder(tf.float32, name="keep_prob")

        self.iter = tf.compat.v1.placeholder(tf.int32)  # training iteration
        self.tst = tf.compat.v1.placeholder(tf.bool)

        self.global_step = tf.Variable(0, trainable=False, name="Global_Step")
        self.epoch_step = tf.Variable(0, trainable=False, name="Epoch_Step")
        self.epoch_increment = tf.compat.v1.assign(self.epoch_step, tf.add(self.epoch_step, tf.constant(1)))
        self.b1 = tf.Variable(tf.ones([self.num_filters]) / 10)
        self.b2 = tf.Variable(tf.ones([self.num_filters]) / 10)
        self.decay_steps, self.decay_rate = decay_steps, decay_rate

        self.instantiate_weights()
        self.logits = self.inference()  # [None, self.label_size]. main computation graph is here.

        self.possibility = tf.nn.softmax(self.logits)
        self.predictions = tf.argmax(self.logits, axis=1, name="predictions")

        self.loss_val = self.loss()
        self.train_op = self.opt()

        # tf.argmax(self.logits, 1)-->[batch_size]
        correct_prediction = tf.equal(tf.cast(self.predictions, tf.int32), self.input_y)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="Accuracy")  # shape=()
        tf.compat.v1.summary.scalar("accuracy", self.accuracy)

    def instantiate_weights(self):
        """define all weights here"""
        with tf.name_scope("embedding"):  # embedding matrix
            # [vocab_size,embed_size] tf.random_uniform([self.vocab_size, self.embed_size],-1.0,1.0)
            self.Embedding = tf.compat.v1.get_variable("Embedding", shape=[self.vocab_size, self.embed_size],
                                                       initializer=self.initializer)
            variable_summaries(self.Embedding)

        with tf.name_scope("weights"):
            # [embed_size,label_size]
            self.W_projection = tf.compat.v1.get_variable("W_projection",
                                                          shape=[self.num_filters_total, self.num_classes],
                                                          initializer=self.initializer)
            variable_summaries(self.W_projection)
        with tf.name_scope("biases"):
            # [label_size] #ADD 2017.06.09
            self.b_projection = tf.compat.v1.get_variable("b_projection", shape=[self.num_classes])
            variable_summaries(self.b_projection)

    def inference(self):

        # [None,sentence_length,embed_size]
        self.embedded_words = tf.nn.embedding_lookup(self.Embedding, self.input_x)

        # [None,sentence_length,embed_size,1). expand dimension so meet input requirement of 2d-conv
        self.sentence_embeddings_expanded = tf.expand_dims(self.embedded_words, -1)

        logger.debug("embedded_words %s" % self.embedded_words)
        logger.debug("sentence_embeddings_expanded %s" % self.sentence_embeddings_expanded)

        # conv2d
        pooled_outputs = []
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.name_scope("convolution-pooling-%s" % filter_size):
                filter = tf.compat.v1.get_variable("filter-%s" % filter_size,
                                                   [filter_size, self.embed_size, 1, self.num_filters],
                                                   initializer=self.initializer)
                conv = tf.nn.conv2d(self.sentence_embeddings_expanded, filter, strides=[1, 1, 1, 1], padding="VALID",
                                    name="conv")
                conv, self.update_ema = self.batchnorm(conv, self.tst, self.iter, self.b1)  # TODO remove it temp

                b = tf.compat.v1.get_variable("b-%s" % filter_size, [self.num_filters])  # ADD 2017-06-09
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                pooled = tf.nn.max_pool2d(h, ksize=[1, self.sequence_length - filter_size + 1, 1, 1],
                                          strides=[1, 1, 1, 1], padding='VALID', name="pool")
                logger.debug('pooled: %s' % pooled)
                pooled_outputs.append(pooled)

        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, self.num_filters_total])
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, rate=1 - self.keep_prob)  # [None,num_filters_total]

            W = tf.compat.v1.get_variable("Dense_w_projection", shape=[self.num_filters_total, self.num_filters_total],
                                          initializer=self.initializer)
            b = tf.compat.v1.get_variable("Dense_b_projection", shape=[self.num_filters_total])

            self.h_drop = tf.nn.tanh(tf.matmul(self.h_drop, W) + b)
            # self.h_drop = tf.layers.dense(self.h_drop, self.num_filters_total, activation=tf.nn.tanh, use_bias=True)

        with tf.name_scope("output"):
            # shape:[None, self.num_classes]==tf.matmul([None,self.embed_size],[self.embed_size,self.num_classes])
            logits = tf.matmul(self.h_drop, self.W_projection) + self.b_projection
            tf.compat.v1.summary.histogram("logists", logits)
        return logits

    def batchnorm(self, Ylogits, is_test, iteration, offset, convolutional=False):
        """
        check:https://github.com/martin-gorner/tensorflow-mnist-tutorial/blob/master/mnist_4.1_batchnorm_five_layers_relu.py#L89
        batch normalization: keep moving average of mean and variance. use it as value for BN when training. when prediction, use value from that batch.
        :param Ylogits:
        :param is_test:
        :param iteration:
        :param offset:
        :param convolutional:
        :return:
        """
        # adding the iteration prevents from averaging across non-existing iterations
        exp_moving_avg = tf.train.ExponentialMovingAverage(0.999, iteration)
        bnepsilon = 1e-5
        if convolutional:
            mean, variance = tf.nn.moments(Ylogits, [0, 1, 2])
        else:
            mean, variance = tf.nn.moments(Ylogits, [0])
        update_moving_averages = exp_moving_avg.apply([mean, variance])
        m = tf.cond(is_test, lambda: exp_moving_avg.average(mean), lambda: mean)
        v = tf.cond(is_test, lambda: exp_moving_avg.average(variance), lambda: variance)
        Ybn = tf.nn.batch_normalization(Ylogits, m, v, offset, None, bnepsilon)
        return Ybn, update_moving_averages

    def loss(self, l2_lambda=0.0001):  # 0.001
        with tf.name_scope("loss"):
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_y, logits=self.logits)
            loss = tf.reduce_mean(losses)
            l2_losses = tf.add_n(
                [tf.nn.l2_loss(v) for v in tf.compat.v1.trainable_variables() if 'bias' not in v.name]) * l2_lambda
            loss = loss + l2_losses
        return loss

    def opt(self):
        """based on the loss, use SGD to update parameter"""
        learning_rate = tf.compat.v1.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps,
                                                             self.decay_rate, staircase=True)
        opt = tf.contrib.layers.optimize_loss(self.loss_val, global_step=self.global_step, learning_rate=learning_rate,
                                              optimizer="Adam", clip_gradients=self.clip_gradients)
        return opt

    def train(self, train, dev, shuffle=True):
        checkpoints_path = None
        saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables())

        dev_X = dev2vec([sent_ for (sent_, tag_) in dev], word_dict=self.vocab, max_seq_len=self.sequence_length)
        dev_y = [self.tag2label[tag_] for (sent_, tag_) in dev]

        with tf.compat.v1.Session(config=self.config) as sess:

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
                    b_x, b_len_x = pad_sequences(seqs)
                    b_y = labels  # PADDING

                    sys.stdout.write(' processing: {} batch / {} batches.'.format(step + 1, num_batches) + '\r')
                    step_num = epoch * num_batches + step + 1

                    summary, loss, possibility, W_projection_value, acc, _ = sess.run(
                        [self.merged, self.loss_val, self.possibility, self.W_projection, self.accuracy, self.train_op],
                        feed_dict={self.input_x: b_x,
                                   self.input_y: b_y,
                                   self.keep_prob: 1 - self.rate,
                                   self.tst: False})

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
                logger.info('====================== validation / test ======================')
                test_summary, test_loss, test_acc, y_pred = sess.run(
                    [self.merged, self.loss_val, self.accuracy, self.predictions],
                    feed_dict={self.input_x: dev_X,
                               self.input_y: dev_y,
                               self.keep_prob: 1.0,
                               self.tst: True})

                _step = (epoch + 1) * num_batches
                test_writer.add_summary(test_summary, _step)
                logger.info("{} <DEV> epoch: {} | step: {} | loss:{} | acc: {} ".format(st, epoch + 1, _step, test_loss,
                                                                                        test_acc))
                print(classification_report(dev_y, y_pred, target_names=self.target_names))

        logger.info("model save in {}".format(checkpoints_path))

    def predict(self, sess, seqs, demo=True):
        """预测标签"""
        if demo:
            input_X = dev2vec(seqs, word_dict=self.vocab, max_seq_len=self.sequence_length)
        else:
            input_X, _ = pad_sequences(seqs)
        predictions = sess.run(self.predictions, feed_dict={self.input_x: input_X, self.keep_prob: 1.0, self.tst: True})
        return predictions

    def predict_prob(self, sess, seqs, demo=True):
        """预测概率"""
        if demo:
            input_X = dev2vec(seqs, word_dict=self.vocab, max_seq_len=self.sequence_length)
        else:
            input_X, _ = pad_sequences(seqs)
        possibility = sess.run(self.possibility, feed_dict={self.input_x: input_X, self.keep_prob: 1.0, self.tst: True})
        return possibility
