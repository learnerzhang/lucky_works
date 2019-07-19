#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019-07-16 19:48
# @Author  : zhangzhen
# @Site    : 
# @File    : dl_utils.py
# @Software: PyCharm
import os
import json
import jieba
import codecs
import random
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from utils.path import DATA_PATH


def conlleval(label_predict, label_path, metric_path):
    """

    :param label_predict:
    :param label_path:
    :param metric_path:
    :return:
    """
    eval_perl = "./conlleval_rev.pl"
    with open(label_path, "w") as fw:
        line = []
        for sent_result in label_predict:
            for char, tag, tag_ in sent_result:
                tag = '0' if tag == 'O' else tag
                char = char.encode("utf-8")
                line.append("{} {} {}\n".format(char, tag, tag_))
            line.append("\n")
        fw.writelines(line)
    os.system("perl {} < {} > {}".format(eval_perl, label_path, metric_path))
    with open(metric_path) as fr:
        metrics = [line.strip() for line in fr]
    return metrics


def read_corpus(random_state=1234, separator='\t', iter=-1, iter_size=10000):
    """
    获取数据, 保证均衡
    :param random_state:
    :param separator:
    :return:
    """
    import collections
    tmp_results = collections.defaultdict(list)
    with codecs.open(DATA_PATH + os.sep + 'train.tsv', encoding='utf-8') as f:
        for line in tqdm(f.readlines()):
            if line.startswith("####") or line.startswith("$$$$"):
                continue
            line = line.strip()
            tokens = line.split(separator)
            if len(tokens) < 2:
                continue
            elif len(tokens) == 2:
                tmp_results[tokens[0]].append((tokens[1], tokens[0]))
            else:
                tmp_results[tokens[0]].append(("".join(tokens[1:None]), tokens[0]))

    # all or part
    train = []
    np.random.seed(random_state)
    for k, v in tmp_results.items():
        if iter < 0:
            train.extend(random.sample(v, iter_size))
        else:
            start = iter * iter_size
            train.extend(v[start: start + iter_size])

    np.random.shuffle(train)
    # DEV
    dev = [(''.join(line.split(separator)[1:None]), line.split(separator)[0]) for line in
           tqdm(codecs.open(DATA_PATH + os.sep + 'dev.tsv', encoding='utf-8').readlines()) if
           not (line.startswith("####") or line.startswith("$$$$")) and len(line.split(separator)) >= 2]
    return train, dev


def read_test_corpus(separator='\t', ):
    results = []
    with codecs.open(DATA_PATH + os.sep + 'test.tsv', encoding='utf-8') as f:
        for line in tqdm(f.readlines()):
            if line.startswith("####") or line.startswith("$$$$"):
                continue
            line = line.strip()
            tokens = line.split(separator)
            if len(tokens) < 2:
                continue
            elif len(tokens) == 2:
                results.append((tokens[1], tokens[0]))
            else:
                results.append(("".join(tokens[1:None]), tokens[0]))
    # np.random.seed(random_state)
    # np.random.shuffle(results)
    return results


def batch_yield(data, batch_size, vocab, tag2label, max_seq_len=128, shuffle=False):
    """
    :param data:
    :param batch_size:
    :param vocab:
    :param tag2label:
    :param shuffle:
    :return:
    """
    if shuffle:
        np.random.shuffle(data)

    seqs, labels = [], []
    for (sent_, tag_) in tqdm(data):
        sent_ = word2id(sent_, vocab, max_seq_len=max_seq_len)
        label_ = tag2label[tag_]

        if len(seqs) == batch_size:
            yield seqs, labels
            seqs, labels = [], []
        seqs.append(sent_)
        labels.append(label_)

    if len(seqs) != 0:
        yield seqs, labels


def pad_sequences(sequences, pad_mark=0, max_sequence_length=50):
    """
    :param sequences:
    :param pad_mark:
    :return:
    """
    max_len = max_sequence_length if max_sequence_length > 0 else max(map(lambda x: len(x), sequences))
    seq_list, seq_len_list = [], []
    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_len] + [pad_mark] * max(max_len - len(seq), 0)
        seq_list.append(seq_)
        seq_len_list.append(min(len(seq), max_len))
    return seq_list, seq_len_list


def data_process(text_str):
    if len(text_str) == 0:
        print('[ERROR] data_process failed! | The params: {}'.format(text_str))
        return None
    text_str = text_str.strip().replace('\s+', ' ', 3)
    return jieba.lcut(text_str)


def load_dict():
    char_dict_re = dict()
    dict_path = os.path.join(DATA_PATH, 'words.dict')
    with open(dict_path, encoding='utf-8') as fin:
        char_dict = json.load(fin)
    for k, v in char_dict.items():
        char_dict_re[v] = k
    return char_dict, char_dict_re


def dev2vec(dev, word_dict, max_seq_len=128):
    return [word2id(text, word_dict, max_seq_len=max_seq_len) for text in dev]


def word2id(text_str, word_dict, max_seq_len=128):
    if len(text_str) == 0 or len(word_dict) == 0:
        print('[ERROR] word2id failed! | The params: {} and {}'.format(text_str, word_dict))
        return None

    sent_list = data_process(text_str)
    sent_ids = list()
    for item in sent_list:
        if item in word_dict:
            sent_ids.append(word_dict[item])
        else:
            sent_ids.append(word_dict['_UNK_'])

    if len(sent_ids) < max_seq_len:
        sent_ids = sent_ids + [word_dict['_PAD_'] for _ in range(max_seq_len - len(sent_ids))]
    else:
        sent_ids = sent_ids[:max_seq_len]
    return sent_ids


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor"""
    with tf.name_scope("summaries"):
        mean = tf.reduce_mean(var)
        tf.compat.v1.summary.scalar("mean", mean)

        with tf.name_scope("stddev"):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))

        tf.compat.v1.summary.scalar("stddev", stddev)
        tf.compat.v1.summary.scalar("max", tf.reduce_mean(var))
        tf.compat.v1.summary.scalar("min", tf.reduce_min(var))
        tf.compat.v1.summary.histogram("histogram", var)
