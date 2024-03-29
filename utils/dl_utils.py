#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019-07-16 19:48
# @Author  : zhangzhen
# @Site    : 
# @File    : dl_utils.py
# @Software: PyCharm
import os
import json
from typing import List

import jieba
import codecs
import random
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import re
from utils.langconv import Converter
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


def read_corpus(filename=None, random_state=1234, separator='\t', iter=-1, iter_size=10000, test_size=0.2):
    """
    获取数据, 保证均衡
    :param random_state:
    :param separator:
    :return:
    """
    import collections

    if filename is not None:
        sents = []
        with codecs.open(os.path.join(DATA_PATH, "emergency_train.tsv"), encoding='utf-8', mode='r') as f:
            for line in tqdm(f.readlines()):
                line = line.strip()
                tokens = line.split('\t')
                sents.append((tokens[1], tokens[0]))
        test_size = len(sents) * test_size
        choices = random.sample(list(range(len(sents))), int(test_size))
        dev = [s for i, s in enumerate(sents) if i in choices]
        train = [s for i, s in enumerate(sents) if i not in choices]
        return train, dev

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


def read_dict():
    word2int, int2word = load_dict()
    vocab_size = len(word2int)
    word2int['_PAD_'], int2word[0] = 0, '_PAD_'
    word2int['_UNK_'], int2word[vocab_size + 1] = vocab_size, '_UNK_'
    return word2int, int2word


def read_target_test_corpus(separator='\t', ):
    tests = []
    with codecs.open(DATA_PATH + os.sep + 'test.tsv', encoding='utf-8') as f:
        for line in tqdm(f.readlines()):
            if line.startswith("####") or line.startswith("$$$$"):
                continue
            line = line.strip()
            tokens = line.split(separator)
            if len(tokens) < 2:
                continue
            elif len(tokens) == 2:
                tests.append((tokens[1], tokens[0]))
            else:
                tests.append(("".join(tokens[1:None]), tokens[0]))

    reply_bad = [(''.join(line.split(separator)[1:None]), line.split(separator)[0]) for line in
                 tqdm(codecs.open(DATA_PATH + os.sep + 'target_reply_bad.tsv', encoding='utf-8').readlines()) if
                 len(line.split(separator)) >= 2]
    reply_good = [(''.join(line.split(separator)[1:None]), line.split(separator)[0]) for line in
                  tqdm(codecs.open(DATA_PATH + os.sep + 'target_reply_good.tsv', encoding='utf-8').readlines()) if
                  len(line.split(separator)) >= 2]
    return reply_good, reply_bad, tests


def persist(lines: List, filepath: str):
    codecs.open(filename=filepath, mode='w+', encoding='utf-8').writelines(lines)


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
        return [word_dict['_PAD_']] * max_seq_len

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


def traditional2simplified(sentence):
    '''
    将sentence中的繁体字转为简体字
    :param sentence: 待转换的句子
    :return: 将句子中繁体字转换为简体字之后的句子
    '''
    sentence = Converter('zh-hans').convert(sentence)
    return sentence


if __name__ == '__main__':
    char_dict, char_dict_re = load_dict()
    print("word nums:", len(char_dict))
    no_zh_ch()
    exit(0)
