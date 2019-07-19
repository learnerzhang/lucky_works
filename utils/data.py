#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019-07-16 14:56
# @Author  : zhangzhen
# @Site    : 
# @File    : data.py
# @Software: PyCharm
import numpy as np
import collections
import codecs
import pickle
import json
import jieba
import os

from sklearn.model_selection import train_test_split
from tqdm import tqdm

root = '/Users/zhangzhen/Downloads/luckin/p'
# SQL 导出
type_file = '60353-数据反馈类型查询.csv'
feedback_file = '60356-意见反馈查询.csv'
comment_file = '60360-订单评论查询.csv'

# 初次整理
comment_good = "comment_good.txt"
comment_bad = "comment_bad.txt"
SPAN = "###############=============={}===============\n"

def optional_labels():
    int2labels = collections.defaultdict(str)

    with codecs.open(root + os.sep + type_file, encoding='utf-8') as f:
        count = 0
        for line in f.readlines():
            count += 1
            if count == 1:
                continue
            line = line.strip()
            tokens1 = line.split('$0')
            tokens2 = tokens1[1].split('$1')
            key = tokens1[0][:-1]
            label = tokens2[0][1:-1]
            # print(key, '$', label)
            int2labels[key] = label
    return int2labels


def read_feedback_opinions():
    """反馈意见处理"""
    int2tags = optional_labels()
    results = []
    with codecs.open(root + os.sep + feedback_file, encoding='utf-8') as f:
        count = 0
        tmp = ""
        for line in tqdm(f.readlines()):
            count += 1
            if count == 1:
                continue
            line = line.strip()
            tmp += line
            if "$0" in line:
                tokens = tmp.split('$0')
                tmp = ""
                assert len(tokens) == 2, "NOT FORMAT"
                content = tokens[0][:-1]
                intlabels = tokens[1][1:]
                if ',' in intlabels:
                    tmp_labels = intlabels[1:-1]
                    tag = "|".join([int2tags.get(l) for l in tmp_labels.split(',') if int2tags.get(l) is not None])
                    # print(tmp_labels)
                else:
                    tag = int2tags.get(intlabels)
                    if tag is None:
                        tag = "UNK"
                # print(tag, "$", content)
                results.append("{}${}\n".format(tag, content))

    # 保存本地
    codecs.open("all_optional.txt", mode='w+', encoding='utf-8').writelines(results, )


def read_comments():
    """订单评论信息处理"""
    # level,$0, label,$1, comment,$2, reply,$3, remark
    # level 0 -> 差评，level 1 -> 好评
    #
    real_bad_comments = []
    real_good_comments = []
    feak_good_comments = []
    good_count = 0
    bad_count = 0
    with codecs.open(root + os.sep + comment_file, encoding='utf-8') as f:
        count = 0
        tmp = ""
        for line in f.readlines():
            count += 1
            if count == 1:
                continue
            line = line.strip()

            tmp += " " + line
            if "$0" in tmp and "$1" in tmp and "$2" in tmp and "$3" in tmp:
                token0 = tmp.split('$0')  # level
                token1 = token0[1].split('$1')  # label
                token2 = token1[1].split('$2')  # comment
                token3 = token2[1].split('$3')  # reply, remark

                level = token0[0][:-1].strip()
                label = token1[0][1:-1]
                comment = token2[0][1:-1]
                reply = token3[0][1:-1]
                remark = token3[1][1:]
                if len(comment) > 1:
                    # print("level:{}\n\tlabel:{}\n\tcomment:{}\n\treply:{}\n\tremark:{}".format(level, label, comment, reply, remark))
                    if level == '1':
                        """满意"""
                        good_count += 1
                        # 根据
                        if len(reply) > 0:
                            feak_good_comments.append(comment + "\n")
                            # real_bad_comments.append(comment + "\n")
                        else:
                            real_good_comments.append(comment + "\n")
                    elif level == '0':
                        """非满意"""
                        bad_count += 1
                        real_bad_comments.append(comment + "\n")
                tmp = ""

    print("good comments: {}, bad comments: {}".format(good_count, bad_count))
    print("real good comments:{}, real bad comments:{}".format(len(real_good_comments), len(real_bad_comments)))

    codecs.open("bad.txt", mode='w+', encoding='utf-8').writelines(real_bad_comments)
    codecs.open("good.txt", mode='w+', encoding='utf-8').writelines(real_good_comments)
    codecs.open("bad.txt", mode='w+', encoding='utf-8').writelines(real_bad_comments)


def filter_comments() -> None:
    """对导出的评论数据 过滤"""
    results = []
    with codecs.open(os.path.join(root, comment_good), encoding='utf-8') as f:
        for line in tqdm(f.readlines()):
            results.append(line)

    print(len(set(results)), len(results))
    codecs.open(os.path.join(root, "filter_comment_good.txt"), mode='w+', encoding='utf-8').writelines(
        list(set(results)))

    results = []
    with codecs.open(os.path.join(root, comment_bad), encoding='utf-8') as f:
        for line in tqdm(f.readlines()):
            results.append(line)

    print(len(set(results)), len(results))
    codecs.open(os.path.join(root, "filter_comment_bad.txt"), mode='w+', encoding='utf-8').writelines(
        list(set(results)))


def split_unit(goods, bads, start=0, end=None, span=2000):
    tmp = []
    tmp.extend(["good\t{}".format(line) for line in goods[start:end]])
    tmp.extend(["bad\t{}".format(line) for line in bads[start:end]])
    print("\nunit num: {} \n\t{}\n\t{}".format(len(tmp), tmp[0], tmp[-1]))
    np.random.shuffle(tmp)

    results = []
    for i, line in tqdm(enumerate(tmp)):
        results.append(line)
        # 添加span分割标识
        if (i + 1) % span == 0:
            results.append(SPAN.format((i+1)/span))
    return results


def corpus_split_train_dev_test(random_state=1234):
    """
    生成统一的语料
    文本格式:
        label_\t_text
    out:
    train x,
    dev 2万,
    test 2万,

    每隔2000行添加 $$$$$$$$$$$$/##########分割
    """
    bads = codecs.open(os.path.join(root, "filter_comment_bad.txt"), encoding='utf-8').readlines()
    goods = codecs.open(os.path.join(root, "filter_comment_good.txt"), encoding='utf-8').readlines()
    print("Good comments nums: {}, \nBad comments nums: {}".format(len(goods), len(bads)))

    np.random.seed(random_state)

    print("\nBefore shuffle \n\tgood:{}".format(goods[0].strip()))
    print("\tbad: {}".format(bads[0].strip()))
    np.random.shuffle(goods)
    np.random.shuffle(bads)
    print("\nAfter shuffle \n\tgood:{}".format(goods[0].strip()))
    print("\tbad: {}".format(bads[0].strip()))

    # choice
    # DEV
    dev = split_unit(goods, bads, start=-20000, end=None)
    # TEST
    test = split_unit(goods, bads, start=-40000, end=-20000)
    # TRAIN
    train = split_unit(goods, bads, start=None, end=-40000)

    # codecs.open("dev.txt", encoding='utf-8', mode='w+').writelines(results)
    codecs.open(root + os.sep + "dev.tsv", encoding='utf-8', mode='w+').writelines(dev)
    codecs.open(root + os.sep + "test.tsv", encoding='utf-8', mode='w+').writelines(test)
    codecs.open(root + os.sep + "train.tsv", encoding='utf-8', mode='w+').writelines(train)


def gen_word2id():
    """根据语料生成字典"""
    vocabularySet = set()
    with codecs.open('dev.txt', encoding='utf-8') as f:
        for line in tqdm(f.readlines()):
            line = line.strip()
            tokens = list(jieba.lcut(line))
            vocabularySet = vocabularySet.union(set(tokens))
    print("Total words:{}".format(len(vocabularySet)))
    # 0 -> PAD
    vocab2int = {voc: idx + 1 for idx, voc in enumerate(vocabularySet)}
    json.dump(vocab2int, codecs.open("words.dict", mode='w+', encoding='utf-8'), ensure_ascii=True)


def read_corpus(sequence_length=50, sparse=True):
    vocabularySet = set()
    labels = ['good', 'bad']
    data = []
    with codecs.open('bad.txt', encoding='utf-8') as f:
        label = 'bad'
        for line in tqdm(f.readlines()[:10000]):
            line = line.strip()
            tokens = list(jieba.lcut(line))
            data.append((tokens, label))
            vocabularySet = vocabularySet.union(set(tokens))

    with codecs.open('good.txt', encoding='utf-8') as f:
        label = 'good'
        for line in tqdm(f.readlines()[:10000]):
            line = line.strip()
            tokens = list(jieba.lcut(line))
            data.append((tokens, label))
            vocabularySet = vocabularySet.union(set(tokens))

    print("total word num:", len(vocabularySet), "total class num:", 2)

    vocab2int = {voc: idx + 1 for idx, voc in enumerate(vocabularySet)}
    int2vocab = {idx: voc for voc, idx in vocab2int.items()}
    vocab2int['PAD'] = 0
    int2vocab[0] = 'PAD'

    label2int = {label: idx for idx, label in enumerate(labels)}
    int2label = {idx: label for label, idx in label2int.items()}

    #
    _data = []
    _labels = []
    for (d, l) in data:
        tmp = []
        for token in d:
            tmp.append(vocab2int[token])
        _data.append(tmp)
        _labels.append(l)
    assert len(_data) == len(_labels), "LEN ERROR"

    X = []
    for tokens in _data:
        if len(tokens) > sequence_length:
            X.append(tokens[:sequence_length])
        else:
            X.append(tokens + (sequence_length - len(tokens)) * [0])

    if sparse:
        y = [label2int[label] for label in _labels]
    else:
        y = []
        for label in labels:
            tmp = np.zeros(len(label2int))
            tmp[label2int[label]] = 1
            y.append(tmp)

    assert len(X) == len(y)
    return vocab2int, int2vocab, label2int, int2label, (X, y)


if __name__ == '__main__':
    # read_feedback_opinions()
    read_comments()
    # filter_comments()
    # corpus_split_train_dev_test()  # 分割语料
    # gen_word2id()

    # vocab2int, int2vocab, label2int, int2label, (_data, _labels) = read_corpus()
    # print(_data[0], _labels[0])
    #
    # X_train, X_test, y_train, y_test = train_test_split(_data, _labels, test_size=0.33, random_state=42)
    # print(len(X_train), len(y_train))
    # print(len(X_test), len(y_test))
    exit(0)
