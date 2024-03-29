#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019-07-16 14:56
# @Author  : zhangzhen
# @Site    : 
# @File    : data.py
# @Software: PyCharm
import re

import numpy as np
import collections
import codecs
import pickle
import itertools
import json
import jieba
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm

root = '/Users/zhangzhen/Downloads/luckin/p'

jieba.load_userdict(os.path.join(root, 'data', 'user.dict'))

mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # enable chinese
mpl.rcParams['font.size'] = 10

# SQL 导出
# root = 'E:\data\luckin\p'
type_file = '60353-数据反馈类型查询.csv'
feedback_file = '60356-意见反馈查询.csv'
comment_file = '60360-订单评论查询.csv'

# 初次整理
comment_good = "good.txt"
comment_bad = "bad.txt"

comments = "all_comments.txt"
opinions = "all_opinions.txt"
emergency_opinion = "emergency_train.tsv"

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
    """订单评论信息处理
        对 csv 数据处理.MD, 分别区分各类数据, 并去除
    """
    # level,$0, label,$1, comment,$2, reply,$3, remark
    # level 0 -> 差评，level 1 -> 好评
    #
    bad_comments = []
    good_comments = []
    bad_good_comments = []
    bad_bad_comments = []
    good_count = 0
    bad_count = 0
    good_reply_count, bad_reply_count = 0, 0

    with codecs.open(root + os.sep + comment_file, encoding='utf-8') as f:
        count = 0
        tmp = ""
        for line in tqdm(f.readlines()):
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
                            good_reply_count += 1
                            bad_good_comments.append(comment + "\n")
                            # real_bad_comments.append(comment + "\n")
                        else:
                            good_comments.append(comment + "\n")
                    elif level == '0':
                        """非满意"""
                        bad_count += 1
                        if len(reply) > 0:
                            bad_reply_count += 1
                            bad_bad_comments.append(comment + "\n")
                        else:
                            bad_comments.append(comment + "\n")
                tmp = ""

    print("[*] total good comments: {}, total bad comments: {}".format(good_count, bad_count))

    print("[*] good cmt reply num: {}, bad cmt reply num: {}".format(good_reply_count, bad_reply_count))
    print("[*] no_reply good comments:{}, no_reply bad comments:{}".format(len(good_comments), len(bad_comments)))
    print("[*] reply good comments:{}, reply bad comments:{}".format(len(bad_good_comments), len(bad_bad_comments)))

    # Filter text
    good_comments = list(set(good_comments))
    bad_comments = list(set(bad_comments))
    bad_good_comments = list(set(bad_good_comments))
    bad_bad_comments = list(set(bad_bad_comments))

    print(
        "[*] filter no_reply good comments:{}, no_reply bad comments:{}".format(len(good_comments), len(bad_comments)))
    print("[*] filter reply good comments:{}, reply bad comments:{}".format(len(bad_good_comments),
                                                                            len(bad_bad_comments)))

    codecs.open(os.path.join(root, "good.txt"), mode='w+', encoding='utf-8').writelines(good_comments)
    codecs.open(os.path.join(root, "bad.txt"), mode='w+', encoding='utf-8').writelines(bad_comments)
    codecs.open(os.path.join(root, "reply_good.txt"), mode='w+', encoding='utf-8').writelines(bad_good_comments)
    codecs.open(os.path.join(root, "reply_bad.txt"), mode='w+', encoding='utf-8').writelines(bad_bad_comments)


def split_unit(goods, bads, start=0, end=None, span=2000):
    tmp = []
    tmp.extend(["0\t{}".format(line) for line in goods[start:end]])
    tmp.extend(["1\t{}".format(line) for line in bads[start:end]])
    print("\nunit num: {} \n\t{}\n\t{}".format(len(tmp), tmp[0], tmp[-1]))
    np.random.shuffle(tmp)

    results = []
    for i, line in tqdm(enumerate(tmp)):
        results.append(line)
        # 添加span分割标识
        if (i + 1) % span == 0:
            results.append(SPAN.format((i + 1) / span))
    return results


def corpus_split_train_dev_test(random_state=1234):
    """
    生成统一的语料
    文本格式:
        Label: 0 -> good; 1 -> bad
        label_\t_text
    out:
    train x,
    dev 2万,
    test 2万,

    每隔2000行添加 $$$$$$$$$$$$/##########分割
    """
    bads = codecs.open(os.path.join(root, "bad.txt"), encoding='utf-8').readlines()
    goods = codecs.open(os.path.join(root, "good.txt"), encoding='utf-8').readlines()
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


def gen_test_target():
    reply_bads = codecs.open(os.path.join(root, "reply_bad.txt"), encoding='utf-8').readlines()
    reply_goods = codecs.open(os.path.join(root, "reply_good.txt"), encoding='utf-8').readlines()
    codecs.open(root + os.sep + "target_reply_good.tsv", encoding='utf-8', mode='w+') \
        .writelines(["1\t{}".format(line) for line in tqdm(reply_goods)])
    codecs.open(root + os.sep + "target_reply_bad.tsv", encoding='utf-8', mode='w+') \
        .writelines(["1\t{}".format(line) for line in tqdm(reply_bads)])


def gen_word2id(mode='all'):
    """
    根据语料生成字典
    :param mode: all, char, word
    :return:
    """
    vocabularySet = set()
    for file in [comments, opinions]:
        with codecs.open(root + os.sep + file, encoding='utf-8') as f:
            for line in tqdm(f.readlines()):
                line = line.strip()
                if '$0' in line:
                    tokens = line.split('$0')
                    line = tokens[1]
                elif '$' in line:
                    tokens = line.split('$')
                    line = tokens[1]

                if mode == 'char':
                    tokens = list(line)
                elif mode == 'word':
                    tokens = list(jieba.lcut(line))
                elif mode == 'all':
                    tokens = list(line) + list(jieba.lcut(line))

                vocabularySet = vocabularySet.union(set(tokens))
                # print(vocabularySet)

    print("Total {}.dict: {}".format(mode, len(vocabularySet)))
    # 0 for PAD
    vocab2int = {voc: idx + 1 for idx, voc in enumerate(vocabularySet)}
    json.dump(vocab2int, codecs.open(os.path.join(root, 'data', 'input', "{}.dict".format(mode)),
                                     mode='w+', encoding='utf-8'), ensure_ascii=True)


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


def static_length():
    """
    统计意见, 评论的长短分布
    :return:
    """
    # TAG = "意见反馈"
    TAG = "订单评论"
    TAG = "紧急工单"
    len_counts = collections.defaultdict(int)
    # with codecs.open(os.path.join(root, opinions), encoding='utf-8') as f:
    # with codecs.open(os.path.join(root, comments), encoding='utf-8') as f:
    with codecs.open(os.path.join(root, emergency_opinion), encoding='utf-8') as f:
        lines = [line.strip() for line in f.readlines() if line.strip() != '']
        lines = list(set(lines))
        for line in lines:
            tokens = line.strip().split('\t')
            if len(tokens[1]) > 0:
                len_counts[len(tokens[1])] += 1

    counts = sorted(len_counts.items(), key=lambda item: item[0])
    sent_length = [c[0] for c in counts]
    sent_freq = [c[1] for c in counts]

    # 绘制句子长度及出现频数统计图
    plt.bar(sent_length, sent_freq)
    plt.title("{}_句子长度及出现频数统计图".format(TAG), )
    plt.xlabel("长度", )
    plt.ylabel("频数", )
    plt.savefig("{}_句子长度及出现频数统计图.png".format(TAG))
    plt.close()

    # 绘制句子长度累积分布函数(CDF)
    sent_pentage_list = [(count / sum(sent_freq)) for count in itertools.accumulate(sent_freq)]

    # 绘制CDF
    plt.plot(sent_length, sent_pentage_list)

    # 寻找分位点为quantile的句子长度
    quantile = 0.91
    # print(list(sent_pentage_list))
    for length, per in zip(sent_length, sent_pentage_list):
        if round(per, 2) == quantile:
            index = length
            break
    print("\n分位点为%s的句子长度:%d." % (quantile, index))

    # 绘制句子长度累积分布函数图
    plt.plot(sent_length, sent_pentage_list)
    plt.hlines(quantile, 0, index, colors="c", linestyles="dashed")
    plt.vlines(index, 0, quantile, colors="c", linestyles="dashed")
    plt.text(0, quantile, str(quantile))
    plt.text(index, 0, str(index))
    plt.title("{}_句子长度累积分布函数图".format(TAG))
    plt.xlabel("长度")
    plt.ylabel("累积频率")
    plt.savefig("{}_句子长度累积分布函数图.png".format(TAG))
    plt.show()
    plt.close()


def no_zh_ch():
    fil = re.compile(u'[^0-9a-zA-Z\u4e00-\u9fa5.，,。“”]', re.UNICODE)
    # fil.findall()
    rs = []
    import collections
    for file in [comments, opinions]:
        with codecs.open(root + os.sep + file, encoding='utf-8') as f:
            for line in tqdm(f.readlines()):
                line = line.strip()
                if '$0' in line:
                    tokens = line.split('$0')
                    line = tokens[1]
                elif '$' in line:
                    tokens = line.split('$')
                    line = tokens[1]

                r = fil.findall(line)
                if r:
                    rs.extend(r)
    counter = collections.Counter(rs)
    print(counter)
    codecs.open(os.path.join(root, "no_normal_char.txt"), encoding='utf-8', mode='w+').writelines(sorted(set(rs)))


if __name__ == '__main__':
    # static_length()
    # read_feedback_opinions()
    # read_comments()
    # corpus_split_train_dev_test()  # 分割语料
    # gen_test_target()
    # gen_word2id(mode='all')
    no_zh_ch()
    # vocab2int, int2vocab, label2int, int2label, (_data, _labels) = read_corpus()
    # print(_data[0], _labels[0])
    #
    # X_train, X_test, y_train, y_test = train_test_split(_data, _labels, test_size=0.33, random_state=42)
    # print(len(X_train), len(y_train))
    # print(len(X_test), len(y_test))
    exit(0)
