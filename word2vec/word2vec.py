#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019-07-16 13:36
# @Author  : zhangzhen
# @Site    : 
# @File    : word2vec.py
# @Software: PyCharm
import logging
import argparse

import multiprocessing
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
logging.root.setLevel(level=logging.INFO)


def arg_helps():
    parser = argparse.ArgumentParser(description='Train Word2Vec use customer data')
    parser.add_argument('--inp', default="word2vec.txt", type=str, help='train source file')
    parser.add_argument('--model', default="model.bin", help="save model path")
    parser.add_argument('--w2v', default='w2v.model', type=str, help="save word2vec parameters path")

    parser.add_argument('--dim', default=300, type=int, help="word2vec dim size")
    parser.add_argument('--sg', default=1, type=int, help="Skip-gram:0 / CBOW:1")
    parser.add_argument('--win', default=4, type=int, help="window for scan")
    parser.add_argument('--min_count', default=5, type=int, help="select num bigger then min count")
    parser.add_argument('--lr', default=0.001, type=float, help="learning rate")
    parser.add_argument('--epochs', default=20, type=int, help="epochs for train")

    return parser.parse_known_args()


def train(args):
    """
    LineSentence(inp)：格式简单：一句话=一行; 单词已经过预处理并被空格分隔。
    size：是每个词的向量维度；
    window：是词向量训练时的上下文扫描窗口大小，窗口为5就是考虑前5个词和后5个词；
    min-count：设置最低频率，默认是5，如果一个词语在文档中出现的次数小于5，那么就会丢弃；
    workers：是训练的进程数（需要更精准的解释，请指正），默认是当前运行机器的处理器核数。这些参数先记住就可以了。
    sg ({0, 1}, optional) – 模型的训练算法: 1: skip-gram 语料小 ;   0: CBOW 语料大
    alpha (float, optional) – 初始学习率
    iter (int, optional) – 迭代次数，默认为5

    :param args:
    :return:
    """
    model = Word2Vec(LineSentence(args.inp),
                     size=args.dim,
                     window=args.win,
                     min_count=args.min_count,
                     sg=args.sg,
                     alpha=args.lr,
                     iter=args.epochs,
                     workers=multiprocessing.cpu_count())
    model.save(args.model)
    # 不以C语言可以解析的形式存储词向量
    model.wv.save_word2vec_format(args.w2v, binary=False)


FLAGS, unparsed = arg_helps()
if __name__ == '__main__':
    train(FLAGS)
