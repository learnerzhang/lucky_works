#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019-07-25 15:27
# @Author  : zhangzhen
# @Site    : 
# @File    : keras_main.py
# @Software: PyCharm
import argparse
import os
import logging
import numpy as np
from keras.utils import np_utils
from sklearn.model_selection import train_test_split

from core.keras_text_rnn import Lstm
from emergency_identification import read_data
from utils.dl_utils import read_dict, dev2vec
from utils.path import MODEL_PATH

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
print('-------------------------------------')


def args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summaries_dir", type=str, default="/tmp/optional",
                        help="Path to save summary logs for TensorBoard.")
    parser.add_argument("--epoches", type=int, default=10, help="epoches")
    parser.add_argument("--num_classes", type=int, default=18, help="the nums of classify")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate for opt")
    parser.add_argument("--num_sampled", type=int, default=5, help="samples")
    parser.add_argument("--batch_size", type=int, default=128, help="each batch contains samples")
    parser.add_argument("--decay_steps", type=int, default=1000, help="each steps decay the lr")
    parser.add_argument("--decay_rate", type=float, default=0.9, help="the decay rate for lr")
    parser.add_argument("--sequence_length", type=int, default=65, help="sequence length")
    parser.add_argument("--vocab_size", type=int, default=150346, help="the num of vocabs")
    parser.add_argument("--embed_size", type=int, default=100, help="embedding size")
    parser.add_argument("--is_training", type=bool, default=True, help='training or not')
    parser.add_argument("--keep_prob", type=float, default=0.9, help='keep prob')
    parser.add_argument("--clip_gradients", type=float, default=5.0, help='clip gradients')
    parser.add_argument("--filter_sizes", type=list, default=[2, 3, 4], help='filter size')
    parser.add_argument("--num_filters", type=int, default=128, help='num filters')
    parser.add_argument('--mode', type=str, default='train', help='train|test|demo|retrain')
    parser.add_argument('--DEMO', type=str, default='keras_emergency', help='model for test and demo')
    return parser.parse_known_args()


FLAGS, unparsed = args()

word2int, int2word = read_dict()
tag2label = {'0': 0, '1': 1}
int2tag = {l: t for t, l in tag2label.items()}


def train():
    sents, tags = read_data()
    x = dev2vec(sents, word2int, max_seq_len=FLAGS.sequence_length)
    y = [tag2label[t] for t in tags]

    model_path = os.path.join(MODEL_PATH, FLAGS.DEMO)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    model_path = os.path.join(model_path, 'm.h5')

    x = np.array(x)
    y = np_utils.to_categorical(y)

    print(x.shape, y.shape)
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=421)
    lstm = Lstm(model_path, word2int, tag2label, input_length=FLAGS.sequence_length)
    lstm.train(train_x, train_y, test_x, test_y)


def demo():

    model_path = os.path.join(MODEL_PATH, FLAGS.DEMO, 'm.h5')
    lstm = Lstm(model_path, word2int, tag2label, input_length=FLAGS.sequence_length)
    lstm.load(model_path)

    print('============= demo =============')
    while True:
        print('Please input your sentence:')
        inp = input()
        if inp == '' or inp.isspace():
            print('See you next time!')
            break
        else:
            inps = [inp.strip()]
            probs = lstm.predict(inps)[0]
            print("\n{}".format(inps))
            pred = np.argmax(probs, axis=0)
            for idx, prob in enumerate(probs):
                print("\t{} -> {}".format(int2tag[idx], prob))
            print("\tTag: {}".format(int2tag[pred]))


if __name__ == '__main__':

    if FLAGS.mode == 'train':
        train()
    elif FLAGS.mode == 'demo':
        demo()
