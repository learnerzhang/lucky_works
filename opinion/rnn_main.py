# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 19:44:02 2017

@author: user
"""
import logging
import argparse
import os
import time

import tensorflow as tf
from opinion.model.tf_text_rnn import TextAttRNN
from utils.dl_utils import load_dict, read_corpus
from utils.path import MODEL_PATH

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
print('-------------------------------------')
print(tf.__version__)


def args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summaries_dir", type=str, default="/tmp/optional",
                        help="Path to save summary logs for TensorBoard.")
    parser.add_argument("--epoches", type=int, default=20, help="epoches")
    parser.add_argument("--num_classes", type=int, default=18, help="the nums of classify")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate for opt")
    parser.add_argument("--num_sampled", type=int, default=5, help="samples")
    parser.add_argument("--batch_size", type=int, default=256, help="each batch contains samples")
    parser.add_argument("--decay_steps", type=int, default=1000, help="each steps decay the lr")
    parser.add_argument("--decay_rate", type=float, default=0.9, help="the decay rate for lr")
    parser.add_argument("--sequence_length", type=int, default=50, help="sequence length")
    parser.add_argument("--vocab_size", type=int, default=150346, help="the num of vocabs")
    parser.add_argument("--embed_size", type=int, default=100, help="embedding size")
    parser.add_argument("--is_training", type=bool, default=True, help='training or not')
    parser.add_argument("--keep_prob", type=float, default=0.9, help='keep prob')
    parser.add_argument("--clip_gradients", type=float, default=5.0, help='clip gradients')
    parser.add_argument("--filter_sizes", type=list, default=[2, 3, 4], help='filter size')
    parser.add_argument("--num_filters", type=int, default=128, help='num filters')
    parser.add_argument('--mode', type=str, default='test', help='train|test|demo')
    parser.add_argument('--DEMO', type=str, default='rnn_iter_0_size_10000_epochs_20', help='model for test and demo')
    return parser.parse_known_args()


FLAGS, unparsed = args()


def cfg():
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.2  # need ~700MB GPU memory
    return config


def read_dict():
    word2int, int2word = load_dict()
    vocab_size = len(word2int)
    word2int['_PAD_'], int2word[0] = 0, '_PAD_'
    word2int['_UNK_'], int2word[vocab_size + 1] = vocab_size, '_UNK_'
    return word2int, int2word


def train():
    tag2label = {'0': 0, '1': 1}
    iter = -1
    iter_size = 10000
    train, dev = read_corpus(random_state=1234, separator='\t', iter=iter, iter_size=iter_size)
    word2int, int2word = read_dict()

    mp = "rnn_iter_{}_size_{}_epochs_{}".format(str(iter + 1), iter_size, FLAGS.epoches)
    textCNN = TextAttRNN(config=cfg(),
                         model_path=os.path.join(MODEL_PATH, 'tf_rnn', mp),
                         vocab=word2int,
                         tag2label=tag2label,
                         batch_size=FLAGS.batch_size,
                         eopches=FLAGS.epoches)

    with tf.compat.v1.Session(config=cfg()) as sess:
        textCNN.train(sess, train, dev, shuffle=True)


def test():
    word2int, int2word = read_dict()

    tag2label = {'0': 0, '1': 1}
    int2tag = {l: t for t, l in tag2label.items()}

    model_path = os.path.join(MODEL_PATH, FLAGS.DEMO, 'checkpoints')
    ckpt_file = tf.train.latest_checkpoint(model_path)
    logging.info("load model from {}".format(ckpt_file))

    textCNN = TextAttRNN(config=cfg(),
                         model_path=ckpt_file,
                         vocab=word2int,
                         tag2label=tag2label,
                         eopches=FLAGS.epoches)

    saver = tf.compat.v1.train.Saver()
    with tf.compat.v1.Session(config=cfg()) as sess:
        print('============= demo =============')
        saver.restore(sess, ckpt_file)

        inps = ['卡布奇诺瑞纳冰已经胜过我爱的星爸爸',
                '非常好，店长给我耐心的推荐不同口味，skr', '比app通知的时间晚太多做好',
                '可以，还可以更好！', '巧克力有点腻', '太冰了没有加热', '商品未送到提前点击已送达，且无电话通知。']
        results = textCNN.predict(sess, inps)
        probs = textCNN.predict_prob(sess, inps)
        for inp, r, prob in zip(inps, results, probs):
            print("\n{}".format(inp))
            for idx, p in enumerate(prob):
                print("\t{} -> {}".format(int2tag[idx], p))
            print("\tTag: {}".format(int2tag[r]))


def demo():
    word2int, int2word = read_dict()

    tag2label = {'good': 0, 'bad': 1}
    int2tag = {l: t for t, l in tag2label.items()}

    model_path = os.path.join(MODEL_PATH, FLAGS.DEMO, 'checkpoints')
    ckpt_file = tf.train.latest_checkpoint(model_path)
    logging.info("load model from {}".format(ckpt_file))

    textCNN = TextAttRNN(config=cfg(),
                         model_path=ckpt_file,
                         vocab=word2int,
                         tag2label=tag2label,
                         eopches=FLAGS.epoches)

    saver = tf.compat.v1.train.Saver()
    with tf.compat.v1.Session(config=cfg()) as sess:
        print('============= demo =============')
        saver.restore(sess, ckpt_file)
        while True:
            print('Please input your sentence:')
            inp = input()
            if inp == '' or inp.isspace():
                print('See you next time!')
                break
            else:
                inps = [inp.strip()]
                pred = textCNN.predict(sess, inps)[0]
                probs = textCNN.predict_prob(sess, inps)[0]

                print("\n{}".format(inps))
                for idx, prob in enumerate(probs):
                    print("\t{} -> {}".format(int2tag[idx], prob))
                print("\tTag: {}".format(int2tag[pred]))


if __name__ == '__main__':

    if FLAGS.mode == 'train':
        train()
    elif FLAGS.mode == 'test':
        test()
    elif FLAGS.mode == 'demo':
        demo()
