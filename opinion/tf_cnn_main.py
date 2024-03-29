# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 19:44:02 2017

@author: user
"""
import logging
import argparse
import os

import tensorflow as tf
from sklearn.metrics import classification_report
from tqdm import tqdm

from utils.dl_utils import load_dict, read_corpus, batch_yield, read_target_test_corpus, persist, read_dict
from core.tf_text_cnn import TextCNN
from utils.path import MODEL_PATH, DATA_PATH

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
print('-------------------------------------')
print(tf.__version__)


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
    parser.add_argument("--sequence_length", type=int, default=50, help="sequence length")
    parser.add_argument("--vocab_size", type=int, default=150346, help="the num of vocabs")
    parser.add_argument("--embed_size", type=int, default=100, help="embedding size")
    parser.add_argument("--is_training", type=bool, default=True, help='training or not')
    parser.add_argument("--keep_prob", type=float, default=0.9, help='keep prob')
    parser.add_argument("--clip_gradients", type=float, default=5.0, help='clip gradients')
    parser.add_argument("--filter_sizes", type=list, default=[2, 3, 4], help='filter size')
    parser.add_argument("--num_filters", type=int, default=128, help='num filters')
    parser.add_argument('--mode', type=str, default='test', help='train|test|demo|retrain')
    parser.add_argument('--DEMO', type=str, default='1563502540', help='model for test and demo')
    return parser.parse_known_args()


FLAGS, unparsed = args()
word2int, int2word = read_dict()
tag2label = {'0': 0, '1': 1}
int2tag = {l: t for t, l in tag2label.items()}
target_names = ["good", "bad"]


def cfg():
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.2  # need ~700MB GPU memory
    return config


def re_train():
    iter = 5
    iter_size = 10000
    train, dev = read_corpus(random_state=1234, separator='\t', iter=iter, iter_size=iter_size)

    mp = "tf_cnn"
    model_path = os.path.join(MODEL_PATH, mp, 'checkpoints')
    ckpt_file = tf.train.latest_checkpoint(model_path)
    logger.info("load pre-train model from {}".format(ckpt_file))
    textCNN = TextCNN(model_path=ckpt_file,
                      vocab=word2int,
                      tag2label=tag2label,
                      sequence_length=FLAGS.sequence_length,
                      eopches=FLAGS.epoches, )

    saver = tf.compat.v1.train.Saver()

    with tf.compat.v1.Session(config=cfg()) as sess:
        saver.restore(sess, ckpt_file)
        textCNN.set_model_path(model_path=os.path.join(MODEL_PATH, mp))
        textCNN.train(sess, train, dev, shuffle=True, re_train=True)


def train():
    iter = 0
    iter_size = 10000
    train, dev = read_corpus(random_state=1234, separator='\t', iter=iter, iter_size=iter_size)

    mp = "iter_{}_size_{}_epochs_{}".format(str(iter + 1), iter_size, FLAGS.epoches)
    textCNN = TextCNN(
        # model_path=os.path.join(MODEL_PATH, str(int(time.time()))),
        model_path=os.path.join(MODEL_PATH, "tf_cnn", mp),
        vocab=word2int,
        tag2label=tag2label,
        batch_size=FLAGS.batch_size,
        eopches=FLAGS.epoches)
    with tf.compat.v1.Session(config=cfg()) as sess:
        textCNN.train(sess, train, dev, shuffle=True)


def test():
    reply_good, reply_bad, test = read_target_test_corpus()

    model_path = os.path.join(MODEL_PATH, FLAGS.DEMO, 'checkpoints')
    # print(model_path)
    ckpt_file = tf.train.latest_checkpoint(model_path)
    logger.info("load model from {}".format(ckpt_file))

    textCNN = TextCNN(model_path=ckpt_file,
                      vocab=word2int,
                      tag2label=tag2label,
                      sequence_length=FLAGS.sequence_length,
                      eopches=FLAGS.epoches, )

    saver = tf.compat.v1.train.Saver()
    with tf.compat.v1.Session(config=cfg()) as sess:
        print('============= TEST RESULT =============')
        saver.restore(sess, ckpt_file)

        test_batch_size = 5000

        true_y = []
        pred_y = []
        for tx, ty in batch_yield(test, test_batch_size, word2int, tag2label, max_seq_len=FLAGS.sequence_length,
                                  shuffle=True):
            preds = textCNN.predict(sess, tx, demo=False)
            true_y.extend(ty)
            pred_y.extend(preds)
            print()
            print(classification_report(ty, preds, target_names=target_names))
            print()

        print('============= FINAL TEST RESULT =============')
        print(classification_report(true_y, pred_y, target_names=target_names))

        print('\n============= TEST REPLY GOOD RESULT =============')
        true_y = []
        pred_y = []
        test_batch_size = 5000
        for tx, ty in batch_yield(reply_good, test_batch_size, word2int, tag2label, max_seq_len=FLAGS.sequence_length,
                                  shuffle=False):
            preds = textCNN.predict(sess, tx, demo=False)
            true_y.extend(ty)
            pred_y.extend(preds)
        print('============= FINAL TEST REPLY GOOD RESULT =============')
        print(classification_report(true_y, pred_y, target_names=target_names))

        print('\n============= TEST REPLY BAD RESULT =============')
        true_y = []
        pred_y = []
        test_batch_size = 5000
        for tx, ty in batch_yield(reply_bad, test_batch_size, word2int, tag2label, max_seq_len=FLAGS.sequence_length,
                                  shuffle=False):
            preds = textCNN.predict(sess, tx, demo=False)
            true_y.extend(ty)
            pred_y.extend(preds)
        print('============= FINAL TEST REPLY BAD RESULT =============')
        print(classification_report(true_y, pred_y, target_names=target_names))
        # probs = textCNN.predict_prob(sess, inps)
        # for inp, r, prob in zip(inps, results, probs):
        #     print("\n{}".format(inp))
        #     for idx, p in enumerate(prob):
        #         print("\t{} -> {}".format(int2tag[idx], p))
        #     print("\tTag: {}".format(int2tag[r]))


def use_for_tagging():
    """通过预测, 对比标注数据"""

    reply_goods, reply_bads, tests = read_target_test_corpus()
    model_path = os.path.join(MODEL_PATH, FLAGS.DEMO, 'checkpoints')
    ckpt_file = tf.train.latest_checkpoint(model_path)
    logger.info("load model from {}".format(ckpt_file))

    textCNN = TextCNN(model_path=ckpt_file,
                      vocab=word2int,
                      tag2label=tag2label,
                      eopches=FLAGS.epoches, )

    saver = tf.compat.v1.train.Saver()
    reply_goods_errors = []

    with tf.compat.v1.Session(config=cfg()) as sess:
        print('============= TAGGING =============')
        saver.restore(sess, ckpt_file)
        for idx, (line, tag) in tqdm(enumerate(reply_goods)):
            inps = [line.strip()]
            pred = textCNN.predict(sess, inps)[0]
            probs = textCNN.predict_prob(sess, inps)[0]
            # print("\n{}".format(inps))
            # for idx, prob in enumerate(probs):
            #     print("\t{} -> {}".format(int2tag[idx], prob))
            # print("\tTag: {}".format(int2tag[pred]))
            if int2tag[pred] != tag:
                reply_goods_errors.append("{}, {}".format(idx + 1, line))

    persist(reply_goods_errors, os.path.join(DATA_PATH, "reply_goods_errors.txt"))
    print("reply_goods_errors nums:", len(reply_goods_errors))


def demo():
    word2int, int2word = read_dict()

    tag2label = {'0': 0, '1': 1}
    int2tag = {l: t for t, l in tag2label.items()}

    model_path = os.path.join(MODEL_PATH, FLAGS.DEMO, 'checkpoints')
    ckpt_file = tf.train.latest_checkpoint(model_path)
    logger.info("load model from {}".format(ckpt_file))

    textCNN = TextCNN(model_path=ckpt_file,
                      vocab=word2int,
                      tag2label=tag2label,
                      eopches=FLAGS.epoches, )

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
    elif FLAGS.mode == 'retrain':
        re_train()
    elif FLAGS.mode == 'test':
        test()
    elif FLAGS.mode == 'demo':
        demo()
    elif FLAGS.mode == 'tag':
        use_for_tagging()
