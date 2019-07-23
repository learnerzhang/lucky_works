# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 19:44:02 2017

@author: user
"""
import logging
import argparse
import os
import time
import torch.nn as nn
import torch
from sklearn.metrics import classification_report
from tqdm import tqdm

from opinion.model.torch_text_rnn import TextAttBiRNN
from utils.dl_utils import load_dict, read_corpus, batch_yield, read_target_test_corpus, persist
from utils.path import MODEL_PATH, DATA_PATH

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
    parser.add_argument("--sequence_length", type=int, default=50, help="sequence length")
    parser.add_argument("--vocab_size", type=int, default=150346, help="the num of vocabs")
    parser.add_argument("--embed_size", type=int, default=100, help="embedding size")
    parser.add_argument("--is_training", type=bool, default=True, help='training or not')
    parser.add_argument("--keep_prob", type=float, default=0.9, help='keep prob')
    parser.add_argument("--clip_gradients", type=float, default=5.0, help='clip gradients')
    parser.add_argument("--filter_sizes", type=list, default=[2, 3, 4], help='filter size')
    parser.add_argument("--num_filters", type=int, default=128, help='num filters')
    parser.add_argument('--mode', type=str, default='train', help='train|test|demo|retrain')
    parser.add_argument('--DEMO', type=str, default='iter_1_size_10000_epochs_10', help='model for test and demo')
    return parser.parse_known_args()


FLAGS, unparsed = args()


def read_dict():
    word2int, int2word = load_dict()
    vocab_size = len(word2int)
    word2int['_PAD_'], int2word[0] = 0, '_PAD_'
    word2int['_UNK_'], int2word[vocab_size + 1] = vocab_size, '_UNK_'
    return word2int, int2word


def re_train():
    word2int, int2word = read_dict()

    tag2label = {'0': 0, '1': 1}
    iter = 2
    iter_size = 10000
    train, dev = read_corpus(random_state=1234, separator='\t', iter=iter, iter_size=iter_size)

    mp = os.path.join("iter_{}_size_{}_epochs_{}".format(str(iter - 1), iter_size, FLAGS.epoches), "model_3.pth")
    model_path = os.path.join(MODEL_PATH, "torch_rnn", mp)

    logger.info("load pre-train model from {}".format(model_path))
    textRNN = TextAttBiRNN(model_path=model_path,
                         vocab=word2int,
                         tag2label=tag2label,
                         bidirectional=True,
                         sequence_length=FLAGS.sequence_length,
                         epoches=FLAGS.epoches,
                         batch_size=FLAGS.batch_size,
                         layer_size=2, )

    textRNN.load_state_dict(torch.load(model_path), strict=False)

    next_mp = "iter_{}_size_{}_epochs_{}".format(str(iter), iter_size, FLAGS.epoches)
    textRNN.set_model_path(model_path=os.path.join(MODEL_PATH, "torch_rnn", next_mp))

    textRNN.train(train, dev, shuffle=True, re_train=True)


def train():
    tag2label = {'0': 0, '1': 1}
    iter = 0
    iter_size = 10000
    train, dev = read_corpus(random_state=1234, separator='\t', iter=iter, iter_size=iter_size)
    word2int, int2word = read_dict()

    mp = "iter_{}_size_{}_epochs_{}".format(str(iter + 1), iter_size, FLAGS.epoches)

    model = TextAttBiRNN(model_path=os.path.join(MODEL_PATH, "torch_rnn", mp),
                         vocab=word2int,
                         tag2label=tag2label,
                         bidirectional=True,
                         sequence_length=FLAGS.sequence_length,
                         epoches=FLAGS.epoches,
                         batch_size=FLAGS.batch_size,
                         layer_size=2, )
    model.train(train, dev, shuffle=True)


def test():
    word2int, int2word = read_dict()
    reply_good, reply_bad, test = read_target_test_corpus()
    # 测试数据
    tag2label = {'0': 0, '1': 1}
    target_names = ['good', 'bad']
    model_path = os.path.join(MODEL_PATH, "torch_rnn", FLAGS.DEMO, "model_3.pth")
    logger.info("load pre-train model from {}".format(model_path))
    # model_path = os.path.join(MODEL_PATH, mp),
    textRNN = TextAttBiRNN(model_path=model_path,
                           vocab=word2int,
                           tag2label=tag2label,
                           bidirectional=True,
                           sequence_length=FLAGS.sequence_length,
                           epoches=FLAGS.epoches,
                           batch_size=FLAGS.batch_size,
                           layer_size=2, )

    # textRNN = nn.DataParallel(textRNN)
    logger.info("Load torch model from {}".format(model_path))
    textRNN.load_state_dict(torch.load(model_path), strict=False)
    print('============= TEST RESULT =============')

    test_batch_size = 5000

    true_y = []
    pred_y = []
    for tx, ty in batch_yield(test, test_batch_size, word2int, tag2label, max_seq_len=FLAGS.sequence_length,
                              shuffle=True):
        preds = textRNN.predict(tx, demo=False)
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
        preds = textRNN.predict(tx, demo=False)
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
        preds = textRNN.predict(tx, demo=False)
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
    word2int, int2word = read_dict()

    tag2label = {'0': 0, '1': 1}
    int2tag = {l: t for t, l in tag2label.items()}

    reply_goods, reply_bads, tests = read_target_test_corpus()
    model_path = os.path.join(MODEL_PATH, FLAGS.DEMO, 'checkpoints')

    logger.info("load model from {}".format(model_path))

    textCNN = TextAttBiRNN(model_path=model_path,
                           vocab=word2int,
                           tag2label=tag2label,
                           eopches=FLAGS.epoches, )

    # reply_goods_errors = []
    #
    # with tf.compat.v1.Session(config=cfg()) as sess:
    #     print('============= TAGGING =============')
    #     saver.restore(sess, ckpt_file)
    #     for idx, (line, tag) in tqdm(enumerate(reply_goods)):
    #         inps = [line.strip()]
    #         pred = textCNN.predict(sess, inps)[0]
    #         probs = textCNN.predict_prob(sess, inps)[0]
    # print("\n{}".format(inps))
    # for idx, prob in enumerate(probs):
    #     print("\t{} -> {}".format(int2tag[idx], prob))
    # print("\tTag: {}".format(int2tag[pred]))
    # if int2tag[pred] != tag:
    #     reply_goods_errors.append("{}, {}".format(idx + 1, line))
    #
    # persist(reply_goods_errors, os.path.join(DATA_PATH, "reply_goods_errors.txt"))
    # print("reply_goods_errors nums:", len(reply_goods_errors))


def demo():
    word2int, int2word = read_dict()

    tag2label = {'0': 0, '1': 1}
    int2tag = {l: t for t, l in tag2label.items()}

    model_path = os.path.join(MODEL_PATH, FLAGS.DEMO, 'checkpoints')
    logger.info("load model from {}".format(model_path))

    textAttRNN = TextAttBiRNN(model_path=model_path,
                              vocab=word2int,
                              tag2label=tag2label,
                              eopches=FLAGS.epoches, )

    print('============= demo =============')
    while True:
        print('Please input your sentence:')
        inp = input()
        if inp == '' or inp.isspace():
            print('See you next time!')
            break
        else:
            inps = [inp.strip()]
            pred = textAttRNN.predict(inps)[0]
            probs = textAttRNN.predict_prob(inps)[0]
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
