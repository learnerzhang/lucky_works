# -*- coding: utf-8 -*
import sys

import os

# 训练数据的路径
DATA_PATH = os.path.join(sys.path[1], 'data', 'input')
DATA_PATH = '/Users/zhangzhen/Downloads/luckin/p'
# DATA_PATH = 'E:\data\luckin\p'
# 模型保存的路径
MODEL_PATH = os.path.join(DATA_PATH, 'data', 'output', 'model')
# MODEL_PATH = '/Users/zhangzhen/Downloads/luckin/p'
# 训练log的输出路径
LOG_PATH = os.path.join(DATA_PATH, 'data', 'output', 'logs')

# print(sys.path)
# print(DATA_PATH, '\n', MODEL_PATH, '\n', LOG_PATH)
MINIST_PATH = '/Users/zhangzhen/data/minist'
