#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019-07-16 11:36
# @Author  : zhangzhen
# @Site    : 
# @File    : utils.py
# @Software: PyCharm
import jieba

jieba.load_userdict("user_dict.txt")


def tihuan_tongyici(string1):
    # tongyici_tihuan.txt是同义词表，每行是一系列同义词，用tab分割
    # 1读取同义词表：并生成一个字典。
    combine_dict = {}
    for line in open("tongyici.txt", "r", encoding='utf-8'):
        # print(line)
        seperate_word = line.strip().split("$")
        num = len(seperate_word)
        for i in range(1, num):
            combine_dict[seperate_word[i]] = seperate_word[0]

    print("同义词典:", combine_dict)
    # 2提升某些词的词频，使其能够被jieba识别出来
    # jieba.suggest_freq("年假", tune=True)

    # 3将语句切分
    seg_list = jieba.cut(string1, cut_all=False)
    f = "$".join(seg_list)  # 不用utf-8编码的话，就不能和tongyici文件里的词对应上
    # print(f)
    # 4
    final_sentence = ""
    for word in f.split("$"):
        if word in combine_dict:
            word = combine_dict[word]
            final_sentence += word
        else:
            final_sentence += word
    # print final_sentence
    return final_sentence


text = "漏了！奶和糖纸巾都没给！"
text = "未经同意，自己确定商品已送达！未按时间送达，延迟10分钟以上！"
text = "接受不了你家这个配送，态度不好不说，还总撒漏，赶快给我解决一下吧，哎，倒霉"
text = "洒漏，而且快递小哥还没到就打电话叫下楼取。。。"
text = "咖啡洒出来了"
text = "●°u°●​」奶思"
text = "作为老客户 咖啡现在没有洒漏赔券后 撒漏现象越来越严重了 希望好好处理这个问题"
# print(tihuan_tongyici(text))

rs = jieba.lcut(text)
print(list(rs))
