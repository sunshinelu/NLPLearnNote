#!/usr/bin/python
#-*-coding:utf-8-*-
'''
Created on 2018-04-25 16:45
@author:wangrs
'''
from seqlib import *
corpuspath = './data/msr.utf8.txt'
input_text = load_file(corpuspath)

#word2vec是一个二维数组
txtwv = [line.split() for line in input_text.split('\n') if line != '']
#word2vec
w2v = trainW2V(txtwv)
w2v.save("./model/word2vector.bin")
