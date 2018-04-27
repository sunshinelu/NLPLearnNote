#!/usr/bin/python
#-*-coding:utf-8-*-
'''
Created on 2018-04-26 8:43
@author:wangrs
'''
from seqlib import *
corpuspath = './data/msr.utf8.txt'
input_text = load_file(corpuspath)

#计算词频
txtnltk = [w for w in input_text.split()] #为计算词频准备的文本格式
freqdf = freq_func(txtnltk)
#print(freqdf)

#建立两个映射词典
word2idx = dict((c,i) for c,i in zip(freqdf['word'],freqdf['idx']))
idx2word = dict((i,c) for c,i in zip(freqdf['word'],freqdf['idx']))
w2v = Word2Vec.load("./model/word2vector.bin")

#初始化向量
init_weight_wv,idx2word,word2idx = init_weightlist(w2v,idx2word,word2idx)

dump(word2idx,open("./data/word2idx.pickle",'wb')) #将python对象序列化保存到本地文件
dump(idx2word,open('./data/idx2word.pickle','wb'))
dump(init_weight_wv,open('./data/init_weight_wv.pickle','wb'))

#读取数据，将格式进行转换为带4种标签BMES
output_file = './data/msr.tagging.txt'
character_tagging(corpuspath,output_file)

#分离word和label
with open(output_file,'r',encoding='utf-8') as f:
    lines = f.readlines()
    train_line = [[word[0] for word in line.split()] for line in lines]
    train_label = [word[2] for line in lines for word in line.split()]

    #将所有训练文本转成数字list
    train_word_num = []
    for line in train_line:
        train_word_num.extend(featContext(line,word2idx))

    #持久化
    dump(train_word_num,open('./data/train_word_num.pickle','wb'))
    dump(train_label,open('./data/train_label.pickle','wb'))
