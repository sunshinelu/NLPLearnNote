#!/usr/bin/python
#-*-coding:utf-8-*-
'''
Created on 2018-04-25 15:41
@author:wangrs
'''
#1.导入模块包和语料库文件
import codecs
from gensim.models.word2vec import Word2Vec
import numpy as np
import nltk
from nltk.probability import FreqDist
import pandas as pd
from pickle import dump,load
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM,GRU,SimpleRNN
from keras.layers.core import Reshape,Flatten,Dropout,Dense,Activation
from keras.regularizers import l1,l2
from keras.layers.convolutional import Convolution2D,MaxPooling2D,MaxPooling1D
from sklearn.cross_validation import train_test_split
from keras.optimizers import SGD,RMSprop,Adagrad
from keras.utils import np_utils

#2.使用分词语料库生成词向量（模型）
def load_file(input_file): #读单个文本
    input_data = codecs.open(input_file,'r',encoding='utf-8')
    input_text = input_data.read()
    return input_text

#使用gensim的word2vec库
def trainW2V(corpus,epochs = 10,num_features=100,sg=1,min_word_count=1,num_works=4,context=4,sample=1e-5,negative=5):
    w2v = Word2Vec(workers=num_works,sample= sample,size=num_features,min_count=min_word_count,window=context)
    np.random.shuffle(corpus) #打乱顺序函数
    w2v.build_vocab(corpus)
    w2v.train(corpus,total_examples=w2v.corpus_count,epochs=epochs)
    print("word2vec DONE.")
    return w2v

#3.语料预处理
def freq_func(input_text): #nltk输入文本，输出词频
    corpus = nltk.Text(input_text)
    fdist = FreqDist(corpus)
    w = list(fdist.keys())
    v = list(fdist.values())
    freqpd = pd.DataFrame({'word':w,'freq':v})
    freqpd.sort_values(by='freq',ascending=False,inplace=True)
    freqpd['idx'] = np.arange(len(v))
    return freqpd

#初始化权重
def init_weightlist(w2v,idx2word,word2idx):
    init_weight_wv = []
    for i in range(len(idx2word)):
        init_weight_wv.append(w2v[idx2word[i]])
    #定义‘U’为未登录新字，‘P’为两头padding用途，并增加两个相应的向量表示
    char_num = len(init_weight_wv)
    idx2word[char_num] = 'U'
    word2idx['U'] = char_num
    idx2word[char_num+1] = 'P'
    word2idx['P'] = char_num+1
    init_weight_wv.append(np.random.randn(100))
    init_weight_wv.append(np.zeros(100))
    return init_weight_wv,idx2word,word2idx

def character_tagging(input_file,output_file): #加入标注标签：BMES(B是词首，M是词中，E是词尾，S是单字词)
    #带BOM的utf-8编码的txt文件时开头会有一个多余的字符\ufeff，BOM被解码为一个字符\ufeff，如何去掉？
    # 修改encoding为utf-8_sig或者utf_8_sig
    input_data =  codecs.open(input_file,'r',encoding='utf-8_sig')
    output_data = codecs.open(output_file,'w',encoding='utf-8')
    for line in input_data.readlines():
        word_list = line.strip().split()
        for word in word_list:
            if len(word) == 1:
                output_data.write(word+"/S ")
            else:
                output_data.write(word[0]+'/B ')
                for w in word[1:len(word)-1]:
                    output_data.write(w+"/M ")
                output_data.write(word[len(word)-1]+'/E ')
        output_data.write('\n')
    output_data.close()
    input_data.close()

def featContext(sentence,word2idx='',context = 7):
    predict_word_num = []
    for w in sentence: #文本中的字如果在词典中则转为数字，如果不在则设置为U
        if w in word2idx:
            predict_word_num.append(word2idx[w])
        else:
            predict_word_num.append(word2idx['U'])
    num = len(predict_word_num) #首尾padding
    pad = int((context-1)*0.5)
    for i in range(pad):
        predict_word_num.insert(0,word2idx['P'])
        predict_word_num.append(word2idx['P'])

    train_x = []
    for i in range(num):
        train_x.append(predict_word_num[i:i+context])
    return train_x

#4.训练语料
class Lstm_Net(object):
    def __init__(self):
        self.init_weight=[]
        self.batch_size = 128
        self.word_dim = 100
        self.maxlen = 7
        self.hidden_units = 100
        self.nb_classes = 0

    def buildnet(self):
        self.maxfeatures = self.init_weight[0].shape[0] #词典大小
        self.model = Sequential()
        print('stacking LSTM .....')#使用了堆叠的LSTM架构
        self.model.add(Embedding(self.maxfeatures,self.word_dim,input_length=self.maxlen))
        self.model.add(LSTM(self.hidden_units,return_sequences=True))
        self.model.add(LSTM(self.hidden_units,return_sequences=False))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(self.nb_classes))
        self.model.add(Activation('softmax'))
        self.model.compile(loss='categorical_crossentropy',optimizer='adam')

    def train(self,modelname):
        result= self.model.fit(self.train_X,self.Y_train,batch_size=self.batch_size,epochs=20,validation_data=(self.test_X,self.Y_test))
        self.model.save_weights(modelname)

    def splitset(self,train_word_num,train_label,train_size=0.9,random_state=1):
        self.train_X,self.test_X,train_y,test_y = train_test_split(train_word_num,train_label,train_size=0.9,random_state=1)
        print(np.shape(self.train_X))
        self.Y_train  = np_utils.to_categorical(train_y,self.nb_classes)
        print(np.shape(self.Y_train))
        self.Y_test = np_utils.to_categorical(test_y,self.nb_classes)

    def predict_num(self,input_num,input_txt,label_dict='',num_dict=''):
        #根据输入得到标注推断
        input_num = np.array(input_num)
        predict_prob = self.model.predict_proba(input_num,verbose=False)
        predict_label = self.model.predict_classes(input_num,verbose=False)
        for i,label in enumerate(predict_label[:-1]):
            if i==0: #如果是首字，不可为E，M
                predict_prob[i,label_dict['E']] = 0
                predict_prob[i,label_dict['M']] = 0
            if label == label_dict['B']: #前字为B，后字不可为B,S
                predict_prob[i+1, label_dict['B']] = 0
                predict_prob[i+1, label_dict['S']] = 0
            if label == label_dict['E']: #前字为E，后字不可为M,E
                predict_prob[i+1, label_dict['M']] = 0
                predict_prob[i+1, label_dict['E']] = 0
            if label == label_dict['M']: #前字为M，后字不可为B,S
                predict_prob[i+1, label_dict['B']] = 0
                predict_prob[i+1, label_dict['S']] = 0
            if label == label_dict['S']:  # 前字为S，后字不可为M,E
                predict_prob[i + 1, label_dict['M']] = 0
                predict_prob[i + 1, label_dict['E']] = 0
            predict_label[i+1] = predict_prob[i+1].argmax()
        predict_label_new = [num_dict[x] for x in predict_label]
        result = [w+'/'+label for w,label in zip(input_txt,predict_label_new)]
        return ' '.join(result)+'\n'

    def getweights(self,wfname):
        return self.model.load_weights(wfname)






