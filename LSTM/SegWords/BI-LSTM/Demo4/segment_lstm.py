#-*-coding:utf-8-*-
'''
Created on 2018-04-26 16:38
@author:wangrs
'''

from seqlib import *

train_word_num = load(open('./data/train_word_num.pickle','rb'))
train_label = load(open('./data/train_label.pickle','rb'))
nb_class = len(np.unique(train_label)) #去除其中重复的元素,返回一个新的无元素重复的元组或者列表
#初始字向量格式准备
init_weight_wv = load(open('./data/init_weight_wv.pickle','rb'))

#建立两个词典
label_dict = dict(zip(np.unique(train_label),range(4)))
num_dict = {n:label for label,n in label_dict.items()}

#将目标变量转为数字
train_label = [label_dict[y] for y in train_label]
print(np.shape(train_label))
train_word_num = np.array(train_word_num)
print(np.shape(train_word_num))

#stacking LSTM 堆叠的LSTM
modelname = './model/my_model_weights.h5'
net = Lstm_Net()
net.init_weight = [np.array(init_weight_wv)]
net.nb_classes = nb_class

net.splitset(train_word_num,train_label)
print('Train----------------------')
net.buildnet()
net.train(modelname)
