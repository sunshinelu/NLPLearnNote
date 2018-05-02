#-*- coding: UTF-8 -*-

import numpy as np
import pandas as pd
import codecs
import re
import os
import sys
import pickle
from tqdm import tqdm
import time
from pyspark.sql import SparkSession

import tensorflow as tf
from tensorflow.contrib import rnn

"""
按照句子进行划分
"""
ipt_file = "/Users/sunlu/Workspaces/PyCharm/Github/NLPLearnNote/LSTM/SegWords/BI-LSTM/Demo3/results/people2014All_character_tagging.txt"

sentences_list = []
input_data = codecs.open(ipt_file,'r',encoding='utf-8_sig')
for line in input_data.readlines():
    sentences = re.split(u'[，。！？、‘’“”]/[BEMS]', line)# 重新以标点来划分
    sentences_list = sentences_list + sentences

"""
提取句子中的word和tag
"""
def get_Xy(sentence):
    """将 sentence 处理成 [word1, w2, ..wn], [tag1, t2, ...tn]"""
    words_tags = re.findall('(.)/(.)', sentence)
    if words_tags:
        words_tags = np.asarray(words_tags)
        words = words_tags[:, 0]
        tags = words_tags[:, 1]
        return words, tags # 所有的字和tag分别存为 data / label
    return None

datas = list()
labels = list()
print 'Start creating words and tags data ...'
for sentence in tqdm(iter(sentences_list)):
    result = get_Xy(sentence)
    if result:
        datas.append(result[0])
        labels.append(result[1])

"""
将words和tags构建dataframe
"""
df_data = pd.DataFrame({'words': datas, 'tags': labels}, index=range(len(datas)))
#　句子长度
df_data['sentence_len'] = df_data['words'].apply(lambda words: len(words))


"""
提取words和tags
"""
# 1.用 chain(*lists) 函数把多个list拼接起来
from itertools import chain
all_words = list(chain(*df_data['words'].values))

# 2.统计所有 word
sr_allwords = pd.Series(all_words)
sr_allwords = sr_allwords.value_counts()
set_words = sr_allwords.index
set_ids = range(1, len(set_words)+1) # 注意从1开始，因为我们准备把0作为填充值
tags = [ 'X', 'S', 'B', 'M', 'E']
tag_ids = range(len(tags))

# 3. 构建 words 和 tags 都转为数值 id 的映射（使用 Series 比 dict 更加方便）
word2id = pd.Series(set_ids, index=set_words)
id2word = pd.Series(set_words, index=set_ids)
tag2id = pd.Series(tag_ids, index=tags)
id2tag = pd.Series(tags, index=tag_ids)

vocab_size = len(set_words)

"""
把 words 和 tags 都转为数值 id
"""
max_len = 32
def X_padding(words):
    """把 words 转为 id 形式，并自动补全位 max_len 长度。"""
    ids = list(word2id[words])
    if len(ids) >= max_len:  # 长则弃掉
        return ids[:max_len]
    ids.extend([0]*(max_len-len(ids))) # 短则补全
    return ids

def y_padding(tags):
    """把 tags 转为 id 形式， 并自动补全位 max_len 长度。"""
    ids = list(tag2id[tags])
    if len(ids) >= max_len:  # 长则弃掉
        return ids[:max_len]
    ids.extend([0]*(max_len-len(ids))) # 短则补全
    return ids

df_data['X'] = df_data['words'].apply(X_padding)
df_data['y'] = df_data['tags'].apply(y_padding)


"""
获取x和y
"""
X = np.asarray(list(df_data['X'].values))
y = np.asarray(list(df_data['y'].values))

"""
保存结果
"""
if not os.path.exists('data/'):
    os.makedirs('data/')

with open('data/data.pkl', 'wb') as outp:
    pickle.dump(X, outp)
    pickle.dump(y, outp)
    pickle.dump(word2id, outp)
    pickle.dump(id2word, outp)
    pickle.dump(tag2id, outp)
    pickle.dump(id2tag, outp)

