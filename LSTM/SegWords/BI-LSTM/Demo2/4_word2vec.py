# -*- coding: utf-8 -*-


import logging
import os.path
import sys
import multiprocessing
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence


# inputFile = "/Users/sunlu/Workspaces/PyCharm/Github/NLPLearnNote/LSTM/SegWords/BI-LSTM/Demo2/results/chars_for_w2v.txt"
# outputFile = "/Users/sunlu/Workspaces/PyCharm/Github/NLPLearnNote/LSTM/SegWords/BI-LSTM/Demo2/results/char_vec.txt"
# modelPath = "/Users/sunlu/Workspaces/PyCharm/Github/NLPLearnNote/LSTM/SegWords/BI-LSTM/Demo2/results/word2vec.model"

inputFile = sys.argv[1]
outputFile = sys.argv[2]
modelPath = sys.argv[3]


# 训练skip-gram模型
model = Word2Vec(LineSentence(inputFile), size=400, window=5, min_count=5,
                 workers=multiprocessing.cpu_count())

# 保存模型
model.save(modelPath)
model.wv.save_word2vec_format(outputFile, binary=False)