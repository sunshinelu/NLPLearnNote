#-*- coding: UTF-8 -*-

from pyspark import SparkContext
from pyspark import SparkConf
import pyspark
import pyspark.sql
from pyspark.sql import SparkSession

"""
注：最终未使用该方法，结果保存出来为unicode编码不是中文字符串，因为saveAsTextFile方法默认将结果保存成unicode。
"""

master= "local"
spark = SparkSession.builder\
    .appName("word_count")\
    .master(master)\
    .getOrCreate()

sc = spark.sparkContext

inp1 = "/home/sunlu/Workspace/bi-lstm_cnn_crf/Corpora/people2014All.txt"
otp = "/home/sunlu/Workspace/bi-lstm_cnn_crf/results/pre_vocab.txt"

text_file = sc.textFile(inp1)
def splitChar(x):
    result = ""
    for i in x:
        result = result + " " + i
    return result

wordCounts = text_file.map(lambda x: splitChar(x)).flatMap(lambda line: line.split(" ")) \
             .map(lambda word: (word, 1)) \
             .reduceByKey(lambda a, b: a + b)
# print(wordCounts.collect)
# wordCounts.saveAsTextFile(otp)
# wordCounts.repartiton(1).saveAsTextFile(otp)
wordCounts2 = wordCounts.filter(lambda x: x[1] >= 3).map(lambda x:[x[0].encode('utf-8'), x[1]])
# wordCounts2.repartiton(1).saveAsTextFile(otp)#AttributeError: 'PipelinedRDD' object has no attribute 'repartiton'
wordCounts2.coalesce(1).saveAsTextFile(otp)

