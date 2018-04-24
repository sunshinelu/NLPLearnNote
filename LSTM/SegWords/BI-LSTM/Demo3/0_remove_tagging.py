#-*- coding: UTF-8 -*-

import re
import codecs

import sys
reload(sys)
sys.setdefaultencoding('utf8')

"""
删除语料库中的词性标注

"""

from pyspark.sql import SparkSession


spark = SparkSession \
    .builder \
    .appName("word_count_df") \
    .getOrCreate()

sc = spark.sparkContext

ipt_file = "/Users/sunlu/Workspaces/PyCharm/Github/NLPLearnNote/LSTM/SegWords/BI-LSTM/Demo2/Corpora/people2014All.txt"
opt_file = "/Users/sunlu/Workspaces/PyCharm/Github/NLPLearnNote/LSTM/SegWords/BI-LSTM/Demo3/results/people2014All_remove_tagging.txt"

rdd = sc.textFile(ipt_file)
rdd1 = rdd.map(lambda x: x.split(" "))

def rm_tagging(ipt_list):
    opt_list = []
    for i in ipt_list:
        opt_list = opt_list + [i.split("/")[0]]
    return " ".join(opt_list)

import re
rdd2 = rdd1.map(lambda x: rm_tagging(x)) \
    .map(lambda x: (re.sub("\[|\]", "",x),))

df1 = rdd2.toDF(["word"])
df1.coalesce(1).toPandas().to_csv(opt_file, header = False, index = False)