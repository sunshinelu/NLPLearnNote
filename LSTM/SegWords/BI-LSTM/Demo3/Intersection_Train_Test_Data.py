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
from pyspark.sql import functions

import tensorflow as tf
from tensorflow.contrib import rnn


reload(sys)
sys.setdefaultencoding('utf8')

# Configure the environment
if 'SPARK_HOME' not in os.environ:
    os.environ['SPARK_HOME'] = '/Users/sunlu/Software/spark-2.0.2-bin-hadoop2.6'
 # Create a variable for our root path
SPARK_HOME = os.environ['SPARK_HOME']

master= "local"
spark = SparkSession.builder\
    .appName("IntersectionDemo")\
    .master(master)\
    .getOrCreate()
sc = spark.sparkContext

ipt_file_1 = "/Users/sunlu/Workspaces/PyCharm/Github/NLPLearnNote/LSTM/SegWords/BI-LSTM/Demo3/results/people2014All_remove_tagging.txt"
ipt_file_2 = "/Users/sunlu/Workspaces/PyCharm/Github/LSTM-CNN-CWS/Corpora/test_gold.txt"

rdd1 = sc.textFile(ipt_file_1)
rdd2 = sc.textFile(ipt_file_2)

df1 = rdd1.map(lambda x:(x,)).toDF(["col1"])
df2 = rdd2.map(lambda x:(x,)).toDF(["col1"])

# df1与df2的差集
df3 = df1.subtract(df2)