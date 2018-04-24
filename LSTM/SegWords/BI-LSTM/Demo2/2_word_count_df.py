#-*- coding: UTF-8 -*-

from pyspark.sql import SparkSession
from pyspark.sql.functions import split, explode
from pyspark.sql.functions import col,desc

import sys
reload(sys)
sys.setdefaultencoding('utf8')
"""
解决以下报错：
UnicodeEncodeError: 'ascii' codec can't encode character u'\u4eca' in position 0: ordinal not in range(128)
"""

spark = SparkSession \
    .builder \
    .appName("word_count_df") \
    .getOrCreate()

sc = spark.sparkContext

# inp1 = "/Users/sunlu/Workspaces/PyCharm/Github/NLPLearnNote/LSTM/SegWords/BI-LSTM/Demo2/results/pre_chars_for_w2v.txt"
# otp = "/Users/sunlu/Workspaces/PyCharm/Github/NLPLearnNote/LSTM/SegWords/BI-LSTM/Demo2/results/pre_vocab.txt"

inp1 = sys.argv[1]
otp = sys.argv[2]

text_rdd = sc.textFile(inp1).map(lambda x: (x, ))
def splitChar(x):
    result = ""
    for i in x:
        result = result + " " + i
    return result

df1 = text_rdd.toDF(["txt"])
df2 = df1.select("txt", explode(split(col("txt"), "\s+")).alias("word"))
# df2.show(5)

# word_count = df2.groupBy('word').count().sort(desc("count"))
word_count = df2.groupBy('word').count().orderBy(desc("count")).filter(col("count") >= 3)
# word_count.printSchema()
# word_count.show(5)

# word_count.coalesce(1).write.mode('overwrite') \
#     .csv(otp ,sep=' ')

word_count.coalesce(1).toPandas().to_csv(otp,sep=' ',
                             index=False,header=False)