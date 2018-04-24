# 运行Demo2

demo来源：

MeteorYee/LSTM-CNN-CWS
<https://github.com/MeteorYee/LSTM-CNN-CWS>


cd /Users/sunlu/Workspaces/PyCharm/Github/NLPLearnNote/LSTM/SegWords/BI-LSTM/Demo2

1. 数据预处理：

python 1_preprocess.py --rootDir /Users/sunlu/Documents/天枢大数据团队/产品/文本分析产品/文本分析数据集/标注语料/人民日报2013年-2014年标注语料库/2014 --corpusAll Corpora/people2014All.txt --resultFile results/pre_chars_for_w2v.txt

2. word cont

python 2_word_count_df.py results/pre_chars_for_w2v.txt results/pre_vocab.txt


3. unk替换

python 3_replace_unk.py results/pre_vocab.txt results/pre_chars_for_w2v.txt results/chars_for_w2v.txt

4. build word2vec model

运行 4_word2vec.py
 
python 4_word2vec.py results/chars_for_w2v.txt results/char_vec.txt results/word2vec.model

5. pre train

python pre_train.py --corpusAll Corpora/people2014All.txt --vecpath results/char_vec.txt \
--train_file Corpora/train.txt --test_file Corpora/test.txt

6. build bi-lstm model

python 6_1_lstm_cnn_train.py --train_data_path Corpora/train.txt --test_data_path Corpora/test.txt --word2vec_path results/char_vec.txt

python 6_2_lstm_crf_train.py --train_data_path Corpora/train.txt --test_data_path Corpora/test.txt --word2vec_path results/char_vec.txt


在26上测试命令：

cd /home/sunlu/Workspace/bi-lstm_cnn_crf

1. 数据预处理：

python 1_preprocess.py --rootDir /home/sunlu/Data/people2014/2014 --corpusAll Corpora/people2014All.txt --resultFile results/pre_chars_for_w2v.txt

输出：
got 103 tags


2. word cont

python 2_word_count_df.py results/pre_chars_for_w2v.txt results/pre_vocab.txt 

3. unk替换

python 3_replace_unk.py results/pre_vocab.txt results/pre_chars_for_w2v.txt results/chars_for_w2v.txt


4. build word2vec model

运行 4_word2vec.py
 
python 4_word2vec.py results/chars_for_w2v.txt results/char_vec.txt results/word2vec.model


5. pre train

python 5_pre_train.py --corpusAll Corpora/people2014All.txt --vecpath results/char_vec.txt --train_file Corpora/train.txt --test_file Corpora/test.txt

输出：
Generating finished, gave up 10708 bad lines


6. build bi-lstm model

python 6_1_lstm_cnn_train.py --train_data_path Corpora/train.txt --test_data_path Corpora/test.txt --word2vec_path results/char_vec.txt
