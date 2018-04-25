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

python 6_2_lstm_crf_train.py --train_data_path Corpora/train.txt --test_data_path Corpora/test.txt --word2vec_path results/char_vec.txt

7. Build model for segmentation

Bi-LSTM-CRF: Freeze graph 

python tools/freeze_graph.py --input_graph Logs/seg_logs/graph.pbtxt --input_checkpoint Logs/seg_logs/model.ckpt --output_node_names "input_placeholder, transitions, Reshape_7" --output_graph Models/lstm_crf_model.pbtxt

报错：

    Traceback (most recent call last):
      File "tools/freeze_graph.py", line 202, in <module>
        app.run(main=main, argv=[sys.argv[0]] + unparsed)
      File "/home/gpuserver/softwares/anaconda2/lib/python2.7/site-packages/tensorflow/python/platform/app.py", line 48, in run
        _sys.exit(main(_sys.argv[:1] + flags_passthrough))
      File "tools/freeze_graph.py", line 134, in main
        FLAGS.variable_names_blacklist)
      File "tools/freeze_graph.py", line 122, in freeze_graph
        variable_names_blacklist=variable_names_blacklist)
      File "/home/gpuserver/softwares/anaconda2/lib/python2.7/site-packages/tensorflow/python/framework/graph_util_impl.py", line 208, in convert_variables_to_constants
        inference_graph = extract_sub_graph(input_graph_def, output_node_names)
      File "/home/gpuserver/softwares/anaconda2/lib/python2.7/site-packages/tensorflow/python/framework/graph_util_impl.py", line 145, in extract_sub_graph
        assert d in name_to_node_map, "%s is not in graph" % d
    AssertionError:  transitions is not in graph


Bi-LSTM-CNN: Freeze graph

python tools/freeze_graph.py --input_graph Logs/seg_cnn_logs/graph.pbtxt --input_checkpoint Logs/seg_cnn_logs/model.ckpt --output_node_names "input_placeholder,Reshape_5" --output_graph Models/lstm_cnn_model.pbtxt

8. Dump Vocabulary 

python tools/vob_dump.py --vecpath results/char_vec.txt --dump_path Models/vob_dump.pk 


9. Seg Script 

python tools/crf_seg.py --result_path results/crf_result.txt

报错：

    Traceback (most recent call last):
      File "tools/crf_seg.py", line 164, in <module>
        args.result_path, args.MAX_LEN, args.batch_size)
      File "tools/crf_seg.py", line 94, in main
        graph = load_graph(model_path)
      File "tools/crf_seg.py", line 22, in load_graph
        graph_def.ParseFromString(f.read())
      File "/home/gpuserver/softwares/anaconda2/lib/python2.7/site-packages/tensorflow/python/lib/io/file_io.py", line 119, in read
        self._preread_check()
      File "/home/gpuserver/softwares/anaconda2/lib/python2.7/site-packages/tensorflow/python/lib/io/file_io.py", line 79, in _preread_check
        compat.as_bytes(self.__name), 1024 * 512, status)
      File "/home/gpuserver/softwares/anaconda2/lib/python2.7/site-packages/tensorflow/python/framework/errors_impl.py", line 473, in __exit__
        c_api.TF_GetCode(self.status.status))
    tensorflow.python.framework.errors_impl.NotFoundError: Models/lstm_crf_model.pbtxt; No such file or directory


python tools/cnn_seg.py --result_path results/cnn_result.txt


10. PRF Scoring 

python PRF_Score.py results/cnn_result.txt Corpora/test_gold.txt

输出：

    Correct words: 379425
    Error words: 31487
    Gold words: 400336
    
    precision: 0.923373
    recall: 0.947766
    F-Value: 0.935411
    error_rate: 0.078651




