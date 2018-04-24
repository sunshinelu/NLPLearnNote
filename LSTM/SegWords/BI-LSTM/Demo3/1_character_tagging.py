#-*- coding: UTF-8 -*-

import re
import codecs

import sys
reload(sys)
sys.setdefaultencoding('utf8')

def replace_number_alphabet(word):
    '''
    将一个单词内的连续的数字，
    或者连续的字母用特殊符号来代替
    '''
    special_word = "$E$"
    special_number = "$N$"
    word = re.sub('([a-zA-Z]+)', special_word, word)
    word = re.sub('([0-9]+)', special_number, word)
    return word


def is_special_word(w1):
    if w1=="$E$":
        return 1
    if w1 == "$N$":
        return 2
    return 0


def character_tagging(input_string):
    '''
    对训练数据进行预处理，
    将每个字根据分词的情况，标记S，B，M，E的标记
    S表示单独一个字；
    B代表一个词的开始
    M代表一个词的中间的字
    E代表一个词的最后一个字

    Agrs:
      input_string: 输入字符串

    输出结果：
    输出结果为对词进行标签后的结果，
    '''

    word_list = input_string.strip().split()
    opt_list = []
    for word in word_list:
        word = replace_number_alphabet(word.split("/")[0])
        if len(word) == 1:
            opt_list = opt_list + [word + "/S "]
        else:
            if len(word) == 3 and is_special_word(word) > 0:
                opt_list = opt_list + [word + "/S "]
                continue
            index = 0
            if len(word) > 3 and is_special_word(word[:3]) > 0:
                opt_list = opt_list + [word[:3] + "/B "]
                index = 3
            else:
                opt_list = opt_list + [word[0] + "/B "]
                index = 1

            while index < (len(word) - 3):
                if is_special_word(word[index:index + 3]) > 0:
                    opt_list = opt_list + [word[index:index + 3] + "/M "]
                    index += 3
                else:
                    opt_list = opt_list + [word[index] + "/M "]
                    index += 1
            if index == (len(word) - 3) and is_special_word(word[index:index + 3]) > 0:
                opt_list = opt_list + [word[index:index + 3] + "/E "]
            else:
                while index < len(word) - 1:
                    opt_list = opt_list + [word[index] + "/M "]
                    index += 1
                opt_list = opt_list + [word[index] + "/E "]
    return "".join(opt_list)


def character_tagging_file(input_file, output_file):
    '''
    对训练数据进行预处理，
    将每个字根据分词的情况，标记S，B，M，E的标记
    S表示单独一个字；
    B代表一个词的开始
    M代表一个词的中间的字
    E代表一个词的最后一个字

    Agrs:
      input_file: 输入文本文件
      output_file: 输出做完标记后的文本文件
    '''

    print("_character_tagging: ", input_file)
    with codecs.open(input_file, 'r', 'utf-8') as input_data, \
            codecs.open(output_file, 'w', 'utf-8') as output_data:
        for line in input_data.readlines():
            word_list = line.strip().split()

            for word in word_list:
                word = replace_number_alphabet(word)
                if len(word) == 1:
                    output_data.write(word + "/S ")
                else:
                    if len(word) == 3 and is_special_word(word) > 0:
                        output_data.write(word + "/S ")
                        continue
                    index = 0
                    if len(word) > 3 and is_special_word(word[:3]) > 0:
                        output_data.write(word[:3] + "/B ")
                        index = 3
                    else:
                        output_data.write(word[0] + "/B ")
                        index = 1

                    while index < (len(word) - 3):
                        if is_special_word(word[index:index + 3]) > 0:
                            output_data.write(word[index:index + 3] + "/M ")
                            index += 3
                        else:
                            output_data.write(word[index] + "/M ")
                            index += 1
                    if index == (len(word) - 3) and is_special_word(word[index:index + 3]) > 0:
                        output_data.write(word[index:index + 3] + "/E ")
                    else:
                        while index < len(word) - 1:
                            output_data.write(word[index] + "/M ")
                            index += 1
                        output_data.write(word[index] + "/E ")
            output_data.write("\n")


input_file = "/Users/sunlu/Workspaces/PyCharm/Github/NLPLearnNote/LSTM/SegWords/BI-LSTM/Demo3/results/people2014All_remove_tagging.txt"
opt_file = "/Users/sunlu/Workspaces/PyCharm/Github/NLPLearnNote/LSTM/SegWords/BI-LSTM/Demo3/results/people2014All_character_tagging.txt"
character_tagging_file(input_file, opt_file)