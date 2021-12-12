# -*- coding:utf-8 -*-
# @Time    : 2021/12/8 20:06
# @Author  : Yinkai Yang
# @FileName: tf_idf_english.py
# @Software: PyCharm
# @Description: this is a program related to TF-IDF, the language is English

from collections import Counter
import math


def tf(word, count):
    return count[word] / sum(count.values())


def idf(word, count_list):
    n_contain = sum([1 for count in count_list if word in count])
    return math.log(len(count_list) / (1 + n_contain))


def tf_idf(word, count, count_list):
    return tf(word, count) * idf(word, count_list)


def word_split(data):
    tmp_words_list = []
    for i in range(len(data)):
        tmp_words_list.append(data[i].split(' '))
    print(tmp_words_list)
    return tmp_words_list


def word_count(words):
    tmp_count_list = []
    for i in range(len(words)):
        count = Counter(words[i])
        tmp_count_list.append(count)
    print(tmp_count_list)
    return tmp_count_list


if __name__ == '__main__':
    # 输入数据
    corpus = ['my name is Yinkai Yang',
              'I love natural language processing',
              'starting now ',
              'her name is Ai Tao']
    words_list = word_split(corpus)
    count_list = word_count(words_list)
    for i, count in enumerate(count_list):
        print("第 {} 个文档 TF-IDF 统计信息".format(i + 1))
        scores = {word: tf_idf(word, count, count_list) for word in count}
        sorted_word = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        for word, score in sorted_word:
            print("\tword: {}, TF-IDF: {}".format(word, round(score, 5)))
