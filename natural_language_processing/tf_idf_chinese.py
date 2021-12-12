# -*- coding:utf-8 -*-
# @Time    : 2021/12/8 20:37
# @Author  : Yinkai Yang
# @FileName: tf_idf_chinese.py
# @Software: PyCharm
# @Description: this is a program related to TF-IDF, the language is Chinese

import jieba

seg_list = jieba.cut("我来到北京清华大学", cut_all=True)
print("Full Mode: " + "/ ".join(seg_list))  # 全模式

seg_list = jieba.cut("我来到北京清华大学", cut_all=False)
print("Default Mode: " + "/ ".join(seg_list))  # 精确模式

seg_list = jieba.cut("他来到了网易杭研大厦")  # 默认是精确模式
print(", ".join(seg_list))

seg_list = jieba.cut_for_search("小明硕士毕业于中国科学院计算所，后在日本京都大学深造")  # 搜索引擎模式
print(", ".join(seg_list))
test_sent = u"永和服装饰品有限公司"
result = jieba.tokenize(test_sent)  # Tokenize：返回词语在原文的起始位置

for tk in result:
    print("word %s\t\t start: %d \t\t end:%d" % (tk[0], tk[1], tk[2]))
    print(tk)
