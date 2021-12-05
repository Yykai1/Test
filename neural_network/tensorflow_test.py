# -*- coding:utf-8 -*-
# @Time    : 2021/12/2 19:29
# @Author  : Yinkai Yang
# @FileName: tensorflow_test.py
# @Software: PyCharm
# @Description: create a new graph

import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def graph_demo():
    g_new = tf.Graph()
    with g_new.as_default():
        a_t = tf.constant(2, name="a_t")
        b_t = tf.constant(3, name="b_t")
        add_t = tf.add(a_t, b_t, name="c_t")

    # this part is expected to create a new name of session, for example new_sess
    with tf.compat.v1.Session(graph=g_new) as sess:
        print(sess.run(add_t))
        print(sess.graph)
        print("a_t:", a_t.graph)
        print("a_t:", a_t)
        print("b_t:", b_t.graph)
        print("b_t:", b_t)
        print("add_t:", add_t.graph)

        # 将图写入本地生成events文件
        tf.compat.v1.summary.FileWriter("./temp/summary", graph=sess.graph)
        # 然后在terminal中切入neural_network空间中运行 tensorboard --logdir="./temp/summary" 即可，然后在浏览器打开 localhost:6006 即可可视化


if __name__ == '__main__':
    graph_demo()