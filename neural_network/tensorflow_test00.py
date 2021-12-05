# -*- coding:utf-8 -*-
# @Time    : 2021/12/1 9:55
# @Author  : Yinkai Yang
# @FileName: tensorflow_test03.py
# @Software: PyCharm
# @Description:


import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

t_1 = tf.constant([1, 2, 3, 4])
t_2 = tf.constant([9, 8, 7, 6])
v_add = tf.add(t_1, t_2)
rand_t = tf.random.uniform([50, 50], 0, 10, seed=10)
'''
define a placeholder,dtype,shape,name
ph_1 = tf.placeholder()
'''
x = tf.compat.v1.placeholder("float")
y = 2 * x
data = tf.random.uniform([4, 5], 10)

# 利用常量来初始化变量，很明显看出来，这是告诉我们变量的初始化方式，必须显式地初始化所有变量
t_a = tf.Variable(rand_t)
t_b = tf.Variable(rand_t)

# 常见的初始化方式
weight = tf.Variable(tf.random.normal([100, 100], stddev=2), name='weight')
bias = tf.Variable(tf.zeros([100]), name='biases')

# 指定一个变量初始化另一个变量
# w2 = tf.Variable(weight.initialized_value(), name='w2')  # initialized_value() will be removed

# 每个变量还可以在运行图中单独使用tf.Variable.initializer来初始化
bias2 = tf.Variable(tf.zeros([100, 100]))

# 计算途中的定义通过生命初始化操作对象来实现
initial_op = tf.compat.v1.global_variables_initializer()

# sess = tf.compat.v1.Session()
# # sess.run(bias2.initializer)  # 完成对bias初始化
# sess.run(initial_op)
# print(bias2)
# print(sess.run(v_add))
# print(sess.run(rand_t))
# sess.close()
with tf.compat.v1.Session() as sess:
    x_data = sess.run(data)
    print(x_data)
    print(sess.run(y, feed_dict={x: x_data}))
    # sess.run(bias2.initializer)
    # sess.run(t_a.initializer)
    # sess.run(t_b.initializer)
    # sess.run(initial_op)
    # print(bias2)
    # print(bias2.eval())
    # print(t_a)
    # print(t_a.eval())
    # print(t_b)
    # print(t_b.eval())