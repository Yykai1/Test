# -*- coding:utf-8 -*-
# @Time    : 2021/11/30 14:30
# @Author  : Yinkai Yang
# @FileName: tensorflow_test02.py
# @Software: PyCharm
# @Description: tensor related to constant


import tensorflow as tf

# close eager execution
tf.compat.v1.disable_eager_execution()

# define some constants
v_1 = tf.constant([1, 2, 3, 4])
v_2 = tf.constant([9, 8, 7, 6])

# define some functions
v_add = tf.add(v_1, v_2)

# define some constants
t_1 = tf.constant(4)
t_2 = tf.constant([1, 2, 3])
zero_t = tf.zeros([2, 3], tf.int32)
# 按照我的理解，就是从1-10生成有10-1个间隔的数据，或者是从1-10这9个数据一共生成10个数据
# 和matlab里面的linespace一样
range1_t = tf.linspace(1, 10, 10)  # 默认float
range2_t = tf.range(1, 10, 1)  # 默认int

random1_t = tf.random.normal([2, 3], seed=12)
random2_t = tf.random.uniform([6, 6], 0, 10, seed=10)  # tips: shape, minimum, maximum, seed

# creat a session
sess = tf.compat.v1.Session()

# run and print
print(sess.run([t_1, t_2, zero_t]))
print("range_1:", sess.run(range1_t))
print("range_2:", sess.run(range2_t))
print(sess.run(random1_t))
print(sess.run(random2_t))
# print(sess.run(v_add))
# print(sess.run([v_1, v_2, v_add]))

# close the session
sess.close()