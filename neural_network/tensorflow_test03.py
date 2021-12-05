# -*- coding:utf-8 -*-
# @Time    : 2021/11/30 18:32
# @Author  : Yinkai Yang
# @FileName: tensorflow_test03.py
# @Software: PyCharm
# @Description:


import tensorflow as tf

tf.compat.v1.disable_eager_execution()

# sess = tf.compat.v1.InteractiveSession()

# 下面的代码中创建了两个不同的张量变量 t_a 和 t_b。两者将被初始化为形状为 [50，50] 的随机均匀分布，最小值=0，最大值=10：
rand_t = tf.random.normal([50, 50], 0, 10, seed=10)
t_a = tf.Variable(rand_t)
t_b = tf.Variable(rand_t)

'''
下面的代码中定义了两个变量的权重和偏置。权重变量使用正态分布随机初始化，均值为 0，权重大小为 100×100。
偏置由 100 个元素组成，每个元素初始化为 0。在这里也使用了可选参数名以给计算图中定义的变量命名：
'''
weights = tf.Variable(tf.random.uniform([100, 100]))
bias = tf.Variable(tf.zeros[100, 1], tf.float32, name='biases')

# 在前面的例子中，都是利用一些常量来初始化变量
# 也可以指定一个变量来初始化另一个变量。下面的语句将利用前面定义的权重来初始化 weight2：
weights2 = tf.Variable(weights.initial_value(), name='w2')

# 变量的定义将指定变量如何被初始化，但是必须显式初始化所有的声明变量。

sess = tf.compat.v1.Session()
# 在计算图的定义中通过声明初始化操作对象来实现：
initial_op = tf.
# print(rand_t.eval())
print(sess.run(rand_t))

sess.close()