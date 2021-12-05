# -*- coding:utf-8 -*-
# @Time    : 2021/12/1 16:25
# @Author  : Yinkai Yang
# @FileName: tensorflow_test05.py
# @Software: PyCharm
# @Description: define a loss function in the tensorflow


import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# define global variables
m = 1000
n = 15
P = 2

# placeholder for the training data
X = tf.compat.v1.placeholder(tf.float32, name='X')
Y = tf.compat.v1.placeholder(tf.float32, name='Y')

# variables for coefficients(系数) initialized to 0
w0 = tf.Variable(0.0)
w1 = tf.Variable(0.0)

# the linear regession model
Y_hat = X*w1 + w0

# loss function
loss = tf.square(Y - Y_hat, name='loss')

# init all the variables
initial_op = tf.compat.v1.global_variables_initializer()

# start the session
with tf.compat.v1.Session() as sess:
    tf.compat.v1.global_variables_initializer(initial_op)
    result = sess.run(loss)
    print(result)