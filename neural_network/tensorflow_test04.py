# -*- coding:utf-8 -*-
# @Time    : 2021/12/1 14:26
# @Author  : Yinkai Yang
# @FileName: tensorflow_test04.py
# @Software: PyCharm
# @Description:

import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# start a interactive Session
sess = tf.compat.v1.InteractiveSession()

# define a 5*5 matrix
I_matrix = tf.eye(4)
print("I_matrix:")
print(I_matrix.eval())

# define a Variable initialied to a 10*10 indentity matrix
X = tf.Variable(tf.eye(6))
X.initializer.run()  # initialize the Variable
print("X:")
print(X.eval())

# define another random 5*10 matrix
A = tf.Variable(tf.random.normal([4, 6]))
A.initializer.run()
print("A:")
print(A.eval())

# multiply A&X matrix
product = tf.matmul(A, X)
print("product:")
print(product.eval())

# creat a random matrix of 1s and 0s,size 5*10
B = tf.Variable(tf.random.uniform([4, 6], 0, 2, dtype=tf.int32))
B.initializer.run()
print("B:")
print(B.eval())

# cast to float32 data type
C = tf.cast(B, dtype=tf.float32)

# add the two matrix
print("A.*X + C:")
t_sum2 = tf.add(product, C)
print(sess.run(t_sum2))
# print("A.*X + B:")
# t_sum1 = tf.add(product, B)
# print(sess.run(t_sum1))

# creat two 4*5 matrix
a = tf.Variable(tf.random.normal([4, 5]))
b = tf.Variable(tf.random.normal([4, 5]))
a.initializer.run()
b.initializer.run()
print("a:")
print(a.eval())
print("b:")
print(b.eval())

# element wise multiplication(标量乘法)
S = a * b
print("S:")
print(sess.run(S))

# multiplication with a scalar(标量) 2
T = tf.scalar_mul(2, S)
print("T:")
print(sess.run(T))

# element division(除法)
V = tf.math.divide(a, b)
print("V:")
print(sess.run(V))

# element wise remainder of division
W = tf.math.mod(a, b)
print("W:")
print(sess.run(W))
