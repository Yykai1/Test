# -*- coding:utf-8 -*-
# @Time    : 2021/12/5 8:40
# @Author  : Yinkai Yang
# @FileName: tensorflow_test06.py
# @Software: PyCharm
# @Description: achieve a easy linear model(1-D)

import tensorflow as tf
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def model_demo():
    # 1）准备数据
    with tf.variable_scope("prepare_data"):
        x = tf.random.normal([100, 1])
        y_true = tf.matmul(x, [[0.8]]) + 0.7

    # 2）构建模型
    with tf.variable_scope("create_model"):
        weights = tf.Variable(tf.random.normal([1, 1]))
        bias = tf.Variable(tf.random.normal([1, 1]))
        y_predict = tf.matmul(x, weights) + bias

    # 3）构造损失函数
    with tf.variable_scope("create_loss_function"):
        loss = tf.reduce_mean(tf.square(y_predict - y_true))

    # 4）优化损失
    optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

    # 收集变量
    tf.compat.v1.summary.scalar("loss", loss)
    tf.compat.v1.summary.histogram("weights", weights)
    tf.compat.v1.summary.histogram("bias", bias)

    # 合并变量
    merged = tf.summary.merge_all()

    # 显式的初始化变量
    init = tf.compat.v1.global_variables_initializer()

    # 开启会话
    with tf.compat.v1.Session() as sess:
        # 初始化操作
        sess.run(init)

        # 创建本地文件
        file_writer = tf.compat.v1.summary.FileWriter("./temp/linear", graph=sess.graph)

        print("-模型初始化-", "weights %f bias %f loss %f" % (weights.eval(), bias.eval(), loss.eval()))
        print("训练中")
        for i in range(1000):
            sess.run(optimizer)
            if i % 10 == 0:
                print("Epoch %4d weights %f bias %f loss %f" % (i, weights.eval(), bias.eval(), loss.eval()))
            summary = sess.run(merged)
            file_writer.add_summary(summary, i)
        print("-模型训练后-", "weights %f bias %f loss %f" % (weights.eval(), bias.eval(), loss.eval()))


if __name__ == '__main__':
    start_time = time.time()
    model_demo()
    end_time = time.time()
    print("The running time: %f s" % (end_time - start_time))
