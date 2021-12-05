# -*- coding:utf-8 -*-
# @Time    : 2021/11/29 18:39
# @Author  : Yinkai Yang
# @FileName: tensorflow_test01.py
# @Software: PyCharm
# @Description: the first time to try learn tensorflow

# 安装TensorFlow
import tensorflow as tf


# 这个是无法执行的，TensorFlow2.0无法兼容1.0
tf.compat.v1.disable_eager_execution()  # 保证sess.run()能够正常运行

# 打印的信息是一个常量字符串，因此使用 tf.constant
message = tf.constant('Welcome to the exciting world of Deep Neural Networks!')
# 这个才是2.0版本的函数，通过定义Session，并使用run来运行
sess = tf.compat.v1.Session()
print(sess.run(message))

'''
其中b表示字节，byte
如果想要把b和''都去掉的话，需要使用对运行结果进行decode() --> print(sess.run(message).decode())
'''

# tensorflow的版本号
print(tf.__version__)

# 下面这个是gpu测试，但是我没有下载，所以结果是false
print(tf.test.is_gpu_available())