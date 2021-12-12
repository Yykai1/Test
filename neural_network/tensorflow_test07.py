# -*- coding:utf-8 -*-
# @Time    : 2021/12/6 18:48
# @Author  : Yinkai Yang
# @FileName: tensorflow_test07.py
# @Software: PyCharm
# @Description:

import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def image_demo(filelist):
    # 构造文件名队列
    file_queue = tf.train.string_input_producer(filelist)
    # 文件读取与解码
    reader = tf.WholeFileReader()
    key, value = reader.read(file_queue)  # key:文件名 value:图片原本的编码格式
    print("key:", key)
    print("value:", value)
    # 解码
    image = tf.image.decode_jpeg(value)
    print("image:", image)

    # 图像的形状、类型修改
    image_resized = tf.image.resize_images(image, [200, 200])
    print("image_resized:", image_resized)

    # 静态形状修改
    image_resized.set_shape([200, 200, 3])
    print("image_resized:", image_resized)

    # 批处理
    image_batch = tf.train.batch([image_resized], batch_size=100, num_threads=1, capacity=100)
    print("image_batch:", image_batch)

    # 开启会话
    with tf.compat.v1.Session() as sess:
        # 开启线程
        # 线程协调员
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord)

        key_new, value_new, image_new, image_resized_new = sess.run([key, value, image, image_resized])
        print("key_new:", key_new)
        print("value_new:", value_new)
        print("image_new:", image_new)
        print("image_resized_new:", image_resized_new)

        coord.request_stop()
        coord.join(threads)
    return None


if __name__ == '__main__':
    # 构造路径+文件名列表
    filename = os.listdir("./cat")
    print(filename)
    # 路径拼接
    file_list = [os.path.join("./cat/", file) for file in filename]
    print(file_list)
    image_demo(file_list)
