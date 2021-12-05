#### 1 安装

> ​		我真的是没有想到，这tensorflow的下载安装时这么困难。
>
> ​		第一次尝试是直接在pycharm里面调用import tensorflow as tf，但是一直都在报错，这个操作实际上是用pip的方式导入，但是因为各种原因，结果是无法导入。尝试了多次，还是没有结果，最终是放弃了！
>
> ​		第二次尝试，我是准备打算利用Anaconda来进行导入的，这个确实是比较方便的，看到网上都是在使用这个方法的，然后我就进入官网下载了Anaconda的安装包，但是在此之前我有重新使用了第一种方法，结果安装包下载结束后，我发现我得tensorflow没有报错了，这是为什么呢？我的运气真的是真么好啊！

> 其他的方法可以参照[TensorFlow安装和下载](http://c.biancheng.net/view/1881.html)，这个介绍还是比较全面的，可以作为参考（本人运气太好了）

#### 2 安装验证

下面是我亲自尝试所踩过的雷，这个真的是烦人啊

```python
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

# tensorflow的版本号
print(tf.__version__)

# 下面这个是gpu测试，但是我没有下载，所以结果是false
print(tf.test.is_gpu_available())
```

运行结果如下：（整体来看是没有问题的，但是里面好像有警告，大概意思就是我里面没有GPU的相关库）

![image-20211130110605487](https://github.com/Yykai1/Test/blob/main/notes/screenshot/image-20211130110605487.png)

###### 2.0.1 **补充知识-1**

eager execution

> 1. tensorflow经典的方式是需要构建计算图，启动会话后，张量在图中流动进行计算。在tf 2.0最大的特色之一就在于eager execution，大大简化了之前这个繁琐的定义和计算过程。
>
> 2. eager execution提供了一种命令式的编程环境，简单地说就是不需要构建图，直接可以运行。在老版本中，需要先构建计算图，然后再执行。在r1.14之后，tensorflow引入了eager execution，为快速搭建和测试算法模型提供了便利。
>
> 3. tf ==2.0== 中，默认情况下，Eager Execution ==处于启用状态==。可以用tf.executing_eargerly()查看Eager Execution当前的启动状态，返回True则是开启，False是关闭。
>
> 4. 启动eager模式==tf.compat.v1.enable_eager_execution()==;关闭eager模式==tf.compat.v1.disable_eager_execution()== 。

#### 3 程序结构

程序分为两个独立的部分，一个是**==计算图的定义==**，另一个是**==计算图的执行==**，这样可以构建神经网络的蓝图。

> **计算图**：是包含节点和边的网络。本节定义所有要使用的数据，也就是**张量**（tensor）对象（==常量、变量和占位符==），同时定义要执行的所有计算，即**运算操作对象**（Operation Object，简称 OP）。==简单来说就是定义数据和计算==
>
> 每个节点可以有零个或多个输入，但**只有一个输出**。网络中的**节点表示对象（张量和运算操作）**，**边表示运算操作之间流动的张量**。计算图定义神经网络的蓝图，但其中的张量还没有相关的数值。
>
> **计算图的执行**：使用==会话==对象来实现计算图的执行。会话对象**封装了评估张量和操作对象的环境**。这里真正实现了运算操作并将信息从网络的一层传递到另外一层。**不同张量对象的值==仅在会话对象中被初始化、访问和保存==。在此之前张量对象只被抽象定义，在==会话中才被赋予实际的意义==。**

#### 4 创建会话

因为内容还是比较简单的，这一部分代码的注释比较少

```python
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
        a_t = tf.constant(2)
        b_t = tf.constant(3)
        add_t = tf.add(a_t, b_t)

    # this part is expected to create a new name of session, for example new_sess
    with tf.compat.v1.Session(graph=g_new) as sess:
        print(sess.run(add_t))
        print(sess.graph)
        print("a_t:", a_t.graph)
        print("b_t:", b_t.graph)
        print("add_t:", add_t.graph)


if __name__ == '__main__':
    graph_demo()
```

#### 5 张量的理解

> 最基本的 [TensorFlow](http://c.biancheng.net/tensorflow/) 提供了一个库来定义和执行对张量的各种数学运算。张量，可理解为一个 n 维矩阵，所有类型的数据，包括标量、矢量和矩阵等都是特殊类型的张量。

| 数据类型 |  张量   |       形状        |
| :------: | :-----: | :---------------: |
|   标量   | 0-D张量 |        []         |
|   向量   | 1-D张量 |      [D~0~]       |
|   矩阵   | 2-D张量 |    [D~0~,D~1~]    |
|   张量   | N-D张量 | [D~0~,...,D~n-1~] |

TensorFlow 支持以下三种类型的张量：

> 1. ==常量==：常量是其值不能改变的张量。
> 2. ==变量==：当一个量在会话中的值需要更新时，使用变量来表示。例如，**在神经网络中，权重需要在训练期间更新，可以通过将权重声明为变量来实现**。==变量在使用前需要被显示初始化==。另外需要注意的是，常量存储在计算图的定义中，每次加载图时都会加载相关变量。换句话说，它们是占用内存的。另一方面，变量又是分开存储的。它们可以存储在参数服务器上。
> 3. ==占位符==：用于将值输入 TensorFlow 图中。它们可以和 feed_dict 一起使用来输入数据。**在训练神经网络时，它们通常用于提供新的训练样本**。在**会话中运行计算图时，可以为占位符赋值**。这样在构建一个计算图时不需要真正地输入数据。需要注意的是，==占位符不包含任何数据，因此不需要初始化它们==。

##### 4.1 **常量**

```python
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

random1_t = tf.random.normal([2, 3], seed=12)  # 种子只能有整数值
```

关于随机数生成，如下图，利用tf.random.去寻找合适的函数

**normal ：正态分布**

**gamma：伽马分布**

**poisson：泊松分布**

**shuffle：随机打乱**

**unform：均匀分布**



![image-20211130181919011](.\screenshot\image-20211130181919011.png)

```python
# -*- coding:utf-8 -*-
# @Time    : 2021/11/30 14:30
# @Author  : Yinkai Yang
# @FileName: tensorflow_test02.py
# @Software: PyCharm
# @Description:


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
```

```python
# -*- coding:utf-8 -*-
# @Time    : 2021/11/30 18:32
# @Author  : Yinkai Yang
# @FileName: tensorflow_test03.py
# @Software: PyCharm
# @Description:


import tensorflow as tf

tf.compat.v1.disable_eager_execution()

sess = tf.compat.v1.InteractiveSession()
rand_t = tf.random.normal([50, 50], 0, 10, seed=10)

print(rand_t.eval())

sess.close()
```

###### 4.1.1 一个小插曲

我在使用tensorflow时出现了各种各样的问题，最后我向现实妥协了。首先是高版本对学习的影响，导致各种api的使用出现各种问题，而且各种warning让人心烦意乱；其次就是学习应该循序渐进，直接上手高版本还是有点累，毕竟网上更多的资源都是和tensorflow1.0版本联系密切。

```python
# -*- coding:utf-8 -*-
# @Time    : 2021/12/1 9:55
# @Author  : Yinkai Yang
# @FileName: tensorflow_test03.py
# @Software: PyCharm
# @Description:


import tensorflow as tf
import os

'''
如果您没有GPU并且希望尽可能多地利用CPU，那么如果您的CPU支持AVX，AVX2和FMA，则应该从针对CPU优化的源构建tensorflow。
在这个问题中已经讨论过这个问题，也是这个GitHub问题。 
Tensorflow使用称为bazel的ad-hoc构建系统，构建它并不是那么简单，但肯定是可行的。在此之后，不仅警告消失，tensorflow性能也应该改善。
————————————————
版权声明：本文为CSDN博主「涛哥带你学编程」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/hq86937375/article/details/79696023
'''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

t_1 = tf.constant([1, 2, 3, 4])
t_2 = tf.constant([9, 8, 7, 6])

v_add = tf.add(t_1, t_2)

sess = tf.compat.v1.Session()
print(sess.run(v_add))
sess.close()
```

下面的插图是我使用python3.6.8（没记错的话）和tensorflow1.5.5来进行学习的，虽然还是遇到了tf.Session()被抛弃，还是要使用tf.compat.v1.Session()，虽然这样，我感觉下面的红色部分消失之后内心还是很开心的。

![image-20211201100507807](.\screenshot\image-20211201100507807.png)

###### 4.1.2 补充知识-2

> import os
>
> os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
>
> '''
> 如果您没有GPU并且希望尽可能多地利用CPU，那么如果您的CPU支持AVX，AVX2和FMA，则应该从针对CPU优化的源构建tensorflow。在这个问题中已经讨论过这个问题，也是这个GitHub问题。 Tensorflow使用称为bazel的ad-hoc构建系统，构建它并不是那么简单，但肯定是可行的。在此之后，不仅警告消失，tensorflow性能也应该改善。
> ————————————————
> 版权声明：本文为CSDN博主「涛哥带你学编程」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
> 原文链接：https://blog.csdn.net/hq86937375/article/details/79696023
> '''

##### 4.2 **变量**

它们通过使用变量类来创建，变量的定义还包括应该初始化的常量/随机值。

```python
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
    # sess.run(bias2.initializer)
    # sess.run(t_a.initializer)
    # sess.run(t_b.initializer)
    sess.run(initial_op)
    print(bias2)
    print(bias2.eval())
    print(t_a)
    print(t_a.eval())
    print(t_b)
    print(t_b.eval())
```

##### 4.3 占位符

用于将数据提供给计算图

```python
x = tf.compat.v1.placeholder("float")  # x就是一个占位符
y = 2 * x
data = tf.random.uniform([4, 5], 10)

with tf.compat.v1.Session() as sess:
    x_data = sess.run(data)  # 随机常量
    print(x_data)
    print(sess.run(y, feed_dict={x: x_data}))  # feed_dict就是用作x的数据集合
```

###### 4.3.1 补充知识-3

> TensorFlow 被设计成与 Numpy 配合运行，因此所有的 TensorFlow 数据类型都是基于 Numpy 的。使用 **tf.convert_to_tensor()** 可以将给定的值转换为张量类型，并将其与 TensorFlow 函数和运算符一起使用。该函数接受 Numpy 数组、[Python](http://c.biancheng.net/python/) 列表和 Python 标量，并允许与张量对象互操作。请注意，与 Python/Numpy 序列不同，**==TensorFlow 序列不可迭代==**。

下表列出了 TensorFlow 支持的常见的数据类型：

|   数据类型   | TensorFLow类型 |                        描述                        |
| :----------: | :------------: | :------------------------------------------------: |
|   DT_FLOAT   |   tf.float32   |                     32位浮点数                     |
|  DT_DOUBLE   |   tf.float64   |                     64位浮点数                     |
|   DT_INT8    |    tf.int8     |                   8位有符号整数                    |
|   DT_UNIT8   |    tf.unit8    |                   8位无符号整数                    |
|  DT_STRING   |   tf.string    | 可变长度的字节数组，每一个张量元素都是一个字节数组 |
|   DT_BOOL    |    tf.bool     |                      布尔类型                      |
| DT_COMPLEX64 |  tf.complex64  |       由两个32位浮点数组成的复数：实部和虚部       |
|  DT_QINT32   |   tf.qint32    |            用于量化的Ops的8位有符号整型            |

#### 6 tensorboard的使用

tensorboard是一种可视化工具，没有过多的介绍，一个简单的程序即可简单了解。

```python
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
```

#### 7 矩阵的简单运算

> 矩阵运算，例如执行**乘法**、**加法**和**减法**，是任何神经网络中信号传播的重要操作。通常在计算中需要**随机矩阵**(random)、**零矩阵**(zeros)、**一矩阵**(ones)或者**单位矩阵**(eye)。其他有用的矩阵操作，如**按元素相乘**、**乘以一个标量**、**按元素相除**、**按元素余数相除**等

```python
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
print("A.*X + B:")
t_sum1 = tf.add(product, B)
print(sess.run(t_sum1))
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
```

#### 8 操作的介绍

操作，即operation，简称OP。下面我们需要区分的就是操作函数和操作对象：

> **操作函数**：就像是我们使用了**tf.constant()**，这个就是一个操作函数，当然，**tf.Variable()**这些都是操作函数
>
> **操作对象**：当我们把**tensor对象**传到操作函数中，我们就生成了一个操作对象，例如==a_t = tf.constant(2)==，**a_t**就是一个操作对象

**常见的操作如下图所示：**

![image-20211203175154186](.\screenshot\image-20211203175154186.png)



#### 9 会话的介绍

会话，即Session，常用sess表示。

> 也不知道该怎么说，因为我这个前面是跟着一些文档学习的，后面的是跟着视频学的，导致我现在看到会话就没有什么需要写的了。前面的很多代码示例都已经有了相关的操作。QAQ

#### 10 实现简单的模型

```python
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
    x = tf.random.normal([100, 1])
    y_true = tf.matmul(x, [[0.8]]) + 0.7

    # 2）构建模型
    weights = tf.Variable(tf.random.normal([1, 1]))
    bias = tf.Variable(tf.random.normal([1, 1]))
    y_predict = tf.matmul(x, weights) + bias

    # 3）构造损失函数
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
```

下面这个tensorboard看起来不是很清晰，内容还是比较乱的，然后我们可以，重新定义命名空间，然后就会发现

![image-20211205101226243](.\screenshot\image-20211205101226243.png)

优化后的图像

![image-20211205101731863](.\screenshot\image-20211205101731863.png)

