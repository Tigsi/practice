# -*- coding: utf-8 -*-
# @Time : 2020/7/31 17:21
# @Author : wangsisi
# @FileName: numpytest.py
# @Software: PyCharm
# -*- coding: utf-8 -*-
# @Time : 2020/7/31 14:43
# @Author : wangsisi
# @FileName: numpytest.py
# @Software: PyCharm
import timeit

import numpy as np

"""
Numpy中的多维数组称为ndarray
"""


#########################################
###           numpy基本的操作           ###
#########################################
# 创建numpy数组
def get_array1():
    """
    基于tuple和list
    :return:
    """
    arr1 = np.array([[1, 2, 3], [4, 5, 6]])
    arr2 = np.array(((1, 2, 3), (4, 5, 6)))
    print(arr1)
    print(arr2)


def get_array2():
    """
    基于np.arange
    :return:
    """
    arr1 = np.arange(1, 4)
    arr2 = np.array([np.arange(1, 4), np.arange(4, 7)])
    print(arr1)
    print(arr2)


def get_array3():
    """
    基于arange以及reshape创建多维数组
    :return:
    """
    arr = np.arange(1, 13).reshape((2, 6, 1))
    print(arr)


def get_array4():
    arr = np.arange(1, 4, dtype=float)
    print(arr)


# Numpy的数值类型
def numtype():
    print(np.int8(12.334))
    print(np.float64(12))
    print(np.float(True))
    print(bool(1))


"""
#########################################
###         ndarray数组的属性           ###
#########################################

a = np.array([[1, 2, 3], [4, 5, 6]])
dtype属性:
    ndarray数组的数据类型，数据类型的种类，前面已描述。
    np.arange(1, 4, dtype=float)
    Output:
        array([1.,  2.,  3.])
ndim属性:
    数组维度的数量
    a.ndim
    Output:
        2
shape属性:
    数组对象的尺度，对于矩阵，即n行m列,shape是一个元组（tuple）
    a.shape
    Output:
        (3, 2)
size属性:
    用来保存元素的数量:
    a.size
    Output:
        6
itemsize属性:
    属性返回数组中各个元素所占用的字节数大小。
    a.itemsize
    Output:
        4
nbytes属性:
    如果想知道整个数组所需的字节数量，可以使用nbytes属性。其值等于数组的size属性值乘以itemsize属性值。
    a.nbytes
    Output:
        24
T属性:
    数组转置
    a.shape
    Output:
        (2,3)    
    a.T.shape
    Output:
        (3,2)
flat属性:
    返回一个numpy.flatiter对象，即可迭代的对象
    for item in a.flat:
        print(item)
    Output:
        1
        2
        3
        4
        5
        6

"""


# ndarray数组的切片和索引
def slice_1_D():
    """
    一维数组的切片和索引与python的list索引类似。
    :return:
    """
    arr = np.arange(1, 7)
    print(arr[:2])
    print(arr[2:])
    print(arr[1:3])


def slice_2_D():
    """
    二维数组切片把内层数组看成一个整体元素
    N维亦是如此
    :return:
    """
    arr = np.arange(1, 7).reshape(3, 2)
    print(arr)
    print(arr[:1])


#########################################
###            处理数组形状             ###
#########################################
# 形状转换
def rshape1():
    """
    函数resize（）的作用跟reshape（）类似，但是会改变所作用的数组，相当于有inplace=True的效果
    :return:
    """
    a = np.arange(1, 13)
    print(a.reshape((4, 3)))
    a.resize((4, 3))
    print(a)


def rshape2():
    """
    ravel()和flatten()，将多维数组转换成一维数组。两者的区别在于返回拷贝（copy）还
    是返回视图（view），flatten()返回一份拷贝，需要分配新的内存空间，对拷贝所做的修
    改不会影响原始矩阵，而ravel()返回的是视图（view），会影响原始矩阵。
    :return:
    """
    a = np.arange(1, 13).reshape((3, 4))
    print(a.ravel())
    print(a.flatten())


def rshape3():
    """
    前面描述了数组转置的属性（T），也可以通过transpose()函数来实现
    :return:
    """
    a = np.arange(1, 13).reshape((3, 4))
    print(a.T)
    print(a.transpose())


# 堆叠数组


def overlay():
    """
    堆叠数组
    concatenate():
        axis=1时，沿水平方向叠加
        axis=0时，沿垂直方向叠加
    深度堆叠(产生一个新的维度):
    a和b均是shape为（2,6）的二维数组，叠加后，arr_dstack是shape为（2,6,2）的三维数
    :return:
    """
    a = np.arange(1, 13).reshape((2, 6))
    b = np.arange(13, 25).reshape((2, 6))
    # 水平堆叠
    print(np.hstack((a, b)))
    np.concatenate((a, b), axis=1)
    # 垂直堆叠
    print(np.vstack((a, b)))
    np.concatenate((a, b), axis=0)
    # 深度堆叠
    print(np.dstack((a, b)))


# 数组的拆分
def splitt():
    """
    分割：堆叠的逆运算
    :return:
    """
    a = np.arange(1, 13).reshape((2, 6))
    b = np.arange(13, 25).reshape((2, 6))
    a = np.arange(1, 25).reshape((2, 12))
    # 水平分割
    print(np.hsplit(a, 2))
    np.split(a, 2, axis=1)
    # 垂直分割
    np.split(a, 2, axis=0)
    print(np.vsplit(a, 2))
    # 深度分割
    np.dsplit()


# 数组的类型转换
def toList():
    a = np.arange(1, 7).reshape((2, 3))
    print(a.tolist())


def astype():
    """
    转换成指定类型，astype()函数
    :return:
    """
    a = np.arange(1, 7).reshape((2, 3))
    print(a.astype(float))


#########################################
###          numpy常用统计函数          ###
#########################################
"""
np.sum()，返回求和
np.mean()，返回均值
np.max()，返回最大值
np.min()，返回最小值
np.ptp()，数组沿指定轴返回最大值减去最小值，即（max-min）
np.std()，返回标准偏差（standard deviation）
np.var()，返回方差（variance）
np.cumsum()，返回累加值
np.cumprod()，返回累乘积值
"""


def statistics():
    a = np.arange(1, 13).reshape((3, 4))
    print(a)
    print(np.sum(a))
    print(np.mean(a))
    print(np.ptp(a))
    print(np.ptp(a, axis=0))
    print(np.ptp(a, axis=1))
    print(np.cumsum(a))


# 数组的广播
def broadcast():
    """
    当数组跟一个标量进行数学运算时，标量需要根据数组的形状进行扩展，然后执行运算。
    这个扩展的过程称为“广播（broadcasting）”
    :return:
    """
    a = np.arange(1, 13).reshape((3, 4))
    a = a + 10
    print(a)


#########################################
### numpy向量化运算效率比不同循环计算效率高  ###
#########################################
def pySum():
    a = list(range(100000))
    b = list(range(100000))
    c = []
    for i in range(len(a)):
        c.append(a[i] ** 2 + b[i] ** 2)
    return c


def npSum():
    a = np.arange(100000)
    b = np.arange(100000)
    c = a ** 2 + b ** 2
    return c


def timetest():
    print(timeit.timeit(stmt="pySum()", setup='from __main__ import pySum', number=10))
    print(timeit.timeit(stmt="npSum()", setup='from __main__ import npSum', number=10))


if __name__ == '__main__':
    statistics()
