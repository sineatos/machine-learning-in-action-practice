# -*- encoding:utf-8 -*-
"""
@file: kNN.py
@author: Sineatos
@time: 2017/10/2 21:00
@contact: sineatos@gmail.com
"""

import numpy as np
import operator


def create_data_set():
    """
    创建一个简单的数据集
    :return: 样本,标签
    """
    group = np.array([
        [1.0, 1.1],
        [1.0, 1.0],
        [0, 0],
        [0, 0.1]
    ])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(inX, data_set, labels, k):
    """
    k近邻算法
    :param inX: 输入向量
    :param data_set: 训练样本集
    :param labels: 标签向量，一个元素对应一个样本的类别
    :param k: 用于选择最近邻居的数目
    :return:
    """
    data_set_size = data_set.shape[0]  # 获取样本数目
    # 将输入向量复制data_set_size次，构成一个data_size_size行，len(inX)列的矩阵，然后每一个行向量与一个训练样本相减
    # 然后作差以后每一位求平方然后求和再开方，实际上就是求出输入向量与各个样本的欧氏距离
    diff_mat = np.tile(inX, (data_set_size, 1)) - data_set
    sq_diff_mat = diff_mat ** 2
    sq_distances = sq_diff_mat.sum(axis=1)  # 矩阵数据按行方向相加
    distances = sq_distances ** 0.5

    sorted_dist_indicies = distances.argsort()  # 对所有的距离从小到大进行排序，返回一个向量，里面第i个元素表示排第i的值在原来的矩阵中的小标为第几

    # 统计输入向量最接近的前k个样本的类别和出现出现的次数
    class_count = {}
    for i in range(k):
        vote_I_label = labels[sorted_dist_indicies[i]]
        class_count[vote_I_label] = class_count.get(vote_I_label, 0) + 1
    # 根据类别的出现次数从大到小排序
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]  # 返回距离输入向量最近的k个样本中出现最多的类别


def file2matrix(filename):
    """
    将文件中读数据转化为矩阵，要求数据格式每一行为4个数字，前三个为属性，最后一个为类别
    :param filename: 文件名
    :return: numpy矩阵(每一行为一个样本的数据，一个样本有3个属性),列表(每一个元素为一个样本的类别)
    """
    fr = open(filename)
    array_of_lines = fr.readlines()
    number_of_lines = len(array_of_lines)
    return_mat = np.zeros((number_of_lines, 3))
    class_label_vector = []
    index = 0
    for line in array_of_lines:
        line = line.strip()
        list_from_line = line.split('\t')
        return_mat[index, :] = list_from_line[0:3]
        class_label_vector.append(int(list_from_line[-1]))
        index += 1
    return return_mat, class_label_vector


def auto_norm(data_set):
    """
    归一化数据
    :param data_set: Numpy矩阵
    :return: 归一化以后的Numpy矩阵，每一个属性(每一列)的取值跨度，每一个属性(每一列的最小值)
    """
    min_vals = data_set.min(0)  # 获取每一列的最小值
    max_vals = data_set.max(0)  # 获取每一列的最大值
    ranges = max_vals - min_vals  # 求出每一个属性的取值跨度
    # norm_data_set = zeros(shape(data_set))
    m = data_set.shape[0]
    norm_data_set = data_set - np.tile(min_vals, (m, 1))  # 每一行的每一个元素都减去对应列的最小值
    norm_data_set = norm_data_set / np.tile(ranges, (m, 1))  # 减去最小值的元素除以取值跨度，得到归一化的矩阵
    return norm_data_set, ranges, min_vals


def img2vector(filename):
    """
    读取手写数字文件，文件中包含0和1两种数字。读取前32行的每一行的前32个字符保存在Numpy数组中返回
    :param filename: 文件路径
    :return: 一个1*1024的Numpy向量
    """
    return_vect = np.zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        line_str = fr.readline()
        for j in range(32):
            return_vect[0, 32 * i + j] = int(line_str[j])
    return return_vect


__all__ = ['create_data_set', 'classify0', 'file2matrix', 'auto_norm', 'img2vector', ]

if __name__ == "__main__":
    pass
