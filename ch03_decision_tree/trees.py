# -*- encoding:utf-8 -*-
"""
@file: trees.py
@author: Sineatos
@time: 2017/10/4 10:46
@contact: sineatos@gmail.com
"""

from math import log
import operator


def calc_Shannon_ent(data_set):
    """
    计算数据集的信息熵
    定义符号x_i出现的概率，或者说选择分类x_i的概率： p(x_i)
    符号x_i的信息定义： l(x_i) = -\log_2 p(x_i)
    熵的定义(信息的期望值)： H = - \sum_{i=1}^n p(x_i) \log_2 p(x_i)
    :param data_set: 数据集，一行是一个样本，样本前面的数据为属性，最后一个为标签(类别)
    :return: 数据集的信息熵
    """
    num_entries = len(data_set)  # 获取样本的数量
    label_counts = {}
    # 统计不同数据集中不同类别的样本分别出现的次数
    for feat_vec in data_set:
        current_label = feat_vec[-1]
        label_counts[current_label] = label_counts.get(current_label, 0) + 1
    shannon_ent = 0.0
    for key in label_counts:
        prob = float(label_counts[key]) / num_entries  # 求出 p(x_i)
        shannon_ent -= prob * log(prob, 2)
    return shannon_ent


def split_data_set(data_set, axis, value):
    """
    分割数据集，从整个数据集中挑选出指定特征属性取值为目标值的样本作为子数据集返回
    :param data_set: 数据样本列表，每一个元素为一个列表，代表一个样本
    :param axis: 划分数据集的特征属性下标
    :param value: 需要返回的特征值
    :return: 抽取后的子数据集
    """
    ret_data_set = []
    for feat_vec in data_set:
        if feat_vec[axis] == value:
            reduced_feat_vec = feat_vec[:axis]
            reduced_feat_vec.extend(feat_vec[axis + 1:])
            ret_data_set.append(reduced_feat_vec)
    return ret_data_set


def choose_best_feature_to_split(data_set):
    """
    选择最好的特征来划分数据集，最好的特征指的是使用该特征划分当前数据集以后当前数据集的熵减去各个子数据集的熵的期望得到的信息增益最大，
    其含义是使用最好的特征划分当前的数据集可以使得数据集的混乱程度下降得最多。
    数据格式要求，输入的data_set必须是一个列表，每一个元素必须是一个列表，代表一个样本，每个样本长度相同且最后一个元素为样本的类别标签。
    :param data_set: 数据集
    :return: 用于分隔数据集的最好特征
    """
    num_features = len(data_set[0]) - 1
    base_entropy = calc_Shannon_ent(data_set)  # 当前数据集的信息熵
    best_info_gain = 0.0  # 用于记录最好的信息增益值
    best_feature = -1  # 用于记录最好的划分特征
    for i in range(num_features):
        feat_list = [example[i] for example in data_set]  # 提取出所有样本的第i个特征的特征值
        unique_vals = set(feat_list)  # 获取第i个特征在集合出现过的取值
        new_entropy = 0.0
        for value in unique_vals:  # 根据第i个特征划分当前数据集以后各个子数据集的信息熵并求出划分后整个数据集的熵的期望
            sub_data_set = split_data_set(data_set, i, value)
            prob = len(sub_data_set) / float(len(data_set))
            new_entropy += prob * calc_Shannon_ent(sub_data_set)
        info_gain = base_entropy - new_entropy  # 求出信息增益
        if info_gain > best_info_gain:  # 找出信息增益最大的特征，用它进行划分
            best_info_gain = info_gain
            best_feature = i
    return best_feature


def majority_cnt(class_list):
    """
    返回类别列表中最多的类别
    :param class_list: 类别类表
    :return: 列表中最多的类别
    """
    class_count = {}
    for vote in class_list:
        class_count[vote] = class_count.get(vote, 0) + 1
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]


def create_tree(data_set, labels):
    """
    创建决策树
    :param data_set: 数据集
    :param labels:  标签集
    :return: 如果是非叶子结点，返回一个树，否则返回属性下标
    """
    class_list = [example[-1] for example in data_set]  # 获取数据集中所有样本的类别
    if class_list.count(class_list[0]) == len(class_list):  # 如果当前数据集的所有样本都属于同一个类别，则返回这个类别
        return class_list[0]
    if len(data_set[0]) == 1:  # 如果当前数据集中的样本只有一个特征属性，则返回当前数据集中最多样本属于的类别
        return majority_cnt(class_list)
    best_feat = choose_best_feature_to_split(data_set)  # 选择最好的特征用以分割数据集
    best_feat_label = labels[best_feat]
    my_tree = {best_feat_label: {}}  # 创建树
    del (labels[best_feat])  # 删除标签集中用来分割数据集的特征
    feat_values = [example[best_feat] for example in data_set]
    unique_vals = set(feat_values)
    for value in unique_vals:  # 枚举最好的特征在当前数据集中的取值
        sub_labels = labels[:]
        # 将当前数据集中特征best_feat取值为value的样本提取出来并去除特征best_feat
        # 递归创建子树
        my_tree[best_feat_label][value] = create_tree(split_data_set(data_set, best_feat, value), sub_labels)
    return my_tree


def create_data_set():
    data_set = [
        [1, 1, 'yes'],
        [1, 1, 'yes'],
        [1, 0, 'no'],
        [0, 1, 'no'],
        [0, 1, 'no'],
    ]
    labels = ['no surfacing', 'flippers']
    return data_set, labels


def classify(input_tree, feat_labels, test_vec):
    """
    使用训练好的决策树进行分类
    :param input_tree: 决策树
    :param feat_labels: 特征标签列表
    :param test_vec: 测试向量
    :return: 测试向量对应的类别
    """
    info = tuple(input_tree.keys())[0]
    children = input_tree[info]
    feat_index = feat_labels.index(info)
    class_label = None
    for child_info, child_root in children.items():
        if test_vec[feat_index] == child_info:
            if isinstance(child_root, dict):
                class_label = classify(child_root, feat_labels, test_vec)
            else:
                class_label = child_root
            break
    return class_label


def store_tree(input_tree, filename):
    """
    将决策树写入文件中
    :param input_tree: 决策树对象
    :param filename: 文件路径
    """
    import pickle
    import os
    parent_path = os.path.split(filename)[0]
    if not os.path.exists(parent_path):
        os.makedirs(parent_path)
    with open(filename, 'wb') as fw:
        pickle.dump(input_tree, fw)


def grab_tree(filename):
    """
    从文件中读取决策树
    :param filename: 文件路径
    :return: 决策树对象
    """
    import pickle
    with open(filename,'rb') as fr:
        return pickle.load(fr)


__all__ = ['calc_Shannon_ent', 'split_data_set', 'choose_best_feature_to_split', 'create_tree', 'majority_cnt',
           'create_data_set', 'classify', 'store_tree', 'grab_tree']

if __name__ == "__main__":
    pass
