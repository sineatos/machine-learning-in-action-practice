# -*- encoding:UTF-8 -*-

import numpy as np


def stump_classify(data_matrix, dimen, thresh_val, thresh_ineq):
    """
    通过阈值对数据进行分类
    :param data_matrix: 数据集
    :param dimen: 需要分类的特征的下标
    :param thresh_val: 阈值
    :param thresh_ineq: 阈值符号
    :return: 标签数组
    """
    ret_array = np.ones((np.shape(data_matrix)[0], 1))
    if thresh_ineq == 'lt':
        ret_array[data_matrix[:, dimen] <= thresh_val] = -1.0
    else:
        ret_array[data_matrix[:, dimen] > thresh_val] = -1.0
    return ret_array


def build_stump(data_arr, class_labels, D):
    """

    :param data_arr: 样本集
    :param class_labels: 标签集
    :param D: 数据的权重向量
    :return: 针对给定的权重向量，得到的最佳单层决策树的相关信息,最小错误率,对应的分类结果
    """
    data_matrix, label_mat = np.mat(data_arr), np.mat(class_labels).T
    m, n = np.shape(data_matrix)
    num_steps = 10.0
    best_stump = {}  # 用于存储给定权重向量D时所得到的最佳单层决策树的相关信息
    best_class_est = np.mat(np.zeros((m, 1)))
    min_error = np.inf  # 用于保存最小错误率
    for i in range(n):
        # 枚举特征
        range_min, range_max = data_matrix[:, i].min(), data_matrix[:, i].max()  # 获取当前枚举特征的最大最小值(因为这是数值型特征)
        step_size = (range_max - range_min) / num_steps  # 确定步长
        for j in range(-1, int(num_steps) + 1):
            # 枚举不同的数值作为阈值，其中-1是设置阈值为一个小于特征最小值的值，num_steps是设置阈值为特征最大值的值
            for inequal in ['lt', 'gt']:
                thresh_val = (range_min + float(j) * step_size)
                predicted_vals = stump_classify(data_matrix, i, thresh_val, inequal)
                err_arr = np.mat(np.ones((m, 1)))
                err_arr[predicted_vals == label_mat] = 0    # 误差向量，如果预测的值跟实际值相同，则误差为0，否则为1
                weighted_error = D.T * err_arr              # 误差向量上每一位都乘以不同的权重
                if weighted_error < min_error:              # 更新最小错误率，保存对应的分类预测值，使用了哪一个特征来分类，对应的阈值以及使用的是哪一种运算(大于还是小于)
                    min_error = weighted_error
                    best_class_est = predicted_vals.copy()
                    best_stump['dim'] = i
                    best_stump['thresh'] = thresh_val
                    best_stump['ineq'] = inequal
    return best_stump, min_error, best_class_est
