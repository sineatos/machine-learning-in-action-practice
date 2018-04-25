# -*- encoding:UTF-8 -*-

import numpy as np
import matplotlib.pyplot as plt


def load_simple_data():
    data_mat = np.matrix([
        [1., 2.1],
        [2., 1.1],
        [1.3, 1.],
        [1., 1.],
        [2., 1.]
    ])
    class_labels = np.array([1.0, 1.0, -1.0, -1.0, 1.0])
    return data_mat, class_labels


def load_data_set(file_name):
    num_feat = len(open(file_name).readline().split('\t'))
    data_mat, label_mat = [], []
    fr = open(file_name)
    for line in fr.readlines():
        line_arr = []
        cur_line = line.strip().split('\t')
        for i in range(num_feat - 1):
            line_arr.append(float(cur_line[i]))
        data_mat.append(line_arr)
        label_mat.append(float(cur_line[-1]))
    return np.array(data_mat), np.array(label_mat)


def draw_graph(X, Y, labels):
    plt.plot(X[labels == 1.0], Y[labels == 1.0], ls='', marker='.')
    plt.plot(X[labels == -1.0], Y[labels == -1.0], ls='', marker='^')
    plt.show()


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
    创建单层决策树(decision stump)，单层决策树是AdaBoost中最流行的弱分类器
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
                err_arr[predicted_vals == label_mat] = 0  # 误差向量，如果预测的值跟实际值相同，则误差为0，否则为1
                weighted_error = D.T * err_arr  # 误差向量上每一位都乘以不同的权重
                if weighted_error < min_error:  # 更新最小错误率，保存对应的分类预测值，使用了哪一个特征来分类，对应的阈值以及使用的是哪一种运算(大于还是小于)
                    min_error = weighted_error
                    best_class_est = predicted_vals.copy()
                    best_stump['dim'] = i
                    best_stump['thresh'] = thresh_val
                    best_stump['ineq'] = inequal
    return best_stump, min_error[0,0], best_class_est


def adaboost_train_ds(data_arr, class_labels, num_it=40):
    # 弱分类器列表
    weak_class_arr = []
    # 样本数
    m = np.shape(data_arr)[0]
    # 数据分布(每一个样本的权重)
    D = np.mat(np.ones((m, 1)) / m)
    # 对每一个样本的分类预测
    agg_class_est = np.mat(np.zeros((m, 1)))
    for i in range(num_it):
        best_stump, error, class_est = build_stump(data_arr, class_labels, D)
        # print("D:", D.T)
        alpha = float(0.5 * np.log((1.0 - error) / max(error, 1e-16)))
        best_stump['alpha'] = alpha
        weak_class_arr.append(best_stump)
        # print("class_est:", class_est.T)
        expon = np.multiply(-1 * alpha * np.mat(class_labels).T, class_est)  # 正确的分类expon=-1*alpha，错误的分类expon=alpha
        D = np.multiply(D, np.exp(expon))
        D = D / D.sum()
        # 将之前的预测结果加上本次修正
        agg_class_est += alpha * class_est
        # print("agg_class_est:", agg_class_est.T)
        agg_errors = np.multiply(np.sign(agg_class_est) != np.mat(class_labels).T, np.ones((m, 1)))
        error_rate = agg_errors.sum() / m
        print("total error: ", error_rate)
        if error_rate == 0.0:
            break
    return weak_class_arr


def adaboost_classify(data_to_class, classifier_arr):
    """
    使用adaboost训练出来的模型进行分类
    :param data_to_class: 数据
    :param classifier_arr: 弱分类器列表，相当于是一个adaboost分类器
    :return: 分类结果
    """
    data_matrix = np.mat(data_to_class)
    m = np.shape(data_matrix)[0]
    agg_class_est = np.mat(np.zeros((m, 1)))
    for i in range(len(classifier_arr)):
        # 使用第i个弱分类器对数据进行分类
        class_est = stump_classify(data_matrix, classifier_arr[i]['dim'], classifier_arr[i]['thresh'],
                                   classifier_arr[i]['ineq'])
        agg_class_est += classifier_arr[i]['alpha'] * class_est  # 将分类结果进行加权
        print(agg_class_est)
    return np.sign(agg_class_est)


if __name__ == '__main__':
    train_data, train_labels = load_data_set('horseColicTraining.txt')
    test_data, test_labels = load_data_set('horseColicTest.txt')
    train_labels[train_labels != 1] = -1
    test_labels[test_labels != 1] = -1
    adaboost_train_ds(train_data, train_labels)
