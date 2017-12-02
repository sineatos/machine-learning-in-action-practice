# -*- encoding:UTF-8 -*-

import numpy as np


def load_data_set(filename):
    data_mat, label_mat = [], []
    fr = open(filename)
    for line in fr.readlines():
        line_arr = line.strip().split('\t')
        data_mat.append([float(line_arr[0]), float(line_arr[1])])
        label_mat.append(float(line_arr[2]))

    return data_mat, label_mat


def select_J_rand(i, m):
    """
    :param i: 第一个alpha的下标
    :param m: 所有alpha的数目
    :return: 下标不等于i的随机一个alpha的下标
    """
    j = i
    while j == i:
        j = int(np.random.uniform(0, m))
    return j


def clip_alpha(aj, H, L):
    """
    用于调整大于H或小于L的alpha值(保证alpha的值不出边界？)
    """
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj


def smo_simple(data_mat_In, class_labels, C, toler, max_iter):
    """
    :param data_mat_In: 输入数据集
    :param class_labels: 标签集
    :param C: 常数C
    :param toler: 容错率
    :param max_iter: 最大迭代次数
    """
    data_matrix = np.mat(data_mat_In)
    label_mat = np.mat(class_labels).transpose()
    b = 0
    m, n = np.shape(data_matrix)
    alphas = np.mat(np.zeros((m, 1)))
    iter = 0
    while iter < max_iter:
        alpha_pairs_changed = 0
        for i in range(m):
            fXi = float(np.multiply(alphas, label_mat).T * (data_matrix * data_matrix[i, :].T)) + b
            Ei = fXi - float(label_mat[i])
            if (label_mat[i] * Ei < -toler and alphas[i] < C) or (label_mat[i] * Ei > toler and alphas[i] > 0):
                j = select_J_rand(i, m)
                fXj = float(np.multiply(alphas, label_mat).T * (data_matrix * data_matrix[j, :].T)) + b
                Ej = fXj - float(label_mat[j])
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                if label_mat[i] != label_mat[j]:
                    L = np.max(0, alphas[j] - alphas[i])
                    H = np.min(C, C + alphas[j] - alphas[i])
                else:
                    L = np.max(0, alphas[j] + alphas[i] - C)
                    H = np.min(C, alphas[j] + alphas[i])
                if L == H:
                    print("L==H")
                    continue
                eta = 2.0 * data_matrix[i, :] * data_matrix[j, :].T - \
                      data_matrix[i, :] * data_matrix[i, :].T - \
                      data_matrix[j, :] * data_matrix[j, :].T
                if eta >= 0:
                    print("eta >= 0")
                    continue
                alphas[j] -= label_mat[j] * (Ei - Ej) / eta
                alphas[j] = clip_alpha(alphas[j], H, L)
                if np.abs(alphas[j] - alphaJold < 1E-5):
                    print("j not moving enough")
                    continue
                alphas[i] += label_mat[j] * label_mat[i] * (alphaJold - alphas[j])
                b1 = b - Ei - label_mat[i] * (alphas[i] - alphaIold) * \
                              data_matrix[i, :] * data_matrix[i, :].T - \
                     label_mat[j] * (alphas[j] - alphaJold) * \
                     data_matrix[i, :] * data_matrix[j, :].T
                b2 = b - Ej - label_mat[i] * (alphas[i] - alphaIold) * \
                              data_matrix[i, :] * data_matrix[j, :].T - \
                     label_mat[j] * (alphas[j] - alphaJold) * \
                     data_matrix[j, :] * data_matrix[j, :].T
                if 0 < alphas[i] and C > alphas[i]:
                    b = b1
                elif 0 < alphas[j] and C > alphas[j]:
                    b = b2
                else:
                    b = (b1 + b2) / 2.0
                alpha_pairs_changed += 1
                print("iter: %d i:%d, pairs changed %d" % (iter, i, alpha_pairs_changed))
        if alpha_pairs_changed == 0:
            iter += 1
        else:
            iter = 0
        print("iteration number: %d" % iter)
    return b, alphas
