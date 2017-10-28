# -*- encoding:utf-8 -*-
"""
@file: bayes.py
@author: Sineatos
@time: 2017/10/27 20:37
@contact: sineatos@gmail.com
"""

import numpy as np


def load_data_set():
    posting_list = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                    ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                    ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                    ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                    ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                    ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']
                    ]
    class_vec = [0, 1, 0, 1, 0, 1]  # 1 侮辱性文字，0 正常言论
    return posting_list, class_vec


def create_vocab_list(data_set):
    vocab_set = set([])
    for document in data_set:
        vocab_set = vocab_set | set(document)
    return list(vocab_set)


def set_of_words2vec(vocab_list, input_set):
    """
    检查词汇是否出现在字典中
    :param vocab_list: 字典
    :param input_set: 输入的词汇集合
    :return: 一个列表，如果在字典中，字典中词汇的对应位置为1，否则为0
    """
    return_vec = [0] * len(vocab_list)
    for word in input_set:
        if word in vocab_list:
            return_vec[vocab_list.index(word)] = 1
        else:
            print("the word: %s is not in my Vocabulary!" % word)
    return return_vec


def train_nb0(train_matrix, train_category):
    num_train_docs = len(train_matrix)  # 样本个数
    num_words = len(train_matrix[0])    # 特征数目(字典大小)
    p_abusive = np.sum(train_category) / float(num_train_docs)  # 统计具有正例的样本占总数的比例
    p0_num = np.zeros(num_words)
    p1_num = np.zeros(num_words)
    p0_denom = 0.0
    p1_denom = 0.0
    for i in range(num_train_docs):
        # 统计每一个特征在正例和负例中分布出现的次数，然后统计样本属于正例和样本属于负例的词汇总数
        if train_category[i] == 1:
            p1_num += train_matrix[i]
            p1_denom += np.sum(train_matrix[i])
        else:
            p0_num += train_matrix[i]
            p0_denom += np.sum(train_matrix[i])
    p1_vect = p1_num / p1_denom
    p0_vect = p0_num / p0_denom
    return p0_vect, p1_vect, p_abusive


if __name__ == "__main__":
    pass
