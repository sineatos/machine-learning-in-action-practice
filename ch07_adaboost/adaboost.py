import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report


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


def stump_classify(datas, dimen, thresh_val, thresh_ineq):
    """
    通过阈值对数据进行分类
    :param datas: 数据集
    :param dimen: 需要分类的特征的下标
    :param thresh_val: 阈值
    :param thresh_ineq: 阈值符号
    :return: 标签数组
    """
    ret_array = np.ones(np.shape(datas)[0])
    if thresh_ineq == 'lt':
        ret_array[datas[:, dimen] <= thresh_val] = -1.0
    else:
        ret_array[datas[:, dimen] > thresh_val] = -1.0
    return ret_array


def build_stump(datas, labels, D):
    """
    创建单层决策树(decision stump)，单层决策树是AdaBoost中最流行的弱分类器
    :param datas: 样本集
    :param labels: 标签集
    :param D: 数据的权重向量
    :return: 针对给定的权重向量，得到的最佳单层决策树的相关信息,最小错误率,对应的分类结果
    """
    m, n = np.shape(datas)
    num_steps = 10.0  # 将取值范围分成多少份
    best_stump = {}  # 用于存储给定权重向量D时所得到的最佳单层决策树的相关信息
    best_class_est = np.zeros(m)
    min_error = np.inf  # 用于保存最小错误率
    for i in range(n):  # 枚举特征
        range_min, range_max = datas[:, i].min(), datas[:, i].max()  # 获取当前枚举特征的最大最小值(因为这是数值型特征)
        step_size = (range_max - range_min) / num_steps  # 确定步长
        for j in range(-1, int(num_steps) + 1):
            # 枚举不同的数值作为阈值，其中-1是设置阈值为一个小于特征最小值的值，num_steps是设置阈值为特征最大值的值
            for inequal in ['lt', 'gt']:
                thresh_val = (range_min + float(j) * step_size)
                predicted_vals = stump_classify(datas, i, thresh_val, inequal)
                err_arr = np.ones(m)
                err_arr[predicted_vals == labels] = 0  # 误差向量，如果预测的值跟实际值相同，则误差为0，否则为1
                weight_error = np.sum(D * err_arr)
                if weight_error < min_error:
                    min_error = weight_error
                    best_class_est = predicted_vals.copy()
                    best_stump['dim'] = i
                    best_stump['thresh'] = thresh_val
                    best_stump['ineq'] = inequal
    return best_stump, min_error, best_class_est


def adaboost_train_ds(datas, labels, num_it=40):
    """
    Adaboost训练
    :param datas: 样本集
    :param labels: 标签集{-1,1}
    :param num_it: 迭代次数
    :return: 弱分类器列表
    """
    # 弱分类器列表
    weak_class_arr = []
    # 样本数
    m = np.shape(datas)[0]
    # 数据分布(每一个样本的权重)
    D = np.ones(m) / m
    # 对每一个样本的分类预测
    agg_class_est = np.zeros(m)
    for i in range(num_it):
        best_stump, error, class_est = build_stump(datas, labels, D)
        alpha = 0.5 * np.log((1.0 - error) / max(error, 1e-16))
        best_stump['alpha'] = alpha
        weak_class_arr.append(best_stump)
        expon = -1 * alpha * labels * class_est
        D = D * np.exp(expon)
        D = D / D.sum()
        # 将之前的预测结果加上本次修正
        agg_class_est += alpha * class_est
        agg_errors = (np.sign(agg_class_est) != labels) * np.ones(m)
        error_rate = agg_errors.sum() / m
        print(error_rate)
        if error_rate == 0.0:
            break
    return weak_class_arr


def adaboost_classify(clf_arr, datas):
    """
    分类
    :param clf_arr: 分类器列表
    :param datas: 数据
    :return: 结果
    """
    m = np.shape(datas)[0]
    agg_class_est = np.zeros(m)
    for clf in clf_arr:
        class_est = stump_classify(datas, clf['dim'], clf['thresh'], clf['ineq'])
        agg_class_est += clf['alpha'] * class_est
    return np.sign(agg_class_est)


def run():
    train_data, train_labels = load_data_set('horseColicTraining.txt')
    test_data, test_labels = load_data_set('horseColicTest.txt')
    train_labels[train_labels != 1] = -1
    test_labels[test_labels != 1] = -1
    clf_arr = adaboost_train_ds(train_data, train_labels)
    predict_arr = adaboost_classify(clf_arr, test_data)
    print("secc:", np.count_nonzero(predict_arr == test_labels) / len(test_labels))


def run_sklearn():
    train_data, train_labels = load_data_set('horseColicTraining.txt')
    test_data, test_labels = load_data_set('horseColicTest.txt')
    clf = AdaBoostClassifier()
    clf.fit(train_data, train_labels)
    print(classification_report(test_labels, clf.predict(test_data)))


if __name__ == '__main__':
    # run()
    run_sklearn()
