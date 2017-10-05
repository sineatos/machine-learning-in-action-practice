# -*- encoding:utf-8 -*-
"""
@file: tree_plotter.py
@author: Sineatos
@time: 2017/10/5 15:25
@contact: sineatos@gmail.com
"""

import matplotlib.pyplot as plt

decision_node = dict(boxstyle="sawtooth", fc="0.8")
leaf_node = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")


def plot_mid_text(current_pt, parent_pt, txt_string):
    """
    在父子节点之间填充文本信息，也可以说是添加父子节点之间的边的信息
    :param current_pt: 孩子节点坐标
    :param parent_pt: 父亲节点坐标
    :param txt_string: 文本信息
    """
    x_mid = (parent_pt[0] - current_pt[0]) / 2.0 + current_pt[0]
    y_mid = (parent_pt[1] - current_pt[1]) / 2.0 + current_pt[1]
    create_plot.ax1.text(x_mid, y_mid, txt_string)


def plot_node(node_text, center_pt, parent_pt, node_type):
    create_plot.ax1.annotate(node_text, xy=parent_pt, xycoords='axes fraction', xytext=center_pt,
                             textcoords='axes fraction', va="center", ha="center", bbox=node_type,
                             arrowprops=arrow_args)


def create_plot(root):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    create_plot.ax1 = plt.subplot(111, frameon=False, **axprops)  # 创建图像
    # 注意：绘制图形的x轴和y轴的有效范围都是0.0~1.0，实际输出的图形并没有x,y坐标，但实际上并不需要担心图形的大小。
    # 通过计算树包含的所有的叶子节点数划分图形的宽度，从而得到当前节点的中心位置。
    plot_tree.total_w = float(get_num_leafs(root))  # 获取树的叶子数目(宽度)
    plot_tree.total_d = float(get_tree_depth(root))  # 获取树的深度
    plot_tree.x_off = -0.5 / plot_tree.total_w  # 用于记录需要绘制的点的x坐标
    plot_tree.y_off = 1.0  # 用于记录需要绘制的点的y坐标
    plot_tree(root, (0.5, 1.0), '')
    plt.show()


def get_num_leafs(root):
    """
    获取整棵树的叶子数目
    :param root: 根节点，要求是一个字典且结构如下：
            {
                特征A : {
                        F的取值a_1 :  类别信息(非字典对象),
                        F的取值a_2 :  {
                                        特征B : { ... }
                                        }
                        ...
                }
            }
    :return: 叶子数目
    """
    num_leafs = 0
    info = tuple(root.keys())[0]
    children = root[info]
    for child_info, child_root in children.items():
        if isinstance(child_root, dict):
            num_leafs += get_num_leafs(child_root)
        else:
            num_leafs += 1
    return num_leafs


def get_tree_depth(root):
    """
    获取树的深度
    :param root: 根节点，要求是一个字典且结构如下：
            {
                特征A : {
                        F的取值a_1 :  类别信息(非字典对象),
                        F的取值a_2 :  {
                                        特征B : { ... }
                                        }
                        ...
                }
            }
    :return: 树的深度
    """
    max_depth = 0
    info = tuple(root.keys())[0]
    children = root[info]
    for child_info, child_root in children.items():
        depth = 1
        if isinstance(child_root, dict):
            depth += get_tree_depth(child_root)
        max_depth = max(max_depth, depth)
    return max_depth


def plot_tree(root, parent_pt, node_txt):
    """
    绘制树
    :param root: 根节点
    :param parent_pt: 父节点坐标(x,y)
    :param node_txt: 根节点的文本信息
    """
    num_leafs = get_num_leafs(root)  # 获取当前子树的叶子节点个数
    depth = get_tree_depth(root)  # 获取当前子树的深度
    info = tuple(root.keys())[0]
    current_pt = (plot_tree.x_off + (1.0 + float(num_leafs)) / 2.0 / plot_tree.total_w, plot_tree.y_off)
    plot_mid_text(current_pt, parent_pt, node_txt)  # 添加边说明
    plot_node(info, current_pt, parent_pt, decision_node)
    children = root[info]
    plot_tree.y_off = plot_tree.y_off - 1.0 / plot_tree.total_d
    for child_info, child_root in children.items():
        if isinstance(child_root, dict):
            plot_tree(child_root, current_pt, str(child_info))
        else:
            plot_tree.x_off = plot_tree.x_off + 1.0 / plot_tree.total_w
            plot_node(child_root, (plot_tree.x_off, plot_tree.y_off), current_pt, leaf_node)
            plot_mid_text((plot_tree.x_off, plot_tree.y_off), current_pt, str(child_info))
    plot_tree.y_off = plot_tree.y_off + 1.0 / plot_tree.total_d


def retrieve_tree(i):
    """
    返回树的样例，一共要两个样例
    :param i: 样例编号
    :return: 树样例
    """
    list_of_trees = [
        {'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
        {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}},
    ]
    return list_of_trees[i]


__all__ = ['create_plot', 'retrieve_tree']

if __name__ == "__main__":
    my_tree = retrieve_tree(0)
    create_plot(my_tree)
