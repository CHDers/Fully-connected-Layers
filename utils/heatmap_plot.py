# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time  : 2023/4/8  10:29
# @Author: Yanjun Hao
# @File  : heatmap_plot.py

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from setuptools.sandbox import save_path

font = {'family': 'Times New Roman', 'size': '14'}  # SimSun宋体 'weight':'bold',
matplotlib.rc('font', **font)


def plot_heatmap(data: pd.DataFrame, save_path: str = None) -> None:
    """
    绘制热力图, 坐标轴刻度为真实x, y值
    :param data: 绘图的数据
    :param save_path: 保存图片的路径
    :return:
    """
    plt.style.use(['science', 'no-latex'])
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(data, xticklabels="auto", yticklabels="auto", cmap='rainbow')
    # 将y轴或x轴进行逆序
    ax.invert_yaxis()
    # ax.set_title('Heatmap')  # 图标题
    ax.set_xlabel('$x$')  # x轴标题
    ax.set_ylabel('$y$')
    # plt.xticks([])
    # plt.yticks([])
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.show()


if __name__ == '__main__':
    data = pd.read_csv("../datasets/4,7,8,8.9.csv", skiprows=18, index_col=0)
    plot_heatmap(data=data, save_path="../run/figure/heatmap.svg")
    print("Coding Ending!!!")
