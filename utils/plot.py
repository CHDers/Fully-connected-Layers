# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time  : 2023/4/8  12:00
# @Author: Yanjun Hao
# @File  : plot.py


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_heatmap(data: pd.DataFrame, save_path: str = None) -> None:
    """
    绘制热力图, 坐标轴刻度为索引
    :param data: 绘图的数据
    :param save_path: 保存图片的路径
    :return:
    """
    plt.style.use(['science', 'no-latex'])
    fig, ax = plt.subplots(figsize=(10, 10))
    # vegetables = ["cucumber", "tomato", "lettuce", "asparagus",
    #               "potato", "wheat", "barley"]
    # farmers = ["Farmer Joe", "Upland Bros.", "Smith Gardening",
    #            "Agrifun", "Organiculture", "BioGoods Ltd.", "Cornylee Corp."]
    #
    # harvest = np.array([[0.8, 2.4, 2.5, 3.9, 0.0, 4.0, 0.0],
    #                     [2.4, 0.0, 4.0, 1.0, 2.7, 0.0, 0.0],
    #                     [1.1, 2.4, 0.8, 4.3, 1.9, 4.4, 0.0],
    #                     [0.6, 0.0, 0.3, 0.0, 3.1, 0.0, 0.0],
    #                     [0.7, 1.7, 0.6, 2.6, 2.2, 6.2, 0.0],
    #                     [1.3, 1.2, 0.0, 0.0, 0.0, 3.2, 5.1],
    #                     [0.1, 2.0, 0.0, 1.4, 0.0, 1.9, 6.3]])

    # plt.xticks(np.arange(data.shape[1])[::500], labels=farmers, rotation=90, rotation_mode="anchor", ha="right")
    # plt.yticks(np.arange(len(vegetables)), labels=vegetables)
    # plt.xticks([200, 800, 1300])
    # plt.title("Harvest of local farmers (in tons/year)")
    plt.xticks([])
    plt.yticks([])

    plt.imshow(data, cmap='rainbow')
    plt.colorbar()
    # plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.show()


if __name__ == '__main__':
    data = pd.read_csv("../datasets/4,7,8,8.9.csv", skiprows=18, index_col=0)  # 改文件路径，画真实图和预测图
    plot_heatmap(data=data, save_path="../run/figure/热力图.svg")
