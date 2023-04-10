# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time  : 2023/4/8  9:56
# @Author: Yanjun Hao
# @File  : predict.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import json
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from rich import print

import matplotlib

font = {'family': 'Times New Roman', 'size': '14'}  # SimSun宋体 'weight':'bold',
matplotlib.rc('font', **font)

from pathlib import Path
import sys

FILE = Path(__file__).resolve()
ROOT_0 = FILE.parents[0]  # project root directory
ROOT_1 = FILE.parents[1]  # project root directory
if str(ROOT_0) not in sys.path:
    sys.path.append(str(ROOT_0))  # add ROOT to PATH
if str(ROOT_1) not in sys.path:
    sys.path.append(str(ROOT_1))  # add ROOT to PATH

from src.models import CustomDataset, Net, train, evaluate, select_line_to_predict
from src.Config import ModelConfig
from src.datasets import DataProcess


def test_model_of_line(model_path: str, val_csv_path: str, features_tuple: tuple = (10, 1, 3, 9),
                       label_name: str = "label"):
    """
    预测单根线
    :param model_path: 模型路径
    :param val_csv_path: 预测的数据集的路径
    :param features_tuple: 常量特征
    :param label_name: 模型的因变量y，默认为"label"
    :return:
    """
    features_num = len(features_tuple) + 2
    device = ModelConfig.DEVICE
    # 加载路径为'model.pth'的模型
    model_state_dict = torch.load(model_path)
    # 创建新的神经网络模型
    model = Net(int(features_num), 1).to(device)
    # 用加载的模型状态字典更新模型参数
    model.load_state_dict(model_state_dict)

    # 构造数据集和数据加载器
    val_origin_df = pd.read_csv(val_csv_path, index_col=0)
    feature1, feature2, feature3, feature4 = features_tuple
    val_dataset_list = DataProcess.iter_df(val_origin_df, feature1, feature2, feature3, feature4)
    val_df = pd.DataFrame(val_dataset_list)
    # feature1, feature2, feature3, feature4 = features_tuple
    features_name_list = [name for name in val_df.columns if name != label_name]
    # 划分数据集
    # 将DataFrame中的x和y转换为列表
    x = val_df[features_name_list].values.tolist()
    y = val_df[label_name].values.tolist()
    # 将x和y对应的数据打包成元组
    val_data = list(zip(x, y))
    val_dataset = CustomDataset(val_data)
    val_loader = DataLoader(val_dataset, batch_size=ModelConfig.BATCH_SIZE, shuffle=False)

    model.eval()
    real_value_list, pred_value_list = [], []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            predict_value = model(inputs)
            real_value_list += [item for sublist in labels.cpu().tolist() for item in sublist]
            pred_value_list += [item for sublist in predict_value.cpu().tolist() for item in sublist]

    # 绘图
    plt.figure(figsize=(10, 6))
    plt.plot(real_value_list, label='real')
    plt.plot(pred_value_list, label='predict')
    plt.legend()
    plt.savefig("./run/figure/predict.svg", bbox_inches='tight')
    plt.show()

    # 保存结果到本地
    with open('./run/result/line_predict_result.json', 'w') as f:
        f.write(json.dumps({"real_value": real_value_list, "pred_value": pred_value_list}))


def test_model_of_surface(model_path: str, val_csv_path: str, features_tuple: tuple = (10, 1, 3, 9),
                          label_name: str = "label"):
    """
    预测热力图(有真实标签)
    :param model_path: 模型路径
    :param val_csv_path: 需要预测的热力图的原始csv数据
    :param label_name: 模型的因变量y，默认为"label"
    :return:
    """
    features_num = len(features_tuple) + 2
    device = ModelConfig.DEVICE
    # 加载路径为'model.pth'的模型
    model_state_dict = torch.load(model_path)
    # 创建新的神经网络模型
    model = Net(int(features_num), 1).to(device)
    # 用加载的模型状态字典更新模型参数
    model.load_state_dict(model_state_dict)

    # 构造数据集和数据加载器
    val_origin_df = pd.read_csv(val_csv_path, skiprows=18, index_col=0)
    feature1, feature2, feature3, feature4 = features_tuple
    val_dataset_list = DataProcess.iter_df(val_origin_df, feature1, feature2, feature3, feature4)
    val_df = pd.DataFrame(val_dataset_list)
    features_name_list = [name for name in val_df.columns if name != label_name]
    # 划分数据集
    # 将DataFrame中的x和y转换为列表
    x = val_df[features_name_list].values.tolist()
    y = val_df[label_name].values.tolist()
    # 将x和y对应的数据打包成元组
    val_data = list(zip(x, y))
    val_dataset = CustomDataset(val_data)
    val_loader = DataLoader(val_dataset, batch_size=ModelConfig.BATCH_SIZE, shuffle=False)

    model.eval()
    pred_value_list = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            predict_value = model(inputs)
            pred_value_list += [item for sublist in predict_value.cpu().tolist() for item in sublist]

    # 保存结果到本地
    pd.DataFrame(np.array(pred_value_list).reshape(val_origin_df.shape[0], val_origin_df.shape[1])).to_csv(
        './run/result/surface_predict_result.csv', encoding="utf_8_sig", index=False
    )


def test_model_of_surface_no_label(model_path: str, features_tuple: tuple = (10, 1, 3, 9), row_li: list = None,
                                   column_li: list = None,
                                   label_name: str = "label"):
    """
    预测热力图(无真实标签)
    :param model_path: 模型路径
    :param features_tuple: 常量特征元组
    :param row_li: 需要预测的x的取值
    :param column_li: 需要预测的y的取值
    :param label_name: 模型的因变量y，默认为"label"
    :return:
    """
    features_num = len(features_tuple) + 2
    device = ModelConfig.DEVICE
    # 加载路径为'model.pth'的模型
    model_state_dict = torch.load(model_path)
    # 创建新的神经网络模型
    model = Net(int(features_num), 1).to(device)
    # 用加载的模型状态字典更新模型参数
    model.load_state_dict(model_state_dict)

    # 构造数据集和数据加载器
    val_origin_df = pd.DataFrame(index=row_li, columns=column_li)
    feature1, feature2, feature3, feature4 = features_tuple
    val_dataset_list = DataProcess.iter_df(val_origin_df, feature1, feature2, feature3, feature4)
    val_df = pd.DataFrame(val_dataset_list)
    features_name_list = [name for name in val_df.columns if name != label_name]
    # 划分数据集
    # 将DataFrame中的x和y转换为列表
    x = val_df[features_name_list].values.tolist()
    y = val_df[label_name].values.tolist()
    # 将x和y对应的数据打包成元组
    val_data = list(zip(x, y))
    val_dataset = CustomDataset(val_data)
    val_loader = DataLoader(val_dataset, batch_size=ModelConfig.BATCH_SIZE, shuffle=False)

    model.eval()
    pred_value_list = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            predict_value = model(inputs)
            pred_value_list += [item for sublist in predict_value.cpu().tolist() for item in sublist]

    # 保存结果到本地
    pd.DataFrame(np.array(pred_value_list).reshape(val_origin_df.shape[0], val_origin_df.shape[0])).to_csv(
        './run/result/surface_predict_result_no_label.csv', encoding="utf_8_sig", index=False
    )


if __name__ == '__main__':
    # SECTION: 预测整个面(有真实值)
    test_model_of_surface(model_path="./run/model/best_model_of_6_features.pth",
                          val_csv_path="./datasets/2,7.4,4,7.csv",
                          features_tuple=(10, 1, 3, 9),
                          label_name="label"
                          )

    # SECTION: 预测整个面(无真实值)
    # test_model_of_surface_no_label(model_path="./run/model/best_model_of_6_features.pth",
    #                                features_tuple=(10, 1, 3, 9),
    #                                row_li=[7.11, 7.22, 7.33, ],
    #                                column_li=[4, 5, 6],
    #                                label_name="label"
    #                                )

    # SECTION: 预测单根线
    # test_model_of_line(model_path="./run/model/best_model_of_6_features.pth",
    #                    val_csv_path="./run/result/2,7.4,4,7_filter_x_df.csv",
    #                    features_tuple=(10, 1, 3, 9),
    #                    label_name="label"
    #                    )

    print("Coding Ending!!!")
