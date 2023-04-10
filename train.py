# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time  : 2023/4/7  21:10
# @Author: Yanjun Hao
# @File  : train.py

import pandas as pd
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


def train_model(epochs: int = 300, file_path: str = None, label_name: str = "label") -> None:
    """
    训练模型
    :param epochs: 模型的训练epoch
    :param file_path: 训练的数据集路径
    :param label_name: 模型的因变量y，默认为"label"
    :return:
    """
    # 设定随机种子和设备
    torch.manual_seed(123)
    device = ModelConfig.DEVICE
    print(f"[italic bold green]-------------------------正在使用{device}训练----------------------------")

    # 构造数据集和数据加载器
    origin_df = pd.read_csv(file_path, index_col=None)
    # 归一化
    # 将需要归一化的列选择出来
    cols_to_norm = [name for name in origin_df.columns if name != label_name]

    # 对需要归一化的列进行归一化
    scaler = MinMaxScaler()
    origin_df[cols_to_norm] = scaler.fit_transform(origin_df[cols_to_norm])

    # 划分数据集
    # 将DataFrame中的x和y转换为列表
    x = origin_df[cols_to_norm].values.tolist()
    y = origin_df[label_name].values.tolist()
    # 将x和y对应的数据打包成元组
    data = list(zip(x, y))
    train_data, val_data = random_split(data, [int(len(data) * 0.8), len(data) - int(len(data) * 0.8)])

    # data = [(torch.randn(4), torch.randn(1)) for i in range(1000)]
    # train_data, val_data = random_split(data, [800, 200])
    train_dataset = CustomDataset(train_data)
    val_dataset = CustomDataset(val_data)
    train_loader = DataLoader(train_dataset, batch_size=ModelConfig.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=ModelConfig.BATCH_SIZE, shuffle=False)

    # 构造模型和损失函数
    model = Net(input_size=int(len(cols_to_norm)), output_size=1, node_size=ModelConfig.NEURONS_NUM).to(device)
    criterion = nn.MSELoss()

    # 构造优化器
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 开始训练模型
    train_loss_list = []
    val_loss_list = []
    best_val_loss = float('inf')
    for epoch in range(epochs):
        train_loss = train(model, optimizer, criterion, train_loader, device)
        train_loss_list.append(train_loss)
        val_loss = evaluate(model, criterion, val_loader, device)
        val_loss_list.append(val_loss)
        if val_loss < best_val_loss:
            torch.save(model.state_dict(), f"./run/model/best_model_of_{len(cols_to_norm)}_features.pth")
        if epoch % 10 == 0:
            print(epoch, ':', train_loss)

    # 绘图
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss_list, label='train_loss')
    plt.plot(val_loss_list, label='val_loss')
    plt.legend()
    plt.savefig("./run/figure/loss.svg", bbox_inches='tight')
    plt.show()

    # 保存loss到本地
    with open('run/result/loss.json', 'w') as f:
        f.write(json.dumps({"train_loss": train_loss_list, "val_loss": val_loss_list}))

    # print("[italic bold green]-------------------------开始预测----------------------------")
    # select_line_to_predict(model_path=f"./run/model/best_model_of_{len(cols_to_norm)}_features.pth",
    #                        input_val_data=val_data, features_num=len(cols_to_norm))


if __name__ == '__main__':
    train_model(epochs=500,
                file_path="run/result/train_dataset_df.csv",
                label_name="label"
                )
    print("[italic bold blue]Code Ending!!!")
