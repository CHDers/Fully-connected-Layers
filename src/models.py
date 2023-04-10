'''
🚀🚀🚀🚀🚀🚀: 
Descripttion: Yanjun Hao的代码
version: 1.0.0
Author: Yanjun Hao
Date: 2023-04-02 10:16:10
LastEditors: Yanjun Hao
LastEditTime: 2023-04-02 23:34:08
'''
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
from src.Config import ModelConfig


# 定义自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        x = torch.Tensor(self.data[index][0])
        y = torch.Tensor([self.data[index][1]])
        return x, y

    def __len__(self):
        return len(self.data)


# 定义全连接神经网络模型
class Net(nn.Module):
    def __init__(self, input_size, output_size, node_size=(2024, 1024, 512, 128)):
        super().__init__()
        self.fc1 = nn.Linear(input_size, node_size[0])
        self.fc2 = nn.Linear(node_size[0], node_size[1])
        self.fc3 = nn.Linear(node_size[1], node_size[2])
        self.fc4 = nn.Linear(node_size[2], node_size[3])

        self.fc5 = nn.Linear(node_size[3], output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        return x


# 定义训练函数
def train(model, optimizer, criterion, train_loader, device):
    model.train()
    train_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * inputs.size(0)
    return train_loss / len(train_loader.dataset)


# 定义验证函数
def evaluate(model, criterion, val_loader, device):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
    return val_loss / len(val_loader.dataset)


def load_model_to_predict(model_path: str, input_val_data: object, features_num: int = 4) -> None:
    """
    加载最优模型进行预测
    :param features_num: 模型的输入特征数
    :param model_path: 最优模型路径
    :param input_val_data: 预测数据(有标签)
    :return:
    """
    device = ModelConfig.DEVICE
    # 加载路径为'model.pth'的模型
    model_state_dict = torch.load(model_path)

    # 创建新的神经网络模型
    model = Net(int(features_num), 1).to(device)

    # 用加载的模型状态字典更新模型参数
    model.load_state_dict(model_state_dict)

    # 进行预测
    val_dataset = CustomDataset(input_val_data)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

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
    plt.plot(real_value_list[:300], label='real')
    plt.plot(pred_value_list[:300], label='predict')
    plt.legend()
    plt.savefig("../run/figure/predict.svg", bbox_inches='tight')
    plt.show()

    # 保存结果到本地
    with open('../run/result/predict_result.json', 'w') as f:
        f.write(json.dumps({"real_value": real_value_list, "pred_value": pred_value_list}))


def predict_single_value(model_path: str, val_data=torch.Tensor(1, 4)) -> None:
    """
    只预测值，不进行验证
    :param model_path: 模型本地路径
    :param val_data: 需要预测的自变量值(无标签)
    :return:
    """
    device = ModelConfig.DEVICE
    # 加载神经网络模型
    model = Net(4, 1).to(device)
    model_state_dict = torch.load(model_path)
    model.load_state_dict(model_state_dict)

    model.eval()
    with torch.no_grad():
        # NOTE: 预测单个值
        pred_value = [item for sublist in model(val_data.to(device)).cpu().tolist() for item in sublist]

    # 保存结果到本地
    with open('../run/predict_value.json', 'w') as f:
        f.write(json.dumps({"pred_value": pred_value}))


def select_line_to_predict(model_path: str, input_val_data: object, features_num: int = 4):
    """
    加载最优模型进行预测
    :param features_num: 模型的输入特征数
    :param model_path: 最优模型路径
    :param input_val_data: 预测数据(有标签)
    :return:
    """
    device = ModelConfig.DEVICE
    # 加载路径为'model.pth'的模型
    model_state_dict = torch.load(model_path)

    # 创建新的神经网络模型
    model = Net(int(features_num), 1).to(device)

    # 用加载的模型状态字典更新模型参数
    model.load_state_dict(model_state_dict)

    # 进行预测
    val_dataset = CustomDataset(input_val_data)
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
    plt.plot(real_value_list[:300], label='real')
    plt.plot(pred_value_list[:300], label='predict')
    plt.legend()
    plt.savefig("../run/figure/predict.svg", bbox_inches='tight')
    plt.show()

    # 保存结果到本地
    with open('../run/result/predict_result.json', 'w') as f:
        f.write(json.dumps({"real_value": real_value_list, "pred_value": pred_value_list}))


# 定义主函数
def main(epochs: int = 300, file_path: str = None) -> None:
    # 设定随机种子和设备
    torch.manual_seed(123)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 构造数据集和数据加载器
    origin_df = pd.read_excel(file_path, index_col=None)
    # 归一化
    # 将需要归一化的列选择出来
    cols_to_norm = ['x_point', 'Temperature', 'Enthalpy value', 'time']

    # 对需要归一化的列进行归一化
    scaler = MinMaxScaler()
    origin_df[cols_to_norm] = scaler.fit_transform(origin_df[cols_to_norm])

    # 划分数据集
    # 将DataFrame中的x和y转换为列表
    x = origin_df[cols_to_norm].values.tolist()
    y = origin_df['actual'].values.tolist()
    # 将x和y对应的数据打包成元组
    data = list(zip(x, y))
    train_data, val_data = random_split(data, [int(len(data) * 0.8), int(len(data) * 0.2)])

    # data = [(torch.randn(4), torch.randn(1)) for i in range(1000)]
    # train_data, val_data = random_split(data, [800, 200])
    train_dataset = CustomDataset(train_data)
    val_dataset = CustomDataset(val_data)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # 构造模型和损失函数
    model = Net(4, 1).to(device)
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
            torch.save(model.state_dict(), "../run/best_model.pth")
        if epoch % 10 == 0:
            print(epoch, ':', train_loss)

    # 绘图
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss_list, label='train_loss')
    plt.plot(val_loss_list, label='val_loss')
    plt.legend()
    plt.savefig("../run/loss.svg", bbox_inches='tight')
    plt.show()

    # 保存loss到本地
    with open('../run/loss.json', 'w') as f:
        f.write(json.dumps({"train_loss": train_loss_list, "val_loss": val_loss_list}))

    print("------------开始预测---------------")
    load_model_to_predict(model_path="../run/best_model.pth", input_val_data=val_data)


if __name__ == '__main__':
    EPOCH = 100
    FILEPATH = "../data/ra.xlsx"
    main(epochs=EPOCH, file_path=FILEPATH)

    # SECTION: 单值预测
    predict_single_value(model_path="../run/best_model.pth", val_data=torch.Tensor(1, 4))
