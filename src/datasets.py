'''
🚀🚀🚀🚀🚀🚀: 
Descripttion: 数据预处理，将数据处理为可以直接导入神经网络的数据
version: 1.0.0
Author: Yanjun Hao
Date: 2023-04-07 14:09:38
LastEditors: Yanjun Hao
LastEditTime: 2023-04-07 20:24:25
'''
import pandas as pd
import os
from rich import print
from pathlib import Path
import sys

FILE = Path(__file__).resolve()
ROOT_0 = FILE.parents[0]  # project root directory
ROOT_1 = FILE.parents[1]  # project root directory
if str(ROOT_0) not in sys.path:
    sys.path.append(str(ROOT_0))  # add ROOT to PATH
if str(ROOT_1) not in sys.path:
    sys.path.append(str(ROOT_1))  # add ROOT to PATH

from src.Config import PredictLine


class DataProcess:
    def __init__(self, base_path):
        """
        文件夹路径，该路径下为需要数据处理的csv文件
        :param base_path:
        """
        self.base_path = base_path
        self.file_name_list = os.listdir(self.base_path)
        self.train_data, self.valid_data = self.read_data()

    @staticmethod
    def iter_df(input_df: pd.DataFrame, feature1: float, feature2: float, feature3: float, feature4: float) -> list:
        """
        将DataFrame根据某几个常量转为可供神经网络运行的list
        :param input_df: 需要转化的DataFrame
        :param feature1: 特征1(常量)
        :param feature2: 特征2(常量)
        :param feature3: 特征3(常量)
        :param feature4: 特征4(常量)
        :return:
        """
        dataset_list = []
        for row_ in input_df.index:
            for column_ in input_df.columns:
                value = input_df.loc[row_, column_]
                new_column_ = column_.replace(" ", "").split('=')[-1]
                dataset_list.append(
                    {"feature1": float(feature1),
                     "feature2": float(feature2),
                     "feature3": float(feature3),
                     "feature4": float(feature4),
                     "row_value": float(str(row_).replace(" ", "")),
                     "column_value": float(new_column_),
                     "label": float(value),
                     }
                )
        return dataset_list

    def read_data(self):
        """
        将某个路径下的多个csv文件合并为一个满足神经网络输入格式的数据集
        :return:
        """
        train_dataset_list, val_dataset_list = [], []
        for file_name in self.file_name_list:
            print(f"Reading file: {os.path.join(self.base_path, file_name)}")
            feature1, feature2, feature3, feature4 = [float(item) for item in
                                            os.path.splitext(file_name.replace(" ", ""))[0].split(',')]
            if file_name.endswith('.csv'):  # 匹配csv格式的文件
                df = pd.read_csv(os.path.join(self.base_path, file_name), skiprows=18, index_col=0)  # 加载csv文件
                # 生成测试数据集
                train_dataset_list += self.iter_df(df, feature1, feature2, feature3, feature4)
                # 生成训练数据集
                if file_name in PredictLine.data:
                    x_idx_list, y_idx_list = PredictLine.data.get(str(file_name)).get("x"), \
                                             PredictLine.data.get(str(file_name)).get("y")
                    filter_x_df = df.iloc[x_idx_list, :]
                    filter_y_df = df.iloc[:, y_idx_list]
                    filter_x_df.to_csv(
                        rf"../run/result/{os.path.splitext(file_name.replace(' ', ''))[0]}_filter_x_df.csv",
                        encoding="utf_8_sig",)
                    filter_y_df.to_csv(
                        rf"../run/result/{os.path.splitext(file_name.replace(' ', ''))[0]}_filter_y_df.csv",
                        encoding="utf_8_sig")
                    val_dataset_list += self.iter_df(filter_x_df, feature1, feature2, feature3, feature4)
                    val_dataset_list += self.iter_df(filter_y_df, feature1, feature2, feature3, feature4)

        train_dataset_df = pd.DataFrame(train_dataset_list).dropna(axis=0, how='any')
        train_dataset_df.to_csv(r"../run/result/train_dataset_df.csv", encoding="utf_8_sig", index=False)
        val_dataset_df = pd.DataFrame(val_dataset_list).dropna(axis=0, how='any')
        val_dataset_df.to_csv(r"../run/result/val_dataset_df.csv", encoding="utf_8_sig", index=False)
        return train_dataset_df, val_dataset_df


if __name__ == '__main__':
    data_obj = DataProcess(r"../datasets")
    print("[italic bold blue]Code Ending!!!")
