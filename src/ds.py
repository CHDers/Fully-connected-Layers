'''
🚀🚀🚀🚀🚀🚀:
Descripttion: Yanjun Hao的代码
version: 1.0.0
Author: Yanjun Hao
Date: 2023-04-04 17:39:28
LastEditors: Yanjun Hao
LastEditTime: 2023-04-04 20:50:04
'''
import pandas as pd
import numpy as np
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

class DataScience:
    def __init__(self, src_data):
        self.src_data = src_data
        self.tga_data = None

    def data_science(self, col_name: str, step:int=10, col_num:int=4) -> None:
        """数据处理

        Args:
            col_name (str):需要提取的列名, 取得是'actual'
            step (int): 取值得步长, 默认为10
            col_num (int): 需要取前几个值, 默认为4
        """
        result_list = []
        for idx in range(0, self.src_data.shape[0] - step, step+1):
            filter_dict = {}
            filter_dict['all value'] = self.src_data.iloc[idx:idx+step, :][col_name].values.tolist()
            for index in range(col_num):
                filter_dict[f'temp_{index+1}'] = filter_dict['all value'][index]
            result_list.append(filter_dict)
            filter_dict['Average value'] = np.mean(filter_dict['all value'])
            filter_dict['actual'] = self.src_data[[col_name]].iloc[idx+step, :].values[0]

        self.tga_data = pd.DataFrame(result_list)


if __name__ == '__main__':
    FILEPATH = "../data/ra.xlsx"
    df = pd.read_excel(FILEPATH, index_col=None)
    data_science_obj = DataScience(src_data=df)
    data_science_obj.data_science(col_name='actual', step=10, col_num=4)
    print(data_science_obj.tga_data)
    data_science_obj.tga_data.to_csv("../data/tga_data.csv", index=False)

