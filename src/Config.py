# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time  : 2023/4/7  22:04
# @Author: Yanjun Hao
# @File  : Config.py
import torch


class ModelConfig:
    BATCH_SIZE = 32
    NEURONS_NUM = (2024, 1024, 512, 128)
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    EPOCHS = 100


class PredictLine:
    data = {
        "2,7.4,4,7.csv": {
            "x": [1000],
            "y": [1005]
        },
        "4,7,8,8.9.csv": {
            "x": [1010],
            "y": [1100]
        },
    }
