# Copyright Xie Fangyuan, Beijing Institute of technology. All rights reserved

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import utils  # 确保你的utils.py包含所有自定义函数
from sklearn.metrics import confusion_matrix
import os.path
from matplotlib import pyplot as plt
from matplotlib import rcParams
import pandas as pd
from torch.utils.data import DataLoader, Dataset

# =========================数据集路径配置=============================
DATASET_FOLDER_PATH = r"D:\PythonProject\Dataset\PaderbornUniversityDataset"
PROJECT_PATH = r"D:\PythonProject\NeuralNetwork\FaultDiagnosis_Paderborn"
MODEL_SAVEDIT_PATH = os.path.join(PROJECT_PATH, '20260819')
IMG_SAVE_PATH = os.path.join(MODEL_SAVEDIT_PATH, "ImageSave")
TIME_IMG_SAVE_PATH = os.path.join(IMG_SAVE_PATH, "Time")
FFT_IMG_SAVE_PATH =os.path.join(IMG_SAVE_PATH, "FFT")
ENVELOPE_IMG_SAVE_PATH = os.path.join(IMG_SAVE_PATH, "Envelope")



for path in [MODEL_SAVEDIT_PATH, IMG_SAVE_PATH,
             TIME_IMG_SAVE_PATH, FFT_IMG_SAVE_PATH, ENVELOPE_IMG_SAVE_PATH,]:
    os.makedirs(path, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# ====================设置全局超参数===========================
batch_size = 64
learning_rate = 0.001
num_epochs = 40
EarlyStopValid = 1 # 是否早停：1-早停；0-不早停
patience = 10           # 早停容忍的epoch数
TimeFrequencyAnalysisVisualization = False # 是否进行时频域可视化图像绘制，True-绘制图像；False-不绘制
# fs=5120
rand_seed=28
rcParams['agg.path.chunksize'] = 20000


# 测试数据集加载
BEARING_SELECT = ['K001', 'KA04', 'KI04']
all_data = {}
for folder in BEARING_SELECT:
    DATA_PATH = os.path.join(DATASET_FOLDER_PATH, f"{folder}\\{folder}")
    print(DATA_PATH)
    # 加载数据
    data = utils.load_paderborn_dataset(DATA_PATH)
    all_data.update(data)



# print("加载的文件列表:", list(all_data.keys()))
# first_key = list(all_data.keys())[0]
# vib = all_data[first_key].get('vibration_1')
# if vib is not None:
#     print(f"振动信号前10个点: {vib[:10]}")
# else:
#     print("该文件中未找到 vibration_1 信号")
#
# phase_cur = all_data[first_key].get('phase_current_1')
# if vib is not None:
#     print(f"相电流信号前10个点: {phase_cur[:10]}")
# else:
#     print("该文件中未找到 phase_current_1 信号")

train_loader, test_loader, num_channels, num_classes, class_names = utils.prepare_paderborn_multichannel_dataloaders(
    all_data,
    signal_names=['vibration_1', 'phase_current_1'],
    sample_length=512,
    step=256,
    test_size=1,
    batch_size=batch_size,
    random_seed=rand_seed,
    normalize=False
)

print(f"通道数: {num_channels}, 类别数: {num_classes}, 类别名称: {class_names}")
print(f"训练批次数: {len(train_loader)}, 测试批次数: {len(test_loader)}")