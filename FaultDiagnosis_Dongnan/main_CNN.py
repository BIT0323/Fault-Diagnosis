# Copyright Xie Fangyuan, Beijing Institute of technology. All rights reserved
#
# ===========================Southeaset University gearbox Dataset introduction===================================
# Gearbox dataset is from Southeast University, China. These data are collected from Drivetrain Dynamic Simulator.
# This dataset contains 2 subdatasets, including bearing data and gear data, which are both acquired on Drivetrain
# Dynamics Simulator (DDS). There are two kinds of working conditions with rotating speed - load configuration set
# to be 20-0 and 30-2. Within each file, there are 8rows of signals which represent: 1-motor vibration, 2,3,4-vibration
# of planetary gearbox in three directions: x, y, and z, 5-motor torque, 6,7,8-vibration of parallel gear box in
# three directions: x, y, and z. Signals of rows 2,3,4 are all effective.
# ================================================================================================================
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os

from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
import utils

# =============================配置全局参数=============================
# 指定数据集路径
DATA_FOLDER =r"D:\PythonProject\Dataset\Mechanical-datasets-master\gearbox\gearset"
MODEL_SAVEDIT_PATH = r"D:\PythonProject\NeuralNetwork\FaultDiagnosis_Dongnan"
IMG_SAVE_PATH = os.path.join(MODEL_SAVEDIT_PATH, "ImageSave")
FIG_SAVE_PATH = os.path.join(IMG_SAVE_PATH, "Figure_CNN")

# 如果指定路径不存在，则创建新文件夹或路径
os.makedirs(IMG_SAVE_PATH, exist_ok=True)
os.makedirs(FIG_SAVE_PATH, exist_ok=True)
# 特征列前缀（根据你的数据列名调整，比如振动信号列：vibration_z_motor,比如电机转矩列：torque_motor）
FEATURE_PREFIX = "vibration_x_gbox"
# 设置随机种子保证结果可复现
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ========== 1. 检查设备 ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# ========== 2. 超参数设置 ==========
batch_size = 32
learning_rate = 0.001
num_epochs = 20

#======================= 3. 读取RS20_L0工况及数据预处理===============================
DATA_FOLDER = os.path.join(DATA_FOLDER, "RS20_L0")
dataset_gearbox = utils.load_multi_csv_data(DATA_FOLDER,visualize_sample=1)
Feature_Dimension = 3
train_loader, test_loader, num_classes, le, scaler = utils.preprocess_data(dataset_gearbox, signal_length=512,
                                                                Feature_Dimension=Feature_Dimension, TEST_SIZE=0.2,
                                                                RANDOM_SEED=RANDOM_SEED, batch_size=batch_size)

# ========== 4. 定义 CNN 模型 ==========
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv1d(Feature_Dimension, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(128 * 64, 256)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(256, 5)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        x = x.view(x.size(0), -1)              # 展平
        x = self.relu4(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleCNN().to(device)

# ========== 5. 损失函数与优化器 ==========
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# ========== 6. 训练或者加载已有模型 ==========
# 创建空列表以绘制训练/测试loss和acc曲线
train_losses = []
train_accs = []
test_losses = []
test_accs = []

model_path = os.path.join(MODEL_SAVEDIT_PATH, "SimpleCNN_FaultDiagnosisCNNFD320260724.pth")
if not os.path.exists(model_path):

    print("开始训练...")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)          # 模型接收 (batch, channels, length)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)  # 获取预测类别索引
            total += labels.size(0)  # 累加本 Batch 的样本数
            correct += (predicted == labels).sum().item()  # 累加本 Batch 猜对的数量

        train_loss = running_loss / len(train_loader)
        train_acc = 100.0 * correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")

        # ============ 7.测试 ===============
        # _, _, test_loss, test_acc = utils.evaluate_pytorch_model(model=model, test_loader=test_loader, device=device,
        #                                            criterion=criterion, epoch=epoch,
        #                                            num_epochs=num_epochs, EachEpoch=True, class_names=le.classes_)
        test_loss, test_acc, y_true, y_pre =utils.evaluate_model(model=model, test_loader=test_loader, device=device,
                                                   criterion=criterion, epoch=epoch,
                                                    num_epochs=num_epochs,verbose=True)

        test_losses.append(test_loss)
        test_accs.append(test_acc)

    torch.save(model.state_dict(), model_path)

    # 绘制训练过程中模型在训练集和测试集上的loss、acc变化曲线
    utils.model_train_curve_plot(num_epochs=num_epochs, train_losses=train_losses, train_accs=train_accs,
                                 test_losses=test_losses, test_accs=test_accs, IMG_SAVE=True, img_save_path=FIG_SAVE_PATH,
                                 title=f"Loss&Accuracy_FD{Feature_Dimension}")

else:
    print(f"发现已保存模型{model_path}, 正在加载...")
    # ========== 7. 测试 ==========
    model = SimpleCNN()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    # metrics, cm, _, _ = utils.evaluate_pytorch_model(model=model, test_loader=test_loader, device=device, criterion=criterion, epoch=None,
    #                                      num_epochs=None, EachEpoch=False, class_names=le.classes_)

    test_loss, test_acc, y_true, y_pre = utils.evaluate_model(model=model, test_loader=test_loader, device=device,
                                                              criterion=criterion, epoch=None,
                                                              num_epochs=None, verbose=None)
# 计算混淆矩阵
cm = confusion_matrix(y_true, y_pre, labels=range(num_classes))
utils.plot_confusion_matrix(cm, classes=le.classes_, normalize=False, title=f"Confusion_Matrix_FD{Feature_Dimension}",
                            IMG_SAVE=True, img_save_path=FIG_SAVE_PATH)
metrics = utils.calculate_multiclass_metrics(cm, verbose=True, class_names=le.classes_)