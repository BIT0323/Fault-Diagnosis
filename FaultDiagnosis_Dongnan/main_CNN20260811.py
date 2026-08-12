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
import numpy as np
import os
import utils  # 确保你的utils.py包含所有自定义函数
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


# ============================= 配置全局参数 =============================
DATA_FOLDER = r"D:\PythonProject\Dataset\Mechanical-datasets-master\gearbox\gearset"
MODEL_SAVEDIT_PATH = r"D:\PythonProject\NeuralNetwork\FaultDiagnosis_Dongnan"
IMG_SAVE_PATH = os.path.join(MODEL_SAVEDIT_PATH, "ImageSave")
FIG_SAVE_PATH = os.path.join(IMG_SAVE_PATH, "Figure_CNN")

os.makedirs(IMG_SAVE_PATH, exist_ok=True)
os.makedirs(FIG_SAVE_PATH, exist_ok=True)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# ========== 超参数 ==========
batch_size = 32
learning_rate = 0.001
num_epochs = 20
patience = 10           # 早停容忍的epoch数
Feature_Dimension = 3  # 输入通道数（三轴振动）
EarlyStopValid = 1 # 是否早停：1-早停；0-不早停

# ========== 数据加载 ==========
data_path = os.path.join(DATA_FOLDER, "RS20_L0")
dataset_gearbox = utils.load_multi_csv_data(data_path, cache_path="gearbox_data_cacheRS20.pkl",visualize_sample=False)
train_loader, test_loader, num_classes, le, scaler = utils.preprocess_data(
    dataset_gearbox,
    signal_length=512,
    Feature_Dimension=Feature_Dimension,
    TEST_SIZE=0.2,
    RANDOM_SEED=RANDOM_SEED,
    batch_size=batch_size
)
print(f"类别数: {num_classes}, 类别映射: {dict(zip(le.classes_, range(num_classes)))}")

# ========== 定义CNN模型（带Dropout） ==========
class SimpleCNN(nn.Module):
    def __init__(self, input_channels, num_classes, dropout_rate=0.5):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(2, 2)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(2, 2)

        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool1d(2, 2)

        self.fc1 = nn.Linear(128 * 64, 256)  # 经过3次池化: 512 -> 64
        self.relu4 = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = self.relu4(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

model = SimpleCNN(input_channels=Feature_Dimension, num_classes=num_classes, dropout_rate=0.5).to(device)

# ========== 损失函数、优化器、学习率调度器 ==========
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

# ========== 训练或加载已有模型 ==========
model_path = os.path.join(MODEL_SAVEDIT_PATH, "SimpleCNN_FaultDiagnosisCNNFD320260811n.pth")

train_losses, train_accs = [], []
test_losses, test_accs = [], []

best_test_loss = float('inf')
patience_counter = 0
start_epoch = 0

# 如果已存在模型且希望继续训练，可加载状态（这里简单处理，直接训练新模型）
# 若你想从检查点恢复，可在此添加加载逻辑
if not os.path.exists(model_path):
    print("开始训练...")
    for epoch in range(start_epoch, num_epochs):
        # ---------- 训练 ----------
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = 100.0 * correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # ---------- 测试 ----------
        test_loss, test_acc, _, _ = utils.evaluate_model(
            model, test_loader, device, criterion,
            epoch=epoch, num_epochs=num_epochs, verbose=True
        )
        test_losses.append(test_loss)
        test_accs.append(test_acc)

        # ---------- 学习率调度 ----------
        scheduler.step(test_loss)


        # ---------- 早停 & 最佳模型保存 ----------
        if EarlyStopValid == 1:
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                patience_counter = 0
                torch.save(model.state_dict(), model_path)
                # print(f"  验证损失降低，保存最佳模型至 {model_path}")
            else:
                patience_counter += 1
                # print(f"  验证损失未改善 ({patience_counter}/{patience})")
                if patience_counter >= patience:
                    print(f"早停触发，停止训练于 epoch {epoch+1}")
                    break

# 加载最佳模型（用于最终测试）
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
print(f"加载最佳模型进行测试: {model_path}")

# # 变负载交叉验证(可选)
# data_path = os.path.join(DATA_FOLDER, "RS30_L2")
# dataset_gearboxRS30_L2 = utils.load_multi_csv_data(data_path, cache_path='gearbox_data_cache_RS30.pkl', visualize_sample=False)
# train_loaderRS30, test_loaderRS30, num_classesRS30, leRS30, scalerRS30 = utils.preprocess_data(
#     dataset_gearboxRS30_L2,
#     signal_length=512,
#     Feature_Dimension=Feature_Dimension,
#     TEST_SIZE=0.2,
#     RANDOM_SEED=RANDOM_SEED,
#     batch_size=batch_size
# )
# train_loader, test_loader, num_classes, le, scaler = train_loaderRS30, test_loaderRS30, num_classesRS30, leRS30, scalerRS30

# ========== 最终测试 ==========
test_loss, test_acc, y_true, y_pre = utils.evaluate_model(
    model, test_loader, device, criterion, verbose=True
)
# test_loss, test_acc, y_true, y_pre = utils.evaluate_model(
#     model, test_loader, device, criterion, verbose=True
# )
print(f"最终测试准确率: {test_acc:.2f}%")

# ========== 混淆矩阵与指标 ==========
cm = confusion_matrix(y_true, y_pre, labels=range(num_classes))
utils.plot_confusion_matrix(
    cm, classes=le.classes_, normalize=False,
    title=f"Confusion_Matrix_FD{Feature_Dimension}",
    IMG_SAVE=True, img_save_path=FIG_SAVE_PATH
)
metrics = utils.calculate_multiclass_metrics(cm, verbose=True, class_names=le.classes_)

# ========== 绘制训练曲线 ==========
if train_losses and test_losses:
    utils.model_train_curve_plot(
        num_epochs=len(train_losses),
        train_losses=train_losses,
        train_accs=train_accs,
        test_losses=test_losses,
        test_accs=test_accs,
        IMG_SAVE_VALID=True,
        img_save_path=FIG_SAVE_PATH,
        title=f"Loss&Accuracy_FD{Feature_Dimension}")
else:
    print("本次直接加载已有模型，跳过训练曲线绘制。")

# 特征提取
X_feat, y_true = utils.extract_features_CNN(model=model, data_loader=test_loader, device=device)
#绘制PCA
utils.PCA_plot(
    X_feat=X_feat,
    y_true=y_true,
    num_classes=num_classes,
    le=le,
    title=f"PCA_FD{Feature_Dimension}_RS20",
    FIG_SAVE_VALID=1,
    FIG_SAVE_PATH=FIG_SAVE_PATH)
# 绘制t-SNE
utils.tSNE_plot(
    X_feat=X_feat,
    y_true=y_true,
    num_classes=num_classes,
    le=le,
    title=f"t-SNE_FD{Feature_Dimension}_RS20",
    FIG_SAVE_VALID=1,
    FIG_SAVE_PATH=FIG_SAVE_PATH)
# 绘制UMAP
utils.UMAP_plot(
    X_feat=X_feat,
    y_true=y_true,
    num_classes=num_classes,
    le=le,
    title=f"UMAP_FD{Feature_Dimension}_RS20",
    FIG_SAVE_VALID=True,
    FIG_SAVE_PATH=FIG_SAVE_PATH
)
plt.show()