# Copyright Xie Fangyuan, Beijing Institute of technology. All rights reserved

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import utils  # 确保你的utils.py包含所有自定义函数
from sklearn.metrics import confusion_matrix
import os.path
from matplotlib import rcParams
import pandas as pd
from matplotlib import pyplot as plt

# =========================数据集路径配置=============================
DATASET_FOLDER_PATH = r"D:\PythonProject\Dataset\PaderbornUniversityDataset"
PROJECT_PATH = r"D:\PythonProject\NeuralNetwork\FaultDiagnosis_Paderborn"
MODEL_SAVEDIT_PATH = os.path.join(PROJECT_PATH, '20260821')
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
learning_rate = 0.0001
num_epochs = 20
EarlyStopValid = False # 是否早停：True-早停；False-不早停
patience = 10           # 早停容忍的epoch数
TimeFrequencyAnalysisVisualization = False # 是否进行时频域可视化图像绘制，True-绘制图像；False-不绘制
fs=5120
rand_seed=28
rcParams['agg.path.chunksize'] = 20000
RANDOM_VALID = True # 非跨工况验证，即比例划分 True-进行非跨轴承验证，存在数据泄露；False-进行跨轴承验证，不存在数据泄露
RANDOM_CWC_VALID = True # 非跨轴承验证下（比例划分下），测试集是否跨轴承（同分布样本是否出现在模型训练及测试中），True-测试集跨轴承，非同分布单独验证；False-测试集比例划分，同分布样本

# 数据集加载
BEARING_SELECT = ['K002', 'KA01', 'KA05', 'KI01', 'KI05', 'K001', 'KA22', 'KI14', 'K003', 'KA06', 'KI07']
all_data = {}
for folder in BEARING_SELECT:
    DATA_PATH = os.path.join(DATASET_FOLDER_PATH, f"{folder}\\{folder}")
    print(DATA_PATH)
    # 加载数据
    data = utils.load_paderborn_dataset(DATA_PATH)
    all_data.update(data)

if not RANDOM_VALID:
    train_loader, test_loader, num_channels, num_classes, class_names, feat_dim = utils.prepare_paderborn_multichannel_dataloaders(
        all_data,
        signal_names=['vibration_1', 'phase_current_1'],
        sample_length=512,
        step=512,
        batch_size=batch_size,
        normalize={'vibration_1': 'sample', 'phase_current_1': 'mean_only'},  # 示例
        train_bearing_codes=['K002', 'KA01', 'KA05', 'KI01', 'KI05'],
        test_bearing_codes=['K001', 'KA22', 'KI14'],
        test_size=None,
        random_seed=rand_seed,
        verbose=True,
        extract_handcrafted=True   # 开启融合
    )
else:
    train_loader, test_loader, num_channels, num_classes, class_names, feat_dim = utils.prepare_paderborn_multichannel_dataloaders(
        all_data,
        signal_names=['vibration_1', 'phase_current_1'],
        sample_length=512,
        step=512,
        batch_size=batch_size,
        normalize={'vibration_1': 'sample', 'phase_current_1': 'mean_only'},  # 保持不变
        train_bearing_codes=None,        # 关键：不指定轴承代码
        test_bearing_codes=None,         # 关键：不指定轴承代码
        test_size=0.2,                   # 测试集比例 20%
        random_seed=rand_seed,                  # 随机种子，保证可复现
        verbose=True,
        extract_handcrafted=True
    )
Feature_Dimension = num_channels

# ========================= 定义双分支模型 =========================
class FusionCNNMLP(nn.Module):
    def __init__(self, input_channels, signal_length, feat_dim, num_classes, dropout_rate):
        super(FusionCNNMLP, self).__init__()
        # ---- CNN 分支 ----
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(2)

        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool1d(2)

        # 计算 CNN 输出长度：经过3次池化，512 -> 64
        cnn_output_len = signal_length // (2**3)  # 512 // 8 = 64
        self.cnn_fc = nn.Linear(128 * cnn_output_len, 256)
        self.cnn_relu = nn.ReLU()

        # ---- MLP 分支（处理手工特征） ----
        self.mlp_fc1 = nn.Linear(feat_dim, 64)
        self.mlp_bn1 = nn.BatchNorm1d(64)
        self.mlp_relu1 = nn.ReLU()
        self.mlp_fc2 = nn.Linear(64, 32)
        self.mlp_bn2 = nn.BatchNorm1d(32)
        self.mlp_relu2 = nn.ReLU()

        # ---- 融合层 ----
        self.fusion_fc = nn.Linear(256 + 32, 128)
        self.fusion_bn = nn.BatchNorm1d(128)
        self.fusion_relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.out = nn.Linear(128, num_classes)

    def forward(self, x_raw, x_feat):
        # CNN 分支
        x = self.pool1(self.relu1(self.bn1(self.conv1(x_raw))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        cnn_out = self.cnn_relu(self.cnn_fc(x))  # (batch, 256)

        # MLP 分支
        y = self.mlp_relu1(self.mlp_bn1(self.mlp_fc1(x_feat)))
        mlp_out = self.mlp_relu2(self.mlp_bn2(self.mlp_fc2(y)))  # (batch, 32)

        # 融合
        fused = torch.cat([cnn_out, mlp_out], dim=1)  # (batch, 288)
        fused = self.fusion_relu(self.fusion_bn(self.fusion_fc(fused)))
        fused = self.dropout(fused)
        out = self.out(fused)
        return out



model = FusionCNNMLP(
    input_channels=num_channels, # 1D-CNN使用的信号通道数量
    signal_length=512,
    feat_dim=feat_dim,          # 直接使用返回的 feat_dim，人工提取的特征数量
    num_classes=num_classes,
    dropout_rate=0.5
).to(device)

# ========== 损失函数、优化器、学习率调度器 ==========
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

# ========== 训练或加载已有模型 ==========
model_path = os.path.join(MODEL_SAVEDIT_PATH, "CNN_20260820Random.pth")

train_losses, train_accs = [], []
test_losses, test_accs = [], []

best_test_loss = float('inf')
patience_counter = 0
start_epoch = 0

if not os.path.exists(model_path):
    print("开始训练...")
    for epoch in range(start_epoch, num_epochs):
        # ---------- 训练 ----------
        model.train()
        running_loss = 0.0
        correct = 0
        total =0

        for raw, feat, labels in train_loader:
            raw, feat, labels = raw.to(device), feat.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(raw, feat)
            loss = criterion(outputs, labels)
            loss.backward()
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = 100.0 * correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")

        # ---------- 测试 ----------
        test_loss, test_acc, _, _ = utils.evaluate_model(
            model, test_loader, device, criterion,
            epoch=epoch, num_epochs=num_epochs, verbose=True, multi_input=True
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

    torch.save(model.state_dict(), model_path)

# 加载最佳模型（用于最终测试）
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
print(f"加载最佳模型进行测试: {model_path}")

if RANDOM_CWC_VALID:
    # =====================非跨轴承训练下的跨工况验证=================================
    _, test_loader,_,_,_,_= utils.prepare_paderborn_multichannel_dataloaders(
        all_data,
        signal_names=['vibration_1', 'phase_current_1'],
        sample_length=512,
        step=512,
        batch_size=batch_size,
        normalize={'vibration_1': 'sample', 'phase_current_1': 'mean_only'},  # 保持不变
        train_bearing_codes=['K002', 'KA01', 'KA05', 'KI01', 'KI05'], # 没有实际使用
        test_bearing_codes=['K003', 'KA06', 'KI07'],
        test_size=None,
        random_seed=rand_seed,                  # 随机种子，保证可复现
        verbose=True,
        extract_handcrafted=True
    )

test_loss, test_acc, y_true, y_pre = utils.evaluate_model(
    model, test_loader, device, criterion, verbose=True, multi_input=True)
print(f"最终测试准确率: {test_acc:.2f}%")

# ========== 混淆矩阵与指标 ==========
FIG_SAVE_PATH = os.path.join(IMG_SAVE_PATH, 'FigureRandom_CWC')
os.makedirs(FIG_SAVE_PATH, exist_ok=True)

cm = confusion_matrix(y_true, y_pre, labels=range(num_classes))
utils.plot_confusion_matrix(
    cm, classes=class_names, normalize=False,
    title=f"Confusion_Matrix",
    IMG_SAVE=True, img_save_path=FIG_SAVE_PATH
)
metrics = utils.calculate_multiclass_metrics(cm, verbose=True, class_names=class_names)

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
        title=f"Loss&Accuracy")
else:
    print("本次直接加载已有模型，跳过训练曲线绘制。")

# 特征提取
X_feat, y_true = utils.extract_features_CNN_Fusion(model=model, data_loader=test_loader, device=device)
#绘制PCA
utils.PCA_plot(
    X_feat=X_feat,
    y_true=y_true,
    num_classes=num_classes,
    le=class_names,
    title=f"PCA",
    FIG_SAVE_VALID=1,
    FIG_SAVE_PATH=FIG_SAVE_PATH,
)
#绘制PCA三主成分可视化
utils.PCA_3D_plot(
    X_feat, y_true, num_classes,
    le=class_names,
    title=f'PCA_3D',
    FIG_SAVE_VALID=True,
    FIG_SAVE_PATH=FIG_SAVE_PATH,
)

# 绘制t-SNE
utils.tSNE_plot(
    X_feat=X_feat,
    y_true=y_true,
    num_classes=num_classes,
    le=class_names,
    title=f"t-SNE",
    FIG_SAVE_VALID=1,
    FIG_SAVE_PATH=FIG_SAVE_PATH,
)
# 绘制UMAP
utils.UMAP_plot(
    X_feat=X_feat,
    y_true=y_true,
    num_classes=num_classes,
    le=class_names,
    title=f"UMAP",
    FIG_SAVE_VALID=True,
    FIG_SAVE_PATH=FIG_SAVE_PATH,
)
plt.show()
