# Copyright Xie Fangyuan, Beijing Institute of Technology. All rights reserved
#
# ===========================Southeaset University gearbox Dataset introduction===================================
# Gearbox dataset is from Southeast University, China. These data are collected from Drivetrain Dynamic Simulator.
# This dataset contains 2 subdatasets, including bearing data and gear data, which are both acquired on Drivetrain
# Dynamics Simulator (DDS). There are two kinds of working conditions with rotating speed - load configuration set
# to be 20-0 and 30-2. Within each file, there are 8rows of signals which represent: 1-motor vibration, 2,3,4-vibration
# of planetary gearbox in three directions: x, y, and z, 5-motor torque, 6,7,8-vibration of parallel gear box in
# three directions: x, y, and z. Signals of rows 2,3,4 are all effective.
# ================================================================================================================

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
import utils
from FaultDiagnosis_Dongnan.utils import extract_features_RNN

# ============================= 配置全局参数 =============================
DATA_FOLDER = r"D:\PythonProject\Dataset\Mechanical-datasets-master\gearbox\gearset"
MODEL_SAVEDIT_PATH = r"D:\PythonProject\NeuralNetwork\FaultDiagnosis_Dongnan"
IMG_SAVE_PATH = os.path.join(MODEL_SAVEDIT_PATH, "ImageSave")
FIG_SAVE_PATH = os.path.join(IMG_SAVE_PATH, "Figure_LSTM")

os.makedirs(IMG_SAVE_PATH, exist_ok=True)
os.makedirs(FIG_SAVE_PATH, exist_ok=True)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# ========== 超参数 ==========
batch_size = 32
learning_rate = 0.001
num_epochs = 70
patience = 5
Feature_Dimension = 3
EarlyStopValid = 0

# ========== 数据加载 ==========
data_path = os.path.join(DATA_FOLDER, "RS20_L0")
dataset_gearbox = utils.load_multi_csv_data(data_path, visualize_sample=False)
train_loader, test_loader, num_classes, le, scaler = utils.preprocess_data(
    dataset_gearbox,
    signal_length=512,
    Feature_Dimension=Feature_Dimension,
    TEST_SIZE=0.2,
    RANDOM_SEED=RANDOM_SEED,
    batch_size=batch_size
)
print(f"类别数: {num_classes}, 类别映射: {dict(zip(le.classes_, range(num_classes)))}")


# ========== 定义 RNN 模型 ==========
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.3):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False
        )
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward_features(self, x):
        x = x.permute(0, 2, 1)
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.layer_norm(out)
        return out

    def forward(self, x):
        out = self.forward_features(x)
        out = self.dropout(out)
        out = self.fc(out)
        return out


model = SimpleRNN(
    input_size=Feature_Dimension,
    hidden_size=128,
    num_layers=2,
    num_classes=num_classes,
    dropout=0.3
).to(device)

# ========== 损失函数、优化器、学习率调度器 ==========
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

# ========== 训练或加载已有模型 ==========
model_path = os.path.join(MODEL_SAVEDIT_PATH, "SimpleRNN_FaultDiagnosisRNNFD20260811.pth")

train_losses, train_accs = [], []
test_losses, test_accs = [], []

best_test_loss = float("inf")
patience_counter = 0

if not os.path.exists(model_path):
    print("未发现已保存模型，开始训练并保存新模型...")
    for epoch in range(num_epochs):
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

        test_loss, test_acc, _, _ = utils.evaluate_model(
            model, test_loader, device, criterion,
            epoch=epoch, num_epochs=num_epochs, verbose=True
        )
        test_losses.append(test_loss)
        test_accs.append(test_acc)

        scheduler.step(test_loss)

        if EarlyStopValid == 1:
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                patience_counter = 0
                torch.save(model.state_dict(), model_path)
                print(f"第 {epoch + 1} 轮验证损失降低，已保存最佳模型: {model_path}")
            else:
                patience_counter += 1
                print(f"验证损失未改善 ({patience_counter}/{patience})")
                if patience_counter >= patience:
                    print(f"早停触发，停止训练于 epoch {epoch + 1}")
                    break

    if EarlyStopValid != 1:
        torch.save(model.state_dict(), model_path)
        print(f"训练完成，已保存模型: {model_path}")
else:
    print(f"发现已保存模型，直接加载并测试: {model_path}")

model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
print(f"加载模型完成: {model_path}")

# ========== 最终测试 ==========
test_loss, test_acc, y_true, y_pre = utils.evaluate_model(
    model, test_loader, device, criterion, verbose=True
)
print(f"最终测试损失: {test_loss:.4f}")
print(f"最终测试准确率: {test_acc:.2f}%")

# ========== 混淆矩阵与指标 ==========
cm = confusion_matrix(y_true, y_pre, labels=range(num_classes))
utils.plot_confusion_matrix(
    cm,
    classes=le.classes_,
    normalize=False,
    title=f"Confusion_Matrix_FD{Feature_Dimension}",
    IMG_SAVE=True,
    img_save_path=FIG_SAVE_PATH
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
        IMG_SAVE=True,
        img_save_path=FIG_SAVE_PATH,
        title=f"Loss&Accuracy_FD{Feature_Dimension}"
    )
else:
    print("本次直接加载已有模型，跳过训练曲线绘制。")


# ========== 提取特征并进行降维可视化 ==========
X_feat, y_feat = extract_features_RNN(model, test_loader, device)

utils.PCA_plot(
    X_feat=X_feat,
    y_true=y_true,
    num_classes=num_classes,
    le=le,
    FIG_SAVE_VALID=1,
    FIG_SAVE_PATH=FIG_SAVE_PATH)

utils.tSNE_plot(
    X_feat=X_feat,
    y_true=y_true,
    num_classes=num_classes,
    le=le,
    FIG_SAVE_VALID=1,
    FIG_SAVE_PATH=FIG_SAVE_PATH)

utils.UMAP_plot(
    X_feat=X_feat,
    y_true=y_true,
    num_classes=num_classes,
    le=le,
    FIG_SAVE_VALID=True,
    FIG_SAVE_PATH=FIG_SAVE_PATH
)
plt.show()
