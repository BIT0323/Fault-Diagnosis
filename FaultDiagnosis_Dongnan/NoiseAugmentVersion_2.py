# Copyright Xie Fangyuan, Beijing Institute of technology. All rights reserved
#
# ===========================Southeast University gearbox Dataset introduction===================================
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
import os.path
from matplotlib import pyplot as plt
from matplotlib import rcParams
import pandas as pd


# =========================数据集路径配置=============================
DATASET_FOLDER_PATH = r"D:\PythonProject\Dataset\Mechanical-datasets-master\gearbox\gearset"
PROJECT_PATH = r"D:\PythonProject\NeuralNetwork\FaultDiagnosis_Dongnan"
MODEL_SAVEDIT_PATH = os.path.join(PROJECT_PATH, '20260813')
IMG_SAVE_PATH = os.path.join(MODEL_SAVEDIT_PATH, "ImageSave")
FIG_SAVE_PATH = os.path.join(IMG_SAVE_PATH, "Figure_TrainNoise_TestNoise")
TIME_IMG_SAVE_PATH = os.path.join(IMG_SAVE_PATH, "Time")
FFT_IMG_SAVE_PATH =os.path.join(IMG_SAVE_PATH, "FFT")
ENVELOPE_IMG_SAVE_PATH = os.path.join(IMG_SAVE_PATH, "Envelope")



for path in [MODEL_SAVEDIT_PATH, IMG_SAVE_PATH, FIG_SAVE_PATH,
             TIME_IMG_SAVE_PATH, FFT_IMG_SAVE_PATH, ENVELOPE_IMG_SAVE_PATH,]:
    os.makedirs(path, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# ====================设置全局超参数===========================
batch_size = 32
learning_rate = 0.001
num_epochs = 40
EarlyStopValid = 1 # 是否早停：1-早停；0-不早停
patience = 10           # 早停容忍的epoch数
Feature_Dimension = 3  # 输入通道数（三轴振动）
TimeFrequencyAnalysisVisualization = False # 是否进行时频域可视化图像绘制，True-绘制图像；False-不绘制

fs=5120
RANDOM_SEED=42
rcParams['agg.path.chunksize'] = 20000
# ========== 噪声配置 ==========
use_fixed_noise = False  # 设为 True 则使用固定噪声，False 则随机
# 固定噪声参数（当 use_fixed_noise=True 时生效）
fixed_noise_config = {
    'noise_type': 'all',          # 可选 'gaussian', 'impulse', 'harmonic', 'all'
    'gaussian_kw': {'snr_db': 20, 'RANDOM_SEED': 42},
    'impulse_kw': {'impulse_prob': 0.01, 'amplitude_scale': 0.5, 'RANDOM_SEED': 42},
    'harmonic_kw': {'freqs': [30, 90], 'amp_ratio': 0.2, 'fs': fs, 'RANDOM_SEED': 42}
}
noise_prob = 0.5

# ====================================时频域分析可视化=====================================
if TimeFrequencyAnalysisVisualization:
    config = {
        # 优先使用 Times New Roman，中文缺失时回退到 SimHei（黑体）
        "font.family": ['Times New Roman', 'SimHei', 'Microsoft YaHei'],
        "mathtext.fontset": 'stix',
        'axes.unicode_minus': False,
        'savefig.dpi': 600,
    }
    rcParams.update(config)
    file_path = utils.dataset_file_path_get(os.path.join(DATASET_FOLDER_PATH,'RS20_L0'), "Chipped_20_0.csv")
    df = utils.data_read(file_path)
    acc_arr = df.iloc[:,7].values
    utils.plt_time_domain(arr=acc_arr,fs=fs, title="Chipped_20_0", img_save_path=TIME_IMG_SAVE_PATH)
    utils.plt_fft_img(acc_arr, fs=fs, title='Chipped_20_0', img_save_path=FFT_IMG_SAVE_PATH, vline=[20, 40, 60, 80, 100], xlim=500)

    # =============仅加入Guassian噪声=====================
    acc_arr_noise = utils.add_gaussian_noise(signal=acc_arr, snr_db=10, RANDOM_SEED=RANDOM_SEED)
    utils.plt_time_domain(arr=acc_arr_noise,fs=fs, title="Chipped_20_0_Guassian", img_save_path=TIME_IMG_SAVE_PATH)
    utils.plt_fft_img(acc_arr_noise, fs=fs, title='Chipped_20_0_Gaussian', img_save_path=FFT_IMG_SAVE_PATH, vline=[20, 40, 60, 80, 100], xlim=500)

    # ==============仅加入Impulse噪声=====================
    acc_arr_noise = utils.add_impulse_noise(signal=acc_arr, impulse_prob=0.01, amplitude_scale=4, RANDOM_SEED=RANDOM_SEED)
    utils.plt_time_domain(arr=acc_arr_noise,fs=fs, title="Chipped_20_0_Impulse", img_save_path=TIME_IMG_SAVE_PATH)
    utils.plt_fft_img(acc_arr_noise, fs=fs, title='Chipped_20_0_Impulse', img_save_path=FFT_IMG_SAVE_PATH, vline=[20, 40, 60, 80, 100], xlim=500)

    # ===============仅加入Harmonic噪声===================
    acc_arr_noise = utils.add_harmonic_interference(signal=acc_arr, freqs=[30, 90],  amp_ratio=0.1, RANDOM_SEED=RANDOM_SEED, fs=fs)
    utils.plt_time_domain(arr=acc_arr_noise,fs=fs, title="Chipped_20_0_Harmonic", img_save_path=TIME_IMG_SAVE_PATH)
    utils.plt_fft_img(acc_arr_noise, fs=fs, title='Chipped_20_0_Harmonic', img_save_path=FFT_IMG_SAVE_PATH, vline=[20, 40, 60, 80, 100], xlim=500)

    # =====================三种噪声混合====================
    acc_arr_noise = utils.add_noise_combination(
        signal=acc_arr,
        noise_type='all',
        gaussian_kw={'snr_db':10},
        impulse_kw={'impulse_prob':0.01, 'amplitude_scale':4},
        harmonic_kw={'freqs':[30, 90],'amp_ratio':0.1, 'RANDOM_SEED':RANDOM_SEED ,'fs':fs}
    )
    utils.plt_time_domain(arr=acc_arr_noise,fs=fs, title="Chipped_20_0_Mix", img_save_path=TIME_IMG_SAVE_PATH)
    utils.plt_fft_img(acc_arr_noise, fs=fs, title='Chipped_20_0_Mix', img_save_path=FFT_IMG_SAVE_PATH, vline=[20, 40, 60, 80, 100], xlim=500)

    # 等待用户输入（按 Enter 键）
    plt.pause(0.1) #图像渲染时间
    input("按 Enter 键关闭所有图形窗口...")
    # 关闭所有 Matplotlib 窗口
    plt.close('all')

# ============================完整数据集加载+统一标签+拼接==========================================
# 1. 分别加载两个工况的数据
df_20 = utils.load_multi_csv_data(os.path.join(DATASET_FOLDER_PATH, "RS20_L0"), cache_path="gearbox_data_cacheRS20.pkl", visualize_sample=False)
df_30 = utils.load_multi_csv_data(os.path.join(DATASET_FOLDER_PATH, "RS30_L2"), cache_path="gearbox_data_cache_RS30.pkl", visualize_sample=False)

df_20['fault_type'] = df_20['fault_type'].apply(utils.unify_label)
df_30['fault_type'] = df_30['fault_type'].apply(utils.unify_label)

# 3. 拼接 DataFrame
combined_df = pd.concat([df_20, df_30], ignore_index=True)

# 4. 统一预处理（此时 LabelEncoder 看到的是一致且完整的标签集合）
train_loader, test_loader, num_classes, le, scaler, cond_train, cond_test= utils.preprocess_data_1(
    combined_df,
    signal_length=512,
    Feature_Dimension=Feature_Dimension,
    TEST_SIZE=0.2,
    RANDOM_SEED=RANDOM_SEED,
    batch_size=batch_size
)

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
model_path = os.path.join(MODEL_SAVEDIT_PATH, "SimpleCNN_FullCondition_NoiseAugment.pth")

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
        total = 0

        for inputs, labels in train_loader:
            inputs_np = inputs.numpy()

            # ---------- 加噪逻辑 ----------
            if use_fixed_noise:
                # 使用固定的噪声类型和参数
                inputs_noisy = utils.add_noise_combination(
                    signal=inputs_np,
                    noise_type=fixed_noise_config['noise_type'],
                    **fixed_noise_config
                )
                # 注意：这里的 **fixed_noise_config 会传入 gaussian_kw 等字典，但 add_noise_combination
                # 对 'all' 类型期望的是 gaussian_kw, impulse_kw, harmonic_kw，所以上述写法正确。
                # 但如果 noise_type 是单一类型，则直接使用对应的键值（例如 'gaussian' 则只传 snr_db 等）
            else:
                # 随机模式（原有代码）
                if np.random.rand() < noise_prob:
                    noise_type = np.random.choice(['gaussian', 'impulse', 'harmonic', 'all'])
                    if noise_type == 'gaussian':
                        inputs_noisy = utils.add_noise_combination(
                            signal=inputs_np, noise_type='gaussian',
                            snr_db=np.random.uniform(15, 30), RANDOM_SEED=None
                        )
                    elif noise_type == 'impulse':
                        inputs_noisy = utils.add_noise_combination(
                            signal=inputs_np, noise_type='impulse',
                            impulse_prob=np.random.uniform(0.005, 0.02),
                            amplitude_scale=np.random.uniform(3.0, 6.0),
                            RANDOM_SEED=None
                        )
                    elif noise_type == 'harmonic':
                        inputs_noisy = utils.add_noise_combination(
                            signal=inputs_np, noise_type='harmonic',
                            freqs=[30, 90],
                            amp_ratio=np.random.uniform(0.05, 0.2),
                            fs=fs, RANDOM_SEED=None
                        )
                    else:  # 'all'
                        inputs_noisy = utils.add_noise_combination(
                            signal=inputs_np, noise_type='all',
                            gaussian_kw={'snr_db': np.random.uniform(15, 25), 'RANDOM_SEED': None},
                            impulse_kw={'impulse_prob': np.random.uniform(0.005, 0.015),
                                        'amplitude_scale': np.random.uniform(3.0, 5.0),
                                        'RANDOM_SEED': None},
                            harmonic_kw={'freqs': [30, 90], 'amp_ratio': np.random.uniform(0.15, 0.25),
                                         'fs': fs, 'RANDOM_SEED': None}
                        )
                    inputs = torch.from_numpy(inputs_noisy).float()
                else:
                    inputs = torch.from_numpy(inputs_np).float()
            # ---------- 加噪结束 ----------

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

# ========== 全工况数据集测试 ==========
test_loader_raw_valid = True #原始测试集加载使能，True-不加载噪声；False-加载噪声
if test_loader_raw_valid:
    # 针对原始测试集验证划分对应保存路径
    FIG_SAVE_PATH = os.path.join(IMG_SAVE_PATH, 'Figure_TrainNoise_TestRaw')
    os.makedirs(FIG_SAVE_PATH, exist_ok=True)
else:
    # 测试集噪声加载
    test_loader_noise = utils.create_noisy_test_loader(test_loader, noise_type='all',
                                                       gaussian_kw={'snr_db': 20},
                                                       impulse_kw={'impulse_prob': 0.01, 'amplitude_scale': 4},
                                                       harmonic_kw={'freqs': [30, 90], 'amp_ratio': 0.1,
                                                                    'RANDOM_SEED': RANDOM_SEED, 'fs': fs})
    test_loader = test_loader_noise

test_loss, test_acc, y_true, y_pre = utils.evaluate_model(
    model, test_loader, device, criterion, verbose=True
)
print(f"最终测试准确率: {test_acc:.2f}%")

# ========== 混淆矩阵与指标 ==========
cm = confusion_matrix(y_true, y_pre, labels=range(num_classes))
utils.plot_confusion_matrix(
    cm, classes=le.classes_, normalize=False,
    title=f"Confusion_Matrix_FD{Feature_Dimension}_Noise_Condition",
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
        title=f"Loss&Accuracy_FD{Feature_Dimension}_Noise")
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
    title=f"PCA_FD{Feature_Dimension}_Noise_Condition",
    FIG_SAVE_VALID=1,
    FIG_SAVE_PATH=FIG_SAVE_PATH,
    condition_labels=cond_test
)
# 绘制t-SNE
utils.tSNE_plot(
    X_feat=X_feat,
    y_true=y_true,
    num_classes=num_classes,
    le=le,
    title=f"t-SNE_FD{Feature_Dimension}_Noise_Condition",
    FIG_SAVE_VALID=1,
    FIG_SAVE_PATH=FIG_SAVE_PATH,
    condition_labels=cond_test
)
# 绘制UMAP
utils.UMAP_plot(
    X_feat=X_feat,
    y_true=y_true,
    num_classes=num_classes,
    le=le,
    title=f"UMAP_FD{Feature_Dimension}_Noise_Condition",
    FIG_SAVE_VALID=True,
    FIG_SAVE_PATH=FIG_SAVE_PATH,
    condition_labels=cond_test
)
plt.show()