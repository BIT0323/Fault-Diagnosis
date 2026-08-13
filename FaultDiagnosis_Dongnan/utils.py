import pandas as pd
import numpy as np
import os
import pickle
import torch
import matplotlib.pyplot as plt
import itertools
import umap
import scipy.signal as signal
from scipy import fftpack
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.ticker as ticker
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# def load_multi_csv_data(data_folder, visualize_sample, cache_path="gearbox_data_cache.pkl", skip_rows=16):
#     """
#     功能描述：用于读取数据库文件夹中的.csv文件
#     变量：
#         data_folder:存放.cvs文件的文件夹地址
#         skip_row:这是由于.cvs文件中的前16行为数据说明行，不含真正数据
#         cache_path:这是缓存文件地址
#     return: combined_df：所有.csv文件中提取的数据组成的numpy数组----(numpy.float64格式数组)
#     """
#     # 为了避免每次运行程序都从.csv文件读取数据，通过缓存逻辑进行处理
#     # 检查缓存文件是否存在
#     if os.path.exists(cache_path):
#         print(f"发现缓存文件，直接读取（跳过数据加载）...")
#         with open(cache_path, 'rb') as f:
#             combined_df = pickle.load(f)
#         print(f"缓存读取完成，数据量: {len(combined_df)}")
#         if visualize_sample:
#             print("前5行数据预览：")
#             print(combined_df.head())
#         return combined_df
#
#     # 无缓存时，正常加载数据
#     print("无缓存文件，开始加载数据")
#     all_data = []
#     for filename in os.listdir(data_folder):
#         if not filename.endswith(".csv"):
#             continue
#
#         fault_type = os.path.splitext(filename)[0]
#         file_path = os.path.join(data_folder, filename)
#
#         # 由于该数据集.csv文件中的所有数据都在第一列，所以只读取第一列，并强制以字符串类型读取
#         df_raw = pd.read_csv(
#             file_path,
#             skiprows=skip_rows,
#             header=None,
#             usecols=[0],  # 仅读第一列
#             dtype={0: str},  # 强制第一列为字符串类型
#             encoding='utf-8',
#             na_filter=False  # 禁用空值过滤，保留原始字符串
#         )
#
#         # 处理空字符串/无效值，再拆分
#         # 1. 去除每行首尾空格
#         df_raw[0] = df_raw[0].str.strip()
#         # 2. 过滤空行
#         df_raw = df_raw[df_raw[0] != '']
#
#         # 拆分制表符分隔的字符串（兼容已为数值的情况）
#         try:
#             # 按制表符拆分，拆分为8列
#             df_features = df_raw[0].str.split('\t', expand=True)
#         except:
#             # 若拆分失败（已为数值），直接转为DataFrame
#             df_features = pd.DataFrame(df_raw[0].values.reshape(-1, 1))
#
#         # 关键修改4：转为数值类型，清理空值，保留前8列
#         df_features = df_features.iloc[:, :8].apply(pd.to_numeric, errors='coerce')
#         df_features = df_features.dropna()  # 过滤含空值的行
#
#         # 添加故障标签
#         df_features['fault_type'] = fault_type
#         all_data.append(df_features)
#
#         print(f"已加载 {filename} | 有效数据行: {len(df_features)} | 故障类型: {fault_type}")
#
#     # 合并所有数据
#     combined_df = pd.concat(all_data, ignore_index=True)
#     # 特征列命名（8个特征）
#     # feature_cols = [f'feature_{i + 1}' for i in range(8)]
#     feature_cols = ['vibration_z_motor','vibration_x_pgbox','vibration_y_pgbox',
#                     'vibration_z_pgbox','torque_motor','vibration_x_gbox',
#                     'vibration_y_gbox','vibration_z_gbox']
#     combined_df.columns = feature_cols + ['fault_type']
#
#     # 保存缓存文件（二进制格式）
#     with open(cache_path, 'wb') as f:
#         pickle.dump(combined_df, f)
#     print(f"数据加载完成，已保存缓存到: {cache_path}")
#
#     if visualize_sample:
#         print(f"\n合并后总数据量: {len(combined_df)} | 特征维度: 8")
#         print("前5行数据预览：")
#         print(combined_df.head())
#
#     return combined_df

def load_multi_csv_data(data_folder, visualize_sample=False, cache_path="gearbox_data_cache.pkl", skip_rows=16):
    """
    功能描述：用于读取数据库文件夹中的.csv文件，并自动从文件夹名提取工况标签

    变量：
        data_folder: 存放.csv文件的文件夹地址
        visualize_sample: 是否打印数据预览
        cache_path: 缓存文件地址
        skip_rows: 跳过前几行（默认16）

    return: combined_df：包含所有数据的DataFrame，新增 'condition' 列（工况标识）
    """
    # 从文件夹路径提取工况名称（例如 'RS20_L0'）
    condition_name = os.path.basename(data_folder)  # 获取最后一级目录名

    # 检查缓存文件是否存在
    if os.path.exists(cache_path):
        print(f"发现缓存文件，直接读取（跳过数据加载）...")
        with open(cache_path, 'rb') as f:
            combined_df = pickle.load(f)
        # 如果缓存中没有 condition 列，则补上（兼容旧缓存）
        if 'condition' not in combined_df.columns:
            combined_df['condition'] = condition_name
            print(f"缓存缺少 'condition' 列，已补填为 '{condition_name}'")
        print(f"缓存读取完成，数据量: {len(combined_df)}")
        if visualize_sample:
            print("前5行数据预览：")
            print(combined_df.head())
        return combined_df

    # 无缓存时，正常加载数据
    print("无缓存文件，开始加载数据")
    all_data = []
    for filename in os.listdir(data_folder):
        if not filename.endswith(".csv"):
            continue

        fault_type = os.path.splitext(filename)[0]  # 例如 'Chipped_20_0'
        file_path = os.path.join(data_folder, filename)

        df_raw = pd.read_csv(
            file_path,
            skiprows=skip_rows,
            header=None,
            usecols=[0],
            dtype={0: str},
            encoding='utf-8',
            na_filter=False
        )

        df_raw[0] = df_raw[0].str.strip()
        df_raw = df_raw[df_raw[0] != '']

        try:
            df_features = df_raw[0].str.split('\t', expand=True)
        except:
            df_features = pd.DataFrame(df_raw[0].values.reshape(-1, 1))

        df_features = df_features.iloc[:, :8].apply(pd.to_numeric, errors='coerce')
        df_features = df_features.dropna()

        # 添加故障标签
        df_features['fault_type'] = fault_type
        # 添加工况标签（从文件夹名获取）
        df_features['condition'] = condition_name

        all_data.append(df_features)
        print(f"已加载 {filename} | 有效数据行: {len(df_features)} | 故障类型: {fault_type} | 工况: {condition_name}")

    combined_df = pd.concat(all_data, ignore_index=True)

    # 特征列命名（8个物理量）
    feature_cols = ['vibration_z_motor', 'vibration_x_pgbox', 'vibration_y_pgbox',
                    'vibration_z_pgbox', 'torque_motor', 'vibration_x_gbox',
                    'vibration_y_gbox', 'vibration_z_gbox']
    combined_df.columns = feature_cols + ['fault_type', 'condition']  # 新增 condition

    # 保存缓存文件
    with open(cache_path, 'wb') as f:
        pickle.dump(combined_df, f)
    print(f"数据加载完成，已保存缓存到: {cache_path}")

    if visualize_sample:
        print(f"\n合并后总数据量: {len(combined_df)} | 特征维度: 8")
        print("前5行数据预览：")
        print(combined_df.head())

    return combined_df

def preprocess_data(combined_df, signal_length, Feature_Dimension, TEST_SIZE,
                               RANDOM_SEED, batch_size, num_workers=0):
    """
    功能描述：对数据进行预处理
    参数：
        combined_df: load_multi_csv_data 返回的DataFrame
        Feature_Dimension: 1 或 3，选择使用多少维振动信号
        TEST_SIZE: 测试集比例
        RANDOM_SEED: 随机种子
        signal_length: 每个样本的时序长度
        batch_size: 批量大小
        num_workers: DataLoader的并行工作线程数（Windows下建议设为0）
    返回：
        train_loader, test_loader, num_classes, label_encoder, le, scaler
    """
    # ===== 2.1 选择特征列 =====
    if Feature_Dimension == 1:
        feature_cols = ['vibration_x_gbox']
    elif Feature_Dimension == 3:
        feature_cols = ['vibration_x_gbox', 'vibration_y_gbox', 'vibration_z_gbox']
    else:
        raise ValueError("Feature_Dimension must be 1 or 3")

    # ===== 2.2 按故障类型切分时序样本 =====
    X_sequence = []
    y_sequence = []
    unique_faults = combined_df['fault_type'].unique()

    for fault in unique_faults:
        fault_data = combined_df[combined_df['fault_type'] == fault]
        signal = fault_data[feature_cols].values
        num_samples = len(signal) // signal_length
        for i in range(num_samples):
            seq = signal[i * signal_length : (i + 1) * signal_length]
            X_sequence.append(seq)
            y_sequence.append(fault)

    X = np.array(X_sequence)          # 形状: (样本数, signal_length, Feature_Dimension)
    y = np.array(y_sequence)          # 形状: (样本数,)

    print(f"X shape: {X.shape}, y shape: {y.shape}")

    # ===== 2.3 标签编码 =====
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    num_classes = len(le.classes_)
    print(f"故障类型映射: {dict(zip(le.classes_, range(num_classes)))}")

    # ===== 2.4 划分训练集和测试集 =====
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y_encoded
    )
    print(f"训练集样本数: {len(X_train)}, 测试集样本数: {len(X_test)}")

    # ===== 2.5 标准化 =====
    scaler = StandardScaler()
    # 将 (样本数, 信号长度, 特征数) 展平为 (样本数*信号长度, 特征数)
    X_train_flat = X_train.reshape(-1, X_train.shape[-1])
    X_train_scaled = scaler.fit_transform(X_train_flat)
    X_train_scaled = X_train_scaled.reshape(X_train.shape)   # 恢复形状

    X_test_flat = X_test.reshape(-1, X_test.shape[-1])
    X_test_scaled = scaler.transform(X_test_flat)
    X_test_scaled = X_test_scaled.reshape(X_test.shape)

    # ===== 2.6 转换为 PyTorch 张量 =====
    # 注意：PyTorch 的 Conv1d 要求输入形状为 (batch, channels, length)
    # 所以把特征维度作为通道，即 (样本数, Feature_Dimension, signal_length)
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).permute(0, 2, 1)
    X_test_tensor  = torch.tensor(X_test_scaled,  dtype=torch.float32).permute(0, 2, 1)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    y_test_tensor  = torch.tensor(y_test,  dtype=torch.long)

    #print(f"训练集张量形状: {X_train_tensor.shape}")

    # ===== 2.7 创建 DataLoader =====
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset  = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers,
                              pin_memory=True)   # 加速GPU传输（如果使用GPU）
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers,
                             pin_memory=True)

    return train_loader, test_loader, num_classes, le, scaler

def preprocess_data_1(combined_df, signal_length, Feature_Dimension, TEST_SIZE,
                    RANDOM_SEED, batch_size, num_workers=0):
    """
    返回：train_loader, test_loader, num_classes, le, scaler, cond_train, cond_test
    """
    # ===== 选择特征列 =====
    if Feature_Dimension == 1:
        feature_cols = ['vibration_x_gbox']
    elif Feature_Dimension == 3:
        feature_cols = ['vibration_x_gbox', 'vibration_y_gbox', 'vibration_z_gbox']
    else:
        raise ValueError("Feature_Dimension must be 1 or 3")

    # ===== 按 (故障类型, 工况) 分组切分样本 =====
    X_sequence = []
    y_sequence = []
    condition_sequence = []

    # 检查 combined_df 是否有 'condition' 列
    if 'condition' not in combined_df.columns:
        raise ValueError("combined_df 必须包含 'condition' 列！")

    grouped = combined_df.groupby(['fault_type', 'condition'])
    print("检测到的分组:", list(grouped.groups.keys()))  # 调试

    for (fault, cond), group in grouped:
        signal = group[feature_cols].values
        num_samples = len(signal) // signal_length
        for i in range(num_samples):
            seq = signal[i * signal_length : (i + 1) * signal_length]
            X_sequence.append(seq)
            y_sequence.append(fault)
            condition_sequence.append(cond)

    X = np.array(X_sequence)
    y = np.array(y_sequence)
    condition = np.array(condition_sequence)

    print(f"X shape: {X.shape}, y shape: {y.shape}")
    print(f"工况分布: {np.unique(condition, return_counts=True)}")

    # ===== 标签编码 =====
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    num_classes = len(le.classes_)
    print(f"故障类型映射: {dict(zip(le.classes_, range(num_classes)))}")

    # ===== 划分数据集（同时划分工况） =====
    X_train, X_test, y_train, y_test, cond_train, cond_test = train_test_split(
        X, y_encoded, condition, test_size=TEST_SIZE,
        random_state=RANDOM_SEED, stratify=y_encoded
    )

    # ===== 标准化 =====
    scaler = StandardScaler()
    X_train_flat = X_train.reshape(-1, X_train.shape[-1])
    X_train_scaled = scaler.fit_transform(X_train_flat)
    X_train_scaled = X_train_scaled.reshape(X_train.shape)

    X_test_flat = X_test.reshape(-1, X_test.shape[-1])
    X_test_scaled = scaler.transform(X_test_flat)
    X_test_scaled = X_test_scaled.reshape(X_test.shape)

    # ===== 转换为 PyTorch 张量 =====
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).permute(0, 2, 1)
    X_test_tensor  = torch.tensor(X_test_scaled,  dtype=torch.float32).permute(0, 2, 1)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    y_test_tensor  = torch.tensor(y_test,  dtype=torch.long)

    # ===== 创建 DataLoader =====
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset  = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers,
                              pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers,
                             pin_memory=True)

    return train_loader, test_loader, num_classes, le, scaler, cond_train, cond_test

def plot_confusion_matrix(cm, classes,
                          IMG_SAVE,  img_save_path,
                          title,
                          normalize=False,
                          cmap=plt.cm.Blues):
    """
    绘制混淆矩阵可视化图
    参数:
    cm : 混淆矩阵（numpy数组）
    classes : 类别名称列表
    IMG_SAVE: 保存图片标志位。True, 保存；False, 不保存。
    img_save_path: 保存图片路径
    normalize : 是否归一化（显示百分比）
    title : 图表标题
    cmap : 颜色映射,可选Blues, Reds...
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Confusion Matrix (Normalized):")
    else:
        print('Confusion Matrix (Non-normalized)')
    print(cm)

    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=16, pad=20)
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0, ha='right')
    plt.yticks(tick_marks, classes)

    plt.gca().xaxis.set_label_position('top')
    plt.gca().xaxis.tick_top()

    plt.ylabel('Actual labels', fontsize=14)
    plt.xlabel('Predict', fontsize=14)

    # 计算阈值（用 NumPy 计算最大值的半值，用于决定文字颜色）
    thresh = np.max(cm) / 2.0

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            text = f'{cm[i, j]:.2f}'
        else:
            text = f'{cm[i, j]:d}'
        plt.text(j, i, text,
                 horizontalalignment="center",
                 verticalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",
                 fontsize=12)
    if IMG_SAVE:
        if img_save_path:
            filename = f"{title}.png"
            img_save_path = os.path.join(img_save_path, filename)
            plt.savefig(img_save_path, dpi=600, bbox_inches = 'tight')

    plt.tight_layout()
    plt.show(block=False)

def calculate_multiclass_metrics(cm, verbose, class_names=None):
    """
    从混淆矩阵计算多分类指标
    参数:
    cm: n×n混淆矩阵，n为类别数
    class_names: 类别名称列表
    verbose: 是否打印。True，打印；False，不打印。
    返回:
    包含总体和各类别指标的字典
    """
    n_classes = cm.shape[0]
    if class_names is None:
        class_names = [f'Class_{i}' for i in range(n_classes)]

    class_metrics = {}
    total_samples = np.sum(cm)

    for i in range(n_classes):
        TP = cm[i, i]
        FP = np.sum(cm[:, i]) - TP
        FN = np.sum(cm[i, :]) - TP
        TN = total_samples - (TP + FP + FN)

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        support = np.sum(cm[i, :])

        class_metrics[class_names[i]] = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'support': support,
            'TP': TP,
            'FP': FP,
            'FN': FN
        }

    macro_precision = np.mean([m['precision'] for m in class_metrics.values()])
    macro_recall = np.mean([m['recall'] for m in class_metrics.values()])
    macro_f1 = np.mean([m['f1_score'] for m in class_metrics.values()])

    weights = [m['support'] for m in class_metrics.values()]
    weighted_precision = np.average([m['precision'] for m in class_metrics.values()], weights=weights)
    weighted_recall = np.average([m['recall'] for m in class_metrics.values()], weights=weights)
    weighted_f1 = np.average([m['f1_score'] for m in class_metrics.values()], weights=weights)

    total_TP = np.sum([m['TP'] for m in class_metrics.values()])
    total_FP = np.sum([m['FP'] for m in class_metrics.values()])
    total_FN = np.sum([m['FN'] for m in class_metrics.values()])

    micro_precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0
    micro_recall = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0
    micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0

    accuracy = np.trace(cm) / total_samples if total_samples > 0 else 0

    metrics = {
        'accuracy': accuracy,
        'class_metrics': class_metrics,
        'macro_avg': {
            'precision': macro_precision,
            'recall': macro_recall,
            'f1_score': macro_f1
        },
        'weighted_avg': {
            'precision': weighted_precision,
            'recall': weighted_recall,
            'f1_score': weighted_f1
        },
        'micro_avg': {
            'precision': micro_precision,
            'recall': micro_recall,
            'f1_score': micro_f1
        }
    }

    if verbose:
        # 打印结果
        print("\n===== 总体准确率 ===== ")
        print(f"Accuracy: {metrics['accuracy']:.4f}")

        print("\n===== 各类别指标 =====")
        for cls, m in metrics['class_metrics'].items():
            print(
                f"{cls}: Precision={m['precision']:.4f}, Recall={m['recall']:.4f}, F1={m['f1_score']:.4f}, Support={m['support']}")

        print("\n===== 宏平均 (Macro) =====")
        print(
            f"Precision: {metrics['macro_avg']['precision']:.4f}, Recall: {metrics['macro_avg']['recall']:.4f}, F1: {metrics['macro_avg']['f1_score']:.4f}")

        print("\n===== 加权平均 (Weighted) =====")
        print(
            f"Precision: {metrics['weighted_avg']['precision']:.4f}, Recall: {metrics['weighted_avg']['recall']:.4f}, F1: {metrics['weighted_avg']['f1_score']:.4f}")

        print("\n===== 微平均 (Micro) =====")
        print(
            f"Precision: {metrics['micro_avg']['precision']:.4f}, Recall: {metrics['micro_avg']['recall']:.4f}, F1: {metrics['micro_avg']['f1_score']:.4f}")

    return metrics

def evaluate_model(model, test_loader, device, criterion, epoch=None, num_epochs=None, verbose=True):
    """
    在测试集上评估模型，返回损失、准确率以及预测和真实标签。
    若提供 epoch 和 num_epochs，则打印当前轮次的测试结果。

    参数:
        model: PyTorch 模型
        test_loader: DataLoader
        device: 'cuda' 或 'cpu'
        criterion: 损失函数
        epoch: 当前轮次（可选）
        num_epochs: 总轮次（可选）
        verbose: 是否打印

    返回:
        test_loss: 平均测试损失
        test_acc: 准确率（百分比）
        y_true: 真实标签 (numpy array)
        y_pred: 预测标签 (numpy array)
    """
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.append(predicted.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    test_loss = test_loss / len(test_loader)
    test_acc = 100.0 * correct / total

    if verbose and epoch is not None and num_epochs is not None:
        print(f"Epoch {epoch+1}/{num_epochs} | Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")

    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_labels, axis=0)
    return test_loss, test_acc, y_true, y_pred

def model_train_curve_plot(num_epochs, train_losses, train_accs, test_losses, test_accs,
                            IMG_SAVE_VALID, img_save_path, title='Loss&Accuracy'):
    """
    根据训练历史绘制模型训练过程中的 训练集损失曲线、训练集准确度曲线、测试集损失曲线、测试集准确度曲线
    参数:
    num_epochs: 超参数，指定的模型训练总轮数
    train_losses：训练损失历史
    train_accs：训练准确度历史
    test_losses：训练测试损失历史
    test_accs：训练测试准确度历史
    IMG_SAVE_VALID：图像存储路径使能
    img_save_path：图像存储路径
    title：图像标题
    返回:
    无
    """
    # ===== 绘制 Loss 曲线（训练 & 测试） =====
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), train_losses, 'b-o', label='Train Loss')
    plt.plot(range(1, num_epochs + 1), test_losses, 'r-s', label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curves')
    plt.legend()
    plt.grid(True)
    # 自动适应横坐标，并强制为整数刻度
    ax = plt.gca()  # 获取当前坐标轴
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # ===== 绘制 Accuracy 曲线（训练 & 测试） =====
    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), train_accs, 'b-o', label='Train Acc')
    plt.plot(range(1, num_epochs + 1), test_accs, 'r-s', label='Test Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy Curves')
    plt.legend()
    plt.grid(True)
    # 自动适应横坐标，并强制为整数刻度
    ax = plt.gca()  # 获取当前坐标轴
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    if IMG_SAVE_VALID:
        if img_save_path:
            filename = f"{title}.png"
            img_save_path = os.path.join(img_save_path, filename)
            plt.savefig(img_save_path, dpi=600, bbox_inches = 'tight')

    plt.tight_layout()
    plt.show(block=False)

def extract_features_CNN(model, data_loader, device):
    """
    提取CNN模型特征用于绘制PCA/t-SNE/UMAP

    :param model: 训练好的模型
    :param data_loader: pytorch的数据加载器
    :param device: 模型训练硬件，cuda 或者 cpu
    :return:
        np.concatenate(features, axis=0): 训练好的模型提取的特征
        np.concatenate(labels, axis=0): 样本特征所对应的标签
    """
    model.eval()
    features = []
    labels = []
    with torch.no_grad():
        for inputs, lbls in data_loader:
            inputs = inputs.to(device)
            # 手动前向传播到fc1
            x = model.pool1(model.relu1(model.bn1(model.conv1(inputs))))
            x = model.pool2(model.relu2(model.bn2(model.conv2(x))))
            x = model.pool3(model.relu3(model.bn3(model.conv3(x))))
            x = x.view(x.size(0), -1)
            x = model.relu4(model.fc1(x))  # 特征
            features.append(x.cpu().numpy())
            labels.append(lbls.numpy())
    return np.concatenate(features, axis=0), np.concatenate(labels, axis=0)

def extract_features_RNN(model, data_loader, device):
    """
    提取RNN模型特征用于绘制PCA/t-SNE/UMAP

    :param model: 训练好的模型
    :param data_loader: pytorch的数据加载器
    :param device: 模型训练硬件，cuda 或者 cpu
    :return:
        np.concatenate(features, axis=0): 训练好的模型提取的特征
        np.concatenate(labels, axis=0): 样本特征所对应的标签
    """
    model.eval()
    features = []
    labels = []
    with torch.no_grad():
        for inputs, lbls in data_loader:
            inputs = inputs.to(device)
            feat = model.forward_features(inputs)
            features.append(feat.cpu().numpy())
            labels.append(lbls.numpy())
    return np.concatenate(features, axis=0), np.concatenate(labels, axis=0)

def PCA_plot(X_feat, y_true, num_classes, le, title, FIG_SAVE_VALID, FIG_SAVE_PATH, condition_labels=None):
    """
    绘制PCA结果，支持可选工况标签

    :param X_feat: 特征矩阵 (n_samples, feature_dim)
    :param y_true: 故障类别标签 (n_samples,)
    :param num_classes: 类别总数
    :param le: LabelEncoder 对象
    :param title: 图片标题
    :param FIG_SAVE_VALID: bool
    :param FIG_SAVE_PATH: 保存目录
    :param condition_labels: 工况标签 (n_samples,)，若为 None 则仅按故障着色
    """
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    from sklearn.preprocessing import StandardScaler

    # 特征标准化（PCA 对尺度敏感）
    scaler = StandardScaler()
    X_feat_norm = scaler.fit_transform(X_feat)

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_feat_norm)

    plt.figure(figsize=(12, 10))
    base_colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    markers = ['o', 's', '^', 'D', 'v', 'P', '*', 'X', 'h', '+']

    if condition_labels is not None:
        unique_conds = np.unique(condition_labels)
        cond_to_marker = {cond: markers[i % len(markers)] for i, cond in enumerate(unique_conds)}

        for i in range(num_classes):
            class_mask = (y_true == i)
            for cond in unique_conds:
                mask = class_mask & (condition_labels == cond)
                if np.sum(mask) > 0:
                    plt.scatter(
                        X_pca[mask, 0], X_pca[mask, 1],
                        label=f'{le.classes_[i]}_{cond}',
                        s=25, alpha=0.7,
                        color=base_colors[i % len(base_colors)],
                        marker=cond_to_marker[cond]
                    )
    else:
        for i in range(num_classes):
            plt.scatter(
                X_pca[y_true == i, 0], X_pca[y_true == i, 1],
                label=le.classes_[i],
                s=20, alpha=0.7,
                color=base_colors[i % len(base_colors)]
            )

    plt.legend(prop={'family': 'Times New Roman', 'size': 16})
    plt.title(label=title, fontsize=16, fontfamily="Times New Roman")
    plt.xlabel('Principal Component 1', fontsize=14)
    plt.ylabel('Principal Component 2', fontsize=14)
    plt.tick_params(axis='both', labelsize=14)
    plt.grid(True, linestyle='--', alpha=0.3)

    if FIG_SAVE_VALID and FIG_SAVE_PATH:
        filename = f"{title}.png"
        img_save_path = os.path.join(FIG_SAVE_PATH, filename)
        plt.savefig(img_save_path, dpi=600, bbox_inches='tight')

    plt.show(block=False)

def tSNE_plot(X_feat, y_true, num_classes, le, title, FIG_SAVE_VALID, FIG_SAVE_PATH, condition_labels=None):
    """
    绘制t-SNE结果，支持可选工况标签

    :param X_feat: 特征矩阵 (n_samples, feature_dim)
    :param y_true: 故障类别标签 (n_samples,)
    :param num_classes: 类别总数
    :param le: LabelEncoder 对象
    :param title: 图片标题
    :param FIG_SAVE_VALID: bool
    :param FIG_SAVE_PATH: 保存目录
    :param condition_labels: 工况标签 (n_samples,)，若为 None 则仅按故障着色
    """
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    from sklearn.preprocessing import StandardScaler

    # t-SNE 对尺度敏感，建议标准化
    scaler = StandardScaler()
    X_feat_norm = scaler.fit_transform(X_feat)

    # 如果样本量很大，可考虑随机采样加速，这里保持全量
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca')
    X_tsne = tsne.fit_transform(X_feat_norm)

    plt.figure(figsize=(12, 10))
    base_colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    markers = ['o', 's', '^', 'D', 'v', 'P', '*', 'X', 'h', '+']

    if condition_labels is not None:
        unique_conds = np.unique(condition_labels)
        cond_to_marker = {cond: markers[i % len(markers)] for i, cond in enumerate(unique_conds)}

        for i in range(num_classes):
            class_mask = (y_true == i)
            for cond in unique_conds:
                mask = class_mask & (condition_labels == cond)
                if np.sum(mask) > 0:
                    plt.scatter(
                        X_tsne[mask, 0], X_tsne[mask, 1],
                        label=f'{le.classes_[i]}_{cond}',
                        s=25, alpha=0.7,
                        color=base_colors[i % len(base_colors)],
                        marker=cond_to_marker[cond]
                    )
    else:
        for i in range(num_classes):
            plt.scatter(
                X_tsne[y_true == i, 0], X_tsne[y_true == i, 1],
                label=le.classes_[i],
                s=20, alpha=0.7,
                color=base_colors[i % len(base_colors)]
            )

    plt.legend(prop={'family': 'Times New Roman', 'size': 16})
    plt.title(label=title, fontsize=16, fontfamily="Times New Roman")
    plt.xlabel('t-SNE Dimension 1', fontsize=14)
    plt.ylabel('t-SNE Dimension 2', fontsize=14)
    plt.tick_params(axis='both', labelsize=14)
    plt.grid(True, linestyle='--', alpha=0.3)

    if FIG_SAVE_VALID and FIG_SAVE_PATH:
        filename = f"{title}.png"
        img_save_path = os.path.join(FIG_SAVE_PATH, filename)
        plt.savefig(img_save_path, dpi=600, bbox_inches='tight')

    plt.show(block=False)

def UMAP_plot(X_feat, y_true, num_classes, le, title, FIG_SAVE_VALID, FIG_SAVE_PATH, condition_labels=None):
    """
    绘制UMAP结果，支持可选工况标签

    :param X_feat: 特征矩阵 (n_samples, feature_dim)
    :param y_true: 故障类别标签 (n_samples,)
    :param num_classes: 类别总数
    :param le: LabelEncoder 对象，用于获取类别名称
    :param title: 图片标题
    :param FIG_SAVE_VALID: bool，是否保存图片
    :param FIG_SAVE_PATH: 保存目录
    :param condition_labels: 工况标签 (n_samples,)，若为 None 则仅按故障着色（默认）
    """
    import umap
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    from sklearn.preprocessing import StandardScaler

    # ----- 可选：特征标准化（提升可视化效果） -----
    scaler = StandardScaler()
    X_feat_norm = scaler.fit_transform(X_feat)

    # ----- UMAP 降维 -----
    reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
    X_umap = reducer.fit_transform(X_feat_norm)

    # ----- 绘图 -----
    plt.figure(figsize=(12, 10))
    base_colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    markers = ['o', 's', '^', 'D', 'v', 'P', '*', 'X', 'h', '+']

    # ---- 判断是否传入工况标签 ----
    if condition_labels is not None:
        # 按 (故障 + 工况) 分别着色
        unique_conds = np.unique(condition_labels)
        cond_to_marker = {cond: markers[i % len(markers)] for i, cond in enumerate(unique_conds)}

        for i in range(num_classes):
            class_mask = (y_true == i)
            for cond in unique_conds:
                mask = class_mask & (condition_labels == cond)
                if np.sum(mask) > 0:
                    plt.scatter(
                        X_umap[mask, 0], X_umap[mask, 1],
                        label=f'{le.classes_[i]}_{cond}',
                        s=25, alpha=0.7,
                        color=base_colors[i % len(base_colors)],
                        marker=cond_to_marker[cond]
                    )
    else:
        # 仅按故障类别着色（原始模式）
        for i in range(num_classes):
            plt.scatter(
                X_umap[y_true == i, 0], X_umap[y_true == i, 1],
                label=le.classes_[i],
                s=20, alpha=0.7,
                color=base_colors[i % len(base_colors)]
            )

    plt.legend(prop={'family': 'Times New Roman', 'size': 16})
    plt.title(label=title, fontsize=16, fontfamily="Times New Roman")
    plt.tick_params(axis='both', labelsize=14)
    plt.grid(True, linestyle='--', alpha=0.3)

    if FIG_SAVE_VALID and FIG_SAVE_PATH:
        filename = f"{title}.png"
        img_save_path = os.path.join(FIG_SAVE_PATH, filename)
        plt.savefig(img_save_path, dpi=600, bbox_inches='tight')

    plt.show(block=False)
    plt.pause(0.1)

def unify_label(label):
    """

    :param label: 加载的数据集标签
    :return: 返回统一的去掉工况后缀的标签
    """
    # 示例规则：取下划线前的部分，即 'Chipped_20L0' -> 'Chipped'
    # 根据你实际的命名规则调整
    return label.split('_')[0] if '_' in label else label

def plt_time_domain(arr, fs=1600, ylabel='Amp($m/s^2$)', title='原始数据时域图', img_save_path=None, x_vline=None,
                    y_hline=None):
    """
    :fun: 绘制时域图模板
    :param arr: 输入一维数组数据
    :param fs: 采样频率
    :param ylabel: y轴标签
    :param title: 图标题
    :return: None
    """

    plt.figure(figsize=(12, 4))
    length = len(arr)
    t = np.linspace(0, length / fs, length)
    plt.plot(t, arr, c='g')
    plt.xlabel('t(s)')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.subplots_adjust(bottom=0.15)

    if x_vline:
        plt.vlines(x=x_vline, ymin=np.min(arr), ymax=np.max(arr), linestyle='--', colors='r')
    if y_hline:
        plt.hlines(y=0.2, xmin=np.min(t), xmax=np.max(t), linestyle=':', colors='y')
    # ===保存图片====#
    if img_save_path:
        filename = f"{title}.png"
        img_save_path = os.path.join(img_save_path, filename)
        plt.savefig(img_save_path, dpi=600, bbox_inches='tight')
    plt.show(block=False)

##========绘制频域信号图========##
def plt_fft_img(arr, fs, ylabel='Amplitude', title='频域图', img_save_path=None, vline=None, hline=None, xlim=None):
    """
    :fun: 绘制频域图模板
    :param arr: 输入一维时域数组数据
    :param fs: 采样频率
    :param ylabel: y轴标签
    :param title: 图标题
    :return: None
    """
    # 计算频域幅值
    length = len(arr)
    t = np.linspace(0, length/fs, length)
    arr = arr - np.mean(arr)
    fft_result = np.fft.fft(arr)
    fft_freq= np.fft.fftfreq(len(arr), d=t[1]-t[0])  # FFT频率
    fft_amp= 2*np.abs(fft_result)/len(t)                     # FFT幅值

    # 绘制频域图
    plt.figure(figsize=(12,4))
    plt.title(title)
    plt.plot(fft_freq[0: int(len(t)/2)], fft_amp[0: int(len(t)/2)], label='Frequency Spectrum', color='b')
    plt.xlabel('频率 (Hz)')
    plt.ylabel(ylabel)
    plt.legend()
    if vline:
        plt.vlines(x=vline, ymin=np.min(fft_amp), ymax=np.max(fft_amp), linestyle='--', colors='r')
    if hline:
        plt.hlines(y=hline, xmin=np.min(fft_freq), xmax=np.max(fft_freq), linestyle=':', colors='y')
    #===保存图片====#
    if xlim: # 图片横坐标是否设置xlim
        plt.xlim(0, xlim)
    if img_save_path:
        filename = f"{title}.png"
        img_save_path = os.path.join(img_save_path, filename)
        plt.savefig(img_save_path, dpi=600, bbox_inches = 'tight')
    plt.tight_layout()
    plt.show(block=False)

def plt_stft_img(arr, fs, ylabel='Amplitude', title='stft时频域图', img_save_path=None, vline=None, hline=None, xlim=None):
    """
    :fun: 绘制stft时频域图模板
    :param arr: 输入一维时域数组数据
    :param fs: 采样频率
    :param ylabel: y轴标签
    :param title: 图标题
    :return: None
    """
    f, t, nd = signal.stft(arr, fs=fs, window='hann', nperseg=128, noverlap=64,nfft=None,
                           detrend=False, return_onesided=True, boundary='odd', padded=False, axis=-1)
    #  fs:时间序列的采样频率,  nperseg:每个段的长度，默认为256(2^n)   noverlap:段之间重叠的点数。如果没有则noverlap=nperseg/2

    #window ： 字符串或元组或数组，可选需要使用的窗。
    # #如果window是一个字符串或元组，则传递给它window是数组类型，直接以其为窗，其长度必须是nperseg。
    # 常用的窗函数有boxcar，triang，hamming， hann等，默认为Hann窗。

    #nfft ： int，可选。如果需要零填充FFT，则为使用FFT的长度。如果为 None，则FFT长度为nperseg。默认为无

    # detrend ： str或function或False，可选
    # 指定如何去除每个段的趋势。如果类型参数传递给False，则不进行去除趋势。默认为False。

    # return_onesided ： bool，可选
    # 如果为True，则返回实际数据的单侧频谱。如果 False返回双侧频谱。默认为 True。请注意，对于复杂数据，始终返回双侧频谱。

    # boundary ： str或None，可选
    # 指定输入信号是否在两端扩展，以及如何生成新值，以使第一个窗口段在第一个输入点上居中。
    # 这具有当所采用的窗函数从零开始时能够重建第一输入点的益处。
    # 有效选项是['even', 'odd', 'constant', 'zeros', None].
    # 默认为‘zeros’,对于补零操作[1, 2, 3, 4]变成[0, 1, 2, 3, 4, 0] 当nperseg=3.

    # padded： bool，可选
    # 指定输入信号在末尾是否填充零以使信号精确地拟合为整数个窗口段，以便所有信号都包含在输出中。默认为True。
    # 填充发生在边界扩展之后，如果边界不是None，则填充为True，默认情况下也是如此。
    # axis ： int，可选
    #=========做希尔伯特变换=======#
    xt = arr
    ht = fftpack.hilbert(xt)
    at = np.sqrt(xt**2+ht**2)   # 获得解析信号at = sqrt(xt^2 + ht^2)
    at = at - np.mean(at)       # 去直流分量
    fft_amp = np.fft.fft(at)         # 对解析信号at做fft变换获得幅值
    fft_amp = np.abs(fft_amp)             # 对幅值求绝对值（此时的绝对值很大）
    fft_amp = fft_amp/len(fft_amp)*2
    fft_amp = fft_amp[0: int(len(fft_amp)/2)]  # 取正频率幅值
    fft_freq = np.fft.fftfreq(len(at), d=1 / fs)  # 获取fft频率，此时包括正频率和负频率
    fft_freq = fft_freq[0:int(len(fft_freq)/2)]  # 获取正频率
    # 绘制STFT时频域图
    plt.figure(figsize=(12,4))
    plt.pcolormesh(t, f, np.abs(nd), vmin = np.min(np.abs(nd)), vmax = np.max(np.abs(nd)))
    plt.title(title)
    plt.xlabel('时间（t）')
    plt.ylabel('频率 (Hz)')
    if vline:
        plt.vlines(x=vline, ymin=np.min(fft_amp), ymax=np.max(fft_amp), linestyle='--', colors='r')
    if hline:
        plt.hlines(y=hline, xmin=np.min(fft_freq), xmax=np.max(fft_freq), linestyle=':', colors='y')
    #===保存图片====#
    if img_save_path:
        filename = f"{title}.png"
        img_save_path = os.path.join(img_save_path, filename)
        plt.savefig(img_save_path, dpi=600, bbox_inches = 'tight')
    if xlim: # 图片横坐标是否设置xlim
        plt.xlim(0, xlim)
    plt.tight_layout()
    plt.show(block=False)

def plt_envelope_spectrum(data, fs, ylabel='Amp(mg)', title='包络谱图', img_save_path=None, vline=None, hline=None, xlim=None):
    '''
    fun: 绘制包络谱图
    param data: 输入数据，1维array
    param fs: 采样频率
    param xlim: 图片横坐标xlim，default = None
    param vline: 图片垂直线，default = None
    '''
    from scipy import fftpack
    #=========做希尔伯特变换=======#
    xt = data
    ht = fftpack.hilbert(xt)
    at = np.sqrt(xt**2+ht**2)   # 获得解析信号at = sqrt(xt^2 + ht^2)
    at = at - np.mean(at)       # 去直流分量
    fft_amp = np.fft.fft(at)         # 对解析信号at做fft变换获得幅值
    fft_amp = np.abs(fft_amp)             # 对幅值求绝对值（此时的绝对值很大）
    fft_amp = fft_amp/len(fft_amp)*2
    fft_amp = fft_amp[0: int(len(fft_amp)/2)]  # 取正频率幅值
    fft_freq = np.fft.fftfreq(len(at), d=1 / fs)  # 获取fft频率，此时包括正频率和负频率
    fft_freq = fft_freq[0:int(len(fft_freq)/2)]  # 获取正频率
    # 绘制包络谱图
    plt.figure(figsize=(12,4))
    plt.title(title)
    plt.plot(fft_freq, fft_amp, color='b')
    plt.xlabel('频率 (Hz)')
    plt.ylabel(ylabel)

    if vline:
        plt.vlines(x=vline, ymin=np.min(fft_amp), ymax=np.max(fft_amp), linestyle='--', colors='r')
    if hline:
        plt.hlines(y=hline, xmin=np.min(fft_freq), xmax=np.max(fft_freq), linestyle=':', colors='y')
    #===保存图片====#
    if xlim: # 图片横坐标是否设置xlim
        plt.xlim(0, xlim)
    if img_save_path:
        filename = f"{title}.png"
        img_save_path = os.path.join(img_save_path, filename)
        plt.savefig(img_save_path, dpi=600, bbox_inches = 'tight')

    plt.tight_layout()
    plt.show(block=False)

def calculate_kurtosis(signal):
    """
    计算信号的峭度
    :param signal: 输入信号
    :return: 峭度值
    """
    x = signal
    X_rms2 = np.sum(x**2)/len(x)  # 3.均方值
    X_rms = np.sqrt(X_rms2)   # 4.均方根值(有效值)
    X_beta = np.mean( np.power(x, 4) )   # 12.峭度
    X_kf = X_beta/X_rms ** 4     # 18.峭度指标
    return X_kf

def data_read(file_path):
    """
    :fun: 读取csv数据
    :param file_path: 文件路径
    :return df:
    """
    try:
        df = pd.read_csv(file_path, header=None)
        df = df.iloc[16:, 0:8]
        df = df.astype(float)
    except:
        df = pd.read_csv(file_path, header=None)
        df.columns = ['acc_data']
        df = df.iloc[16:,:]
        # 首先，检查一下是否每个单元格都是字符串类型，如果不是，转换为字符串
        df['acc_data'] = df['acc_data'].astype(str)

        # 使用str.split来分割列，参数expand=True表示分割后的结果将作为新的列添加到DataFrame中
        df[['col1', 'col2', 'col3', 'col4', 'col5', 'col6', 'col7', 'col8', 'col9']] = df['acc_data'].str.split('\t', expand=True)
        df = df[['col1', 'col2', 'col3', 'col4', 'col5', 'col6', 'col7', 'col8']]
        df = df.astype(float)
    return df

def dataset_file_path_get(DATA_FOLDER, subset):
    """

    """
    assert subset in ["Chipped_20_0.csv", "Chipped_30_2.csv",
                      "Health_20_0.csv", "Health_30_2.cvs",
                      "Miss_20_0.csv", "Miss_30_2.csv",
                      "Root_20_0.csv", "Root_30_2.csv",
                      "Surface_20_0.csv", "Surface_30_2"]
    file_path = os.path.join(DATA_FOLDER, subset)

    return file_path

def add_gaussian_noise(signal, snr_db=20, RANDOM_SEED=None):
    """
    添加高斯白噪声（信噪比可控）
    :param signal: numpy 数组，形状 (..., length)
    :param snr_db: 信噪比 (dB)，越大噪声越小
    :param RANDOM_SEED: 随机种子
    :return: 加噪后的信号
    """
    if RANDOM_SEED is not None:
        np.random.seed(RANDOM_SEED)
    # 计算信号功率
    signal_power = np.mean(signal ** 2)
    # 根据 SNR 计算噪声功率
    snr_linear = 10 ** (snr_db / 10.0)
    noise_power = signal_power / snr_linear
    # 生成高斯噪声
    noise = np.random.normal(0, np.sqrt(noise_power), signal.shape)
    return signal + noise

def add_impulse_noise(signal, impulse_prob=0.01, amplitude_scale=5.0, RANDOM_SEED=None):
    """
    添加随机脉冲噪声（尖峰）
    :param signal: numpy 数组
    :param impulse_prob: 每个时间点产生脉冲的概率
    :param amplitude_scale: 脉冲幅值相对于信号标准差的倍数
    :param RANDOM_SEED: 随机种子
    :return: 加噪信号
    """
    if RANDOM_SEED is not None:
        np.random.seed(RANDOM_SEED)
    # 生成脉冲位置掩码
    mask = np.random.random(signal.shape) < impulse_prob
    # 脉冲幅值：正负随机，幅值为信号标准差的 amplitude_scale 倍
    impulse_amp = np.random.normal(0, amplitude_scale * np.std(signal), signal.shape)
    impulse = mask * impulse_amp
    return signal + impulse

def add_harmonic_interference(signal, freqs=[50, 100, 150], amp_ratio=0.2, amplitudes=None, fs=5120, RANDOM_SEED=None):
    """
    添加谐波干扰（固定频率正弦波叠加）

    :param signal: numpy 数组，形状 (..., length)
    :param freqs: 干扰频率列表 (Hz)
    :param amplitudes: 幅值控制，支持三种形式：
                        - None（默认）：自动根据信号 RMS 生成（每个频率幅值为 RMS × 0.1~0.3 随机）
                        - 单个数值（int/float）：所有频率使用该相同幅值
                        - 列表/元组（list/tuple）：长度须与 freqs 一致，按顺序指定每个频率的幅值
    :param fs: 采样频率（Hz），需与数据集采样率一致
    :param RANDOM_SEED: 随机种子，若为 None 则每次不同
    :return: 加噪后的信号（与 signal 形状相同）
    """
    if RANDOM_SEED is not None:
        np.random.seed(RANDOM_SEED)

    length = signal.shape[-1]
    t = np.arange(length) / fs

    # ----- 1. 处理 amplitudes 参数 -----
    if amplitudes is None:
        # 自动生成：基于信号标准差（或RMS）的 0.1~0.3 倍随机值
        base_amp = np.std(signal) * amp_ratio
        amplitudes = [base_amp * (0.5 + np.random.rand()) for _ in freqs]
    elif isinstance(amplitudes, (int, float)):
        # 单个数值 -> 所有频率使用同一幅值
        amplitudes = [amplitudes] * len(freqs)
    elif isinstance(amplitudes, (list, tuple)):
        # 列表/元组 -> 检查长度是否匹配
        if len(amplitudes) != len(freqs):
            raise ValueError(f"amplitudes 长度 ({len(amplitudes)}) 必须与 freqs 长度 ({len(freqs)}) 一致")
    else:
        raise TypeError("amplitudes 必须为 None、数值或数值列表/元组")

    # ----- 2. 生成并叠加谐波 -----
    interference = np.zeros_like(signal)
    # 预计算广播形状，适用于任意维度的信号（最内层为时间轴）
    broadcast_shape = (1,) * (len(signal.shape) - 1) + (-1,)

    for freq, amp in zip(freqs, amplitudes):
        phase = 2 * np.pi * np.random.rand()  # 随机初始相位
        wave = amp * np.sin(2 * np.pi * freq * t + phase)
        # 将 wave 扩展为与 signal 相同维数（广播到 batch/channels 维度）
        interference += wave.reshape(broadcast_shape)

    return signal + interference

def add_noise_combination(signal, noise_type='all', **kwargs):
    """
    组合噪声添加（方便统一调用）
    :param signal: numpy 数组
    :param noise_type: 'gaussian', 'impulse', 'harmonic', 'all'
    :param kwargs: 各噪声函数的参数，如 snr_db, impulse_prob, freqs 等
    :return: 加噪信号
    """
    if noise_type == 'gaussian':
        return add_gaussian_noise(signal, **kwargs)
    elif noise_type == 'impulse':
        return add_impulse_noise(signal, **kwargs)
    elif noise_type == 'harmonic':
        return add_harmonic_interference(signal, **kwargs)
    elif noise_type == 'all':
        # 按顺序叠加（顺序可能有影响，但结果差异不大）
        signal = add_gaussian_noise(signal, **kwargs.get('gaussian_kw', {}))
        signal = add_impulse_noise(signal, **kwargs.get('impulse_kw', {}))
        signal = add_harmonic_interference(signal, **kwargs.get('harmonic_kw', {}))
        return signal
    else:
        return signal


def create_noisy_test_loader(test_loader, noise_type='all', random_seed=42,
                             gaussian_kw=None, impulse_kw=None, harmonic_kw=None):
    """
    创建添加固定噪声的测试集 DataLoader
    :param test_loader: 原始测试集 DataLoader
    :param noise_type: 噪声类型，可选 'gaussian', 'impulse', 'harmonic', 'all'
    :param random_seed: 固定随机种子，确保噪声可重现
    :param gaussian_kw: 高斯噪声参数字典，如 {'snr_db': 20}
    :param impulse_kw: 脉冲噪声参数字典，如 {'impulse_prob': 0.01, 'amplitude_scale': 4.0}
    :param harmonic_kw: 谐波干扰参数字典，如 {'freqs': [50,100], 'amp_ratio': 0.2, 'fs': 10000}
    :return: 带噪测试集 DataLoader
    """
    # 固定随机种子
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    noisy_inputs = []
    noisy_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs_np = inputs.numpy()  # (batch, channels, length)

            # 调用通用加噪函数
            inputs_noisy = add_noise_combination(
                signal=inputs_np,
                noise_type=noise_type,
                gaussian_kw=gaussian_kw or {},
                impulse_kw=impulse_kw or {},
                harmonic_kw=harmonic_kw or {}
            )
            noisy_inputs.append(torch.from_numpy(inputs_noisy).float())
            noisy_labels.append(labels)

    # 合并所有 batch
    X_noisy = torch.cat(noisy_inputs, dim=0)
    y_noisy = torch.cat(noisy_labels, dim=0)
    noisy_dataset = TensorDataset(X_noisy, y_noisy)

    # 保持原始 batch_size
    batch_size = test_loader.batch_size
    return DataLoader(noisy_dataset, batch_size=batch_size, shuffle=False)