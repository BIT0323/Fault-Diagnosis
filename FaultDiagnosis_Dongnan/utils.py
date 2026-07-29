import pandas as pd
import numpy as np
import os
import pickle
import torch
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.ticker as ticker

def load_multi_csv_data(data_folder, visualize_sample, cache_path="gearbox_data_cache.pkl", skip_rows=16):
    """
    功能描述：用于读取数据库文件夹中的.csv文件
    变量：
        data_folder:存放.cvs文件的文件夹地址
        skip_row:这是由于.cvs文件中的前16行为数据说明行，不含真正数据
        cache_path:这是缓存文件地址
    return: combined_df：所有.csv文件中提取的数据组成的numpy数组----(numpy.float64格式数组)
    """
    # 为了避免每次运行程序都从.csv文件读取数据，通过缓存逻辑进行处理
    # 检查缓存文件是否存在
    if os.path.exists(cache_path):
        print(f"发现缓存文件，直接读取（跳过数据加载）...")
        with open(cache_path, 'rb') as f:
            combined_df = pickle.load(f)
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

        fault_type = os.path.splitext(filename)[0]
        file_path = os.path.join(data_folder, filename)

        # 由于该数据集.csv文件中的所有数据都在第一列，所以只读取第一列，并强制以字符串类型读取
        df_raw = pd.read_csv(
            file_path,
            skiprows=skip_rows,
            header=None,
            usecols=[0],  # 仅读第一列
            dtype={0: str},  # 强制第一列为字符串类型
            encoding='utf-8',
            na_filter=False  # 禁用空值过滤，保留原始字符串
        )

        # 处理空字符串/无效值，再拆分
        # 1. 去除每行首尾空格
        df_raw[0] = df_raw[0].str.strip()
        # 2. 过滤空行
        df_raw = df_raw[df_raw[0] != '']

        # 拆分制表符分隔的字符串（兼容已为数值的情况）
        try:
            # 按制表符拆分，拆分为8列
            df_features = df_raw[0].str.split('\t', expand=True)
        except:
            # 若拆分失败（已为数值），直接转为DataFrame
            df_features = pd.DataFrame(df_raw[0].values.reshape(-1, 1))

        # 关键修改4：转为数值类型，清理空值，保留前8列
        df_features = df_features.iloc[:, :8].apply(pd.to_numeric, errors='coerce')
        df_features = df_features.dropna()  # 过滤含空值的行

        # 添加故障标签
        df_features['fault_type'] = fault_type
        all_data.append(df_features)

        print(f"已加载 {filename} | 有效数据行: {len(df_features)} | 故障类型: {fault_type}")

    # 合并所有数据
    combined_df = pd.concat(all_data, ignore_index=True)
    # 特征列命名（8个特征）
    # feature_cols = [f'feature_{i + 1}' for i in range(8)]
    feature_cols = ['vibration_z_motor','vibration_x_pgbox','vibration_y_pgbox',
                    'vibration_z_pgbox','torque_motor','vibration_x_gbox',
                    'vibration_y_gbox','vibration_z_gbox']
    combined_df.columns = feature_cols + ['fault_type']

    # 保存缓存文件（二进制格式）
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
        train_loader, test_loader, num_classes, label_encoder, scaler
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

    print(f"训练集张量形状: {X_train_tensor.shape}")

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
    cmap : 颜色映射
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
            plt.savefig(img_save_path, dpi=500, bbox_inches = 'tight')

    plt.tight_layout()
    plt.show()

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
                            IMG_SAVE, img_save_path, title='Loss&Accuracy'):
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

    if IMG_SAVE:
        if img_save_path:
            filename = f"{title}.png"
            img_save_path = os.path.join(img_save_path, filename)
            plt.savefig(img_save_path, dpi=500, bbox_inches = 'tight')

    plt.tight_layout()
    plt.show()