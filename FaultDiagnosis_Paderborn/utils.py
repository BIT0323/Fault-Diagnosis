import pandas as pd
import numpy as np
import os
import pickle
import torch
import matplotlib.pyplot as plt
import itertools
import scipy.signal as signal
from scipy import fftpack
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.ticker as ticker
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import scipy.io as sio
import h5py
from pathlib import Path
from typing import Dict, List, Optional, Union

# ----------- 数据集.mat文件中信号类型索引 ----------------
SIGNAL_INDEX_MAP = {
    'force': 0,
    'phase_current_1': 1,
    'phase_current_2': 2,
    'speed': 3,
    'temp_2_bearing_m': 4,
    'torque': 5,
    'vibration_1': 6,
}
# ---------- 轴承代码 -> 故障类别映射（基于文档） ----------
# 类别: 0=健康, 1=外圈故障, 2=内圈故障
BEARING_LABEL_MAP = {
    # 健康轴承 (6个)
    'K001': 0, 'K002': 0, 'K003': 0, 'K004': 0, 'K005': 0, 'K006': 0,
    # 外圈故障 (12个)
    'KA01': 1, 'KA03': 1, 'KA05': 1, 'KA06': 1,
    'KA07': 1, 'KA08': 1, 'KA09': 1, 'KA04': 1,
    'KA15': 1, 'KA16': 1, 'KA22': 1, 'KA30': 1,
    'KB27': 1,  # 外圈主导
    # 内圈故障 (14个)
    'KI01': 2, 'KI03': 2, 'KI05': 2, 'KI07': 2,
    'KI08': 2, 'KI04': 2, 'KI14': 2, 'KI16': 2,
    'KI17': 2, 'KI18': 2, 'KI21': 2, 'KB23': 2,
    'KB24': 2,
}

def load_paderborn_mat_file(
    file_path: str,
    signals_to_extract: List[str],
    signal_index_map: Optional[Dict[str, int]] = None
) -> Dict[str, np.ndarray]:
    """
    读取单个 .mat 文件，从结构体数组 Y 中提取指定信号

    功能描述:
        1. 加载 .mat 文件（支持 v7 和 v7.3 格式）
        2. 获取顶层变量名（与文件名相同）
        3. 访问该变量的 Y 字段（结构体数组）
        4. 根据信号索引映射提取对应的 Data 字段作为一维数组

    :param file_path: .mat 文件路径
    :param signals_to_extract: 要提取的信号名称列表
    :param signal_index_map: 信号名称到 Y 数组索引（0基）的映射，若为 None 则使用默认映射
    :return: 字典 {信号名: numpy 数组}
    """
    if signal_index_map is None:
        signal_index_map = SIGNAL_INDEX_MAP

    # ----- 1. 加载 .mat 文件 -----
    try:
        data = sio.loadmat(file_path, struct_as_record=False, squeeze_me=True)
    except NotImplementedError:
        # v7.3 格式使用 h5py
        data = {}
        with h5py.File(file_path, 'r') as f:
            for key in f.keys():
                if not key.startswith('__'):
                    data[key] = f[key][:]
        # 对于 h5py，需将 HDF5 数据集转换为 numpy 数组并解析结构
        # 此处暂不处理结构体解析，因为 v7.3 的结构体解析较复杂，可先用 scipy.io 或提供备用方案
        # 若您的文件为 v7.3，可能需要使用 mat73 库或手动解析 HDF5 结构
        # 此处暂时抛出提示
        raise NotImplementedError("暂不支持 v7.3 格式的结构体解析，请使用 scipy.io 版本或转换格式")

    # ----- 2. 获取顶层变量名（与文件名相同） -----
    file_basename = Path(file_path).stem  # 例如 'N09_M07_F10_K001_1'
    # 在 data 中查找与文件名相同的键
    if file_basename not in data:
        # 若未找到，尝试查找第一个非元数据键
        keys = [k for k in data.keys() if not k.startswith('__')]
        if not keys:
            raise KeyError(f"文件中未找到任何数据变量: {file_path}")
        var_name = keys[0]
        print(f"未找到变量名 {file_basename}，改用 {var_name}")
    else:
        var_name = file_basename

    # ----- 3. 获取结构体数组 Y -----
    struct_var = data[var_name]
    # 检查是否有 Y 属性（字段）
    if not hasattr(struct_var, 'Y'):
        # 如果 struct_var 是 numpy 对象数组，可能直接是 Y 的列表？
        # 尝试直接当作数组处理
        if isinstance(struct_var, np.ndarray) and struct_var.dtype.names is not None:
            # 可能是一个具有多个字段的结构体
            # 尝试获取 'Y' 字段
            if 'Y' in struct_var.dtype.names:
                Y = struct_var['Y']
            else:
                raise ValueError(f"变量 {var_name} 中未找到 Y 字段")
        else:
            raise ValueError(f"变量 {var_name} 不是期望的结构体")
    else:
        Y = struct_var.Y

    # Y 应是一个结构体数组，每个元素有 Data 字段
    # 如果 Y 是 numpy 对象数组，每个元素为 numpy.void 或对象
    # 我们需提取每个元素的 Data

    # ----- 4. 按索引提取信号 -----
    extracted = {}
    for sig in signals_to_extract:
        if sig in signal_index_map:
            idx = signal_index_map[sig]
            # 检查 idx 是否在 Y 范围内
            if idx < len(Y):
                # 获取第 idx 个元素的 Data
                elem = Y[idx]
                # Data 可能是 numpy 数组，或嵌套结构，我们取其值
                if hasattr(elem, 'Data'):
                    data_val = elem.Data
                elif isinstance(elem, np.void) and 'Data' in elem.dtype.names:
                    data_val = elem['Data']
                else:
                    # 若 elem 本身是数组，直接使用
                    data_val = np.array(elem)
                # 确保为一维
                data_val = np.asarray(data_val).flatten()
                extracted[sig] = data_val
            else:
                print(f"警告: 索引 {idx} 超出 Y 范围（长度为 {len(Y)}），信号 {sig} 未提取")
        else:
            print(f"警告: 信号 '{sig}' 未在索引映射中定义，请检查")

    return extracted


def load_paderborn_dataset(
    data_folder: str,
    signals_to_extract: Optional[List[str]] = None,
    file_extension: str = '.mat',
    signal_index_map: Optional[Dict[str, int]] = None
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    批量加载 Paderborn 数据集文件夹中的所有 .mat 文件

    :param data_folder: 数据集文件夹路径
    :param signals_to_extract: 要提取的信号名称列表，默认使用常用信号
    :param file_extension: 文件扩展名
    :param signal_index_map: 信号索引映射，若为 None 则使用默认映射
    :return: 字典，键为文件名，值为该文件的信号字典
    """
    if signals_to_extract is None:
        signals_to_extract = [
            'force', 'phase_current_1', 'phase_current_2',
            'speed', 'temp_2_bearing_m', 'torque', 'vibration_1'
        ]
    if signal_index_map is None:
        signal_index_map = SIGNAL_INDEX_MAP

    data_dict = {}
    file_paths = Path(data_folder).glob(f'*{file_extension}')

    for file_path in file_paths:
#        print(f"正在加载: {file_path.name}")
        try:
            sig_data = load_paderborn_mat_file(
                str(file_path),
                signals_to_extract,
                signal_index_map
            )
            data_dict[file_path.stem] = sig_data
        except Exception as e:
            print(f"加载文件 {file_path.name} 失败: {e}")

#    print(f"成功加载 {len(data_dict)} 个文件")
    return data_dict

def parse_paderborn_filename(filename):
    """解析文件名，返回工况和轴承代码"""
    import re
    stem = filename.stem if hasattr(filename, 'stem') else filename.split('.')[0]
    pattern = r'(N\d+)_(M\d+)_(F\d+)_([A-Z]+\d+)_(\d+)'
    match = re.match(pattern, stem)
    if not match:
        raise ValueError(f"无法解析文件名: {stem}")
    speed_str, torque_str, force_str, bearing_code, meas = match.groups()
    return {
        'speed': int(speed_str[1:]),
        'torque': float(torque_str[1:]) / 100,
        'force': int(force_str[1:]) * 100,
        'bearing_code': bearing_code,
        'measurement': int(meas)
    }

def prepare_paderborn_multichannel_dataloaders(
    all_data,
    signal_names,
    sample_length=512,
    step=None,
    batch_size=64,
    normalize='standard',
    train_bearing_codes=None,
    test_bearing_codes=None,
    test_size=0.2,
    random_seed=42,
    verbose=True,
    extract_handcrafted=False
):
    """
    按文件划分训练/测试集，支持为每个通道独立设置归一化方式
    若 extract_handcrafted=True，则返回融合 DataLoader（每个 batch 返回 raw, feat, label）

    参数:
        ... 同上 ...
        extract_handcrafted: bool, 是否提取手工特征并返回融合 DataLoader

    返回:
        当 extract_handcrafted=False 时:
            train_loader, test_loader, num_channels, num_classes, class_names, None
        当 extract_handcrafted=True 时:
            train_loader, test_loader, num_channels, num_classes, class_names, feat_dim
        其中 train_loader 和 test_loader 的每个 batch 为 (raw, feat, labels)
    """
    if step is None:
        step = sample_length

    # ----- 1. 构建文件标签和轴承代码映射 -----
    file_names = list(all_data.keys())
    file_label_map = {}
    file_bearing_map = {}

    for fname in file_names:
        try:
            info = parse_paderborn_filename(fname)
        except ValueError:
            continue
        bearing_code = info['bearing_code']
        if bearing_code in BEARING_LABEL_MAP:
            file_label_map[fname] = BEARING_LABEL_MAP[bearing_code]
            file_bearing_map[fname] = bearing_code

    valid_files = list(file_label_map.keys())
    if len(valid_files) == 0:
        raise RuntimeError("没有找到任何可用的文件，请检查标签映射。")

    file_to_idx = {fname: idx for idx, fname in enumerate(valid_files)}

    # ----- 2. 根据轴承代码划分文件 -----
    if train_bearing_codes is not None and test_bearing_codes is not None:
        train_files = []
        test_files = []
        for fname in valid_files:
            code = file_bearing_map[fname]
            if code in train_bearing_codes:
                train_files.append(fname)
            elif code in test_bearing_codes:
                test_files.append(fname)

        assigned = set(train_files) | set(test_files)
        if len(assigned) < len(valid_files):
            unassigned = set(valid_files) - assigned
            if verbose:
                print(f"警告: 以下文件未被分配到训练或测试集，将被忽略: {list(unassigned)[:5]}...")

        if not train_files or not test_files:
            raise ValueError(
                f"根据轴承代码未找到足够文件。训练文件数: {len(train_files)}, 测试文件数: {len(test_files)}。"
                f"请检查 train_bearing_codes={train_bearing_codes}, test_bearing_codes={test_bearing_codes} 是否正确。"
            )

        train_file_indices = [file_to_idx[f] for f in train_files]
        test_file_indices = [file_to_idx[f] for f in test_files]

        if verbose:
            print(f"使用轴承代码划分: 训练轴承 {len(train_bearing_codes)} 个, 测试轴承 {len(test_bearing_codes)} 个")
            print(f"  训练文件数: {len(train_files)}, 测试文件数: {len(test_files)}")

    else:
        if verbose:
            print("未指定轴承代码，使用随机按比例划分（文件级别）...")
        unique_file_indices = np.arange(len(valid_files))
        file_labels_for_split = [file_label_map[f] for f in valid_files]

        if isinstance(test_size, int) and test_size >= 1:
            test_count = test_size
        elif isinstance(test_size, float):
            test_count = int(len(valid_files) * test_size)
        else:
            raise ValueError("test_size 必须为浮点比例或整数个数")

        if test_count <= 0 or test_count >= len(valid_files):
            raise ValueError(f"测试集大小 {test_count} 无效，请调整 test_size 或提供更多文件。")

        train_file_indices, test_file_indices = train_test_split(
            unique_file_indices,
            test_size=test_count,
            random_state=random_seed,
            stratify=file_labels_for_split
        )
        if verbose:
            print(f"随机划分: 训练文件数 {len(train_file_indices)}, 测试文件数 {len(test_file_indices)}")

    # ----- 3. 切分样本 -----
    X_list = []
    y_list = []
    file_ids = []

    for fname, sig_dict in all_data.items():
        if fname not in file_to_idx:
            continue

        file_idx = file_to_idx[fname]
        label = file_label_map[fname]

        missing = [s for s in signal_names if s not in sig_dict]
        if missing:
            if verbose:
                print(f"警告: 文件 {fname} 缺少信号 {missing}，跳过")
            continue

        lengths = [len(sig_dict[s]) for s in signal_names]
        if len(set(lengths)) > 1:
            if verbose:
                print(f"警告: 文件 {fname} 中各信号长度不一致，跳过")
            continue

        total_len = lengths[0]
        num_samples = (total_len - sample_length) // step + 1
        if num_samples <= 0:
            if verbose:
                print(f"警告: 文件 {fname} 信号长度 {total_len} 小于样本长度 {sample_length}，跳过")
            continue

        for i in range(num_samples):
            start = i * step
            slices = []
            for s in signal_names:
                seg = sig_dict[s][start:start + sample_length]
                slices.append(seg)
            multi_channel = np.stack(slices, axis=0)  # (channels, length)
            X_list.append(multi_channel)
            y_list.append(label)
            file_ids.append(file_idx)

    if len(X_list) == 0:
        raise RuntimeError("未提取到任何样本。")

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)
    file_ids = np.array(file_ids)

    num_channels = X.shape[1]

    # 根据文件索引划分
    train_mask = np.isin(file_ids, train_file_indices)
    test_mask = np.isin(file_ids, test_file_indices)

    X_train = X[train_mask]
    y_train = y[train_mask]
    X_test = X[test_mask]
    y_test = y[test_mask]

    # ========== 解析归一化配置 ==========
    if isinstance(normalize, str):
        norm_config = [normalize] * num_channels
    elif isinstance(normalize, list):
        if len(normalize) != num_channels:
            raise ValueError(f"normalize 列表长度 ({len(normalize)}) 必须等于通道数 ({num_channels})")
        norm_config = normalize
    elif isinstance(normalize, dict):
        norm_config = []
        for s in signal_names:
            if s in normalize:
                norm_config.append(normalize[s])
            else:
                norm_config.append('standard')
    else:
        raise TypeError("normalize 必须为字符串、列表或字典")

    # ========== 执行归一化 ==========
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    for c in range(num_channels):
        method = norm_config[c]
        if method == 'standard':
            train_flat = X_train[:, c, :].reshape(-1, 1)
            test_flat = X_test[:, c, :].reshape(-1, 1)
            scaler = StandardScaler()
            scaler.fit(train_flat)
            X_train_scaled[:, c, :] = scaler.transform(train_flat).reshape(X_train.shape[0], -1)
            X_test_scaled[:, c, :] = scaler.transform(test_flat).reshape(X_test.shape[0], -1)
        elif method == 'sample':
            eps = 1e-8
            for i in range(X_train.shape[0]):
                sig = X_train[i, c, :]
                mean = sig.mean()
                std = sig.std()
                X_train_scaled[i, c, :] = (sig - mean) / (std + eps)
            for i in range(X_test.shape[0]):
                sig = X_test[i, c, :]
                mean = sig.mean()
                std = sig.std()
                X_test_scaled[i, c, :] = (sig - mean) / (std + eps)
        elif method == 'mean_only':
            for i in range(X_train.shape[0]):
                sig = X_train[i, c, :]
                X_train_scaled[i, c, :] = sig - sig.mean()
            for i in range(X_test.shape[0]):
                sig = X_test[i, c, :]
                X_test_scaled[i, c, :] = sig - sig.mean()
        elif method == 'minmax':
            train_flat = X_train[:, c, :].reshape(-1, 1)
            test_flat = X_test[:, c, :].reshape(-1, 1)
            min_val = train_flat.min()
            max_val = train_flat.max()
            if max_val - min_val < 1e-12:
                X_train_scaled[:, c, :] = 0
                X_test_scaled[:, c, :] = 0
            else:
                X_train_scaled[:, c, :] = 2 * (X_train[:, c, :] - min_val) / (max_val - min_val) - 1
                X_test_scaled[:, c, :] = 2 * (X_test[:, c, :] - min_val) / (max_val - min_val) - 1
        elif method == 'none':
            pass
        else:
            raise ValueError(f"未知的归一化方式: {method}")

    # ========== 打印摘要 ==========
    if verbose:
        print("\n" + "="*60)
        print("数据加载完成 - 信息摘要")
        print("="*60)
        print(f"归一化配置 (按通道): {dict(zip(signal_names, norm_config))}")
        print(f"信号通道数 (num_channels): {num_channels}")
        print(f"  通道名称: {signal_names}")
        print(f"样本长度 (sample_length): {sample_length}")
        print(f"滑动步长 (step): {step}")
        print(f"总样本数: {X.shape[0]}")
        print(f"训练样本数: {X_train_scaled.shape[0]}")
        print(f"测试样本数: {X_test_scaled.shape[0]}")
        print(f"类别数 (num_classes): {len(np.unique(y))}")
        print(f"类别名称: {['Healthy', 'Outer_Race', 'Inner_Race'][:len(np.unique(y))]}")
        print(f"训练集类别分布: {dict(zip(*np.unique(y_train, return_counts=True)))}")
        print(f"测试集类别分布: {dict(zip(*np.unique(y_test, return_counts=True)))}")
        print("="*60 + "\n")

    # ========== 提取手工特征（若需要） ==========
    feat_dim = None
    if extract_handcrafted:
        # 将原始信号转为 (N, L, C) 以适配特征提取函数
        X_train_for_feat = X_train_scaled.transpose(0, 2, 1)  # (N, L, C)
        X_test_for_feat = X_test_scaled.transpose(0, 2, 1)
        X_train_feat = extract_handcrafted_features_from_samples(X_train_for_feat)  # (N, feat_dim)
        X_test_feat = extract_handcrafted_features_from_samples(X_test_for_feat)
        # 标准化手工特征
        scaler_feat = StandardScaler()
        X_train_feat = scaler_feat.fit_transform(X_train_feat)
        X_test_feat = scaler_feat.transform(X_test_feat)
        feat_dim = X_train_feat.shape[1]

        # 转为 Tensor
        X_train_feat_tensor = torch.tensor(X_train_feat, dtype=torch.float32)
        X_test_feat_tensor = torch.tensor(X_test_feat, dtype=torch.float32)
    else:
        # 占位，不使用
        X_train_feat_tensor = None
        X_test_feat_tensor = None

    # ----- 转为 Tensor -----
    X_train_raw_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)  # (N, C, L)
    X_test_raw_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    # ----- 构建 Dataset 和 DataLoader -----
    if extract_handcrafted:
        # 融合 Dataset
        class FusionDataset(torch.utils.data.Dataset):
            def __init__(self, raw, feat, labels):
                self.raw = raw
                self.feat = feat
                self.labels = labels
            def __len__(self):
                return len(self.labels)
            def __getitem__(self, idx):
                return self.raw[idx], self.feat[idx], self.labels[idx]

        train_dataset = FusionDataset(X_train_raw_tensor, X_train_feat_tensor, y_train_tensor)
        test_dataset = FusionDataset(X_test_raw_tensor, X_test_feat_tensor, y_test_tensor)
    else:
        # 普通单输入 Dataset
        train_dataset = TensorDataset(X_train_raw_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_raw_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    num_classes = len(np.unique(y))
    class_names = ['Healthy', 'Outer_Race', 'Inner_Race'][:num_classes]

    return train_loader, test_loader, num_channels, num_classes, class_names, feat_dim

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

def evaluate_model(model, test_loader, device, criterion, epoch=None, num_epochs=None, verbose=True, multi_input=False):
    """
    在测试集上评估模型，支持单输入和多输入（如融合模型）
    参数与之前相同，新增 multi_input: 若为 True，则 test_loader 返回 (input1, input2, label)
    """
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        if multi_input:
            for inputs1, inputs2, labels in test_loader:
                inputs1, inputs2, labels = inputs1.to(device), inputs2.to(device), labels.to(device)
                outputs = model(inputs1, inputs2)   # 双输入调用
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_preds.append(predicted.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        else:
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

def extract_features_CNN_Fusion(model, data_loader, device):
    """
    :function
        提取融合模型的特征向量（融合层输出）
    :param
        model: FusionCNNMLP 模型
        data_loader: DataLoader，返回 (raw, feat, labels) 三元组
        device: 计算设备
    :return
        features: numpy array，形状 (样本数, 特征维度)
        y_true: numpy array，真实标签
    """
    model.eval()
    features = []
    labels = []

    with torch.no_grad():
        for raw, feat, lbl in data_loader:
            raw, feat = raw.to(device), feat.to(device)

            # ---- 手动前向传播到融合特征层 ----
            # CNN 分支
            x = model.pool1(model.relu1(model.bn1(model.conv1(raw))))
            x = model.pool2(model.relu2(model.bn2(model.conv2(x))))
            x = model.pool3(model.relu3(model.bn3(model.conv3(x))))
            x = x.view(x.size(0), -1)
            cnn_out = model.cnn_relu(model.cnn_fc(x))

            # MLP 分支
            y = model.mlp_relu1(model.mlp_bn1(model.mlp_fc1(feat)))
            mlp_out = model.mlp_relu2(model.mlp_bn2(model.mlp_fc2(y)))

            # 融合
            fused = torch.cat([cnn_out, mlp_out], dim=1)
            fused = model.fusion_relu(model.fusion_bn(model.fusion_fc(fused)))
            # 此时 fused 即为特征向量，形状 (batch, 128)

            features.append(fused.cpu().numpy())
            labels.append(lbl.numpy())

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
    :param le: 对象名称(class_names)
    :param title: 图片标题
    :param FIG_SAVE_VALID: bool
    :param FIG_SAVE_PATH: 保存目录
    :param condition_labels: 工况标签 (n_samples,)，若为 None 则仅按故障着色
    """

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
                        label=f'{le[i]}_{cond}',
                        s=25, alpha=0.7,
                        color=base_colors[i % len(base_colors)],
                        marker=cond_to_marker[cond]
                    )
    else:
        for i in range(num_classes):
            plt.scatter(
                X_pca[y_true == i, 0], X_pca[y_true == i, 1],
                label=le[i],
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


def PCA_3D_plot(X_feat, y_true, num_classes, le, title, FIG_SAVE_VALID, FIG_SAVE_PATH, condition_labels=None):
    """
    绘制三维 PCA 结果，支持可选工况标签

    :param X_feat: 特征矩阵 (n_samples, feature_dim)
    :param y_true: 故障类别标签 (n_samples,)
    :param num_classes: 类别总数
    :param le: 对象名称(class_names)
    :param title: 图片标题
    :param FIG_SAVE_VALID: bool
    :param FIG_SAVE_PATH: 保存目录
    :param condition_labels: 工况标签 (n_samples,)，若为 None 则仅按故障着色
    """
    # 特征标准化（PCA 对尺度敏感）
    scaler = StandardScaler()
    X_feat_norm = scaler.fit_transform(X_feat)

    # 提取前 3 个主成分
    pca = PCA(n_components=3, random_state=42)
    X_pca = pca.fit_transform(X_feat_norm)

    # 打印解释方差比例（帮助你判断三维是否足够）
    explained_var = pca.explained_variance_ratio_
    print(f"PC1 解释方差: {explained_var[0]:.2%}|PC2 解释方差: {explained_var[1]:.2%}|PC3 解释方差: {explained_var[2]:.2%}")
    print(f"前三维累计解释方差: {sum(explained_var[:3]):.2%}")

    # 创建 3D 画布
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection='3d')

    # 颜色与标记
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
                    ax.scatter(
                        X_pca[mask, 0],  # PC1
                        X_pca[mask, 1],  # PC2
                        X_pca[mask, 2],  # PC3
                        label=f'{le[i]}_{cond}',
                        s=30, alpha=0.7,
                        color=base_colors[i % len(base_colors)],
                        marker=cond_to_marker[cond],
                        depthshade=True  # 开启阴影增加立体感
                    )
    else:
        for i in range(num_classes):
            ax.scatter(
                X_pca[y_true == i, 0],
                X_pca[y_true == i, 1],
                X_pca[y_true == i, 2],
                label=le[i],
                s=25, alpha=0.7,
                color=base_colors[i % len(base_colors)]
            )

    # 坐标轴标签
    ax.set_xlabel(f'Principal Component 1 ({explained_var[0]:.1%})', fontsize=14)
    ax.set_ylabel(f'Principal Component 2 ({explained_var[1]:.1%})', fontsize=14)
    ax.set_zlabel(f'Principal Component 3 ({explained_var[2]:.1%})', fontsize=14)
    ax.set_title(label=title, fontsize=16, fontfamily="Times New Roman")

    # 图例（放在图外右侧）
    ax.legend(prop={'family': 'Times New Roman', 'size': 12}, bbox_to_anchor=(1.05, 1), loc='upper left')

    # 保存图片
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
    :param le: 对象名称(class_names)
    :param title: 图片标题
    :param FIG_SAVE_VALID: bool
    :param FIG_SAVE_PATH: 保存目录
    :param condition_labels: 工况标签 (n_samples,)，若为 None 则仅按故障着色
    """

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
                        label=f'{le[i]}_{cond}',
                        s=25, alpha=0.7,
                        color=base_colors[i % len(base_colors)],
                        marker=cond_to_marker[cond]
                    )
    else:
        for i in range(num_classes):
            plt.scatter(
                X_tsne[y_true == i, 0], X_tsne[y_true == i, 1],
                label=le[i],
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
    :param le: 对象名称(class_names)
    :param title: 图片标题
    :param FIG_SAVE_VALID: bool，是否保存图片
    :param FIG_SAVE_PATH: 保存目录
    :param condition_labels: 工况标签 (n_samples,)，若为 None 则仅按故障着色（默认）
    """

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
                        label=f'{le[i]}_{cond}',
                        s=25, alpha=0.7,
                        color=base_colors[i % len(base_colors)],
                        marker=cond_to_marker[cond]
                    )
    else:
        # 仅按故障类别着色（原始模式）
        for i in range(num_classes):
            plt.scatter(
                X_umap[y_true == i, 0], X_umap[y_true == i, 1],
                label=le[i],
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

def compute_handcrafted_features(signal):
    """
    :function
        从一维信号中提取时域统计特征（峰值、RMS、峰因子、脉冲因子、裕度因子、偏度、峭度、Hjorth三参数）
    :param
        signal: 一维 numpy 数组 (长度 L)
    :return
        features: dict，包含所有特征值的字典
    """
    import numpy as np
    signal = np.asarray(signal, dtype=np.float64)
    mean = np.mean(signal)
    std = np.std(signal, ddof=1)
    rms = np.sqrt(np.mean(signal**2))
    peak = np.max(np.abs(signal))
    crest_factor = peak / rms if rms > 0 else 0
    mean_abs = np.mean(np.abs(signal))
    impulse_factor = peak / mean_abs if mean_abs > 0 else 0
    sqrt_abs_mean = np.mean(np.sqrt(np.abs(signal)))**2
    clearance_factor = peak / sqrt_abs_mean if sqrt_abs_mean > 0 else 0
    skewness = np.mean(((signal - mean) / std)**3) if std > 0 else 0
    kurtosis = np.mean(((signal - mean) / std)**4) if std > 0 else 0

    # Hjorth 参数
    activity = np.var(signal, ddof=1)
    diff_signal = np.diff(signal)
    if len(diff_signal) > 0:
        var_diff = np.var(diff_signal, ddof=1)
        mobility = np.sqrt(var_diff / activity) if activity > 0 else 0
    else:
        mobility = 0
    if len(diff_signal) > 1:
        diff2 = np.diff(diff_signal)
        var_diff2 = np.var(diff2, ddof=1)
        mobility_diff = np.sqrt(var_diff2 / var_diff) if var_diff > 0 else 0
        complexity = mobility_diff / mobility if mobility > 0 else 0
    else:
        complexity = 0

    return {
        'peak': peak,
        'rms': rms,
        'crest_factor': crest_factor,
        'impulse_factor': impulse_factor,
        'clearance_factor': clearance_factor,
        'skewness': skewness,
        'kurtosis': kurtosis,
        'hjorth_activity': activity,
        'hjorth_mobility': mobility,
        'hjorth_complexity': complexity
    }

def extract_handcrafted_features_from_samples(X):
    """
    :function
        从多维样本中提取手工特征（支持单通道或多通道）
    :param
        X: numpy array，形状 (样本数, 信号长度, 通道数) 或 (样本数, 信号长度)
    :return
        features: numpy array，形状 (样本数, 特征总数)
    """
    import numpy as np
    if X.ndim == 3:
        n_samples, n_length, n_channels = X.shape
        all_feats = []
        for ch in range(n_channels):
            for i in range(n_samples):
                feats = compute_handcrafted_features(X[i, :, ch])
                vals = [feats[k] for k in ['peak', 'rms', 'crest_factor', 'impulse_factor',
                                           'clearance_factor', 'skewness', 'kurtosis',
                                           'hjorth_activity', 'hjorth_mobility', 'hjorth_complexity']]
                all_feats.append(vals)
        return np.array(all_feats).reshape(n_samples, n_channels * 10)
    else:
        features_list = []
        for i in range(X.shape[0]):
            feats = compute_handcrafted_features(X[i])
            vals = [feats[k] for k in ['peak', 'rms', 'crest_factor', 'impulse_factor',
                                       'clearance_factor', 'skewness', 'kurtosis',
                                       'hjorth_activity', 'hjorth_mobility', 'hjorth_complexity']]
            features_list.append(vals)
        return np.array(features_list)
