import os.path
from matplotlib import pyplot as plt
from matplotlib import rcParams
import numpy as np
import pandas as pd
from pandas.core.common import random_state
import utils

# =========================数据集路径配置=============================
DATASET_FILE_PATH = r"D:\PythonProject\Dataset\Mechanical-datasets-master\gearbox\gearset\RS20_L0"
img_save_path = r"D:\PythonProject\NeuralNetwork\FaultDiagnosis_Dongnan\ImageSave"
time_img_save_path = os.path.join(img_save_path, "Time")
fft_img_save_path =os.path.join(img_save_path, "FFT")
envelope_img_save_path = os.path.join(img_save_path, "Envelope")

for path in [time_img_save_path, fft_img_save_path, envelope_img_save_path]:
    os.makedirs(path, exist_ok=True)

config = {
    # 优先使用 Times New Roman，中文缺失时回退到 SimHei（黑体）
    "font.family": ['Times New Roman', 'SimHei', 'Microsoft YaHei'],
    "mathtext.fontset": 'stix',
    'axes.unicode_minus': False,
    'savefig.dpi': 600,
}
rcParams.update(config)

# ====================设置全局超参数===========================
fs=5120
random_seed=42
rcParams['agg.path.chunksize'] = 20000

file_path = utils.dataset_file_path_get(DATASET_FILE_PATH, "Chipped_20_0.csv")
df = utils.data_read(file_path)
acc_arr = df.iloc[:,7].values
utils.plt_time_domain(arr=acc_arr,fs=fs, title="Chipped_20_0", img_save_path=time_img_save_path)
utils.plt_fft_img(acc_arr, fs=fs, title='Chipped_20_0', img_save_path=fft_img_save_path, vline=[20, 40, 60, 80, 100], xlim=500)

# =============仅加入Guassian噪声=====================
acc_arr_noise = utils.add_gaussian_noise(signal=acc_arr, snr_db=10, random_state=random_seed)
utils.plt_time_domain(arr=acc_arr_noise,fs=fs, title="Chipped_20_0_Guassian", img_save_path=time_img_save_path)
utils.plt_fft_img(acc_arr_noise, fs=fs, title='Chipped_20_0_Gaussian', img_save_path=fft_img_save_path, vline=[20, 40, 60, 80, 100], xlim=500)

# ==============仅加入Impulse噪声=====================
acc_arr_noise = utils.add_impulse_noise(signal=acc_arr, impulse_prob=0.01, amplitude_scale=5, random_state=random_seed)
utils.plt_time_domain(arr=acc_arr_noise,fs=fs, title="Chipped_20_0_Impulse", img_save_path=time_img_save_path)
utils.plt_fft_img(acc_arr_noise, fs=fs, title='Chipped_20_0_Impulse', img_save_path=fft_img_save_path, vline=[20, 40, 60, 80, 100], xlim=500)

# ===============仅加入Harmonic噪声===================
acc_arr_noise = utils.add_harmonic_interference(signal=acc_arr, freqs=[50, 100, 150],  random_state=random_seed)
utils.plt_time_domain(arr=acc_arr_noise,fs=fs, title="Chipped_20_0_Harmonic", img_save_path=time_img_save_path)
utils.plt_fft_img(acc_arr_noise, fs=fs, title='Chipped_20_0_Harmonic', img_save_path=fft_img_save_path, vline=[20, 40, 60, 80, 100], xlim=500)

# =====================三种噪声混合====================
acc_arr_noise = utils.add_noise_combination(
    signal=acc_arr,
    noise_type='all',
    gaussian_kw={'snr_db':10},
    impulse_kw={'impulse_prob':0.01, 'amplitude_scale':5},
    harmonic_kw={'freqs':[50, 100, 150]}
)
utils.plt_time_domain(arr=acc_arr_noise,fs=fs, title="Chipped_20_0_Mix", img_save_path=time_img_save_path)
utils.plt_fft_img(acc_arr_noise, fs=fs, title='Chipped_20_0_Mix', img_save_path=fft_img_save_path, vline=[20, 40, 60, 80, 100], xlim=500)
plt.show()