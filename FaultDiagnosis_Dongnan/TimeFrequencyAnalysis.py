import os.path

from fontTools.unicodedata import block
from matplotlib import pyplot as plt
from matplotlib import rcParams
import numpy as np
import pandas as pd

DATASET_FILE_PATH = r"D:\PythonProject\Dataset\Mechanical-datasets-master\gearbox\gearset\RS20_L0"
img_save_path = r"D:\PythonProject\NeuralNetwork\FaultDiagnosis_Dongnan\ImageSave"
time_img_save_path = os.path.join(img_save_path, "Time")
fft_img_save_path =os.path.join(img_save_path, "FFT")
envelope_img_save_path = os.path.join(img_save_path, "Envelope")

for path in [img_save_path, fft_img_save_path, envelope_img_save_path]:
    os.makedirs(path, exist_ok=True)

print(f"{time_img_save_path}\n{fft_img_save_path}\n{envelope_img_save_path}")
config = {
    "font.family": 'serif', # 衬线字体
    "font.size": 14, # 相当于小四大小
    "font.serif": ['SimSun'], # 宋体
    "mathtext.fontset": 'stix', # matplotlib渲染数学字体时使用的字体，和Times New Roman差别不大
    'axes.unicode_minus': False # 处理负号，即-号
}
rcParams.update(config)

##========绘制时域信号图========
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
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
    plt.rcParams['axes.unicode_minus'] = False  # 显示负号
    font = {'family': 'Times New Roman', 'size': '20', 'color': '0.5', 'weight': 'bold'}

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
        plt.savefig(img_save_path, dpi=500, bbox_inches='tight')
    plt.show(block=False)

##========绘制频域信号图========##
def plt_fft_img(arr, fs, ylabel='Amp(mg)', title='频域图', img_save_path=None, vline=None, hline=None, xlim=None):
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
        plt.savefig(img_save_path, dpi=500, bbox_inches = 'tight')
    plt.tight_layout()
    plt.show(block=False)

def plt_stft_img(arr, fs, ylabel='Amp(mg)', title='stft时频域图', img_save_path=None, vline=None, hline=None, xlim=None):
    """
    :fun: 绘制stft时频域图模板
    :param arr: 输入一维时域数组数据
    :param fs: 采样频率
    :param ylabel: y轴标签
    :param title: 图标题
    :return: None
    """
    import scipy.signal as signal
    import numpy as np
    import matplotlib.pyplot as plt

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
        plt.savefig(img_save_path, dpi=500, bbox_inches = 'tight')
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
        plt.savefig(img_save_path, dpi=500, bbox_inches = 'tight')

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


fs = 5120
# ##=====================================时域分析=================================
# ##=======缺损=====##
# file_path = dataset_file_path_get(DATASET_FILE_PATH, "Chipped_20_0.csv")
# df = data_read(file_path)
# acc_arr= df.iloc[:,7].values
# plt_time_domain(acc_arr, fs=fs, title="Chipped_20_0", img_save_path=time_img_save_path)
# kurtosis_value = calculate_kurtosis(acc_arr)
# print(f"Kurtosis of the signal: {kurtosis_value}")
# ##======断齿======##
# file_path = dataset_file_path_get(DATASET_FILE_PATH, "Miss_20_0.csv")
# df = data_read(file_path)
# acc_arr= df.iloc[:,7].values
# plt_time_domain(acc_arr, fs=fs, title="Miss_20_0", img_save_path=time_img_save_path)
# kurtosis_value = calculate_kurtosis(acc_arr)
# print(f"Kurtosis of the signal: {kurtosis_value}")
# ##=======齿根磨损=====##
# file_path = dataset_file_path_get(DATASET_FILE_PATH, "Root_20_0.csv")
# df = data_read(file_path)
# acc_arr= df.iloc[:,7].values
# plt_time_domain(acc_arr, fs=fs, title="Root_20_0", img_save_path=time_img_save_path)
# kurtosis_value = calculate_kurtosis(acc_arr)
# print(f"Kurtosis of the signal: {kurtosis_value}")
# ##=======齿面磨损=====##
# file_path = dataset_file_path_get(DATASET_FILE_PATH, "Surface_20_0.csv")
# df = data_read(file_path)
# acc_arr= df.iloc[:,7].values
# plt_time_domain(acc_arr, fs=fs, title="Surface_20_0", img_save_path=time_img_save_path)
# kurtosis_value = calculate_kurtosis(acc_arr)
# print(f"Kurtosis of the signal: {kurtosis_value}")
# ##=======健康齿轮=====##
# file_path = dataset_file_path_get(DATASET_FILE_PATH, "Health_20_0.csv")
# df = data_read(file_path)
# acc_arr= df.iloc[:,7].values
# plt_time_domain(acc_arr, fs=fs, title="Health_20_0", img_save_path=time_img_save_path)
# kurtosis_value = calculate_kurtosis(acc_arr)
# print(f"Kurtosis of the signal: {kurtosis_value}")
#
##==========================================频域分析========================================
##=======缺损=====##
file_path = dataset_file_path_get(DATASET_FILE_PATH, "Chipped_20_0.csv")
df = data_read(file_path)
acc_arr= df.iloc[:,7].values
plt_fft_img(acc_arr, fs=fs, title='Chipped_20_0', img_save_path=fft_img_save_path, vline=[20, 40, 60, 80, 100], xlim=500)
##=======断齿=====##
file_path = dataset_file_path_get(DATASET_FILE_PATH, "Miss_20_0.csv")
df = data_read(file_path)
acc_arr= df.iloc[:,7].values
plt_fft_img(acc_arr, fs=fs, title='Miss_20_0', img_save_path=fft_img_save_path, vline=[20, 40, 60, 80, 100], xlim=500)
##=======齿根磨损=====##
file_path = dataset_file_path_get(DATASET_FILE_PATH, "Root_20_0.csv")
df = data_read(file_path)
acc_arr= df.iloc[:,7].values
plt_fft_img(acc_arr, fs=fs, title='Root_20_0', img_save_path=fft_img_save_path, vline=[20, 40, 60, 80, 100], xlim=500)
##=======齿面磨损=====##
file_path = dataset_file_path_get(DATASET_FILE_PATH, "Surface_20_0.csv")
df = data_read(file_path)
acc_arr= df.iloc[:,7].values
plt_fft_img(acc_arr, fs=fs, title='Surface_20_0', img_save_path=fft_img_save_path, vline=[20, 40, 60, 80, 100], xlim=500)
##=======健康齿轮=====##
file_path = dataset_file_path_get(DATASET_FILE_PATH, "Health_20_0.csv")
df = data_read(file_path)
acc_arr= df.iloc[:,7].values
plt_fft_img(acc_arr, fs=fs, title='Health_20_0', img_save_path=fft_img_save_path, vline=[20, 40, 60, 80, 100], xlim=500)

##=========================================包络谱分析=======================================
##=======缺损=====##
file_path = dataset_file_path_get(DATASET_FILE_PATH, "Chipped_20_0.csv")
df = data_read(file_path)
acc_arr= df.iloc[:,7].values
plt_envelope_spectrum(acc_arr, fs=fs, xlim=50, title='Chipped_20_0', img_save_path= envelope_img_save_path)
##=======断齿=====##
file_path = dataset_file_path_get(DATASET_FILE_PATH, "Miss_20_0.csv")
df = data_read(file_path)
acc_arr= df.iloc[:,7].values
plt_envelope_spectrum(acc_arr, fs=fs, xlim=50, title='Miss_20_0', img_save_path= envelope_img_save_path)
##=======齿根磨损=====##
file_path = dataset_file_path_get(DATASET_FILE_PATH, "Root_20_0.csv")
df = data_read(file_path)
acc_arr= df.iloc[:,7].values
plt_envelope_spectrum(acc_arr, fs=fs, xlim=50, title='Root_20_0', img_save_path= envelope_img_save_path)
##=======齿面磨损=====##
file_path = dataset_file_path_get(DATASET_FILE_PATH, "Surface_20_0.csv")
df = data_read(file_path)
acc_arr= df.iloc[:,7].values
plt_envelope_spectrum(acc_arr, fs=fs, xlim=50, title='Surface_20_0', img_save_path= envelope_img_save_path)
##=======健康齿轮=====##
file_path = dataset_file_path_get(DATASET_FILE_PATH, "Health_20_0.csv")
df = data_read(file_path)
acc_arr= df.iloc[:,7].values
plt_envelope_spectrum(acc_arr, fs=fs, xlim=50, title='Health_20_0', img_save_path= envelope_img_save_path)

plt.show()