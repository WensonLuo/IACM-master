"""
@Time : 2021/3/31 20:23
@Author : 罗望成
@File : preprocessing.py
@Desc : v3
    1. 8类划分，只留心拍，不要心律
    2. 10倍交叉验证
    3. 只做带通滤波
    4. 不做SMOTE
"""
import matplotlib
from IPython.display import display
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import pickle as dill
import os
from tqdm import tqdm
from collections import OrderedDict, Counter
import shutil
import posixpath

import wfdb
from wfdb import processing
import scipy.signal as signal
from sklearn import preprocessing


## 去掉前后不满足心跳长度的波峰
def discard_short_peak(peak, label):
    # 输入一维数组all_peak[2274,], all_label
    end = 650000  # dat的长度固定650000
    for i in range(peak.shape[0] - 1, -1, -1):  # 下标i从2273开始
        if peak[i] < 100 or end - peak[i] < 156 or \
                (label[i] != 'N' and label[i] != 'L' and label[i] != 'R' and label[i] != 'A' \
                 and label[i] != 'V' and label[i] != '/' and label[i] != 'E' and label[i] != '!'):
            peak = np.delete(peak, i)
            label = np.delete(label, i)
    return peak, label


# 将一维数组Y里边的字符转int，然后构造出 [x,1] 形状的数组
def char2int(in_label):
    out_label = []
    for i in range(in_label.shape[0]):  # 一维数组长度
        if in_label[i] == 'N':
            out_label.append(0)  # 类型N
        elif in_label[i] == 'L':
            out_label.append(1)  # 类型L
        elif in_label[i] == 'R':
            out_label.append(2)  # 类型R
        elif in_label[i] == 'A':
            out_label.append(3)  # 类型APB
        elif in_label[i] == 'V':
            out_label.append(4)  # 类型PVC
        elif in_label[i] == '/':
            out_label.append(5)  # 类型PAB
        elif in_label[i] == 'E':
            out_label.append(6)  # 类型VEB
        elif in_label[i] == '!':
            out_label.append(7)  # 类型VFW
        else:
            print("Illegal Type ERROR: " + in_label[i])
    out_label = np.array(out_label)  # list转nparray
    return out_label


# denoise
def filter_channel(x):
    signal_freq = 360

    ### candidate channels for ECG
    P_wave = (0.67, 5)
    QRS_complex = (10, 50)
    T_wave = (1, 7)
    muscle = (5, 50)
    resp = (0.12, 0.5)
    ECG_preprocessed = (0.5, 50)
    wander = (0.001, 0.5)
    noise = 50

    ### use low (wander), middle (ECG_preprocessed) and high (noise) for example
    bandpass_list = [wander, ECG_preprocessed]
    highpass_list = [noise]

    nyquist_freq = 0.5 * signal_freq
    filter_order = 1

    bandpass = ECG_preprocessed
    low = bandpass[0] / nyquist_freq
    high = bandpass[1] / nyquist_freq
    b, a = signal.butter(filter_order, [low, high], btype="band")
    y = signal.lfilter(b, a, x)

    return y


# 读取数据、波峰、标签保存到mit-bih-database.pkl
def read_dataset(data_path):
    #  read data and label
    all_data = []
    all_peak = []
    all_label = []
    filenames = pd.read_csv(os.path.join(data_path, 'RECORDS'), header=None)
    filenames = filenames.iloc[:, 0].values  # 拿第一列的值
    print(filenames)  # 打印文件名
    label_count = []
    peak_len = []
    for filename in tqdm(filenames):
        # read data
        dat = wfdb.rdrecord(os.path.join(data_path, '{0}'.format(filename)), channels=[0])
        dat = np.array(dat.p_signal)  # [650000,1]
        x = []
        for i in dat:
            x.append(i[0])
        all_data.append(np.array(x))
        # read label
        atr = wfdb.rdann(os.path.join(data_path, '{0}'.format(filename)), 'atr')
        peak = atr.sample
        label = np.array(atr.symbol)
        peak, label = discard_short_peak(peak, label)  # 丢弃长度小于256的心拍
        label = char2int(label)  # 按AAMI标准将char转int
        all_peak.append(peak)
        all_label.append(label)
        label_count = np.hstack([label_count, label])  # 一维数组
        peak_len.append(peak.shape[0])
    all_data = np.array(all_data)  # list转nparray
    # all_peak = np.array(all_peak)
    # all_label = np.array(all_label)
    print(Counter(label_count))  # 各类心拍的数量统计
    print(label_count.shape)  # 每个患者有多少心拍
    print(peak_len)  # 每个患者有多少心拍
    res = {'data': all_data, 'peak': all_peak, 'label': all_label}  # res 作为json格式保存数据和标签
    display(res)
    with open(os.path.join(output_path, 'mit-bih-database.pkl'), 'wb') as fout:
        dill.dump(res, fout)


def preprocess_dataset(data_path):
    # read pkl
    with open(os.path.join(data_path, 'mit-bih-database.pkl'), 'rb') as fin:
        res = dill.load(fin)

    ## 去噪
    all_data = res['data']

    for i in range(all_data.shape[0]):
        tmp_data = all_data[i]
        # tmp_std = np.std(tmp_data)
        # tmp_mean = np.mean(tmp_data)
        # all_data[i] = (tmp_data - tmp_mean) / tmp_std  # normalize
        all_data[i] = filter_channel(tmp_data).astype(np.float16)  # filter
    all_data = res['data']
    all_peak = res['peak']
    all_label = res['label']

    ## 切分心拍 [x,256]
    p_beat = []  # 单个患者的所有心拍
    for i in tqdm(range(len(all_peak))):
        for j in range(all_peak[i].shape[0]):
            beat = all_data[i, all_peak[i][j] - 100:all_peak[i][j] + 156]
            if len(p_beat):
                p_beat = np.row_stack((p_beat, beat))
            else:
                p_beat = beat
    all_beat = p_beat

    # 将list48的标签，转 arr[x,1]
    tmp_label = []
    for arr in all_label:
        arr = arr.reshape(-1, 1)  # [x,1]
        if len(tmp_label):
            tmp_label = np.vstack((tmp_label, arr))
        else:
            tmp_label = arr
    # [x,1]
    all_label = tmp_label

    # split train val test 8:1:1
    n_sample = all_label.shape[0]
    split_idx_1 = int(0.8 * n_sample)
    split_idx_2 = int(0.9 * n_sample)

    # 随机重排
    shuffle_idx = np.random.permutation(n_sample)  # 按心跳序列随机重排序
    all_beat = all_beat[shuffle_idx]
    all_label = all_label[shuffle_idx]

    # 训练集：验证集：测试集 = 8:1:1
    X_train = all_beat[:split_idx_1, :]
    X_val = all_beat[split_idx_1:split_idx_2, :]
    X_test = all_beat[split_idx_2:, :]
    Y_train = all_label[:split_idx_1, :]
    Y_val = all_label[split_idx_1:split_idx_2, :]
    Y_test = all_label[split_idx_2:, :]

    # save
    fout = open(os.path.join(data_path, 'mit_X_train.bin'), 'wb')
    np.save(fout, X_train)
    fout.close()

    fout = open(os.path.join(data_path, 'mit_X_val.bin'), 'wb')
    np.save(fout, X_val)
    fout.close()

    fout = open(os.path.join(data_path, 'mit_X_test.bin'), 'wb')
    np.save(fout, X_test)
    fout.close()

    fout = open(os.path.join(data_path, 'mit_Y_train.bin'), 'wb')
    np.save(fout, Y_train)
    fout.close()

    fout = open(os.path.join(data_path, 'mit_Y_val.bin'), 'wb')
    np.save(fout, Y_val)
    fout.close()

    fout = open(os.path.join(data_path, 'mit_Y_test.bin'), 'wb')
    np.save(fout, Y_test)
    fout.close()


''' 先运行read_dataset，完成后，再运行 preprocess_dataset'''
output_path = '../output/'
data_path = '../input/mit-bih-arrhythmia-database-1.0.0/'
read_dataset(data_path)
# preprocess_dataset(output_path)
