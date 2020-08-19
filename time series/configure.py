"""
Description     : 本文件用来控制各种敞亮，阈值等
"""
import os
import  sys
import torch
import matplotlib.pyplot as plt

savepath = '/root/data/save/data_analysis'
datapath = '/root/save/data'
figsavepath = '/root/data/save/data_analysis/figure'  # 存图路径
logsavepath = '/root/data/logs'
#savepath = '/root/data/save/data_analysis'
#datapath = '/root/data'
#figsavepath = '/root/save/figure'  # 存图路径
developemode = 0
device_num = 8
channel_num = 9
pattern1_value_threshold = [137,250]
pattern2_value_threshold = 1
pattern3_value_threshold = [250,0]
pattern1_count_threshold = 10
pattern2_count_threshold = 5
pattern3_count_threshold = 1000
time_threshold =10
data_pattern=[[1]*5,[]]
flag_count = 11
cc_count = 6
sep = 200
predict_len = 8000
train_len = 24000
train_splite=0.6
val_splite = 0.2





