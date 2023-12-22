import subprocess

import numpy as np

from util import util

# 定义不同的 step_size 值
# batch_size_list = [4, 8, 1]
# global_epoch = [0, 10, 50]
#
# for epoch in global_epoch:
#     for batch_size in batch_size_list:
#         experiment_name = "dlg_lstm_batch{}_globalepoch{}".format(batch_size, epoch)
#         command = f"D:\\app\\anaconda3\\envs\\newtorch\\python.exe D:\\pythoncode\\dlg-defense\\attack.py --batch_size {batch_size} --experiment_name {experiment_name} --dlg_attack_global_epoch {epoch}"
#         subprocess.call(command, shell=True)
#
# for epoch in global_epoch:
#     for batch_size in batch_size_list:
#         experiment_name = "dlg_lstm_batch{}_globalepoch{}_enhance".format(batch_size, epoch)
#         command = f"D:\\app\\anaconda3\\envs\\newtorch\\python.exe D:\\pythoncode\\dlg-defense\\attack.py --batch_size {batch_size} --experiment_name {experiment_name} --dlg_attack_global_epoch {epoch} --is_dlg_enhance 1"
#         subprocess.call(command, shell=True)

attack_methods = ['idlg', 'cpl', 'sapag', 'invgrad', 'stgia']
datasets = ['tokyoci']  # , 'gowallaci']
for dataset in datasets:
    for attack_method in attack_methods:
        experiment_name = f"{attack_method}_lstm_batch8_{dataset}"
        command = f"C:\\Users\\LLZHENG\\anaconda3\\envs\\fedsimu\\python.exe E:\\workspace\\dlg-defense\\main.py " \
                  f"--experiment_name {experiment_name} " \
                  f"--attack_method {attack_method} "
        subprocess.call(command, shell=True)

    experiment_name = f"stgia_lstm_batch8_random_{dataset}"
    command = f"C:\\Users\\LLZHENG\\anaconda3\\envs\\fedsimu\\python.exe E:\\workspace\\dlg-defense\\main.py " \
              f"--experiment_name {experiment_name} " \
              f"--attack_method stgia " \
              f"--is_use_last_result 0 "
    subprocess.call(command, shell=True)

    experiment_name = f"stgia_lstm_batch8_without_road_{dataset}"
    command = f"C:\\Users\\LLZHENG\\anaconda3\\envs\\fedsimu\\python.exe E:\\workspace\\dlg-defense\\main.py " \
              f"--experiment_name {experiment_name} " \
              f"--attack_method stgia " \
              f"--is_use_road 0 "

    subprocess.call(command, shell=True)

# lat_candidate_set = np.load('util/lat_candidate_set.npy', allow_pickle=True)
# lnt_candidate_set = np.load('util/lnt_candidate_set.npy', allow_pickle=True)
# print(lat_candidate_set)
# # print(lnt_candidate_set)
# for i in range(len(lat_candidate_set) - 1):
#     print(lat_candidate_set[i + 1] - lat_candidate_set[i])
#
# print(lnt_candidate_set)
# # print(lnt_candidate_set)
# for i in range(len(lnt_candidate_set) - 1):
#     print(lnt_candidate_set[i + 1] - lnt_candidate_set[i])
#
# print(util.loc_distance([0, 0], [0.006, 0.006]))
