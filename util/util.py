import random

import torch
import math
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from math import radians, cos, sin, asin, sqrt
from sklearn.preprocessing import MinMaxScaler
# from GeoPrivacy.mechanism import random_laplace_noise

from model.LSTM import LSTMModel
from model.PMF import PMFModel
from model.DeepMove import TrajPreAttnAvgLongUser
from util.data_util import *
from util.embedding_util import *


def init_dataset(args):
    if args.dataset_name == 'nybc':
        dataset = NYBDataset(data_dir=args.data_dir, usernum=args.usernum, order=2, model_name=args.model_name)
        data_dir = 'data/raw_data_1706_np.npy'
        raw_data_np = np.load(data_dir, allow_pickle=True)
        columns_to_normalize = [2, 3]
        scaler = MinMaxScaler()
        # raw_data_np[:, columns_to_normalize] = scaler.fit_transform(raw_data_np[:, columns_to_normalize])
    elif args.dataset_name == 'tokyoci':
        dataset = TokyoCheckinDataset()
        data_dir = 'data/tokyoci.npy'
        raw_data_np = np.load(data_dir, allow_pickle=True)
        columns_to_normalize = [2, 3]
        scaler = MinMaxScaler()
        raw_data_np[:, columns_to_normalize] = scaler.fit_transform(raw_data_np[:, columns_to_normalize])
    elif args.dataset_name == 'gowallaci':
        dataset = GowallaCheckinDataset()
        data_dir = 'data/gowallaci.npy'
        raw_data_np = np.load(data_dir, allow_pickle=True)
        columns_to_normalize = [2, 3]
        scaler = MinMaxScaler()
        raw_data_np[:, columns_to_normalize] = scaler.fit_transform(raw_data_np[:, columns_to_normalize])
    else:
        raise ValueError("No such dataset: " + args.dataset_name + "!")
    return dataset, scaler


def init_model(args, model_name, input_size, hidden_size, output_size):
    if model_name == 'LSTM':
        model = LSTMModel(input_size=input_size, hidden_size=hidden_size, output_size=output_size)
    elif model_name == 'PMF':
        model = PMFModel(hidden_size=hidden_size, output_size=788)
    else:
        raise ValueError("No such model: " + model_name + "!")
    return model


def init_loss(model_name):
    if model_name == 'LSTM':
        loss = nn.MSELoss()
    elif model_name == 'PMF':
        loss = nn.CrossEntropyLoss()
        # loss = nn.MSELoss()
    else:
        raise ValueError("No such model: " + model_name + "!")
    return loss


def GeoGraphI(global_iter, dataset, batch_size, cover, epsilon):
    # gt_data_np
    gt_data = []
    for b in range(batch_size):
        ord_list = list()
        p_list = list()
        for i in range(-cover, cover):
            if global_iter + i >= 0 and global_iter + i < len(dataset) and i != 0:
                ord_list.append(global_iter + i)
                x_loc = dataset[global_iter + i, [2, 3]]
                t_loc = dataset[global_iter, [2, 3]]
                dist = ((x_loc[0] - t_loc[0]) ** 2 + (x_loc[1] - t_loc[1]) ** 2) ** 0.5
                p_list.append(math.exp(- epsilon * dist / 2))
        p_sum = sum(p_list)
        p_list = [p / p_sum for p in p_list]
        sampled_idx = random.choices(range(len(p_list)), p_list)[0]
        gt_data.append(dataset[ord_list[sampled_idx], [2, 3]])
    gt_data_np = np.array(gt_data)
    gt_label = []
    ord_list = list()
    p_list = list()
    for i in range(-cover, cover):
        if global_iter + batch_size + i >= 0 and global_iter + batch_size + i < len(dataset) and i != 0:
            ord_list.append(global_iter + batch_size + i)
            x_loc = dataset[global_iter + batch_size + i, [2, 3]]
            t_loc = dataset[global_iter + batch_size, [2, 3]]
            dist = ((x_loc[0] - t_loc[0]) ** 2 + (x_loc[1] - t_loc[1]) ** 2) ** 0.5
            p_list.append(math.exp(- epsilon * dist / 2))
    p_sum = sum(p_list)
    p_list = [p / p_sum for p in p_list]
    sampled_idx = random.choices(range(len(p_list)), p_list)[0]
    gt_label.append(dataset[ord_list[sampled_idx], [2, 3]])
    gt_label_np = np.array(gt_label)
    return gt_data_np, gt_label_np


def APGEM(global_iter, dataset, batch_size, cover, epsilon, used_epsilon, asr_t, ait_t):
    gama_t = 0.5 * asr_t + 0.5 / ait_t
    remain_epsilon = epsilon - used_epsilon
    epsilon_t = math.exp(-gama_t) * remain_epsilon
    used_epsilon += epsilon_t

    # gt_data_np
    gt_data = []
    for b in range(batch_size):
        ord_list = list()
        p_list = list()
        for i in range(-cover, cover):
            if global_iter + i >= 0 and global_iter + i < len(dataset) and i != 0:
                ord_list.append(global_iter + i)
                x_loc = dataset[global_iter + i, [2, 3]]
                t_loc = dataset[global_iter, [2, 3]]
                dist = ((x_loc[0] - t_loc[0]) ** 2 + (x_loc[1] - t_loc[1]) ** 2) ** 0.5
                p_list.append(math.exp(- epsilon_t * dist / 2))
        p_sum = sum(p_list)
        p_list = [p / p_sum for p in p_list]
        sampled_idx = random.choices(range(len(p_list)), p_list)[0]
        gt_data.append(dataset[ord_list[sampled_idx], [2, 3]])
    gt_data_np = np.array(gt_data)
    gt_label = []
    ord_list = list()
    p_list = list()
    for i in range(-cover, cover):
        if global_iter + batch_size + i >= 0 and global_iter + batch_size + i < len(dataset) and i != 0:
            ord_list.append(global_iter + batch_size + i)
            x_loc = dataset[global_iter + batch_size + i, [2, 3]]
            t_loc = dataset[global_iter + batch_size, [2, 3]]
            dist = ((x_loc[0] - t_loc[0]) ** 2 + (x_loc[1] - t_loc[1]) ** 2) ** 0.5
            p_list.append(math.exp(- epsilon_t * dist / 2))
    p_sum = sum(p_list)
    p_list = [p / p_sum for p in p_list]
    sampled_idx = random.choices(range(len(p_list)), p_list)[0]
    gt_label.append(dataset[ord_list[sampled_idx], [2, 3]])
    gt_label_np = np.array(gt_label)
    return gt_data_np, gt_label_np, used_epsilon


def train_for_loss(args,
                   model_name,
                   model,
                   loss_fn,
                   batch_size,
                   gt_data_np,
                   gt_tim_np,
                   gt_label_np,
                   global_iter,
                   dataset,
                   device):
    model.train()
    if model_name == 'LSTM':
        gt_data_np_final = gt_data_np.copy()
        gt_label_np_final = gt_label_np.copy()
        # ================= Geo-Indistinguishability ==================== #
        # if args.is_geo:
        #    for i in range(len(gt_data_np_final)):
        #        gt_data_np_final[i] += np.array(random_laplace_noise(args.geo_epsilon), dtype=np.float32)
        #    gt_label_np_final += np.array(random_laplace_noise(args.geo_epsilon), dtype=np.float32)
        # ======================================================= #
        # ================ Geo-Graph-Indistinguishability ================ #
        if args.is_geogi:
            gt_data_np_final, gt_label_np_final = GeoGraphI(global_iter, dataset, batch_size, args.geogi_cover,
                                                            args.geogi_epsilon)
        # ======================================================= #
        gt_data = torch.from_numpy(gt_data_np_final.reshape(1, batch_size, 2).astype(np.float32)).to(device)
        gt_label = torch.from_numpy(gt_label_np_final.reshape(1, 2).astype(np.float32)).to(device)
        out = model(gt_data)
        true_loss = loss_fn(out, gt_label)
    else:
        raise ValueError("No such model: " + model_name + "!")
    return true_loss


def init_dummy_data(batch_size, model_name, device):
    if model_name == 'LSTM':
        dummy_data = torch.rand(1, batch_size, 2).to(device).requires_grad_(True)
        dummy_label = torch.rand(1, 2).to(device).requires_grad_(True)
    elif model_name == 'PMF':
        dummy_data = torch.rand(1, batch_size, 977).to(device).requires_grad_(True)
        dummy_label = torch.rand(1, 2).to(device).requires_grad_(True)
    else:
        raise ValueError("No such model: " + model_name + "!")
    return dummy_data, dummy_label


def init_dummy_data_embedding(true_time):
    res_list = []
    for i in range(len(true_time)):
        random_lat_code = np.zeros(361)
        random_lat_code[random.randint(0, 360)] = 1
        random_lnt_code = np.zeros(427)
        random_lnt_code[random.randint(0, 426)] = 1

        random_location_code = np.concatenate((random_lat_code, random_lnt_code))
        random_all_code = location_time_embedding_single(random_location_code, true_time[i])
        res_list.append(random_all_code)

    return np.array(res_list)


def init_dummy_label_embedding():
    random_lat_code = np.zeros(361)
    random_lat_code[random.randint(0, 360)] = 1
    random_lnt_code = np.zeros(427)
    random_lnt_code[random.randint(0, 426)] = 1
    random_location_code = np.concatenate((random_lat_code, random_lnt_code))
    return random_location_code


def find_best_record(attack_record, dlg_attack_round, gt_label):
    label_dist = []
    edist = nn.PairwiseDistance(p=2)
    for r in range(dlg_attack_round):
        label_dist.append(edist(attack_record[r]['last_dummy_label'], gt_label).item())
    best_index = label_dist.index(min(label_dist))
    return best_index, label_dist[best_index]


def find_best_record2(attack_record, dlg_attack_round, gt_label):
    label_dist_list = []
    edist = nn.PairwiseDistance(p=2)
    for r in range(dlg_attack_round):
        best_idx = 10001
        best_dist = 10001
        for i in range(len(attack_record[r]['dummy_label_list'])):
            label_dist = edist(
                torch.from_numpy(attack_record[r]['dummy_label_list'][i].reshape(1, 2).astype(np.float32)).cuda(),
                gt_label).item()
            if label_dist < best_dist:
                best_idx = i
                best_dist = label_dist
        label_dist_list.append([best_idx, best_dist])
    # print("label_dist_list: {}".format(label_dist_list))
    ridx = np.argmin(np.array(label_dist_list)[:, 1])
    return ridx, label_dist_list[ridx][0], label_dist_list[ridx][1]


def save_args(args, save_path):
    file = open(save_path + r'\args.txt', 'w')
    for arg in vars(args):
        file.write("{}==>{}\n".format(arg, str(getattr(args, arg))))
    file.close()


def loc_distance(loc1, loc2):
    lng1 = loc1[1]
    lat1 = loc1[0]
    lng2 = loc2[1]
    lat2 = loc2[0]
    lng1, lat1, lng2, lat2 = map(radians, [float(lng1), float(lat1), float(lng2), float(lat2)])
    dlon = lng2 - lng1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    distance = 2 * asin(sqrt(a)) * 6371 * 1000
    distance = round(distance / 1000, 3)
    return distance


def list_avg(x):
    if len(x) == 0:
        return 0
    return sum(x) / len(x)


def avg_dy_dx_list(dy_dx_list):
    # 假设 dy_dx_list 是一个包含多个梯度的列表
    # dy_dx_list = [dy_dx1, dy_dx2, ...]

    # 初始化累加梯度的变量
    total_gradient = [torch.zeros_like(gradient) for gradient in dy_dx_list[0]]

    # 累加所有梯度
    for gradient in dy_dx_list:
        for i in range(len(gradient)):
            total_gradient[i] += gradient[i]

    # 计算平均梯度
    num_gradients = len(dy_dx_list)
    average_gradient = [gradient / num_gradients for gradient in total_gradient]
    return average_gradient


def draw_arrow(traj_data, plt_layer, color):
    for i in range(len(traj_data) - 1):
        x_start, y_start = traj_data[i]
        x_end, y_end = traj_data[i + 1]
        dx = x_end - x_start
        dy = y_end - y_start
        plt_layer.arrow(x_start, y_start, dx, dy, head_width=0.02, head_length=0.02, color=color)
