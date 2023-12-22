import pickle
import numpy as np
from util.util import *


def get_scaler(model_name):
    if model_name == 'nycb':
        data_dir = 'data/raw_data_1706_np.npy'
        raw_data_np = np.load(data_dir, allow_pickle=True)
        columns_to_normalize = [2, 3]
        scaler = MinMaxScaler()
        raw_data_np[:, columns_to_normalize] = scaler.fit_transform(raw_data_np[:, columns_to_normalize])
    if model_name == 'tokyoci':
        data_dir = 'data/tokyoci.npy'
        raw_data_np = np.load(data_dir, allow_pickle=True)
        columns_to_normalize = [2, 3]
        scaler = MinMaxScaler()
        raw_data_np[:, columns_to_normalize] = scaler.fit_transform(raw_data_np[:, columns_to_normalize])
    if model_name == 'gowallaci':
        data_dir = 'data/gowallaci.npy'
        raw_data_np = np.load(data_dir, allow_pickle=True)
        columns_to_normalize = [2, 3]
        scaler = MinMaxScaler()
        raw_data_np[:, columns_to_normalize] = scaler.fit_transform(raw_data_np[:, columns_to_normalize])
    return scaler


model_name = 'gowallaci'
result_name = 'testDTdp10'

scaler = get_scaler(model_name)

expe_rounds = 100
global_iter = 2000
attack_iter = 200
grad_threshold = 0.005
dist_threshold = 1

attack_success_rounds = 0
attack_sum_success_iterations = 0

for er in range(1, expe_rounds):
    attack_record = pickle.load(
        open('result_process/result/expe_[{}]/attack_record_er={}_gloiter=1_dlground=0.pickle'.format(result_name, er),
             'rb'))
    grad_loss_list = attack_record['grad_loss_list']
    # dummy_label_list = attack_record['dummy_label_list']
    # true_label = attack_record['gt_label']
    # true_label = scaler.inverse_transform(true_label.reshape(1, 2)).reshape(2)
    # for ai in range(attack_iter):
    #     dummy_label = dummy_label_list[ai]
    #     dummy_label = scaler.inverse_transform(dummy_label.reshape(1, 2)).reshape(2)
    #     dist = loc_distance(true_label, dummy_label)
    #     print("attack_iter: {}, grad_loss: {}, dummy_label: {}, true_label: {}, dist: {}".format(ai, grad_loss_list[ai], dummy_label, true_label, dist))
    # raise ValueError("hhhh")


    for ai in range(attack_iter):
        if ai < 5:
            continue
        if abs(grad_loss_list[ai] - grad_loss_list[ai - 1]) < grad_threshold and \
                abs(grad_loss_list[ai - 1] - grad_loss_list[ai - 2]) < grad_threshold and \
                abs(grad_loss_list[ai - 2] - grad_loss_list[ai - 3]) < grad_threshold:
            break
    dummy_label = attack_record['dummy_label_list'][ai]
    dummy_label = scaler.inverse_transform(dummy_label.reshape(1, 2)).reshape(2)
    true_label = attack_record['gt_label']
    true_label = scaler.inverse_transform(true_label.reshape(1, 2)).reshape(2)
    dist = loc_distance(dummy_label, true_label)
    print("expe_iter: {}, attack_iter: {}, dummy_label: {}, ture_label: {}, dist: {}".format(er, ai, dummy_label, true_label, dist))
    if dist <= dist_threshold:
        attack_success_rounds += 1
        attack_sum_success_iterations += ai


test_dist = 0
for er in range(expe_rounds):
    test_label_list = pickle.load(open('result_process/result/expe_[{}]/test_label_list_er={}.pickle'.format(result_name, er), 'rb'))
    test_pred_list = pickle.load(open('result_process/result/expe_[{}]/test_pred_list_er={}.pickle'.format(result_name, er), 'rb'))
    for ti in range(100):
        test_label = test_label_list[ti]
        test_label = scaler.inverse_transform(test_label.reshape(1, 2)).reshape(2)
        test_pred = test_pred_list[ti]
        test_pred = scaler.inverse_transform(test_pred.reshape(1, 2)).reshape(2)
        dist = loc_distance(test_pred, test_label)
        test_dist += dist

print("success rate: {}".format(attack_success_rounds/expe_rounds))
print("pred average dist: {}".format(test_dist/(expe_rounds*100)))