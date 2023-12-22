import pickle
import numpy as np
from util.util import *


def get_scaler(model_name):
    if model_name == 'nycb':
        data_dir = '../data/raw_data_1706_np.npy'
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


model_name = 'nycb'
result_name = 'sapag'

scaler = get_scaler(model_name)

expe_rounds = 10
global_iter = 1
attack_iter = 400
grad_threshold = 0.005

#grad_threshold = 1e-15
dist_threshold = 1
step_size = 1

attack_success_rounds = 0
attack_sum_success_iterations = 0

for er in range(0, expe_rounds):
    attack_record = pickle.load(
        # open('../result/expe_[{}]_10-24-19-55-09/attack_record_er={}_gloiter=1_{}round=0_stepsize=1.pickle'.format(result_name,er,
        #                                                                                             result_name
        #                                                                                             ),
        #      'rb'))
        # open('../result/expe_[{}]_10-25-15-07-58/attack_record_er={}_gloiter=1_{}round=0_stepsize=1.pickle'.format(
        #     result_name, er,
        #     result_name
        #     ),
        #      'rb'))
        # open('../result/expe_[dlg]/attack_record_er=0_gloiter=1_dlground=0_stepsize=1.pickle','rb'))
        #open('../result/expe_[dlg_lstm]_10-30-14-20-44/attack_record_er={}_gloiter=1_dlground=0.pickle'.format(er),'rb'))
        #open('../result/expe_[sapag_lstm]_10-30-16-06-17/attack_record_er={}_gloiter=1_dlground=0.pickle'.format(er),'rb'))
        #open('../result/expe_[invgrad]_10-24-19-55-09/attack_record_er={}_gloiter=1_invgradround=0_stepsize=1.pickle'.format(er),'rb'))
        #open('../result/expe_[sapag]_10-25-15-07-58/attack_record_er={}_gloiter=1_sapaground=0_stepsize=1.pickle'.format(er),'rb'))

        # open(
        #     '../result/expe_[cpl_pmf]_11-13-14-52-06/attack_record_er={}_gloiter=1_cplround=0_stepsize=1.pickle'.format(
        #         er), 'rb'))
        # open(
        #     '../result/expe_[idlg_pmf]_11-13-15-46-30/attack_record_er={}_gloiter=1_idlground=0_stepsize=1.pickle'.format(
        #         er), 'rb'))
        #open('../result/expe_[cpl_lstm]_11-13-17-10-57/attack_record_er={}_gloiter=1_cplround=0.pickle'.format(er),'rb'))

        # open(
        #     '../result/expe_[idlg_lstm]_11-13-17-32-25/attack_record_er={}_gloiter=1_idlground=0.pickle'.format(
        #         er), 'rb'))
    #open('../result/expe_[invgrad_lstm]_10-30-14-47-10/attack_record_er={}_gloiter=1_dlground=0.pickle'.format(er),'rb'))
        # open(
        #     '../result/expe_[dlg_pmf]_10-30-15-05-11/attack_record_er={}_gloiter=1_dlground=0_stepsize=1.pickle'.format(
        #         er),
        #     'rb'))

        #open('../result/expe_[dlg_lstm]_10-30-14-20-44/attack_record_er={}_gloiter=1_dlground=0.pickle'.format(er),'rb'))
        open('../result/expe_[dlg_lstm_l12]_11-27-12-41-05/attack_record_er={}_gloiter=1_dlground=0.pickle'.format(er), 'rb'))

    grad_loss_list = attack_record['grad_loss_list']
    #print(grad_loss_list)
    attack_iter = len(grad_loss_list)
    for ai in range(attack_iter):
        if ai < 5:
            continue
        if abs(grad_loss_list[ai] - grad_loss_list[ai - 1]) < grad_threshold and \
                abs(grad_loss_list[ai - 1] - grad_loss_list[ai - 2]) < grad_threshold and \
                abs(grad_loss_list[ai - 2] - grad_loss_list[ai - 3]) < grad_threshold:
            break
        # if grad_loss_list[ai] < 1:
        #     break

    #ai = len(grad_loss_list) - 1
    ai = 1000
    dummy_data = attack_record['dummy_data_list'][ai]  # for pmf
    true_data = attack_record['gt_data'][:, [2, 3]]    # for pmf
    #dummy_label = attack_record['dummy_label_list'][ai]  # for pmf

    #dummy_data = attack_record['dummy_data_list'][ai][0]  # for lstm
    #true_data = attack_record['gt_data']   # for lstm
    # dummy_label = attack_record['dummy_label_list'][ai][0]  # for lstm

    true_label = attack_record['gt_label']

    data_dist = 0
    for i in range(step_size):
        # print(loc_distance(dummy_data[i], true_data[i]))
        data_dist += loc_distance(dummy_data[i], true_data[i])
    print("expe_iter: {}, attack_iter: {}, dummy_data: {}, ture_data: {}, dist: {}".format(er, ai, dummy_data,
                                                                                           true_data, data_dist))
    # label_dist = loc_distance(dummy_label, true_label)
    # print("dummy_label: {}, ture_label: {}, dist: {}".format(dummy_label, true_label, label_dist))
    # if data_dist <= dist_threshold:
    #     attack_success_rounds += 1
    #     attack_sum_success_iterations += ai

# test_dist = 0
# for er in range(expe_rounds):
#     test_label_list = pickle.load(open('../result/expe_[{}]/test_label_list_er={}.pickle'.format(result_name, er), 'rb'))
#     test_pred_list = pickle.load(open('../result/expe_[{}]/test_pred_list_er={}.pickle'.format(result_name, er), 'rb'))
#     for ti in range(100):
#         test_label = test_label_list[ti]
#         test_label = scaler.inverse_transform(test_label.reshape(1, 2)).reshape(2)
#         test_pred = test_pred_list[ti]
#         test_pred = scaler.inverse_transform(test_pred.reshape(1, 2)).reshape(2)
#         dist = loc_distance(test_pred, test_label)
#         test_dist += dist

print("success rate: {}".format(attack_success_rounds / expe_rounds))
# print("pred average dist: {}".format(test_dist/(expe_rounds*100)))
