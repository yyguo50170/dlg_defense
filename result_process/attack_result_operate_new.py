import pickle
from util.util import *


def get_scaler(model_name):
    if model_name == 'nycb':
        data_dir = '../data/raw_data_1706_np.npy'
        raw_data_np = np.load(data_dir, allow_pickle=True)
        columns_to_normalize = [2, 3]
        scaler = MinMaxScaler()
        raw_data_np[:, columns_to_normalize] = scaler.fit_transform(raw_data_np[:, columns_to_normalize])
    if model_name == 'tokyoci':
        data_dir = '../data/tokyoci.npy'
        raw_data_np = np.load(data_dir, allow_pickle=True)
        columns_to_normalize = [2, 3]
        scaler = MinMaxScaler()
        raw_data_np[:, columns_to_normalize] = scaler.fit_transform(raw_data_np[:, columns_to_normalize])
    if model_name == 'gowallaci':
        data_dir = '../data/gowallaci.npy'
        raw_data_np = np.load(data_dir, allow_pickle=True)
        columns_to_normalize = [2, 3]
        scaler = MinMaxScaler()
        raw_data_np[:, columns_to_normalize] = scaler.fit_transform(raw_data_np[:, columns_to_normalize])
    return scaler


model_name = 'tokyoci'
result_name = 'sapag'

scaler = get_scaler(model_name)

expe_rounds = 4
global_iter = 1
grad_threshold = 0.005

# grad_threshold = 1e-15
dist_threshold = 2.5
step_size = 8

attack_methods = ['dlg', 'idlg', 'cpl', 'sapag', 'invgrad', 'stgia']
print(f'dist_threshold:{dist_threshold}')
for attack_method in attack_methods:
    exp_name = f"expe_[{attack_method}_lstm_batch8_tokyoci]"
    attack_success_rounds = 0
    attack_sum_success_iterations = 0
    sum_dist = 0
    attack_success_count = 0
    all_count = 0
    for it in [10, 20, 30]: #, 40, 50]:
        it = it - 1
        attack_success_count = 0
        all_count = 0
        for er in range(0, expe_rounds):
            name = 'dlg'
            attack_record = pickle.load(
                open(f'../result/2/{exp_name}/attack_record_er={er}_gloiter={it}_{name}round=0.pickle', 'rb'))

            grad_loss_list = attack_record['grad_loss_list']
            attack_iter = len(grad_loss_list)
            ai = attack_iter - 1
            dummy_data = attack_record['dummy_data_list'][ai][0]  # for lstm
            dummy_data = scaler.inverse_transform(dummy_data)
            true_data = attack_record['gt_data']  # for lstm
            true_data = scaler.inverse_transform(true_data)
            # dummy_label = attack_record['dummy_label_list'][ai][0]  # for lstm
            # true_label = attack_record['gt_label']
            data_dist = 0
            for i in range(step_size):
                now_dist = loc_distance(dummy_data[i], true_data[i])
                all_count += 1
                if now_dist <= dist_threshold:
                    # data_dist += now_dist
                    attack_success_count += 1
        # print("er:{},dist:{}".format(er, data_dist))
        # print("er:{},data:{}".format(er, dummy_data))
        # sum_dist += data_dist
        # print(attack_success_count)
        print(f"iter:{it},attack method:{attack_method},success rate:{attack_success_count / all_count}")
    # print("average dist:", sum_dist / attack_success_count)
    # print(f"attack method:{attack_method},success rate:{attack_success_count / all_count}")
    # print(attack_success_count)
    # print(all_count)

    # for er in range(0, expe_rounds):
    #     iters = 50
    #     if attack_method == 'dlg':
    #         iters = 30;
    #     for it in range(0, iters):
    #         # for it in [1, 5, 10, 15, 20, 25, 30]:
    #         # it = it - 1
    #         name = 'dlg'
    #         if attack_method in ['cpl', 'idlg']:
    #             name = attack_method
    #         name = 'dlg'
    #         attack_record = pickle.load(
    #             open(f'../result/2/{exp_name}/attack_record_er={er}_gloiter={it}_{name}round=0.pickle', 'rb'))
    #
    #         grad_loss_list = attack_record['grad_loss_list']
    #         attack_iter = len(grad_loss_list)
    #         ai = attack_iter - 1
    #         dummy_data = attack_record['dummy_data_list'][ai][0]  # for lstm
    #         dummy_data = scaler.inverse_transform(dummy_data)
    #         true_data = attack_record['gt_data']  # for lstm
    #         true_data = scaler.inverse_transform(true_data)
    #         # dummy_label = attack_record['dummy_label_list'][ai][0]  # for lstm
    #         # true_label = attack_record['gt_label']
    #         data_dist = 0
    #         for i in range(step_size):
    #             now_dist = loc_distance(dummy_data[i], true_data[i])
    #             all_count += 1
    #             if now_dist <= dist_threshold:
    #                 # data_dist += now_dist
    #                 attack_success_count += 1
    #     # print("er:{},dist:{}".format(er, data_dist))
    #     # print("er:{},data:{}".format(er, dummy_data))
    #     # sum_dist += data_dist
    # # print("average dist:", sum_dist / attack_success_count)
    # print(f"attack method:{attack_method},success rate:{attack_success_count / all_count}")
    # # print(attack_success_count)
    # # print(all_count)
