import pickle

from util.util import *

model_name = 'nycb'
result_name = 'sapag'

expe_rounds = 1

attack_success_iter = 0
attack_success_rounds = 0
attack_sum_success_iterations = 0
sum_dist = 0
attack_success_count = 0
all_count = 0
# 定义不同的 step_size 值
batch_size_list = [1, 4, 8]
global_epoch = [0, 10, 50]

for epoch in global_epoch:
    for batch_size in batch_size_list:
        print(f'batch_size:{batch_size},global_epoch:{epoch}')
        # experiment_name = "../result/expe_[dlg_lstm_batch{}_globalepoch{}_enhance]/".format(batch_size, epoch)
        experiment_name = "../result/expe_[dlg_lstm_batch{}_globalepoch{}_enhance]/".format(batch_size, epoch)

        attack_success_count = 0
        attack_success_iter = 0
        sum_dist = 0

        for er in range(0, expe_rounds):
            attack_success_flag = [False for x in range(batch_size)]

            path = experiment_name + 'attack_record_er={}_gloiter=10_dlground=0.pickle'.format(er)
            attack_record = pickle.load(open(path, 'rb'))
            grad_loss_list = attack_record['grad_loss_list']
            attack_iter = len(grad_loss_list)
            # for ai in range(attack_iter):
            for ai in [attack_iter - 1]:
                # 对每一个ai的数据
                dummy_data = attack_record['dummy_data_list'][ai][0]  # for lstm
                true_data = attack_record['gt_data']  # for lstm
                # dummy_label = attack_record['dummy_label_list'][ai][0]  # for lstm
                # true_label = attack_record['gt_label']
                data_dist = 0
                for i in range(batch_size):
                    if dummy_data[i][0] > 1 or dummy_data[i][0] < 0 or dummy_data[i][1] > 1 or dummy_data[i][1] < 0:
                        print("数据溢出")
                    if ai == 0 or attack_success_flag[i] is False:
                        now_dist = loc_distance(dummy_data[i], true_data[i])
                        if now_dist <= 0.01:
                            # print(f"expe:{er}ai:{ai},i:{i},now_dist:{now_dist},dummy_data[i]:{dummy_data}, true_data[i]:{true_data}")
                            data_dist += now_dist
                            attack_success_count += 1
                            attack_success_iter += ai
                            attack_success_flag[i] = True
            # sum_dist += data_dist
        # print("average dist:", sum_dist / attack_success_count)
        print("average attack success iter:", attack_success_iter / attack_success_count)
        print("success rate:", attack_success_count / (expe_rounds * batch_size))
