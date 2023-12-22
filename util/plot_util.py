import matplotlib.pyplot as plt
import time
import os
import numpy as np
import random
import pickle
import torch
import torch.nn as nn

def attack_record_plot(args,
                       start_time,
                       global_iter,
                       attack_record,
                       best_index,
                       best_dist,
                       gt_data,
                       gt_label,
                       device,
                       save_path,
                       ):

    plt.figure(figsize=(17, 9))

    AR_path = save_path
    if not os.path.exists(AR_path):
        os.mkdir(AR_path)

    # 基本信息
    plt.subplot(2, 3, 1)
    # 计算攻击成功次数
    scount = 0
    for i in range(len(attack_record)):
        if attack_record[i]['if_attack_success']:
            scount += 1
    # 计算距离
    true_data = torch.from_numpy(gt_data.astype(np.float32)).to(device)
    dummy_data = attack_record[best_index]['last_dummy_data'].reshape((args.batch_size, 2))
    edist = nn.PairwiseDistance(p=2)
    data_dist = torch.mean(edist(true_data, dummy_data)).item()
    # true_label = torch.from_numpy(gt_label.astype(np.float32)).to(device)
    # dummy_label = attack_record[best_index]['last_dummy_label'].reshape((1,2))
    # label_dist = torch.mean(edist(true_label, dummy_label)).item()
    # plot
    plt.text(x=0.1,  # 文本x轴坐标
             y=9.9,  # 文本y轴坐标
             s="============Attack Record=============" + '\n' +
               "In the global iteration: {}".format(global_iter) + '\n' +
               "The dlg attack rounds: {}".format(args.dlg_attack_rounds) + '\n' +
               "The dlg training iterations: {}".format(args.dlg_iterations) + '\n' +
               "The dlg learning rate: {}".format(args.dlg_lr) + '\n' +
               "The dlg threshold :{}".format(args.dlg_threshold) + '\n' +
               "user number: {}".format(args.usernum) + '\n' +
               "batch size: {}".format(args.batch_size) + '\n' +
               "model name: {}".format(args.model_name) + '\n' +
               "learning rate for OT: {}".format(args.lr) + '\n' +
               "global iterations for OT: {}".format(args.global_iterations) + '\n' +
               "--------------------------------------" + '\n' +
               "Success Attack Round: {}".format(scount) + '\n' +
               "Success Attack Rate: {}".format(scount / len(attack_record)) + '\n' +
               "--------------------------------------" + '\n' +
               "The best round: {}".format(best_index) + '\n' +
               "Is attack successful: {}".format(attack_record[best_index]['if_attack_success']) + '\n' +
               "Its training iteration: {}".format(attack_record[best_index]['training_iteration']) + '\n' +
               "The gradient loss: {}".format(attack_record[best_index]['grad_loss_list'][-1]) + '\n' +
               "The average distance btw d_data and t_data: {}".format(data_dist) + '\n' +
               "The distance btw d_label and t_label: {}".format(best_dist)
             ,  # 文本内容
             rotation=0,  # 文字旋转
             ha='left',  # x=2.2是文字的左端位置，可选'center', 'right', 'left'
             va='top',  # y=8是文字的低端位置，可选'center', 'top', 'bottom', 'baseline', 'center_baseline'
             fontdict=dict(fontsize=7, color='black',
                           family='monospace',  # 字体,可选'serif', 'sans-serif', 'cursive', 'fantasy', 'monospace'
                           weight='normal',  # 磅值，可选'light', 'normal', 'medium', 'semibold', 'bold', 'heavy', 'black'
                           ))
    plt.xlim(0, 10)
    plt.ylim(0, 10)
    plt.axis('off')

    # 所有轮的轨迹
    plt.subplot(2, 3, 2)
    plt.plot(gt_data[:, 0], gt_data[:, 1], color='red', linewidth=1)
    for i in range(len(attack_record)):
        x = [a[0] for a in attack_record[i]['dummy_label_list']]
        y = [a[1] for a in attack_record[i]['dummy_label_list']]
        plt.scatter(x, y, marker='.', c=range(len(y)), cmap='plasma')
    plt.scatter(gt_data[0, 0], gt_data[0, 1], color='red', marker='x')
    plt.scatter(gt_data[-1, 0], gt_data[-1, 1], color='red', marker='o')
    plt.scatter(gt_label[0], gt_label[1], marker='^', color='green')

    # 最优的轨迹
    plt.subplot(2, 3, 3)
    plt.plot(gt_data[:, 0], gt_data[:, 1], color='red', linewidth=1)
    x = [a[0] for a in attack_record[best_index]['dummy_label_list']]
    y = [a[1] for a in attack_record[best_index]['dummy_label_list']]
    plt.scatter(x, y, marker='.', c=range(len(y)), cmap='plasma')
    plt.scatter(gt_data[0, 0], gt_data[0, 1], color='red', marker='x')
    plt.scatter(gt_data[-1, 0], gt_data[-1, 1], color='red', marker='o')
    plt.scatter(gt_label[0], gt_label[1], marker='^', color='green')

    # 最优的loss
    plt.subplot(2, 3, 4)
    x = np.array(range(len(attack_record[best_index]['grad_loss_list'])), dtype='i4')
    y = attack_record[best_index]['grad_loss_list']
    plt.plot(x, y, label='loss of {}'.format(best_index))
    plt.xlim(0, len(attack_record[best_index]['grad_loss_list'])-1)
    # plt.ylim(0, 1)
    plt.title('gradient loss of the best DLG attack')
    plt.xlabel('training iteration')
    plt.ylabel('gradient loss')
    plt.legend()

    # 最优的data轨迹
    plt.subplot(2, 3, 5)
    x = gt_data[:, 0]
    y = gt_data[:, 1]
    plt.plot(x, y, color='red', linewidth=1)

    for i in range(args.batch_size):
        r = random.random()
        g = random.random()
        b = random.random()
        x = [a[2 * i] for a in attack_record[best_index]['dummy_data_list']]
        y = [a[2 * i + 1] for a in attack_record[best_index]['dummy_data_list']]
        plt.scatter(x, y, marker='.', color=(r, g, b))
        plt.scatter(x[0], y[0], marker='x', color=(r, g, b))
        plt.scatter(gt_data[i, 0], gt_data[i, 1], color=(r, g, b), marker='*')

    x = [a[0] for a in attack_record[best_index]['dummy_label_list']]
    y = [a[1] for a in attack_record[best_index]['dummy_label_list']]
    plt.scatter(x, y, marker='.', c=range(len(y)), cmap='plasma')
    plt.scatter(gt_label[0], gt_label[1], marker='^', color='green')

    # dlg_attack_iteration, dist of data/label
    ax = plt.subplot(2, 3, 6)
    # 获取数据
    seq = list(range(args.dlg_attack_rounds))
    train_iters = []
    success = []
    data_dists = []
    label_dists = []
    true_data = torch.from_numpy(gt_data.astype(np.float32)).to(device)
    true_label = torch.from_numpy(gt_label.astype(np.float32)).to(device)
    for r in range(args.dlg_attack_rounds):
        train_iters.append(attack_record[r]['training_iteration'])
        success.append(attack_record[r]['if_attack_success'])
        data_dists.append(torch.mean(edist(true_data, attack_record[r]['last_dummy_data'].reshape((args.batch_size, 2)))).item())
        label_dists.append(torch.mean(edist(true_label, attack_record[r]['last_dummy_label'].reshape((1, 2)))).item())
    # 双轴柱状图
    # ax
    for i in range(args.dlg_attack_rounds):
        if success[i] == True:
            pl = ax.bar(seq[i]-0.3, train_iters[i], color='green', width=0.3)
        else:
            pl = ax.bar(seq[i]-0.3, train_iters[i], color='red', width=0.3)
        ax.bar_label(pl, label_type='center')
    ax.set_xticks(seq)
    ax.set_xlabel('round')
    ax.set_ylabel('attack iterations')
    # ax1
    ax1 = ax.twinx()
    pl = ax1.bar(seq, data_dists, color='orange', width=0.3)
    ax1.bar_label(pl, label_type='edge')
    pl = ax1.bar([s+0.3 for s in seq], label_dists, color='yellow', width=0.3)
    ax1.bar_label(pl, label_type='edge')
    ax1.set_ylabel('E_dist')
    ax1.set_ylim(0, 0.15)

    plt.savefig(AR_path + '/attack_record_gloiter={}.png'.format(global_iter), dpi=750, bbox_inches='tight')





def track_plot(args,
               start_time,
               true_loss_list,
               best_dist_list,
               true_track,
               dummy_track,
               save_path,
               ):

    plt.figure(figsize=(23, 5))

    plt.subplot(1, 4, 1)
    plt.text(x=0.1,  # 文本x轴坐标
             y=9.9,  # 文本y轴坐标
             s="============ Setting information =============" + '\n' +
               "user number: {}".format(args.usernum) + '\n' +
               "batch size: {}".format(args.batch_size) + '\n' +
               "model name: {}".format(args.model_name) + '\n' +
               "learning rate for OT: {}".format(args.lr) + '\n' +
               "global iterations for OT: {}".format(args.global_iterations) + '\n' +
               "The dlg attack rounds: {}".format(args.dlg_attack_rounds) + '\n' +
               "The dlg training iterations: {}".format(args.dlg_iterations) + '\n' +
               "The dlg learning rate: {}".format(args.dlg_lr) + '\n' +
               "The dlg threshold :{}".format(args.dlg_threshold)
             ,  # 文本内容
             rotation=0,  # 文字旋转
             ha='left',  # x=2.2是文字的左端位置，可选'center', 'right', 'left'
             va='top',  # y=8是文字的低端位置，可选'center', 'top', 'bottom', 'baseline', 'center_baseline'
             fontdict=dict(fontsize=7, color='black',
                           family='monospace',  # 字体,可选'serif', 'sans-serif', 'cursive', 'fantasy', 'monospace'
                           weight='normal',  # 磅值，可选'light', 'normal', 'medium', 'semibold', 'bold', 'heavy', 'black'
                           ))
    plt.xlim(0, 10)
    plt.ylim(0, 10)
    plt.axis('off')


    # true loss
    plt.subplot(1, 4, 2)
    x = np.array(range(len(true_loss_list)), dtype='i4')
    y = true_loss_list
    plt.plot(x, y, label='true loss')
    plt.xlim([0, len(true_loss_list)-1])
    plt.ylim([0, 1])
    plt.title('the loss of original task')
    plt.xlabel('global iteration')
    plt.ylabel('loss')
    plt.legend()


    # distance
    plt.subplot(1, 4, 3)
    x = np.array(range(len(best_dist_list)), dtype='i4')
    y = best_dist_list
    plt.plot(x, y, label='straight-line distance')
    plt.xlim([0, len(best_dist_list) - 1])
    plt.ylim([0, 5])
    plt.title('the distance between gt_label and dummy_label')
    plt.xlabel('global iteration')
    plt.ylabel('distance')
    plt.legend()


    # track
    plt.subplot(1, 4, 4)

    x1 = [t[0] for t in true_track]
    y1 = [t[1] for t in true_track]
    plt.plot(x1, y1, color='green', label='true track')
    plt.scatter(true_track[0][0], true_track[0][1], color='green', marker='x')
    plt.scatter(true_track[-1][0], true_track[-1][1], color='green', marker='o')

    x2 = [d[0] for d in dummy_track]
    y2 = [d[1] for d in dummy_track]
    plt.plot(x2, y2, color='red', label='dummy track')
    plt.scatter(dummy_track[0][0], dummy_track[0][1], color='red', marker='x')
    plt.scatter(dummy_track[-1][0], dummy_track[-1][1], color='red', marker='o')

    plt.title('true track and dummy track')
    plt.legend()

    S_path = save_path
    if not os.path.exists(S_path):
        os.mkdir(S_path)
    plt.savefig(S_path + '/Result {}.png'.format(time.strftime('%m-%d-%H-%M-%S', time.localtime(start_time))), dpi=750, bbox_inches='tight')