import pickle

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from util.util import *


def sapag_attack(args,
                 batch_size,
                 model,
                 true_dy_dx,
                 dlg_attack_round,
                 dlg_iteration,
                 dlg_lr,
                 global_iter,
                 model_name,
                 gt_data_np,
                 gt_tim_np,
                 gt_label_np,
                 save_path,
                 device,
                 dataset,
                 er,
                 ):
    # 数据记录
    attack_record_list = list()
    F_loss = init_loss(model_name)

    model.train()

    # 进行 dlg_attack_round 次攻击
    for r in range(dlg_attack_round):
        # 初始化每次攻击需要收集的数据条目
        attack_record = {
            'grad_loss_list': [],
            'dummy_data_list': [],
            'dummy_label_list': [],
            'last_dummy_data': [],
            'last_dummy_label': [],
            'global_iteration': global_iter,
            'model_name': model_name,
            'gt_data': gt_data_np,
            'gt_tim': gt_tim_np,
            'gt_label': gt_label_np
        }

        # 随机生成虚假数据
        dummy_data, dummy_label = init_dummy_data(batch_size=batch_size, model_name=model_name, device=device)

        optimizer = torch.optim.LBFGS([dummy_data, dummy_label], lr=dlg_lr)
        # 进行 dlg_iteration 轮训练
        for iters in tqdm(range(dlg_iteration)):
            # 记录虚假数据
            attack_record['dummy_data_list'].append(dummy_data.cpu().detach().numpy())
            attack_record['dummy_label_list'].append(dummy_label.cpu().detach().numpy())

            def closure():
                optimizer.zero_grad()

                # 计算dummy gradients
                if model_name == 'LSTM' or model_name == 'PMF':
                    dummy_pred = model(dummy_data)
                dummy_loss = F_loss(dummy_pred, dummy_label)
                dummy_dy_dx = torch.autograd.grad(dummy_loss, model.parameters(), create_graph=True)

                # 计算dummy gradients与true gradients之间的loss
                grad_loss = 0
                # E loss
                for (d_gy, t_gy) in zip(dummy_dy_dx, true_dy_dx):
                    sigmafang = cal_var(t_gy)
                    mask = (sigmafang == 0) | torch.isnan(sigmafang)
                    sigmafang[mask] = 1
                    gaussian_distance = gaussian_kernel(d_gy, t_gy, sigmafang)
                    #print(gaussian_distance)
                    # 计算公式中的值
                    # grad_loss += Q * (torch.ones_like(gaussian_distance) - gaussian_distance)
                    grad_loss += (torch.ones_like(gaussian_distance) - gaussian_distance).sum()
                    #print(grad_loss)

                grad_loss.backward()
                return grad_loss

            grad_loss = closure()
            if iters % 10 == 0:
                print("dlg iters:{} , loss:{}".format(iters, grad_loss.item()))
            attack_record['grad_loss_list'].append(grad_loss.item())
            # 记录最后的数据
            attack_record['last_dummy_data'] = dummy_data
            attack_record['last_dummy_label'] = dummy_label
            # 对dummy_data和dummy_label进行反向传播
            optimizer.step(closure)

        attack_record_list.append(attack_record)
        pickle.dump(attack_record,
                    open(save_path + '/attack_record_er={}_gloiter={}_dlground={}.pickle'.format(er, global_iter, r),
                         'wb'))

    return attack_record_list


def sapag_attack_onehot(args,
                        step_size,
                        model,
                        true_dy_dx,
                        idlg_attack_round,
                        idlg_iteration,
                        idlg_lr,
                        global_iter,
                        model_name,
                        gt_data_np,
                        gt_tim_np,
                        gt_label_np,
                        save_path,
                        device,
                        dataset,
                        er,
                        gt_label_code,
                        true_time,
                        ):
    # 数据记录
    attack_record_list = list()

    F_loss = init_loss(model_name)

    model.train()

    # 进行 dlg_attack_round 次攻击
    for r in range(idlg_attack_round):
        # 初始化每次攻击需要收集的数据条目
        attack_record = {
            'grad_loss_list': [],
            'dummy_data_list': [],
            'dummy_label_list': [],
            'last_dummy_data': [],
            'last_dummy_label': [],
            'global_iteration': global_iter,
            'model_name': model_name,
            'gt_data': gt_data_np,
            'gt_tim': gt_tim_np,
            'gt_label': gt_label_np,
        }

        # 随机生成虚假数据
        # dummy_data, dummy_label = init_dummy_data(batch_size=batch_size, model_name=model_name, device=device)
        dummy_data_code_np = init_dummy_data_embedding(true_time)
        dummy_data_code = torch.from_numpy(dummy_data_code_np.reshape((1, step_size, 977)).astype(np.float32)).to(
            device).requires_grad_(True)

        dummy_label_code_np = init_dummy_label_embedding()
        dummy_label_code = torch.from_numpy(dummy_label_code_np.reshape((1, 788)).astype(np.float32)).to(
            device).requires_grad_(True)

        optimizer = torch.optim.LBFGS([dummy_data_code, dummy_label_code], lr=idlg_lr)

        # 进行 dlg_iteration 轮训练
        for iters in tqdm(range(idlg_iteration)):
            # 记录虚假数据
            attack_record['dummy_data_list'].append(inv_embedding_location(dummy_data_code.cpu().detach().numpy()[0]))

            attack_record['dummy_label_list'].append(
                inv_embedding_location_single(dummy_label_code.cpu().detach().numpy()[0]))

            def closure():
                optimizer.zero_grad()

                # 计算dummy gradients
                if model_name == 'LSTM' or model_name == 'PMF':
                    dummy_pred = model(dummy_data_code)
                dummy_loss = F_loss(dummy_pred, gt_label_code)
                dummy_dy_dx = torch.autograd.grad(dummy_loss, model.parameters(), create_graph=True)

                # 计算dummy gradients与true gradients之间的loss
                grad_loss = 0
                # E loss
                for (d_gy, t_gy) in zip(dummy_dy_dx, true_dy_dx):
                    # 计算高斯核距离
                    sigmafang = cal_var(t_gy)
                    mask = (sigmafang == 0) | torch.isnan(sigmafang)
                    sigmafang[mask] = 1
                    gaussian_distance = gaussian_kernel(d_gy, t_gy, sigmafang)

                    # 计算公式中的值
                    # grad_loss += Q * (torch.ones_like(gaussian_distance) - gaussian_distance)
                    grad_loss += (torch.ones_like(gaussian_distance) - gaussian_distance).sum()

                grad_loss.backward()
                return grad_loss

            grad_loss = closure()
            if iters % 10 == 0:
                print("idlg iters:{} , loss:{}".format(iters, grad_loss.item()))
            attack_record['grad_loss_list'].append(grad_loss.item())
            # 记录最后的数据
            attack_record['last_dummy_data'] = inv_embedding_location(dummy_data_code.cpu().detach().numpy()[0])
            attack_record['last_dummy_label'] = inv_embedding_location_single(
                dummy_label_code.cpu().detach().numpy()[0])
            # 对dummy_data和dummy_label进行反向传播
            optimizer.step(closure)

        print(attack_record['last_dummy_data'])
        attack_record_list.append(attack_record)
        pickle.dump(attack_record,
                    open(save_path + '/attack_record_er={}_gloiter={}_sapaground={}_stepsize={}.pickle'.format(er,
                                                                                                               global_iter,
                                                                                                               r,
                                                                                                               step_size),
                         'wb'))

    return attack_record_list


def cal_var(dy_dx):
    variance = torch.var(dy_dx, dim=0)
    return variance


def gaussian_kernel(x, y, sigmafang):
    return torch.exp(-torch.norm(x - y) ** 2 / (sigmafang))
