import pickle

from util.util import *

expe_rounds = 1
attack_success_iter = 0
attack_success_rounds = 0
attack_sum_success_iterations = 0
sum_dist = 0
attack_success_count = 0
all_count = 0
thread_hold = 0.003
enhance = True


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


scaler = get_scaler("nycb")

if __name__ == "__main__":
    print(f"thread_hold:{thread_hold},is_enhance:{enhance}")
    dis_list = []
    for er in range(5):
        for base in range(23):
            point_id = base + 7
            data_list = []
            true = None
            for it in range(base, base + 8):
                path = f'../result/expe_[dlg_lstm_batch8]_12-08-08-18-55/attack_record_er={er}_gloiter={it}_dlground=0.pickle'
                if enhance:
                    path = f'../result/expe_[dlg_lstm_batch8_enhance]_12-07-21-31-17/attack_record_er={er}_gloiter={it}_dlground=0.pickle'
                attack_record = pickle.load(open(path, 'rb'))
                grad_loss_list = attack_record['grad_loss_list']
                attack_iter = len(grad_loss_list)
                ai = attack_iter - 1
                dummy_data = attack_record['dummy_data_list'][ai][0]  # for lstm
                dummy_data = scaler.inverse_transform(dummy_data)
                data_list.append(dummy_data[point_id - it])
                true_data = attack_record['gt_data']  # for lstm
                true_data = scaler.inverse_transform(true_data)
                true = true_data[point_id - it]
            all_count += 1
            min_dist = 10000
            min_dist_id = base
            for i in range(8):
                now_dist = loc_distance(data_list[i], true)
                if now_dist < 0.03:
                    dis_list.append(now_dist)
                if now_dist < min_dist:
                    min_dist = now_dist
                    min_dist_id = base + i
            if min_dist <= thread_hold:
                attack_success_count += 1
                print(f"er:{er},point:{point_id}")
                it = min_dist_id
                path = f'../result/expe_[dlg_lstm_batch8]_12-08-08-18-55/attack_record_er={er}_gloiter={it}_dlground=0.pickle'
                if enhance:
                    path = f'../result/expe_[dlg_lstm_batch8_enhance]_12-07-21-31-17/attack_record_er={er}_gloiter={it}_dlground=0.pickle'
                attack_record = pickle.load(open(path, 'rb'))
                for ai in range(200):
                    dummy_data = attack_record['dummy_data_list'][ai][0]  # for lstm
                    dummy_data = scaler.inverse_transform(dummy_data)
                    true_data = attack_record['gt_data']  # for lstm
                    true_data = scaler.inverse_transform(true_data)
                    true = true_data[point_id - it]
                    now_dist = loc_distance(dummy_data[point_id - it], true)
                    if now_dist <= thread_hold:
                        attack_success_iter += ai
                        break
    print(dis_list)
    print(attack_success_count)
    print(all_count)
    print("攻击成功率:", attack_success_count / all_count)
    print("攻击成功所需轮次", attack_success_iter / attack_success_count)
    # plt.figure()
    # plt.hist(dis_list, bins=10, color='blue', edgecolor='black')
    # plt.title('Histogram of Data')
    # plt.xlabel('Value')
    # plt.ylabel('Frequency')
    # plt.grid(True)
    # plt.show()

    # x = [point[0] for point in data_list]
    # y = [point[1] for point in data_list]
    # # 创建散点图
    # plt.figure(base)
    # plt.scatter(x, y, label='dummy_data', marker='o')
    # plt.scatter(true[0], true[1], label='true_data', color='red', marker='x')
    #
    # # 添加图例
    # plt.legend()
    #
    # # 设置坐标轴标签
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # title = f"point_{point_id}"
    # if enhance:
    #     title += "_enhance"
    # plt.title(title)
    # save_path = f"../pic/dlg/er{er}/"
    # if enhance:
    #     save_path = f"../pic/enhance/er{er}/"
    # if not os.path.exists(save_path):
    #     os.mkdir(save_path)
    # save_path += f"{title}.png"
    # plt.savefig(save_path)
    # plt.close(base)
