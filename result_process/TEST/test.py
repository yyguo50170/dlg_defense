import pickle
import numpy as np
import matplotlib.pyplot as plt

from util.embedding_util import *
from util.util import *

data_dir = 'data/raw_data_1706_np.npy'
raw_data_np = np.load(data_dir, allow_pickle=True)
columns_to_normalize = [2, 3]
scaler = MinMaxScaler()
raw_data_np[:, columns_to_normalize] = scaler.fit_transform(raw_data_np[:, columns_to_normalize])

None_dist = []
attack_record = pickle.load(open('result_process/expe_[testdefence]_09-03-11-06-28/attack_record_gloiter=1_dlground=0.pickle ', 'rb'))
grad_loss_list = attack_record['grad_loss_list']
dummy_label_list = attack_record['dummy_label_list']
true_label = attack_record['gt_label']
true_label = scaler.inverse_transform(true_label.reshape(1, 2))
for i in range(len(grad_loss_list)):
    dummy_label = scaler.inverse_transform(dummy_label_list[i].reshape(1, 2))
    dist = loc_distance(dummy_label.reshape(2), true_label.reshape(2))
    None_dist.append(dist)
    print("i: {}, grad_loss: {}, dummy_label: {}, true_label: {}, dist: {}".format(i, grad_loss_list[i], dummy_label, true_label, dist))

print("#" * 80)

DPSGD_dist = []
attack_record = pickle.load(open('result_process/expe_[testdefence]_09-03-15-34-18/attack_record_gloiter=1_dlground=0.pickle', 'rb'))
grad_loss_list = attack_record['grad_loss_list']
dummy_label_list = attack_record['dummy_label_list']
true_label = attack_record['gt_label']
true_label = scaler.inverse_transform(true_label.reshape(1, 2))
for i in range(len(grad_loss_list)):
    dummy_label = scaler.inverse_transform(dummy_label_list[i].reshape(1, 2))
    dist = loc_distance(dummy_label.reshape(2), true_label.reshape(2))
    DPSGD_dist.append(dist)
    print("i: {}, grad_loss: {}, dummy_label: {}, true_label: {}, dist: {}".format(i, grad_loss_list[i], dummy_label, true_label, dist))

print("#" * 80)

Geo_dist = []
attack_record = pickle.load(open('result_process/expe_[testdefence]_09-03-15-22-28/attack_record_gloiter=1_dlground=0.pickle', 'rb'))
grad_loss_list = attack_record['grad_loss_list']
dummy_label_list = attack_record['dummy_label_list']
true_label = attack_record['gt_label']
true_label = scaler.inverse_transform(true_label.reshape(1, 2))
for i in range(len(grad_loss_list)):
    dummy_label = scaler.inverse_transform(dummy_label_list[i].reshape(1, 2))
    dist = loc_distance(dummy_label.reshape(2), true_label.reshape(2))
    Geo_dist.append(dist)
    print("i: {}, grad_loss: {}, dummy_label: {}, true_label: {}, dist: {}".format(i, grad_loss_list[i], dummy_label, true_label, dist))

print("#" * 80)

Geogi_dist = []
attack_record = pickle.load(open('result_process/expe_[testdefence]_09-03-14-44-43/attack_record_gloiter=1_dlground=0.pickle', 'rb'))
grad_loss_list = attack_record['grad_loss_list']
dummy_label_list = attack_record['dummy_label_list']
true_label = attack_record['gt_label']
true_label = scaler.inverse_transform(true_label.reshape(1, 2))
for i in range(len(grad_loss_list)):
    dummy_label = scaler.inverse_transform(dummy_label_list[i].reshape(1, 2))
    dist = loc_distance(dummy_label.reshape(2), true_label.reshape(2))
    Geogi_dist.append(dist)
    print("i: {}, grad_loss: {}, dummy_label: {}, true_label: {}, dist: {}".format(i, grad_loss_list[i], dummy_label, true_label, dist))


plt.plot(range(len(None_dist)), None_dist, label='None')
plt.plot(range(len(DPSGD_dist)), DPSGD_dist, label='DPSGD')
plt.plot(range(len(Geo_dist)), Geo_dist, label='Geo')
plt.plot(range(len(Geogi_dist)), Geogi_dist, label='Geogi')
plt.legend()
plt.show()