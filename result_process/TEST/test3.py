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

G1_dist = []
attack_record = pickle.load(open('result_process/expe_[testg1]_09-03-16-09-03/attack_record_gloiter=1_dlground=0.pickle', 'rb'))
grad_loss_list = attack_record['grad_loss_list']
dummy_label_list = attack_record['dummy_label_list']
true_label = attack_record['gt_label']
true_label = scaler.inverse_transform(true_label.reshape(1, 2))
for i in range(len(grad_loss_list)):
    dummy_label = scaler.inverse_transform(dummy_label_list[i].reshape(1, 2))
    dist = loc_distance(dummy_label.reshape(2), true_label.reshape(2))
    G1_dist.append(dist)

G4_dist = []
attack_record = pickle.load(open('result_process/expe_[testg4]_09-03-16-10-09/attack_record_gloiter=1_dlground=0.pickle', 'rb'))
grad_loss_list = attack_record['grad_loss_list']
dummy_label_list = attack_record['dummy_label_list']
true_label = attack_record['gt_label']
true_label = scaler.inverse_transform(true_label.reshape(1, 2))
for i in range(len(grad_loss_list)):
    dummy_label = scaler.inverse_transform(dummy_label_list[i].reshape(1, 2))
    dist = loc_distance(dummy_label.reshape(2), true_label.reshape(2))
    G4_dist.append(dist)

G8_dist = []
attack_record = pickle.load(open('result_process/expe_[testg8]_09-03-16-11-46/attack_record_gloiter=1_dlground=0.pickle', 'rb'))
grad_loss_list = attack_record['grad_loss_list']
dummy_label_list = attack_record['dummy_label_list']
true_label = attack_record['gt_label']
true_label = scaler.inverse_transform(true_label.reshape(1, 2))
for i in range(len(grad_loss_list)):
    dummy_label = scaler.inverse_transform(dummy_label_list[i].reshape(1, 2))
    dist = loc_distance(dummy_label.reshape(2), true_label.reshape(2))
    G8_dist.append(dist)

plt.plot(range(len(None_dist)), None_dist, label='N')
plt.plot(range(len(G1_dist)), G1_dist, label='1')
plt.plot(range(len(G4_dist)), G4_dist, label='4')
plt.plot(range(len(G8_dist)), G8_dist, label='8')
plt.legend()
plt.show()