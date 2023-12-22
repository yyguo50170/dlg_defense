import pickle
import numpy as np
import matplotlib.pyplot as plt

from util.embedding_util import *
from util.util import *

data_dir = 'data/tokyoci.npy'
raw_data_np = np.load(data_dir, allow_pickle=True)
columns_to_normalize = [2, 3]
scaler = MinMaxScaler()
raw_data_np[:, columns_to_normalize] = scaler.fit_transform(raw_data_np[:, columns_to_normalize])

test_label_list = pickle.load(open('result_process/expe_[test]_09-03-16-52-48/test_label_list.pickle', 'rb'))
test_pred_list = pickle.load(open('result_process/expe_[test]_09-03-16-52-48/test_pred_list.pickle', 'rb'))

for i in range(len(test_pred_list)):
    label = np.array(test_label_list[i]).reshape((1, 2))
    pred = np.array(test_pred_list[i]).reshape((1, 2))
    dist = loc_distance(label.reshape(2), pred.reshape(2))
    print("i: {}, dist: {}".format(i, dist))