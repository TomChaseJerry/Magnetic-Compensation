import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from utils import *
from tqdm import tqdm
import sys
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def scaled_mse_loss(y_pred, y_true, scaling_factor):
    return torch.mean((y_pred - y_true) ** 2) / (scaling_factor * 100)


class MagCompDataset(Dataset):
    def __init__(self, xyzs, inds, features, lpf, mode):
        arrays_x = [[] for n in range(len(features))]
        arrays_y = []
        self.std = StandardScaler()
        for flight, xyz in xyzs.items():
            if mode == 'train':
                sub_diurnal = xyz['diurnal'][~inds[flight]]
                sub_igrf = xyz['igrf'][~inds[flight]]
                arrays_y.extend(xyz['mag_1_c'][~inds[flight]] - sub_diurnal - sub_igrf)
            else:
                sub_diurnal = xyz['diurnal'][inds[flight]]
                sub_igrf = xyz['igrf'][inds[flight]]
                arrays_y.extend(xyz['mag_1_c'][inds[flight]] - sub_diurnal - sub_igrf)

            for i, key in enumerate(features):
                if mode == 'train':
                    value = xyz[key][~inds[flight]]
                else:
                    value = xyz[key][inds[flight]]

                if key in ['mag_2_uc', 'mag_3_uc', 'mag_4_uc', 'mag_5_uc', 'flux_d_x', 'flux_d_y', 'flux_d_z']:
                    arrays_x[i].extend(value - sub_diurnal - sub_igrf)
                elif key in ['cur_com_1', 'cur_strb', 'cur_outpwr', 'cur_ac_lo']:
                    lpf_sig = bpf_data(value, bpf=lpf)
                    arrays_x[i].extend(lpf_sig)
                else:
                    arrays_x[i].extend(value)

        for i in range(len(features)):
            arrays_x[i] = self.std.fit_transform(np.array(arrays_x[i]).reshape(-1, 1))
        arrays_x = np.array(arrays_x).reshape(-1, len(features))
        self.x = torch.tensor(arrays_x, dtype=torch.float32)
        self.y = torch.tensor(np.array(arrays_y), dtype=torch.float32)
        self.length = self.y.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return self.length

    def get_y_max(self):
        return np.max(abs(np.array(self.y)))


class Model1(nn.Module):
    def __init__(self, input_dim):
        super(Model1, self).__init__()
        self.fc1 = nn.Linear(input_dim, 8)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(8, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x