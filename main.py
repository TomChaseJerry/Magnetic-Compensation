import numpy as np
import pandas as pd
import os
import torch
from data import MagCompData, MagCompDataset
from model import MagneticCompensationLoss, PINN_TLNET
from utils import compute_std_delta_mag, get_bpf, inverse_transform, plot_model_vs_real
from config import config
from torch.utils.data.dataset import random_split
import torch.optim as optim
from torch.utils.data import DataLoader

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

if __name__ == '__main__':
    config = config()
    test_ttlim = config.test_ttlim
    MagCompData = MagCompData(config)
    xyzs = MagCompData.xyzs
    train_inds = MagCompData.train_inds
    test_inds = MagCompData.test_inds

    device = config.device
    batch_size = config.batch_size
    lr = config.lr
    epochs = config.epochs

    selected_features = ['utm_x', 'utm_y', 'utm_z', 'ins_pitch', 'ins_roll', 'ins_yaw', 'cur_ac_hi', 'cur_strb', 'cur_heat',
                         'vol_bat_1', 'vol_block', 'cur_com_1', 'cur_ac_lo', 'cur_tank', 'cur_flap', 'vol_bat_2', 'mag_4_uc',
                         'mag_5_uc', 'flux_d_x', 'flux_d_y', 'flux_d_z']

    lpf = get_bpf(pass1=0.0, pass2=0.2, fs=10.0)
    custom_dataset = MagCompDataset(xyzs=xyzs, train_inds=train_inds, test_inds=test_inds, features=selected_features, lpf=lpf,
                                    mode='train')
    train_size = int(custom_dataset.__len__() * 14 / 17)
    val_size = custom_dataset.__len__() - train_size
    dataset_train, dataset_val = random_split(custom_dataset, [train_size, val_size])
    dataset_test = MagCompDataset(xyzs=xyzs, train_inds=train_inds, test_inds=test_inds, features=selected_features, lpf=lpf,
                                  mode='test')
    print("train num:", train_size)
    print("val num:", val_size)
    print("test num:", dataset_test.__len__())
    dataloader_train = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True, num_workers=0)
    dataloader_val = DataLoader(dataset=dataset_val, batch_size=batch_size, shuffle=False, num_workers=0)
    dataloader_test = DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=False, num_workers=0)

    model_type = 'PINN_TLNET'
    print("Feature selected : ", selected_features)
    if custom_dataset.is_pca:
        print("Using PCA, input dim : ", custom_dataset.x.size(1))
    model = PINN_TLNET(custom_dataset.x.size(1)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_function = MagneticCompensationLoss()
    print("=" * 50)
    print("{} : Start training and validation...\n".format(model_type))
    minerror = np.inf
    for epoch in range(epochs):
        model.train()
        sum_loss = 0.0
        for i, data in enumerate(dataloader_train):
            x, y, B_total = data
            x, y, B_total = x.to(device), y.to(device), B_total.to(device)
            y_hat = model(x)
            loss = loss_function(y_hat, y, B_total)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sum_loss += loss.item()
        print("Epoch{}, train loss:{:.4f}".format(epoch, sum_loss / dataloader_train.__len__()))

        if epoch % 5 == 0:
            model.eval()
            val_y = []
            val_y_hat = []
            with torch.no_grad():
                for data in dataloader_val:
                    x, y, _ = data
                    x, y = x.to(device), y.to(device)
                    y_hat = model(x)
                    val_y.extend(np.array(y.cpu()))
                    val_y_hat.extend(np.array(y_hat.cpu()))
            B_pred = inverse_transform(np.array(val_y_hat), custom_dataset.get_std_output())
            B_real = inverse_transform(np.array(val_y), custom_dataset.get_std_output())
            magerror = compute_std_delta_mag(B_real, B_pred)
            if magerror < minerror:
                minerror = magerror
                torch.save(model, "results/{}.pt".format(model_type))
            print("{}'s MagError in validation:{}".format(model_type, magerror))

    print("=" * 50)
    print("Start testing...\n")
    model.eval()
    val_y = []
    val_y_hat = []
    with torch.no_grad():
        for data in dataloader_test:
            x, y = data
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            val_y.extend(np.array(y.cpu()))
            val_y_hat.extend(np.array(y_hat.cpu()))
    B_pred = inverse_transform(np.array(val_y_hat), custom_dataset.get_std_output())
    B_real = inverse_transform(np.array(val_y), dataset_test.get_std_output())
    plot_model_vs_real(test_ttlim['Flt1003'][0], B_pred, B_real, model_type)
    magerror = compute_std_delta_mag(B_real, B_pred)
    print("{}'s MagError in testing:{}".format(model_type, magerror))
