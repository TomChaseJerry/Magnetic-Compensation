import argparse

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data.dataset import random_split
from torch.utils.data import DataLoader

from data import AeroMagneticCompensationDataset, SequentialDataset
from model import MLP, CNN
from utils import compute_std_dev, compute_rmse
from config import config

import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# all_sensors = ['mag_2_uc', 'mag_3_uc', 'mag_4_uc', 'mag_5_uc', 'cur_com_1', 'cur_ac_hi', 'cur_ac_lo', 'cur_tank', 'cur_flap',
#                    'cur_strb', 'cur_srvo_o', 'cur_srvo_m', 'cur_srvo_i', 'cur_heat', 'cur_acpwr', 'cur_outpwr', 'cur_bat_1',
#                    'cur_bat_2', 'vol_acpwr', 'vol_outpwr', 'vol_bat_1', 'vol_bat_2', 'vol_res_p', 'vol_res_n', 'vol_back_p',
#                    'vol_back_n', 'vol_gyro_1', 'vol_gyro_2', 'vol_acc_p', 'vol_acc_n', 'vol_block', 'vol_back', 'vol_srvo',
#                    'vol_cabt', 'vol_fan']

def get_parser():
    parser = argparse.ArgumentParser(description="航磁补偿")
    parser.add_argument("-flights", type=list, default=[2, 3, 4, 6, 7], help="flights to select")
    parser.add_argument("-path", type=str, default="./datasets/data/processed/Flt_data.h5", help="data path")
    parser.add_argument("-sensors", type=list,
                        default=['mag_3_c', 'mag_4_c', 'mag_5_c', 'vol_bat_1', 'vol_bat_2', 'ins_vn', 'ins_vw', 'ins_vu',
                                 'cur_heat',
                                 'cur_flap', 'cur_ac_lo', 'cur_tank', 'ins_pitch', 'ins_roll', 'ins_yaw', 'baro', 'line',
                                 'mag_1_bpf'],
                        help="sensors to select")
    parser.add_argument("-batch", type=int, default=128, help="batch_size")
    parser.add_argument("-lr", type=int, default=1e-3, help="learning rate")
    parser.add_argument("-wd", type=int, default=1e-3, help="weight_decay")
    parser.add_argument("-epochs", type=int, default=30, help="training epochs")
    parser.add_argument("-model", type=str, default='CNN', help="model")
    parser.add_argument("-test", type=str, default='1007', help="[1007, 1003]")
    parser.add_argument("-mode", type=str, default='train', help="[train, test]")
    parser.add_argument("-win", type=int, default=20, help="sliding window‘s length if 1DCNN")

    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    if args.model == 'CNN':
        custom_dataset = SequentialDataset(args)
        shuffle = False
        model = CNN(custom_dataset.x.shape[1], args.win).to(device)

    train_size = int(custom_dataset.__len__() * 14 / 17)
    val_size = custom_dataset.__len__() - train_size
    dataset_train, dataset_val = random_split(custom_dataset, [train_size, val_size])
    print("train num:", train_size)
    print("val num:", val_size)

    dataloader_train = DataLoader(dataset=dataset_train, batch_size=args.batch, shuffle=shuffle)
    dataloader_val = DataLoader(dataset=dataset_val, batch_size=args.batch, shuffle=shuffle)

    print("Feature selected : ", args.sensors)
    minerror = np.inf

    y_scaler = custom_dataset.get_scaler_params()
    y_mean = torch.tensor(y_scaler.mean_, dtype=torch.float32, device=device)
    y_scale = torch.tensor(y_scaler.scale_, dtype=torch.float32, device=device)

    loss_function = torch.nn.MSELoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    print("=" * 50)
    print("Start training and validation...\n")
    for epoch in range(1, args.epochs + 1):
        model.train()
        sum_loss = 0.0
        batch_idx = 0
        for data in dataloader_train:

            x, y = data
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            loss = loss_function(y_hat, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            sum_loss += loss.item()
            batch_idx += 1
        print("Epoch{}, train loss:{:.4f}".format(epoch, sum_loss / batch_idx))

        if epoch % 5 == 0:
            model.eval()
            B_preds, B_reals = np.empty((0, 1)), np.empty((0, 1))
            with torch.no_grad():
                for data in dataloader_val:
                    x, y = data
                    x, y = x.to(device), y.to(device)
                    y_hat = model(x)
                    val_loss = loss_function(y_hat, y)
                    B_pred = np.array(y_scaler.inverse_transform(y_hat.cpu()))

                    B_preds = np.vstack((B_preds, B_pred))
                    B_real = np.array((y * y_scale + y_mean).cpu())
                    B_reals = np.vstack((B_reals, B_real))

            std = compute_std_dev(B_preds, B_reals)
            rmse = compute_rmse(B_preds, B_reals)
            if std < minerror:
                minerror = std
                torch.save(model, "results/models/{}.pt".format(args.model))
            print("===Validation===   loss:{:.4f}; STD:{}nT; RMSE:{}nT".format(val_loss, std, rmse))

            scheduler.step(val_loss)
