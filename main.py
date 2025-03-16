import numpy as np
import os
import torch
from data import MagCompData, MagCompDataset
from model import MagneticCompensationLoss, PINN_TLNET
from utils import compute_std_delta_mag, inverse_transform
from config import config
from torch.utils.data.dataset import random_split
import torch.optim as optim
from torch.utils.data import DataLoader

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

if __name__ == '__main__':
    config = config()
    data = MagCompData(config)
    selected_features = ['utm_x', 'utm_y', 'utm_z', 'ins_pitch', 'ins_roll', 'ins_yaw', 'cur_ac_hi', 'cur_strb', 'cur_heat',
                         'vol_bat_1', 'vol_block', 'cur_com_1', 'cur_ac_lo', 'cur_tank', 'cur_flap', 'vol_bat_2', 'mag_5_uc']

    custom_dataset = MagCompDataset(data, mode='train')
    train_size = int(custom_dataset.__len__() * 14 / 17)
    val_size = custom_dataset.__len__() - train_size
    dataset_train, dataset_val = random_split(custom_dataset, [train_size, val_size])
    print("train num:", train_size)
    print("val num:", val_size)
    dataloader_train = DataLoader(dataset=dataset_train, batch_size=config.batch_size, shuffle=True, num_workers=0)
    dataloader_val = DataLoader(dataset=dataset_val, batch_size=config.batch_size, shuffle=False, num_workers=0)

    model_type = 'PINN_TLNET'
    print("Feature selected : ", selected_features)
    if config.is_pca:
        print("Using PCA, input dim : ", custom_dataset.x.shape[1])
    device = config.device
    model = PINN_TLNET(custom_dataset.x.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    loss_function = MagneticCompensationLoss()
    # loss_function = torch.nn.MSELoss()
    print("=" * 50)
    print("{} : Start training and validation...\n".format(model_type))
    minerror = np.inf
    beta_TL = torch.tensor(custom_dataset.beta_TL, dtype=torch.float32, device=device)
    for epoch in range(config.epochs):
        model.train()
        sum_loss = 0.0
        for i, data in enumerate(dataloader_train):
            x, y, A, mag4, mag1 = data
            x, y, A, mag4, mag1 = x.to(device), y.to(device), torch.tensor(A, dtype=torch.float32).to(device), torch.tensor(mag4,
                                                                                                                            dtype=torch.float32).to(
                device), torch.tensor(mag1, dtype=torch.float32).to(device)
            c = model(x)
            # loss = loss_function(y, torch.diag(torch.matmul(A, (beta_TL + c).T)))
            B_pred = mag4 - torch.diag(torch.matmul(A, (beta_TL + c).T))
            B_real = mag1
            B_tl = mag4 - torch.diag(torch.matmul(A, beta_TL.T))
            loss = loss_function(B_pred, B_real, B_tl)
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
                    x, y, A, _, _ = data
                    x, y, A = x.to(device), y.to(device), torch.tensor(A, dtype=torch.float32).to(device)
                    c = model(x)
                    val_y.extend(y.cpu())
                    val_y_hat.extend(torch.diag(torch.matmul(A, (beta_TL + c).T)).cpu())
            magerror = compute_std_delta_mag(np.array(val_y), np.array(val_y_hat))
            if magerror < minerror:
                minerror = magerror
                torch.save(model, "results/{}.pt".format(model_type))
            print("{}'s MagError in validation:{}nT".format(model_type, magerror))
