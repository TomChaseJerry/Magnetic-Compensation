import numpy as np
import os
import torch
from data import MagCompData, MagCompDataset, SequentialDataset, PINNsDataset, PINNsSequentialDataset
from model import CustomLoss, TLMLP, TLCNN, MagTransformer
from utils import compute_std_dev, compute_rmse
from config import config
from torch.utils.data.dataset import random_split
import torch.optim as optim
from torch.utils.data import DataLoader

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

if __name__ == '__main__':
    config = config()
    device = config.device

    data = MagCompData(config)

    if config.model_type == 'MLP':
        if config.is_PINNs:
            custom_dataset = PINNsDataset(data, mode='train')
            shuffle = True
            model = TLMLP(custom_dataset.x.shape[1]).to(device)

        else:
            custom_dataset = MagCompDataset(data, mode='train')
            shuffle = True
            model = TLMLP(custom_dataset.x.shape[1]).to(device)
    elif config.model_type == 'CNN':
        if config.is_PINNs:
            custom_dataset = PINNsSequentialDataset(data, mode='train')
            shuffle = False
            model = TLCNN(custom_dataset.x.shape[1], config.seq_len).to(device)
        else:
            custom_dataset = SequentialDataset(data, mode='train')
            shuffle = False
            model = TLCNN(custom_dataset.x.shape[1], config.seq_len).to(device)
    elif config.model_type == 'Transformer':
        if config.is_PINNs:
            custom_dataset = PINNsSequentialDataset(data, mode='train')
            shuffle = False
            model = MagTransformer(custom_dataset.x.shape[1]).to(device)
        else:
            custom_dataset = SequentialDataset(data, mode='train')
            shuffle = False
            model = MagTransformer(custom_dataset.x.shape[1]).to(device)
    else:
        raise ValueError("model_type must be in ['MLP', 'CNN', 'Transformer']")

    train_size = int(custom_dataset.__len__() * 14 / 17)
    val_size = custom_dataset.__len__() - train_size
    dataset_train, dataset_val = random_split(custom_dataset, [train_size, val_size])
    print("train num:", train_size)
    print("val num:", val_size)

    dataloader_train = DataLoader(dataset=dataset_train, batch_size=config.batch_size, shuffle=shuffle)
    dataloader_val = DataLoader(dataset=dataset_val, batch_size=config.batch_size, shuffle=shuffle)

    model_name = 'CNN'
    print("Feature selected : ", config.selected_features)
    if config.is_pca:
        print("Using PCA, input dim : ", custom_dataset.x.shape[1])
    minerror = np.inf

    std_y = custom_dataset.get_std_y()
    y_mean = torch.tensor(std_y.mean_, dtype=torch.float32, device=device)
    y_scale = torch.tensor(std_y.scale_, dtype=torch.float32, device=device)
    print("std_y : mean{}, scale{}".format(std_y.mean_, std_y.scale_))

    if config.is_PINNs:
        std_mag4c = custom_dataset.get_std_mag4c()
        print("std_mag4c : mean{}, scale{}".format(std_mag4c.mean_, std_mag4c.scale_))
        beta = torch.tensor(custom_dataset.beta_tl_4, dtype=torch.float32, device=device)
        loss_function = CustomLoss(beta, std_y)
    else:
        loss_function = torch.nn.MSELoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    print("=" * 50)
    print("{} : Start training and validation...\n".format(model_name))
    for epoch in range(1, config.epochs + 1):
        model.train()
        sum_loss = 0.0
        batch_idx = 0
        for data in dataloader_train:
            if config.is_PINNs:
                x, y, A, mag4uc, mag4c = data
                x, y, A, mag4uc, mag4c = x.to(device), y.to(device), A.to(device), mag4uc.to(device), mag4c.to(device)
                c = model(x)
                y_hat = (mag4uc - torch.diag(torch.matmul(A, (beta + c).T)).reshape(-1, 1) - y_mean) / y_scale
                loss = loss_function(c, y, A, mag4uc, mag4c)
            else:
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
                    if config.is_PINNs:
                        x, y, A, mag4uc, mag4c = data
                        x, y, A, mag4uc, mag4c = x.to(device), y.to(device), A.to(device), mag4uc.to(device), mag4c.to(device)
                        c = model(x)
                        y_hat = (mag4uc - torch.diag(torch.matmul(A, (beta + c).T)).reshape(-1, 1) - y_mean) / y_scale
                        val_loss = loss_function(c, y, A, mag4uc, mag4c)
                        B_pred = np.array((mag4uc - torch.diag(torch.matmul(A, (beta + c).T)).reshape(-1, 1)).cpu())
                    else:
                        x, y = data
                        x, y = x.to(device), y.to(device)
                        y_hat = model(x)
                        val_loss = loss_function(y_hat, y)
                        B_pred = np.array(std_y.inverse_transform(y_hat.cpu()))
                    B_preds = np.vstack((B_preds, B_pred))
                    B_real = np.array((y * y_scale + y_mean).cpu())
                    B_reals = np.vstack((B_reals, B_real))

            std = compute_std_dev(B_preds, B_reals)
            rmse = compute_rmse(B_preds, B_reals)
            if std < minerror:
                minerror = std
                torch.save(model, "results/models/{}.pt".format(model_name))
            print("===Validation===   loss:{:.4f}; STD:{}nT; RMSE:{}nT".format(val_loss, std, rmse))

            scheduler.step(val_loss)
