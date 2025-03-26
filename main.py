import numpy as np
import os
import torch
from data import MagCompData, MagCompDataset, SequentialDataset
from model import CustomLoss, TLMLP, TLCNN
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
        custom_dataset = MagCompDataset(data, mode='train')
        shuffle = True
        model = TLMLP(custom_dataset.x.shape[1]).to(device)
    elif config.model_type == 'CNN':
        custom_dataset = SequentialDataset(data, mode='train')
        shuffle = False
        model = TLCNN(custom_dataset.x.shape[1], config.seq_len).to(device)
    else:
        raise ValueError("model_type must be in ['MLP', 'CNN']")

    train_size = int(custom_dataset.__len__() * 14 / 17)
    val_size = custom_dataset.__len__() - train_size
    dataset_train, dataset_val = random_split(custom_dataset, [train_size, val_size])
    print("train num:", train_size)
    print("val num:", val_size)

    dataloader_train = DataLoader(dataset=dataset_train, batch_size=config.batch_size, shuffle=shuffle)
    dataloader_val = DataLoader(dataset=dataset_val, batch_size=config.batch_size, shuffle=shuffle)

    model_name = 'TLMLP'
    print("Feature selected : ", config.selected_features)
    if config.is_pca:
        print("Using PCA, input dim : ", custom_dataset.x.shape[1])
    minerror = np.inf

    std_y = custom_dataset.get_std_y()

    beta_TL = custom_dataset.beta_TL.to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    loss_function = CustomLoss(beta_TL)

    print("=" * 50)
    print("{} : Start training and validation...\n".format(model_name))
    for epoch in range(1, config.epochs + 1):
        model.train()
        sum_loss = 0.0
        for data in dataloader_train:
            x, y, _, _, A, Btl = data
            x, y, A, Btl = x.to(device), y.to(device), A.to(device), Btl.to(device)
            model_output = model(x)
            loss = loss_function(model_output, y, A, Btl)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sum_loss += loss.item()
        print("Epoch{}, train loss:{:.4f}".format(epoch, sum_loss / dataloader_train.__len__()))

        if epoch % 5 == 0:
            model.eval()
            B_preds, B_reals = np.empty((0, 1)), np.empty((0, 1))
            with torch.no_grad():
                for data in dataloader_val:
                    x, _, mag4uc, mag1c, A, _ = data
                    x, A = x.to(device), A.to(device)

                    model_output = model(x)
                    c, d = model_output[:, :18], model_output[:, 18].reshape(-1, 1)

                    tmp_value = torch.diag(torch.matmul(A, (beta_TL + c).T)).reshape(-1, 1) + d
                    B_pred = mag4uc - std_y.inverse_transform(tmp_value.cpu())
                    B_pred = np.array(B_pred)
                    B_preds = np.vstack((B_preds, B_pred))
                    B_real = np.array(mag1c)
                    B_reals = np.vstack((B_reals, B_real))

            std = compute_std_dev(B_preds, B_reals)
            rmse = compute_rmse(B_preds, B_reals)
            if std < minerror:
                minerror = std
                torch.save(model, "results/models/{}.pt".format(model_name))
            print("===Validation===   STD:{}nT; RMSE:{}nT".format(std, rmse))
