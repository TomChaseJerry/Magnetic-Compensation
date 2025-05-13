from config import config
from data import MagCompData, get_XYZ, get_ind, MagCompDataset, SequentialDataset, PINNsDataset, PINNsSequentialDataset
import pandas as pd
import numpy as np
from TL_model import create_TL_coef, create_TL_A
from utils import detrend, plot_model_vs_real, compute_std_dev, compute_rmse, inverse_transform
import torch
from torch.utils.data import DataLoader


def TLNET_predict(config, model_path, result_name, std_y, is_save=True):
    print("=" * 50)
    print("Start testing...\n")

    data = MagCompData(config)
    device = config.device

    if config.model_type == 'MLP':
        dataset_test = MagCompDataset(data, mode='test')
    elif config.model_type == 'CNN':
        dataset_test = SequentialDataset(data, mode='test')
    else:
        raise ValueError("model_type must be in ['MLP', 'CNN']")

    print("test num:", dataset_test.__len__())
    dataloader_test = DataLoader(dataset=dataset_test, batch_size=config.batch_size, shuffle=False, num_workers=0)

    model = torch.load(model_path, weights_only=False)
    model.eval()
    B_preds, B_reals = np.empty((0, 1)), np.empty((0, 1))
    with torch.no_grad():
        for data in dataloader_test:
            x, y = data
            x = x.to(device)

            y_hat = model(x)

            B_pred = np.array(inverse_transform(y_hat.cpu(), std_y))
            B_preds = np.vstack((B_preds, B_pred))
            B_reals = np.vstack((B_reals, y))

        std = compute_std_dev(B_preds, B_reals)
        rmse = compute_rmse(B_preds, B_reals)
        print("===Test===   STD:{}nT; RMSE:{}nT".format(std, rmse))
        plot_model_vs_real([0, 10000], B_preds, B_reals, result_name, std, rmse, is_save)


def PINNs_predict(config, model_path, result_name, is_save=True):
    print("=" * 50)
    print("Start testing...\n")

    data = MagCompData(config)
    device = config.device

    if config.model_type == 'MLP':
        dataset_test = PINNsDataset(data, mode='test')
    elif config.model_type == 'CNN':
        dataset_test = PINNsSequentialDataset(data, mode='test')
    elif config.model_type == 'Transformer':
        dataset_test = PINNsSequentialDataset(data, mode='train')
    else:
        raise ValueError("model_type must be in ['MLP', 'CNN', 'Transformer']")
    beta = torch.tensor(dataset_test.beta_tl_4, dtype=torch.float32, device=device)
    print("test num:", dataset_test.__len__())
    dataloader_test = DataLoader(dataset=dataset_test, batch_size=config.batch_size, shuffle=False, num_workers=0)

    model = torch.load(model_path, weights_only=False)
    model.eval()
    B_preds, B_reals = np.empty((0, 1)), np.empty((0, 1))

    with torch.no_grad():
        for data in dataloader_test:
            x, y, A, mag4uc, mag4c = data
            x, y, A, mag4uc, mag4c = x.to(device), y.to(device), A.to(device), mag4uc.to(device), mag4c.to(device)

            c = model(x)
            y_hat = mag4uc - torch.diag(torch.matmul(A, (beta + c).T)).reshape(-1, 1)
            B_pred = np.array(y_hat.cpu())
            B_preds = np.vstack((B_preds, B_pred))
            B_real = np.array(y.cpu())
            B_reals = np.vstack((B_reals, B_real))

        std = compute_std_dev(B_preds, B_reals)
        rmse = compute_rmse(B_preds, B_reals)
        print("===Test===   STD:{}nT; RMSE:{}nT".format(std, rmse))
        plot_model_vs_real([0, 10000], B_preds, B_reals, result_name, std, rmse, is_save)


if __name__ == '__main__':
    config = config()
    # TLmodel_predict([train_flight, test_flight], config.test_ttlim[test_flight])
    model_name = 'CNN'
    model_path = "results/models/{}.pt".format(model_name)
    result_name = '{}_FLt1007'.format(model_name)
    std_y = [-26.53752596, 251.33759672]
    TLNET_predict(config, model_path, result_name, std_y)
