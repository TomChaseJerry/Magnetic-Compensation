from config import config
from data import MagCompData, get_XYZ, get_ind, MagCompDataset, SequentialDataset
import pandas as pd
import numpy as np
from TL_model import create_TL_coef, create_TL_A
from utils import detrend, plot_model_vs_real, compute_std_dev, compute_rmse
import torch
from torch.utils.data import DataLoader
from model import TLMLP, TLCNN


def TLmodel_predict(flight, test_interval, is_save=True):
    train_flight = flight[0]
    test_flight = flight[1]
    data = MagCompData(config)
    TL_ind = data.train_inds[train_flight]
    lambd = 0.025  # ridge parameter for ridge regression
    use_vec = "flux_d"  # selected vector (flux) magnetometer
    use_sca = "mag_4_uc"
    terms_A = ["permanent", "induced", "eddy"]  # Tolles-Lawson terms to use
    Bx = data.xyzs[train_flight].get(use_vec + '_x')  # load Flux D data
    By = data.xyzs[train_flight].get(use_vec + '_y')
    Bz = data.xyzs[train_flight].get(use_vec + '_z')
    Bt = data.xyzs[train_flight].get(use_vec + '_t')
    TL_d_4 = create_TL_coef(Bx[TL_ind], By[TL_ind], Bz[TL_ind], Bt[TL_ind], data.xyzs[train_flight].get(use_sca)[TL_ind],
                            lambd=lambd, terms=terms_A)  # coefficients with Flux D & Mag 4

    for index in range(len(test_interval)):
        time = test_interval[index]
        print("TL model train in {} and test in {}".format(config.train_ttlim[train_flight], time))

        ind = get_ind(data.xyzs[test_flight], tt_lim=time)
        Bx = data.xyzs[test_flight].get(use_vec + '_x')  # load Flux D data
        By = data.xyzs[test_flight].get(use_vec + '_y')
        Bz = data.xyzs[test_flight].get(use_vec + '_z')
        Bt = data.xyzs[test_flight].get(use_vec + '_t')
        A = create_TL_A(Bx[ind], By[ind], Bz[ind], Bt=Bt[ind])  # Tolles-Lawson `A` matrix for Flux D
        mag_4_uc = data.xyzs[test_flight]['mag_4_uc'][ind]  # uncompensated Mag 4
        mag_4_c = np.array(mag_4_uc - np.dot(A, TL_d_4))
        std = compute_std_dev(mag_4_c, data.xyzs[test_flight]['mag_1_c'][ind])
        rmse = compute_rmse(mag_4_c, data.xyzs[test_flight]['mag_1_c'][ind])
        print("===Test===   STD:{}nT; RMSE:{}nT".format(std, rmse))
        plot_model_vs_real(time, mag_4_c, data.xyzs[test_flight]['mag_1_c'][ind], 'TL_model_{}_{}'.format(test_flight, time),
                           std, rmse, is_save)


def TLNET_predict(config, flight, model_path, model_name, is_save=True):
    print("=" * 50)
    print("Start testing...\n")

    data = MagCompData(config)
    device = config.device
    batch_size = config.batch_size

    if config.model_type == 'MLP':
        custom_dataset = MagCompDataset(data, mode='train')
        dataset_test = MagCompDataset(data, mode='test')
    elif config.model_type == 'CNN':
        custom_dataset = SequentialDataset(data, mode='train')
        dataset_test = SequentialDataset(data, mode='test')
    else:
        raise ValueError("model_type must be in ['MLP', 'CNN']")

    print("Model train in {} and test in {}".format(config.train_ttlim[flight[0]], config.test_ttlim[flight[1]]))
    print("test num:", dataset_test.__len__())
    dataloader_test = DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=False, num_workers=0)

    model = torch.load(model_path, weights_only=False)
    model.eval()
    B_preds, B_reals = np.empty((0, 1)), np.empty((0, 1))
    std_y = custom_dataset.get_std_y()
    beta_TL = custom_dataset.beta_TL.to(device)
    with torch.no_grad():
        for data in dataloader_test:
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
        print("===Test===   STD:{}nT; RMSE:{}nT".format(std, rmse))
        plot_model_vs_real([0, 10000], B_preds, B_reals, model_name, std, rmse, is_save)


if __name__ == '__main__':
    config = config()
    train_flight = "Flt1003"
    test_flight = "Flt1003"
    # TLmodel_predict([train_flight, test_flight], config.test_ttlim[test_flight])
    model_name = 'TLMLP'
    model_path = "results/models/{}.pt".format(model_name)

    TLNET_predict(config, [train_flight, test_flight], model_path,
                  '{}_{}_{}'.format(model_name, test_flight, config.test_ttlim[test_flight][0]))
