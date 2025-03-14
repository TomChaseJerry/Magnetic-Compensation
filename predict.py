from config import config
from data import MagCompData, get_XYZ, get_ind, MagCompDataset
import pandas as pd
import numpy as np
from TL_model import create_TL_coef, create_TL_A
from utils import detrend, plot_model_vs_real, compute_std_delta_mag, inverse_transform
import torch
from torch.utils.data import DataLoader
from model import PINN_TLNET


def TL_model_predict(flight, test_interval, is_save=True):
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
        magerror = compute_std_delta_mag(data.xyzs[test_flight]['mag_1_c'][ind], mag_4_c)
        print("{}'s MagError in testing : {}nT".format('TL_model_{}_{}'.format(test_flight, time), magerror))

        plot_model_vs_real(time, mag_4_c, data.xyzs[test_flight]['mag_1_c'][ind], 'TL_model_{}_{}'.format(test_flight, time),
                           magerror, is_save)


def Model_predict(config, flight, model_path, model_type, is_save=True):
    print("=" * 50)
    print("Start testing...\n")

    data = MagCompData(config)
    device = config.device
    batch_size = config.batch_size

    custom_dataset = MagCompDataset(data, mode='train')
    dataset_test = MagCompDataset(data, mode='test')
    print("Model train in {} and test in {}".format(config.train_ttlim[flight[0]], config.test_ttlim[flight[1]]))
    print("test num:", dataset_test.__len__())
    dataloader_test = DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=False, num_workers=0)

    model = torch.load(model_path, weights_only=False)
    model.eval()
    val_y = []
    val_y_hat = []
    with torch.no_grad():
        beta_TL = torch.tensor(custom_dataset.beta_TL, dtype=torch.float32, device=device)
        for data in dataloader_test:
            x, y, A = data
            x, y, A = x.to(device), y.to(device), torch.tensor(A, dtype=torch.float32).to(device)
            c = model(x)
            val_y.extend(y.cpu())
            val_y_hat.extend(torch.diag(torch.matmul(A, (beta_TL + c).T)).cpu())

    val_y = np.array(val_y)
    val_y_hat = np.array(val_y_hat)
    magerror = compute_std_delta_mag(val_y, val_y_hat)
    B_pred = dataset_test.mag4uc - val_y_hat
    B_real = dataset_test.mag1c
    plot_model_vs_real([0, 10000], B_pred, B_real, model_type, magerror, is_save)
    print("{}'s MagError in testing:{}nT".format(model_type, magerror))


if __name__ == '__main__':
    config = config()
    train_flight = "Flt1003"
    test_flight = "Flt1003"
    # TL_model_predict([train_flight, test_flight], config.test_ttlim[test_flight])
    model_path = "results/TLNET.pt"
    Model_predict(config, [train_flight, test_flight], model_path, 'TLNET_{}_{}'.format(test_flight, config.test_ttlim[test_flight][0]))
