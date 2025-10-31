from config import config
from data import AeroMagneticCompensationDataset, SequentialDataset
import argparse
import numpy as np
from TL_model import create_TL_coef, create_TL_A
from utils import detrend, plot_comparison, compute_std_dev, compute_rmse
import torch
from torch.utils.data import DataLoader
import joblib

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict(args, is_save=True):
    print("=" * 50)
    print("Start testing...\n")

    model_path = "results/models/{}.pt".format(args.model)
    y_scaler_path = "results/logs/y_scaler_{}.pkl".format(args.test)

    if args.model == 'CNN':
        dataset_test = SequentialDataset(args)

    print("test num:", dataset_test.__len__())
    dataloader_test = DataLoader(dataset=dataset_test, batch_size=args.batch, shuffle=False, num_workers=0)

    model = torch.load(model_path, weights_only=False)
    y_scaler = joblib.load(y_scaler_path)
    model.eval()
    B_preds, B_reals = np.empty((0, 1)), np.empty((0, 1))
    with torch.no_grad():
        for data in dataloader_test:
            x, y = data
            x = x.to(device)

            y_hat = model(x)

            B_pred = np.array(y_scaler.inverse_transform(y_hat.cpu()))
            B_preds = np.vstack((B_preds, B_pred))
            B_reals = np.vstack((B_reals, y))

        std = compute_std_dev(B_preds, B_reals)
        rmse = compute_rmse(B_preds, B_reals)
        print("===Test===   STD:{}nT; RMSE:{}nT".format(std, rmse))

    before = np.array(dataset_test.flights[7]['mag_4_bpf'])
    plot_comparison(before[1000:10000], B_preds[1000:10000], ['7', '补偿前磁场值', '补偿后磁场值'])

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
    parser.add_argument("-model", type=str, default='CNN', help="model")
    parser.add_argument("-test", type=str, default='1007', help="[1007, 1003]")
    parser.add_argument("-mode", type=str, default='test', help="[train, test]")
    parser.add_argument("-win", type=int, default=20, help="sliding window‘s length if 1DCNN")

    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    predict(args)
