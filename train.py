import os
import torch
import numpy as np
import argparse
from data import AeroMagneticCompensationDataset
from model import CNN1D
from utils import plot_loss_curve, set_seed, inverse_from_11, mean_squared_error_np, plot_signals
from torch.utils.data import DataLoader
from tqdm import tqdm


def train(args, model, train_loader, val_loader=None, save_path=None, val_interval=10):
    print('-' * 10 + 'train and validation' + '-' * 10)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.8)
    criterion = torch.nn.MSELoss()

    best_val_loss = np.inf
    train_losses = []
    epochs_list = []

    epoch_iter = tqdm(range(1, args.epochs + 1), desc="Epochs")
    for epoch in epoch_iter:
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            x, y = batch
            x, y = x.to(device), y.to(device)

            y_pred = model(x)
            loss = criterion(y_pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        epochs_list.append(epoch)
        epoch_iter.set_postfix(train_loss="{:.6f}".format(avg_train_loss))

        # ================= Validation =================
        if val_loader is not None and (epoch % val_interval == 0 or epoch == args.epochs):
            model.eval()
            y_true_all = []
            y_pred_all = []
            bpf_all = []
            with torch.no_grad():
                for batch in val_loader:
                    x, y = batch
                    x, y = x.to(device), y.to(device)
                    y_pred = model(x)

                    y_true_all.append(y.cpu().numpy())
                    y_pred_all.append(y_pred.cpu().numpy())
                    bpf_all.append(x[:, 0, -1].cpu().numpy().reshape(-1, 1))

            y_true_all = np.concatenate(y_true_all, axis=0)
            y_pred_all = np.concatenate(y_pred_all, axis=0)
            bpf_all = np.concatenate(bpf_all, axis=0)
            val_loss = np.sqrt(mean_squared_error_np(y_true_all, y_pred_all))
            y_true_denorm = inverse_from_11(y_true_all, y_scaler)
            y_pred_denorm = inverse_from_11(y_pred_all, y_scaler)
            if isinstance(x_scaler, dict) and "min" in x_scaler and "max" in x_scaler:
                bpf_scaler = {
                    "min": np.asarray([x_scaler["min"][0]]),
                    "max": np.asarray([x_scaler["max"][0]]),
                }
                bpf_denorm = inverse_from_11(bpf_all, bpf_scaler)
            else:
                bpf_denorm = bpf_all
            std_raw = np.std(bpf_denorm, ddof=1)
            std_tl = np.std(y_true_denorm, ddof=1)
            std_model = np.std(y_true_denorm - y_pred_denorm, ddof=1)
            print("epoch{}:".format(epoch))
            print("STD raw:{}; STD tl:{}; IR:{}".format(std_raw, std_tl, std_raw / std_tl))
            print("STD raw:{}; STD tl+ourmodel:{}; IR:{}".format(std_raw, std_model, std_raw / std_model))
            # plot_signals([bpf_denorm, y_true_denorm, y_true_denorm - y_pred_denorm], ['bpf', 'tl', 'tl+ourmodel'])

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if save_path is not None:
                    torch.save(model.state_dict(), save_path)

    plot_loss_curve(epochs_list, train_losses, save_path)


def get_parser():
    used_sensors = ['time', 'mag_1_uc', 'flux_b_x', 'flux_b_y', 'flux_b_z', 'line']

    parser = argparse.ArgumentParser(description="航磁补偿")
    parser.add_argument("-path", type=str, default="./data/data.txt", help="data path")
    parser.add_argument("-sensors", type=list, default=used_sensors, help="sensors to select")
    parser.add_argument("-batch", type=int, default=64, help="batch_size")
    parser.add_argument("-lr", type=int, default=1e-4, help="learning rate")
    parser.add_argument("-wd", type=int, default=1e-3, help="weight_decay")
    parser.add_argument("-epochs", type=int, default=100, help="training epochs")
    parser.add_argument("-model", type=str, default='1DCNN', help="model")
    parser.add_argument("-win", type=int, default=64, help="sliding window‘s length if 1DCNN")
    parser.add_argument("-lambd", type=int, default=0.025, help="ridge parameter for ridge regression")
    parser.add_argument("-save", type=str, default='./results/models/1DCNN.pt', help="model save path")
    parser.add_argument("-seed", type=int, default=2026, help="random seed")
    parser.add_argument("-cali", type=list, default=[46370.00, 47600.00], help="calibration window")
    parser.add_argument("-smoo", type=list, default=[66560.00, 67850.00], help="smooth window")

    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    custom_dataset = AeroMagneticCompensationDataset(args)
    traindataset, valdataset = custom_dataset.build_custom_dataset()
    train_loader = DataLoader(traindataset, batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(valdataset, batch_size=args.batch, shuffle=False)
    y_scaler = custom_dataset.y_scaler
    x_scaler = custom_dataset.x_scaler
    model = CNN1D(in_dim=19, window_size=args.win)
    train(args, model, train_loader, val_loader=val_loader, save_path=args.save)

    # testdataset = custom_dataset.build_test_dataset()
