import numpy as np

from get_XYZ import *
from model import *
from utils import compute_std_delta_mag
from torch.utils.data.dataset import random_split

if __name__ == '__main__':
    # flights_20 = ["Flt1002", "Flt1003", "Flt1004", "Flt1005", "Flt1006", "Flt1007"]
    # flights_21 = ["Flt2001", "Flt2002", "Flt2004", "Flt2005", "Flt2006", "Flt2007", "Flt2008", "Flt2015", "Flt2016", "Flt2017"]
    # flights_all = flights_20 + flights_21
    flights_all = ["Flt1002", "Flt1004"]
    test_lines = {"Flt1002": 1002.08, "Flt1003": 1003.01, "Flt1004": 4016.0, "Flt1005": 4002.0, "Flt1006": 1006.08,
                  "Flt1007": 1007.07,
                  "Flt2001": 2001.00, "Flt2002": 2002.02, "Flt2004": 2001.11, "Flt2005": 2004.00, "Flt2006": 2004.07,
                  "Flt2007": 2005.07, "Flt2008": 2001.22, "Flt2015": 2005.09, "Flt2016": 2005.19, "Flt2017": 2005.20}
    df_flight_path = "datasets/dataframes/df_flight.csv"
    df_flight = pd.read_csv(df_flight_path)
    df_all_path = "datasets/dataframes/df_all.csv"
    df_all = pd.read_csv(df_all_path)

    xyzs = {}
    test_inds = {}
    for flight in flights_all:
        xyz = get_XYZ(flight, df_flight, silent=True)
        xyzs[flight] = xyz
        test_line = test_lines[flight]
        print("test line in {}: {}\n".format(flight, test_line))
        test_ind = get_ind(xyz, tt_lim=[df_all[df_all['line'] == test_line]['t_start'].item(),
                                        df_all[df_all['line'] == test_line]['t_end'].item()])  # get Boolean indices
        test_inds[flight] = test_ind

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 32
    lr = 0.01
    epochs = 2

    train_features = ['cur_ac_hi', 'cur_strb', 'cur_heat', 'vol_bat_1', 'vol_block', 'cur_com_1', 'cur_ac_lo', 'cur_tank',
                      'cur_flap', 'vol_bat_2', 'mag_2_uc', 'mag_3_uc', 'mag_4_uc', 'mag_5_uc', 'flux_d_x', 'flux_d_y', 'flux_d_z']

    lpf = get_bpf(pass1=0.0, pass2=0.2, fs=10.0)
    custom_dataset = MagCompDataset(xyzs, test_inds, train_features, lpf, 'train')
    train_size = int(custom_dataset.__len__() * 14 / 17)
    val_size = len(custom_dataset) - train_size
    dataset_train, dataset_val = torch.utils.data.random_split(custom_dataset, [train_size, val_size])
    dataset_test = MagCompDataset(xyzs, test_inds, train_features, lpf, 'test')
    print("train num:", train_size)
    print("val num:", val_size)
    print("test num:", dataset_test.__len__())
    dataloader_train = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True, num_workers=0)
    dataloader_val = DataLoader(dataset=dataset_val, batch_size=batch_size, shuffle=False, num_workers=0)
    dataloader_test = DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=False, num_workers=0)

    model_type = 'Model1'
    print("input dim:", len(train_features))
    model = Model1(len(train_features)).to(device)
    criterion = nn.MSELoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    print("=" * 50)
    print("Start training and validation...\n")
    sum_loss = 0.0
    model.train()
    for epoch in range(epochs):
        with tqdm(total=len(dataloader_train), desc='epoch{} [train]'.format(epoch + 1), file=sys.stdout) as t:
            for i, data in enumerate(dataloader_train):
                x, y = data
                x, y = x.to(device), y.to(device)
                y_hat = model(x).reshape(-1)
                loss = criterion(y_hat, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                sum_loss += loss.item()
                t.set_postfix(loss=sum_loss / (i + 1), lr=scheduler.get_last_lr()[0])
                t.update(1)
            scheduler.step()
        if epoch % 5 == 0:
            model.eval()
            val_y = []
            val_y_hat = []
            with torch.no_grad():
                for data in dataloader_val:
                    x, y = data
                    x, y = x.to(device), y.to(device)
                    y_hat = model(x)
                    val_y.extend(np.array(y.cpu()))
                    val_y_hat.extend(np.array(y_hat.cpu()))
            print("{}'s MagError:{}".format(model_type, np.round(compute_std_delta_mag(np.array(val_y), np.array(val_y_hat)), 2)))

    print("=" * 50)
    print("Start testing...\n")
    model.eval()
    val_y = []
    val_y_hat = []
    with torch.no_grad():
        for data in dataloader_val:
            x, y = data
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            val_y.extend(np.array(y.cpu()))
            val_y_hat.extend(np.array(y_hat.cpu()))
    print("{}'s MagError:{}".format(model_type, np.round(compute_std_delta_mag(np.array(val_y), np.array(val_y_hat)), 2)))
