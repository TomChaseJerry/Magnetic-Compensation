import pandas as pd
import numpy as np
import h5py
from pyproj import Transformer
import torch
import joblib

from TL_model import create_TL_coef, create_TL_A
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from utils import get_bpf, bpf_data, plot_comparison, detrend


class AeroMagneticCompensationDataset(Dataset):
    def __init__(self, args):
        self.args = args
        self.flights = dict()
        self.features = self.args.sensors
        self.bpf = get_bpf(pass1=0.1, pass2=0.9, fs=10.0, pole=4)

        self.read()
        if self.args.test == '1007':
            self.train_lines = self.flights[2].line.unique().tolist() + self.flights[3].line.unique().tolist() + self.flights[
                4].line.unique().tolist() + self.flights[6].line.unique().tolist()
            self.test_lines = self.flights[7].line.unique().tolist()
        else:
            self.train_lines = self.flights[2].line.unique().tolist() + self.flights[4].line.unique().tolist() + self.flights[
                6].line.unique().tolist() + self.flights[7].line.unique().tolist()
            self.test_lines = self.flights[3].line.unique().tolist()

        self.lines = self.train_lines if self.args.mode == 'train' else self.test_lines

        self.y_scaler = StandardScaler()
        self.apply_TL()
        self.data_processing()

    def read(self):
        for flight_num in self.args.flights:
            data_path = self.args.path
            df = pd.read_hdf(data_path, key="Flt100{}".format(flight_num))
            # apply bpf
            df['mag_3_bpf'] = bpf_data(df['mag_3_uc'], self.bpf)
            df['mag_4_bpf'] = bpf_data(df['mag_4_uc'], self.bpf)
            df['mag_5_bpf'] = bpf_data(df['mag_5_uc'], self.bpf)
            df['mag_1_bpf'] = bpf_data(df['mag_1_c'], self.bpf)
            # plot_comparison(df['mag_4_bpf'], detrend(df['mag_4_uc']), [str(flight_num), 'mag4bpf', 'mag4uc'])
            self.flights[flight_num] = df

    def apply_TL(self):
        mask_tl = (self.flights[2].line == 1002.20)
        tl_data = self.flights[2][mask_tl]
        lambd = 0.025  # ridge parameter for ridge regression
        terms_A = ["permanent", "induced", "eddy"]  # Tolles-Lawson terms to use
        self.beta_tl_3 = create_TL_coef(tl_data['flux_b_x'], tl_data['flux_b_y'], tl_data['flux_b_z'], tl_data['flux_b_t'],
                                        tl_data['mag_3_bpf'], lambd=lambd, terms=terms_A)
        self.beta_tl_4 = create_TL_coef(tl_data['flux_b_x'], tl_data['flux_b_y'], tl_data['flux_b_z'], tl_data['flux_b_t'],
                                        tl_data['mag_4_bpf'], lambd=lambd, terms=terms_A)
        self.beta_tl_5 = create_TL_coef(tl_data['flux_b_x'], tl_data['flux_b_y'], tl_data['flux_b_z'], tl_data['flux_b_t'],
                                        tl_data['mag_5_bpf'], lambd=lambd, terms=terms_A)

        for flight_num, temp_df in self.flights.items():
            A = create_TL_A(temp_df['flux_b_x'], temp_df['flux_b_y'], temp_df['flux_b_z'], temp_df['flux_b_t'])
            temp_df['mag_3_c'] = temp_df['mag_3_bpf'].values - np.dot(A, self.beta_tl_3)
            temp_df['mag_4_c'] = temp_df['mag_4_bpf'].values - np.dot(A, self.beta_tl_4)
            temp_df['mag_5_c'] = temp_df['mag_5_bpf'].values - np.dot(A, self.beta_tl_5)
            self.flights[flight_num] = temp_df

    def data_processing(self):
        x = pd.DataFrame()
        for flight_num in self.args.flights:
            # select features
            df = self.flights[flight_num]
            df = df[self.features]

            # concat
            x = pd.concat([x, df], ignore_index=True, axis=0)

        x = x.loc[x.line.isin(self.lines)]

        scaler = StandardScaler()
        y = x.loc[:, 'mag_1_bpf']
        x = x.drop(columns=['line', 'mag_1_bpf'])
        self.x = torch.tensor(scaler.fit_transform(x.to_numpy()), dtype=torch.float32)
        if self.args.mode == 'train':
            self.y = torch.tensor(self.y_scaler.fit_transform(y.to_numpy().reshape(-1, 1)), dtype=torch.float32)
            self.length = self.y.size()[0]
            joblib.dump(self.y_scaler, '{}/y_scaler_{}.pkl'.format('./results/logs/', self.args.test))
        else:
            self.y = y.to_numpy().reshape(-1, 1)
            self.length = self.y.shape[0]

    def get_scaler_params(self):
        return self.y_scaler

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return self.length


class SequentialDataset(AeroMagneticCompensationDataset):
    def __init__(self, args):
        super().__init__(args)
        self.x = self.x.unfold(0, self.args.win, 1)
        self.y = self.y[self.args.win - 1:]
        self.length = self.length - self.args.win + 1


def read_check(data, silent=False):
    """
        Internal helper function to check for NaNs or missing data (returned as NaNs) in opened HDF5 file.
    Prints out warning for any field that contains NaNs.

    **Arguments:**
        - `data`:    opened HDF5 file
        - `silent`: (optional) if true, no print outs

    **Returns:**
        - `val`: data returned for `field`
    """

    for feature in data.keys():
        val = np.array(data[feature])  # Read the data
        if not silent and np.isnan(val).any():
            print("{} field contains NaNs".format(feature))


def euler2dcm(roll, pitch, yaw, order="body2nav"):
    """
        euler2dcm(roll, pitch, yaw, order::Symbol = :body2nav)

    Converts a (Euler) roll-pitch-yaw (`X`-`Y`-`Z`) right-handed body to navigation
    frame rotation (or the opposite rotation), to a DCM (direction cosine matrix).
    Yaw is synonymous with azimuth and heading here.
    If frame 1 is rotated to frame 2, then the returned DCM, when pre-multiplied,
    rotates a vector in frame 1 into frame 2. There are 2 use cases:

    1) With `order = :body2nav`, the body frame is rotated in the standard
    -roll, -pitch, -yaw sequence to the navigation frame. For example, if v1 is
    a 3x1 vector in the body frame [nose, right wing, down], then that vector
    rotated into the navigation frame [north, east, down] would be v2 = dcm * v1.

    2) With `order = :nav2body`, the navigation frame is rotated in the standard
    yaw, pitch, roll sequence to the body frame. For example, if v1 is a 3x1
    vector in the navigation frame [north, east, down], then that vector rotated
    into the body frame [nose, right wing, down] would be v2 = dcm * v1.

    Reference: Titterton & Weston, Strapdown Inertial Navigation Technology, 2004,
    Section 3.6 (pg. 36-41 & 537).

    **Arguments:**
    - `roll`:  length-`N` roll  angle [rad], right-handed rotation about x-axis
    - `pitch`: length-`N` pitch angle [rad], right-handed rotation about y-axis
    - `yaw`:   length-`N` yaw   angle [rad], right-handed rotation about z-axis
    - `order`: (optional) rotation order {`:body2nav`,`:nav2body`}

    **Returns:**
    - `dcm`: `3` x `3` x `N` direction cosine matrix [-]
    """
    r = np.array(roll)
    p = np.array(pitch)
    y = np.array(yaw)

    cr = np.cos(r)
    sr = np.sin(r)
    cp = np.cos(p)
    sp = np.sin(p)
    cy = np.cos(y)
    sy = np.sin(y)

    dcm = np.zeros((3, 3, len(roll)))

    if order == "body2nav":
        dcm[0, 0, :] = cp * cy
        dcm[0, 1, :] = -cr * sy + sr * sp * cy
        dcm[0, 2, :] = sr * sy + cr * sp * cy
        dcm[1, 0, :] = cp * sy
        dcm[1, 1, :] = cr * cy + sr * sp * sy
        dcm[1, 2, :] = -sr * cy + cr * sp * sy
        dcm[2, 0, :] = -sp
        dcm[2, 1, :] = sr * cp
        dcm[2, 2, :] = cr * cp
    else:
        raise ValueError(f"DCM rotation {order} order not defined")
    if len(roll) == 1:
        return dcm[:, :, 0]
    else:
        return dcm


def read_flight_data(flight_nums, verbose=False):
    """
        Read h5 flight data and convert it to a pandas dataframe

        Arguments:
        - `flight_number` : number of the flight we want to convert to a dataframe
        - `verbose` : print keys with NaNs and infos

        Returns:
        - `df` : pandas dataframe containing data from the flight
    """
    for flight_num in flight_nums:
        file_path = "datasets/data/Flt100{}_train.h5".format(flight_num)
        h5 = h5py.File(file_path, 'r')

        df = pd.DataFrame()
        for key in h5.keys():
            data = h5[key]
            if data.shape != ():
                df[key] = data[:]
                if df[key].isnull().any() & verbose:
                    print("{} contains NaNs".format(key))

        datafields = pd.read_csv("datasets/fields_sgl_2020.csv", header=None).iloc[:, 0].tolist()
        df = df.reindex(columns=datafields)
        df = df.sort_values(by=['tt'])
        df.index = df['tt']
        df.index.name = 'Time (s)'

        WGS_to_UTC = Transformer.from_crs(crs_from=4326,  # EPSG:4326 World Geodetic System 1984, https://epsg.io/4326
                                          crs_to=32618)  # EPSG:32618 WGS 84/UTM zone 18N, https://epsg.io/32618

        # Transfom (LAT, LONG) -> (X_UTM, Y_UTM)
        UTM_X_pyproj, UTM_Y_pyproj = WGS_to_UTC.transform(df.lat.values,
                                                          df.lon.values)

        print("Check if the converted coordinates and the dataset coordinates are equal (+/- 1.4cm) : ",
              all(np.sqrt((df.utm_x - UTM_X_pyproj) ** 2 + (df.utm_y - UTM_Y_pyproj) ** 2) < 0.014))

        df.to_hdf("./datasets/data/processed/Flt_data.h5", key="Flt100{}".format(flight_num))
