import pandas as pd
import numpy as np
import h5py
from pyproj import Transformer
import torch
from utils import get_ind, bpf_data, detrend
from TL_model import fdm, create_TL_coef, create_TL_A
from Math import sortperm, deg2rad
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from utils import get_bpf


class MagCompData:
    def __init__(self, config):
        self.config = config
        self.flights = {}
        self.beta_tl_4 = None
        self.beta_tl_5 = None
        self.Read()
        self.create_tl_comp_data()
        self.create_igrf_data()
        self.create_model_y()

    def Read(self):
        for flight_num in self.config.flight_nums:
            df = pd.read_hdf("./datasets/data/processed/Flt_data.h5", key="Flt100{}".format(flight_num))
            self.flights[flight_num] = df

    def create_tl_comp_data(self):
        mask_tl = (self.flights[2].line == 1002.20)
        tl_data = self.flights[2][mask_tl]
        lambd = 0.025  # ridge parameter for ridge regression
        terms_A = ["permanent", "induced", "eddy"]  # Tolles-Lawson terms to use
        self.beta_tl_3 = create_TL_coef(tl_data['flux_b_x'], tl_data['flux_b_y'], tl_data['flux_b_z'], tl_data['flux_b_t'],
                                        tl_data['mag_3_uc'], lambd=lambd, terms=terms_A)
        self.beta_tl_4 = create_TL_coef(tl_data['flux_b_x'], tl_data['flux_b_y'], tl_data['flux_b_z'], tl_data['flux_b_t'],
                                        tl_data['mag_4_uc'], lambd=lambd, terms=terms_A)
        self.beta_tl_5 = create_TL_coef(tl_data['flux_b_x'], tl_data['flux_b_y'], tl_data['flux_b_z'], tl_data['flux_b_t'],
                                        tl_data['mag_5_uc'], lambd=lambd, terms=terms_A)

        for flight_num in self.config.flight_nums:
            A = create_TL_A(self.flights[flight_num]['flux_b_x'], self.flights[flight_num]['flux_b_y'],
                            self.flights[flight_num]['flux_b_z'], self.flights[flight_num]['flux_b_t'])
            self.flights[flight_num]['mag_3_c'] = self.flights[flight_num]['mag_3_uc'].values - np.dot(A, self.beta_tl_3) \
                                                  + np.mean(np.dot(A, self.beta_tl_3))
            self.flights[flight_num]['mag_4_c'] = self.flights[flight_num]['mag_4_uc'].values - np.dot(A, self.beta_tl_4) \
                                                  + np.mean(np.dot(A, self.beta_tl_4))
            self.flights[flight_num]['mag_5_c'] = self.flights[flight_num]['mag_5_uc'].values - np.dot(A, self.beta_tl_5) \
                                                  + np.mean(np.dot(A, self.beta_tl_5))
            for i in range(A.shape[1]):
                self.flights[flight_num]['A{}'.format(i)] = A[:, i].tolist()

    def create_igrf_data(self):
        for flight_num in self.config.flight_nums:
            df = self.flights[flight_num]
            df['igrf'] = df['mag_1_dc'] - df['mag_1_igrf']

    def create_model_y(self):
        for flight_num in self.config.flight_nums:
            df = self.flights[flight_num]
            df['model_y'] = df['mag_1_c'] - df['igrf'] - df['diurnal']


class MagCompDataset(Dataset):
    def __init__(self, data, mode):
        self.flights = data.flights
        self.mode = mode
        self.config = data.config
        self.features = self.config.selected_features
        self.x = None
        self.y = None
        self.std_y = StandardScaler()
        if self.config.test_is_Flt1007:
            self.train_lines = self.flights[2].line.unique().tolist() + self.flights[3].line.unique().tolist() + self.flights[
                4].line.unique().tolist() + self.flights[6].line.unique().tolist()
            self.test_lines = self.flights[7].line.unique().tolist()
        else:
            self.train_lines = self.flights[2].line.unique().tolist() + self.flights[4].line.unique().tolist() + self.flights[
                6].line.unique().tolist() + self.flights[7].line.unique().tolist()
            self.test_lines = self.flights[3].line.unique().tolist()
        self.lines = self.train_lines if mode == 'train' else self.test_lines
        self.length = None
        self.data_processing()

    def data_processing(self):
        x = pd.DataFrame()
        for flight_num in self.config.flight_nums:
            # IGRF and diurnal correct
            df = self.flights[flight_num]
            df['mag_4_c'] = df['mag_4_c'] - df['diurnal'] - df['igrf']
            df['mag_5_c'] = df['mag_5_c'] - df['diurnal'] - df['igrf']

            # concat
            df = df[self.features]
            x = pd.concat([x, df], ignore_index=True, axis=0)

        x = x.loc[x.line.isin(self.lines)]

        scaler = StandardScaler()
        y = x.loc[:, 'model_y']
        x = x.drop(columns=['line', 'model_y'])
        self.x = torch.tensor(scaler.fit_transform(x.to_numpy()), dtype=torch.float32)
        if self.mode == 'train':
            self.y = torch.tensor(self.std_y.fit_transform(y.to_numpy().reshape(-1, 1)), dtype=torch.float32)
            self.length = self.y.size()[0]
        else:
            self.y = y.to_numpy().reshape(-1, 1)
            self.length = self.y.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return self.length

    def get_std_y(self):
        return self.std_y


class SequentialDataset(MagCompDataset):
    def __init__(self, data, mode):
        super().__init__(data, mode)
        self.x = self.x.unfold(0, self.config.seq_len, 1)
        self.y = self.y[self.config.seq_len - 1:]
        self.length = self.length - self.config.seq_len + 1


class PINNsDataset(Dataset):
    def __init__(self, data, mode):
        self.flights = data.flights
        self.mode = mode
        self.config = data.config
        self.features = self.config.selected_features
        self.x = None
        self.y = None
        self.A = None
        self.mag4uc = None
        self.mag4c = None
        self.beta_tl_4 = data.beta_tl_4.reshape(1, -1)
        self.beta_tl_5 = data.beta_tl_5.reshape(1, -1)
        self.std_y = StandardScaler()
        self.std_mag4c = StandardScaler()
        if self.config.test_is_Flt1007:
            self.train_lines = self.flights[2].line.unique().tolist() + self.flights[3].line.unique().tolist() + self.flights[
                4].line.unique().tolist() + self.flights[6].line.unique().tolist()
            self.test_lines = self.flights[7].line.unique().tolist()
        else:
            self.train_lines = self.flights[2].line.unique().tolist() + self.flights[4].line.unique().tolist() + self.flights[
                6].line.unique().tolist() + self.flights[7].line.unique().tolist()
            self.test_lines = self.flights[3].line.unique().tolist()
        self.lines = self.train_lines if mode == 'train' else self.test_lines
        self.length = None
        self.data_processing()

    def data_processing(self):
        x, mag4uc, mag4c = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        for flight_num in self.config.flight_nums:
            # IGRF and diurnal correct
            df = self.flights[flight_num]
            df['mag_4_c'] = df['mag_4_c'] - df['diurnal'] - df['igrf']
            df['mag_5_c'] = df['mag_5_c'] - df['diurnal'] - df['igrf']
            df['mag_4_uc'] = df['mag_4_c'] - df['diurnal'] - df['igrf']
            df['mag_5_uc'] = df['mag_5_c'] - df['diurnal'] - df['igrf']

            # concat
            df = df[self.features + ['mag_4_uc', 'mag_5_uc'] + ['A{}'.format(i) for i in range(18)]]
            x = pd.concat([x, df], ignore_index=True, axis=0)

        x = x.loc[x.line.isin(self.lines)]

        scaler = StandardScaler()
        y = x.loc[:, 'model_y']
        A = np.column_stack([x.loc[:, 'A{}'.format(i)] for i in range(18)])
        mag4uc = x.loc[:, 'mag_4_uc']
        mag4c = x.loc[:, 'mag_4_c']
        x = x.drop(columns=['line', 'model_y'] + ['A{}'.format(i) for i in range(18)])
        self.x = torch.tensor(scaler.fit_transform(x.to_numpy()), dtype=torch.float32)
        self.A = torch.tensor(A.reshape(-1, 18), dtype=torch.float32)
        self.mag4uc = torch.tensor(mag4uc.to_numpy().reshape(-1, 1), dtype=torch.float32)
        self.mag4c = torch.tensor(self.std_mag4c.fit_transform(mag4c.to_numpy().reshape(-1, 1)), dtype=torch.float32)
        if self.mode == 'train':
            self.y = torch.tensor(self.std_y.fit_transform(y.to_numpy().reshape(-1, 1)), dtype=torch.float32)
            self.length = self.y.size()[0]
        else:
            self.y = y.to_numpy().reshape(-1, 1)
            self.length = self.y.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.A[idx], self.mag4uc[idx], self.mag4c[idx]

    def __len__(self):
        return self.length

    def get_std_y(self):
        return self.std_y

    def get_std_mag4c(self):
        return self.std_mag4c


class PINNsSequentialDataset(PINNsDataset):
    def __init__(self, data, mode):
        super().__init__(data, mode)
        self.x = self.x.unfold(0, self.config.seq_len, 1)
        self.y = self.y[self.config.seq_len - 1:]
        self.A = self.A[self.config.seq_len - 1:]
        self.mag4uc = self.mag4uc[self.config.seq_len - 1:]
        self.mag4c = self.mag4c[self.config.seq_len - 1:]
        self.length = self.length - self.config.seq_len + 1


def read_check(xyz, field, N, silent=False):
    """
        Internal helper function to check for NaNs or missing data (returned as NaNs) in opened HDF5 file.
    Prints out warning for any field that contains NaNs.

    **Arguments:**
        - `xyz`:    opened HDF5 file
        - `field`:  data field to read
        - `N`:      number of samples (instances)
        - `silent`: (optional) if true, no print outs

    **Returns:**
        - `val`: data returned for `field`
    """

    field = str(field)  # Ensure field is a string
    if field in xyz.keys():
        val = np.array(xyz[field])  # Read the data
    else:
        val = np.full(N, np.nan)  # Create an array of NaNs

    if not silent and np.isnan(val).any():
        print("{} field contains NaNs".format(field))

    return val


def xyz_fields(flight):
    """
        Internal helper function to get field names for given SGL flight.
    - Valid for SGL flights:
        - `:Flt1001`
        - `:Flt1002`
        - `:Flt1003`
        - `:Flt1004_1005`
        - `:Flt1004`
        - `:Flt1005`
        - `:Flt1006`
        - `:Flt1007`
        - `:Flt1008`
        - `:Flt1009`
        - `:Flt1001_160Hz`
        - `:Flt1002_160Hz`
        - `:Flt2001_2017`
        - `:Flt2001`
        - `:Flt2002`
        - `:Flt2004`
        - `:Flt2005`
        - `:Flt2006`
        - `:Flt2007`
        - `:Flt2008`
        - `:Flt2015`
        - `:Flt2016`
        - `:Flt2017`

    **Arguments:**
        - `flight`: flight name (e.g., `:Flt1001`)

    **Returns:**
        - `fields`: vector of data field names (Symbols)
    """

    # Load CSV files containing field definitions
    fields20 = "./datasets/fields_sgl_2020.csv"
    fields21 = "./datasets/fields_sgl_2021.csv"
    d = {
        "fields20": pd.read_csv(fields20, header=None).squeeze("columns").astype(str).tolist(),
        "fields21": pd.read_csv(fields21, header=None).squeeze("columns").astype(str).tolist(),
        "fields160": []
    }

    if flight in d:
        return d[flight]
    # Handle specific flight cases
    elif flight in ["Flt1001", "Flt1002"]:
        # No mag_6_uc or flux_a for these flights
        exc = ["mag_6_uc", "flux_a_x", "flux_a_y", "flux_a_z", "flux_a_t"]
        return [field for field in d["fields20"] if field not in exc]

    elif flight in ["Flt1003", "Flt1004_1005", "Flt1004", "Flt1005",
                    "Flt1006", "Flt1007"]:
        # No mag_6_uc for these flights
        exc = ["mag_6_uc"]
        return [field for field in d["fields20"] if field not in exc]

    elif flight in ["Flt1008", "Flt1009"]:
        return d["fields20"]

    elif flight in ["Flt1001_160Hz", "Flt1002_160Hz"]:
        # No mag_6_uc or flux_a for these flights
        exc = ["mag_6_uc", "flux_a_x", "flux_a_y", "flux_a_z", "flux_a_t"]
        return [field for field in d["fields160"] if field not in exc]

    elif flight in ["Flt2001_2017", "Flt2001", "Flt2002", "Flt2004", "Flt2005",
                    "Flt2006", "Flt2007", "Flt2008", "Flt2015", "Flt2016", "Flt2017"]:
        return d["fields21"]

    else:
        raise ValueError(f"{flight} flight not defined")


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


def get_XYZ20(xyz_h5, flight, info=None, tt_sort=True, silent=False):
    """
    get_XYZ20(xyz_h5::String;
              info::String  = splitpath(xyz_h5)[end],
              tt_sort::Bool = true,
              silent::Bool  = false)
    Get `XYZ20` flight data from saved HDF5 file. Based on 2020 SGL data fields.

    **Arguments:**
    - `xyz_h5`:  path/name of flight data HDF5 file (`.h5` extension optional)
    - `info`:    (optional) flight data information
    - `tt_sort`: (optional) if true, sort data by time (instead of line)
    - `silent`:  (optional) if true, no print outs

    **Returns:**
    - `xyz`: `XYZ20` flight data struct
    """

    info = info or xyz_h5.split("/")[-1]

    fields = flight

    if not silent:
        print(f"Reading in XYZ20 data: {xyz_h5}")

    with h5py.File(xyz_h5, 'r') as xyz:
        N = len(xyz['tt'])
        d = {}
        ind = np.argsort(read_check(xyz, 'tt', N, silent)) if tt_sort else np.arange(N)

        for field in xyz_fields(fields):
            if field != 'ignore':
                d[field] = read_check(xyz, field, N, silent)[ind]

        d['info'] = info
        d['N'] = N

        for field in ['aux_1', 'aux_2', 'aux_3']:
            d[field] = read_check(xyz, field, N, True)

        dt = np.round(d['tt'][1] - d['tt'][0], 9) if N > 1 else 0.1

        # using [rad] exclusively
        for field in ['lat', 'lon', 'ins_roll', 'ins_pitch', 'ins_yaw',
                      'roll_rate', 'pitch_rate', 'yaw_rate']:
            d[field] = deg2rad(d.get(field, np.zeros(N)))

        # provided IGRF for convenience
        d['igrf'] = d.get('mag_1_dc', np.zeros(N)) - d.get('mag_1_igrf', np.zeros(N))

        # trajectory velocities & specific forces from position
        d['vn'] = fdm(d.get('utm_y', np.zeros(N))) / dt
        d['ve'] = fdm(d.get('utm_x', np.zeros(N))) / dt
        d['vd'] = -fdm(d.get('utm_z', np.zeros(N))) / dt
        d['fn'] = fdm(d['vn']) / dt
        d['fe'] = fdm(d['ve']) / dt
        d['fd'] = fdm(d['vd']) / dt - 9.81  # Assuming g_earth = 9.81

        # Cnb direction cosine matrix (body to navigation) from roll, pitch, yaw
        d['Cnb'] = np.zeros((3, 3, N))  # unknown
        d['ins_Cnb'] = euler2dcm(d['ins_roll'], d['ins_pitch'], d['ins_yaw'])
        d['ins_P'] = np.zeros((1, 1, N))  # unknown

        # INS velocities in NED direction
        d['ins_ve'] = -d.get('ins_vw', np.zeros(N))
        d['ins_vd'] = -d.get('ins_vu', np.zeros(N))

        # INS specific forces from measurements, rotated wander angle (CW for NED)
        ins_f = np.zeros((N, 3))
        for i in range(N):
            ins_f[i, :] = euler2dcm([0], [0], -d['ins_wander'][i]) @ \
                          np.array([d['ins_acc_x'][i], -d['ins_acc_y'][i], -d['ins_acc_z'][i]])

        d['ins_fn'], d['ins_fe'], d['ins_fd'] = ins_f[:, 0], ins_f[:, 1], ins_f[:, 2]

        # INS specific forces from finite differences
        # push!(d,:ins_fn => fdm(-d[:ins_vn]) / dt)
        # push!(d,:ins_fe => fdm(-d[:ins_ve]) / dt)
        # push!(d,:ins_fd => fdm(-d[:ins_vd]) / dt .- g_earth)

        return d


def get_XYZ21(xyz_h5, flight, info=None, tt_sort=True, silent=False):
    """
        get_XYZ21(xyz_h5::String;
                  info::String  = splitpath(xyz_h5)[end],
                  tt_sort::Bool = true,
                  silent::Bool  = false)

    Get `XYZ21` flight data from saved HDF5 file. Based on 2021 SGL data fields.

    **Arguments:**
    - `xyz_h5`:  path/name of flight data HDF5 file (`.h5` extension optional)
    - `info`:    (optional) flight data information
    - `tt_sort`: (optional) if true, sort data by time (instead of line)
    - `silent`:  (optional) if true, no print outs

    **Returns:**
    - `xyz`: `XYZ21` flight data struct
    """
    info = info or xyz_h5.split("/")[-1]

    fields = flight

    if not silent:
        print(f"Reading in XYZ21 data: {xyz_h5}")

    with h5py.File(xyz_h5, 'r') as xyz:
        N = len(xyz['tt'])
        d = {}
        ind = np.argsort(read_check(xyz, 'tt', N, silent)) if tt_sort else np.arange(N)
        for field in xyz_fields(fields):
            if field != 'ignore':
                d[field] = read_check(xyz, field, N, silent)[ind]

        d['info'] = info
        d['N'] = N

        for field in ['aux_1', 'aux_2', 'aux_3']:
            d[field] = read_check(xyz, field, N, True)

        dt = np.round(d['tt'][1] - d['tt'][0], 9) if N > 1 else 0.1

        # using [rad] exclusively
        for field in ['lat', 'lon', 'ins_roll', 'ins_pitch', 'ins_yaw']:
            d[field] = deg2rad(d.get(field, np.zeros(N)))

        # provided IGRF for convenience
        d['igrf'] = d.get('mag_1_dc', np.zeros(N)) - d.get('mag_1_igrf', np.zeros(N))

        # trajectory velocities & specific forces from position
        d['vn'] = fdm(d.get('utm_y', np.zeros(N))) / dt
        d['ve'] = fdm(d.get('utm_x', np.zeros(N))) / dt
        d['vd'] = -fdm(d.get('utm_z', np.zeros(N))) / dt
        d['fn'] = fdm(d['vn']) / dt
        d['fe'] = fdm(d['ve']) / dt
        d['fd'] = fdm(d['vd']) / dt - 9.81  # Assuming g_earth = 9.81

        # Cnb direction cosine matrix (body to navigation) from roll, pitch, yaw
        d['Cnb'] = np.zeros((3, 3, N))  # unknown
        d['ins_Cnb'] = euler2dcm(d['ins_roll'], d['ins_pitch'], d['ins_yaw'])
        d['ins_P'] = np.zeros((1, 1, N))  # unknown

        # INS velocities in NED direction
        d['ins_ve'] = -d.get('ins_vw', np.zeros(N))
        d['ins_vd'] = -d.get('ins_vu', np.zeros(N))

        # INS specific forces from measurements, rotated wander angle (CW for NED)
        ins_f = np.zeros((N, 3))
        for i in range(N):
            ins_f[i, :] = euler2dcm([0], [0], -d['ins_wander'][i]) @ \
                          np.array([d['ins_acc_x'][i], -d['ins_acc_y'][i], -d['ins_acc_z'][i]])

        d['ins_fn'], d['ins_fe'], d['ins_fd'] = ins_f[:, 0], ins_f[:, 1], ins_f[:, 2]

        # INS specific forces from finite differences
        # push!(d,:ins_fn => fdm(-d[:ins_vn]) / dt)
        # push!(d,:ins_fe => fdm(-d[:ins_ve]) / dt)
        # push!(d,:ins_fd => fdm(-d[:ins_vd]) / dt .- g_earth)

        return d


def xyz_reorient_vec(xyz):
    pass


def get_XYZ(flight, df_flight, tt_sort=True, reorient_vec=False, silent=False):
    """
    **Arguments:**
        - `flight`:    flight name (e.g., `:Flt1001`)
        - `df_flight`: lookup table (DataFrame) of flight data files
            |**Field**|**Type**|**Description**
            |:--|:--|:--
            `flight`  |`Symbol`| flight name (e.g., `:Flt1001`)
            `xyz_type`|`Symbol`| subtype of `XYZ` to use for flight data {`:XYZ0`,`:XYZ1`,`:XYZ20`,`:XYZ21`}
            `xyz_set` |`Real`  | flight dataset number (used to prevent improper mixing of datasets, such as different magnetometer locations)
            `xyz_file`|`String`| path/name of flight data CSV, HDF5, or MAT file (`.csv`, `.h5`, or `.mat` extension required)
        - `tt_sort`:      (optional) if true, sort data by time (instead of line)
        - `reorient_vec`: (optional) if true, align vector magnetometer measurements with body frame
        - `silent`:       (optional) if true, no print outs

    **Returns:**
        - `xyz`: `XYZ` flight data struct
    """

    # Find the index of the first matching flight
    ind = df_flight[df_flight['flight'] == flight].index[0]

    # Extract xyz_file and xyz_type
    xyz_file = str(df_flight.at[ind, 'xyz_file'])
    xyz_type = df_flight.at[ind, 'xyz_type']

    # Determine the function to call based on xyz_type
    if xyz_type == "XYZ20":
        xyz = get_XYZ20(xyz_file, silent=silent, flight=flight)
    elif xyz_type == "XYZ21":
        xyz = get_XYZ21(xyz_file, silent=silent, flight=flight)
    else:
        raise ValueError("{} xyz_type not defined".format(xyz_type))

    # Reorient vector if requested
    if reorient_vec:
        xyz_reorient_vec(xyz)

    print(xyz['info'], xyz.keys())
    return xyz


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


if __name__ == '__main__':
    # flight_nums = [2, 3, 4, 5, 6, 7]
    # read_flight_data(flight_nums)
    # flight_num = 3
    # df2 = pd.read_hdf("./datasets/data/processed/Flt_data.h5", key="Flt100{}".format(flight_num))
    # print(df2)

    # from config import config
    #
    # config = config()
    # data = MagCompData(config)
    # dataset_train = MagCompDataset(data, 'train')
    # dataset_test = MagCompDataset(data, 'test')
    # print(1)

    a = np.random.rand(1, 3)
    b = np.random.rand(2, 3)
    print(a)
    print(b)
    print(a + b)
