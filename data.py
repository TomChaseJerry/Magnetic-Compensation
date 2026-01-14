import pandas as pd
import numpy as np
import h5py
from pyproj import Transformer
import torch
from TLmodel import caculate_coef, get_tl_data, build_A
from utils import get_bpf, plot_signals, apply_bpf


class AeroMagneticCompensationDataset:
    def __init__(self, args):
        self.args = args
        self.flights = []
        self.features = self.args.sensors
        self.bpf = get_bpf(0.2, 0.8)

        self.read()
        self.apply_TL()
        # self.precompute_data = self._precompute_data()

    def read(self):
        file_path = getattr(self.args, "path", None)
        if not file_path or not str(file_path).lower().endswith(".txt"):
            file_path = "data/data.txt"
        df_raw = pd.read_csv(file_path, sep=r"\s+", header=None, engine="python")

        df = pd.DataFrame({
            'time': df_raw.iloc[:, 4],
            'flux_b_x': df_raw.iloc[:, 10],
            'flux_b_y': df_raw.iloc[:, 11],
            'flux_b_z': df_raw.iloc[:, 12],
            'mag_1_uc': df_raw.iloc[:, 15],
            'line': df_raw.iloc[:, 16],
        })

        df = df.set_index('time', drop=True)

        cali_window = getattr(self.args, "cali", None)
        smooth_window = getattr(self.args, "smoo", None)

        def slice_window(source_df, window):
            start, end = window
            return source_df.loc[start:end].copy()

        self.cali = slice_window(df, cali_window)
        self.smoo = slice_window(df, smooth_window)
        self.flights = [self.cali, self.smoo]

    def apply_TL(self):
        tl_data = self.cali.copy()

        lambd = self.args.lambd
        A_tl, B_1_tl = get_tl_data(tl_data, self.bpf)
        self.beta = caculate_coef(B_1_tl, A_tl, lambd)

        for df in self.flights:
            df['mag_1_bpf'] = apply_bpf(df['mag_1_uc'].values, self.bpf)
            B_vector = np.vstack([df['flux_b_x'].values, df['flux_b_y'].values, df['flux_b_z'].values]).T
            A_raw = build_A(B_vector, df['mag_1_uc'])
            A_f = apply_bpf(A_raw, self.bpf)
            df['mag_1_tl'] = df['mag_1_bpf'] - np.dot(A_f, self.beta)
            column_names = [f'A{i}' for i in range(1, 19)]
            df[column_names] = A_f

            plot_signals([df['mag_1_tl'].values, df['mag_1_bpf'].values], ['mag_1_tl', 'mag_1_bpf'])
            std_raw = np.std(df['mag_1_bpf'].values, ddof=1)
            std_tl = np.std(df['mag_1_tl'].values, ddof=1)
            print("STD raw:{}; STD tl:{}; IR:{}".format(std_raw, std_tl, std_raw / std_tl))

    def _scale_to_11(self, x, scaler=None):
        x = np.asarray(x, dtype=np.float32)
        if scaler is None:
            xmin = np.min(x, axis=0)
            xmax = np.max(x, axis=0)
            denom = xmax - xmin
            if np.isscalar(denom) or getattr(denom, "ndim", 0) == 0:
                if denom == 0:
                    denom = 1.0
            else:
                denom = np.where(denom == 0, 1.0, denom)
            x_scaled = 2 * (x - xmin) / denom - 1
            scaler = {"min": xmin, "max": xmax}
            return x_scaled, scaler
        xmin = scaler["min"]
        xmax = scaler["max"]
        denom = xmax - xmin
        if np.isscalar(denom) or getattr(denom, "ndim", 0) == 0:
            if denom == 0:
                denom = 1.0
        else:
            denom = np.where(denom == 0, 1.0, denom)
        x_scaled = 2 * (x - xmin) / denom - 1
        return x_scaled

    def build_custom_dataset(self):
        window_size = self.args.win
        df = self.cali
        if df.empty:
            raise ValueError("No calibration data found to build dataset.")

        x_cols = ["mag_1_bpf"] + [f"A{i}" for i in range(1, 19)]
        missing = [c for c in x_cols + ["mag_1_tl"] if c not in df.columns]
        if missing:
            raise ValueError("Missing columns: {}".format(missing))

        x = df[x_cols].values.astype(np.float32)
        y = df["mag_1_tl"].values.astype(np.float32)

        x_scaled, self.x_scaler = self._scale_to_11(x)
        y_scaled, self.y_scaler = self._scale_to_11(y)
        if len(x) < window_size:
            raise ValueError("Not enough samples for window_size.")

        x_seq = []
        y_seq = []
        for i in range(len(x_scaled) - window_size + 1):
            x_seq.append(x_scaled[i:i + window_size].T)
            y_seq.append(y_scaled[i + window_size - 1])

        x_tensor = torch.tensor(np.stack(x_seq), dtype=torch.float32)
        y_tensor = torch.tensor(np.array(y_seq).reshape(-1, 1), dtype=torch.float32)

        split_idx = int(len(x_tensor) * 0.8)
        train_dataset = torch.utils.data.TensorDataset(x_tensor[:split_idx], y_tensor[:split_idx])
        val_dataset = torch.utils.data.TensorDataset(x_tensor[split_idx:], y_tensor[split_idx:])
        return train_dataset, val_dataset

    def build_test_dataset(self):
        window_size = self.args.win
        df = self.smoo
        if df.empty:
            raise ValueError("No smooth data found to build test dataset.")

        x_cols = ["mag_1_bpf"] + [f"A{i}" for i in range(1, 19)]
        missing = [c for c in x_cols + ["mag_1_tl"] if c not in df.columns]
        if missing:
            raise ValueError("Missing columns: {}".format(missing))

        x = df[x_cols].values.astype(np.float32)
        y = df["mag_1_tl"].values.astype(np.float32)

        if len(x) < window_size:
            raise ValueError("Not enough samples for window_size.")

        x_scaled, self.x_scaler = self._scale_to_11(x)
        y_scaled, self.y_scaler = self._scale_to_11(y)
        x_seq = []
        y_seq = []
        for i in range(len(x_scaled) - window_size + 1):
            x_seq.append(x_scaled[i:i + window_size].T)
            y_seq.append(y_scaled[i + window_size - 1])

        x_tensor = torch.tensor(np.stack(x_seq), dtype=torch.float32)
        y_tensor = torch.tensor(np.array(y_seq).reshape(-1, 1), dtype=torch.float32)
        return torch.utils.data.TensorDataset(x_tensor, y_tensor)


def create_txt():
    df = pd.read_hdf('data/Flt_data.h5', key="Flt1002")
    used_sensors = ['tt', 'mag_1_uc', 'flux_b_x', 'flux_b_y', 'flux_b_z', 'line']
    df = df[used_sensors]
    df = df[df['line'].isin([1002.02, 1002.20])]

    n_rows = len(df)
    out = np.zeros((n_rows, 17), dtype=float)
    out[:, 4] = df['tt'].to_numpy()
    out[:, 10] = df['flux_b_x'].to_numpy()
    out[:, 11] = df['flux_b_y'].to_numpy()
    out[:, 12] = df['flux_b_z'].to_numpy()
    out[:, 15] = df['mag_1_uc'].to_numpy()
    out[:, 16] = df['line'].to_numpy()

    np.savetxt('data/data.txt', out, fmt="%.2f")
