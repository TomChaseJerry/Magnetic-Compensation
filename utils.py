import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy.signal import butter, filtfilt
import numpy as np
import random
import torch
import os


def set_seed(seed=2026):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def get_bpf(pass1=0.2, pass2=0.8, fs=10.0, pole=4):
    """
    get_bpf(; pass1 = 0.1, pass2 = 0.9, fs = 10.0, pole::Int = 4)

    Create a Butterworth bandpass (or low-pass or high-pass) filter object. Set
    `pass1 = -1` for low-pass filter or `pass2 = -1` for high-pass filter.

    **Arguments:**
    - `pass1`: (optional) first  passband frequency [Hz]
    - `pass2`: (optional) second passband frequency [Hz]
    - `fs`:    (optional) sampling frequency [Hz]
    - `pole`:  (optional) number of poles for Butterworth filter

    **Returns:**
    - `bpf`: filter object
    """
    nyquist = fs / 2  # Nyquist frequency

    # Determine filter type and cutoff frequencies
    if 0 < pass1 < nyquist and 0 < pass2 < nyquist:
        # Bandpass filter
        btype = 'bandpass'
        cutoff = [pass1 / nyquist, pass2 / nyquist]
    elif (pass1 <= 0 or pass1 >= nyquist) and 0 < pass2 < nyquist:
        # Lowpass filter
        btype = 'lowpass'
        cutoff = pass2 / nyquist
    elif 0 < pass1 < nyquist and (pass2 <= 0 or pass2 >= nyquist):
        # Highpass filter
        btype = 'highpass'
        cutoff = pass1 / nyquist
    else:
        raise ValueError(f"{pass1} and {pass2} passband frequencies are invalid")

    # Design Butterworth filter
    b, a = butter(pole, cutoff, btype=btype)
    return b, a


def apply_bpf(x, bpf):
    """
        bpf_data(x::AbstractMatrix; bpf=get_bpf())

    Bandpass (or low-pass or high-pass) filter columns of matrix.

    **Arguments:**
    - `x`:   data matrix (e.g., Tolles-Lawson `A` matrix)
    - `bpf`: (optional) filter object

    **Returns:**
    - `x_f`: data matrix, filtered
    """

    x_f = np.copy(x)  # Create a deep copy of the input matrix
    if x.ndim == 1:
        if np.std(x) > np.finfo(x.dtype).eps:  # Check if std deviation is greater than machine epsilon
            x_f = filtfilt(bpf[0], bpf[1], x)  # Apply the bandpass filter
    elif x.ndim > 1:
        for i in range(x.shape[1]):  # Iterate over columns
            if np.std(x[:, i]) > np.finfo(x.dtype).eps:  # Check if std deviation is greater than machine epsilon
                x_f[:, i] = filtfilt(bpf[0], bpf[1], x[:, i])  # Apply the bandpass filter
    else:
        print("bpf_data: input's ndim cant be zero.")

    return x_f


def inverse_from_11(x, scaler):
    if hasattr(scaler, "inverse_transform"):
        return scaler.inverse_transform(x)
    if isinstance(scaler, dict) and "min" in scaler and "max" in scaler:
        xmin = scaler["min"]
        xmax = scaler["max"]
        denom = xmax - xmin
        if np.isscalar(denom) or getattr(denom, "ndim", 0) == 0:
            if denom == 0:
                denom = 1.0
        else:
            denom = np.where(denom == 0, 1.0, denom)
        return (x + 1) * 0.5 * denom + xmin
    raise TypeError("Unsupported scaler type for inverse transform.")


def mean_squared_error_np(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    diff = y_true - y_pred
    return np.mean(diff * diff)


def plot_signals(sigs, names=None, is_save=False, fs=10):
    """
    通用多曲线绘图函数

    Parameters
    ----------
    sigs : list of np.ndarray
        信号列表，例如 [sig1, sig2, sig3, ...]，每个 shape = (N,) 或 (N,1)
    names : list of str 或 None
        每条曲线的名称，如 ["raw", "filter", "compensated"]
        若为 None，则自动命名为 signal_0, signal_1, ...
    is_save : bool
        是否保存图片
    fs : float
        采样频率，用于时间轴
    """
    assert isinstance(sigs, (list, tuple)) and len(sigs) > 0, "sigs 必须是非空列表/元组"

    # 全部拉平成一维，并检查长度一致
    proc_sigs = []
    for i, s in enumerate(sigs):
        s = np.asarray(s).reshape(-1)
        proc_sigs.append(s)
        if i == 0:
            N = len(s)
        else:
            assert len(s) == N, "所有信号长度必须一致"

    # 处理名字
    if names is None:
        names = [f"signal_{i}" for i in range(len(proc_sigs))]
    else:
        assert len(names) == len(proc_sigs), "names 长度必须与 sigs 相同"

    # 时间轴
    t = np.arange(N) / fs

    plt.figure(figsize=(12, 6))

    font_candidates = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "PingFang SC",
        "WenQuanYi Zen Hei",
        "Arial Unicode MS",
    ]
    for font_name in font_candidates:
        try:
            font_path = font_manager.findfont(font_name, fallback_to_default=False)
        except Exception:
            font_path = None
        if font_path:
            plt.rcParams["font.family"] = font_name
            break

    # 使用默认颜色循环
    for s, name in zip(proc_sigs, names):
        plt.plot(
            t, s,
            label=name,
            linewidth=1,
            linestyle='-',
            alpha=0.9
        )

    plt.xticks(rotation=45)
    plt.xlabel('Time [s]', fontsize=12)
    plt.ylabel('Band-pass magnetic field [nT]', fontsize=12)

    plt.legend(fontsize=12, framealpha=1, edgecolor='black', loc='upper right')
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.gca().set_facecolor('#f5f5f5')

    for spine in plt.gca().spines.values():
        spine.set_visible(True)
        spine.set_edgecolor('gray')
        spine.set_linewidth(0.5)

    plt.tight_layout()

    if is_save:
        fname = "_".join(names)
        plt.savefig(f"./results/{fname}.png", dpi=300, bbox_inches='tight')

    plt.show()


def plot_loss_curve(epochs, train_losses, model_path):
    plt.figure(figsize=(10, 6))

    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training Loss Curve', fontsize=14)  # 英文标题
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')

    base = os.path.splitext(os.path.basename(model_path))[0]
    loss_plot_path = "./results/analysis/" + base + "_loss.png"
    plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
    print(f"Loss curve saved to: {loss_plot_path}")
    plt.close()
