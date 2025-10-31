import matplotlib.pyplot as plt
import scipy
from scipy.signal import butter, sosfiltfilt
from Math import *
from sklearn.metrics import mean_squared_error


def get_bpf(pass1=0.1, pass2=0.9, fs=10.0, pole=4):
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
    sos = butter(pole, cutoff, btype=btype, output='sos')
    return sos


def bpf_data(x, sos):
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
            x_f = sosfiltfilt(sos, x, padlen=3 * max(1, len(sos) * 10), padtype='odd')  # Apply the bandpass filter
    elif x.ndim > 1:
        for i in range(x.shape[1]):  # Iterate over columns
            if np.std(x[:, i]) > np.finfo(x.dtype).eps:  # Check if std deviation is greater than machine epsilon
                x_f[:, i] = sosfiltfilt(sos, x[:, i], padlen=3 * max(1, len(sos) * 10),
                                        padtype='odd')  # Apply the bandpass filter
    else:
        print("bpf_data: input's ndim cant be zero.")

    return x_f


def detrend(x, type="linear"):  # type{linear/constant}
    return scipy.signal.detrend(x, type=type)


def plot(tt, mag, detrend_data=False, detrend_type="linear"):
    start_time = tt[0]
    end_time = tt[1]
    timestamps = np.linspace(0, 1, len(mag))
    time_series = [start_time + (end_time - start_time) * t for t in timestamps]

    plt.figure()
    plt.xlabel("time")
    plt.ylabel("magnetic field [nT]")
    if detrend_data:
        mag = detrend(mag, type=detrend_type)
    plt.plot(time_series, mag)
    plt.show()


def plot_comparison(sig1, sig2, name, is_save=True):
    '''
    **Arguments:**
    - name:   sig names eg:"[1007, filter, filter - T-L]"
    '''
    assert sig1.shape[0] == sig2.shape[0], "The dimensions of the data1 and the data2 must be the same"

    fs = 10
    N = len(sig1)
    t = np.arange(N) / fs

    plt.figure(figsize=(12, 6))
    plt.plot(t, sig1,
             label=name[1],
             color='blue',
             linewidth=1,
             linestyle='--',
             alpha=0.8)
    plt.plot(t, sig2,
             label=name[2],
             color='red',
             linewidth=1,
             linestyle='-',
             alpha=0.9)

    plt.xticks(rotation=45)
    plt.xlabel('Time [s]', fontsize=12)
    plt.ylabel('带通内磁场值 [nT]', fontsize=12)

    plt.legend(fontsize=12, framealpha=1, edgecolor='black', loc='upper right')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.gca().set_facecolor('#f5f5f5')

    for spine in plt.gca().spines.values():
        spine.set_visible(True)
        spine.set_edgecolor('gray')
        spine.set_linewidth(0.5)

    plt.tight_layout()

    if is_save:
        plt.savefig("./results/{}.png".format(name[0] + '_' + name[1] + '&' + name[2]), dpi=300, bbox_inches='tight')
    plt.show()


def min_max_normalize(array):
    min_val = np.min(array)
    max_val = np.max(array)
    normalized = (array - min_val) / (max_val - min_val)
    return normalized


def z_score_normalize(array):
    mean = np.mean(array)
    std = np.std(array)
    normalized = (array - mean) / std
    return normalized


def compute_std_dev(B_pred, B_real):  # Standard deviation of magnetic signal error
    errors = B_pred - B_real
    mu_delta_mag = np.mean(errors)
    sigma_delta_mag = np.sqrt(np.mean((errors - mu_delta_mag) ** 2))
    sigma_delta_mag = round(sigma_delta_mag, 2)
    return sigma_delta_mag


def compute_rmse(B_pred, B_real):
    error = sqrt(mean_squared_error(B_pred, B_real))
    error = round(error, 2)
    return error


def compute_improvement_ratio(sigma_uc, sigma_c):  # Improvement Ratio

    return sigma_uc / sigma_c


def compute_snr(sigma_magtruth, sigma_delta_mag):  # SNR, Signal-to-Noise Ratio
    return sigma_magtruth / sigma_delta_mag
