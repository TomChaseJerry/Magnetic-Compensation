import matplotlib.pyplot as plt
import scipy
from scipy.signal import butter, filtfilt
from Math import *
from sklearn.metrics import mean_squared_error


def get_ind(xyz, line=None, ind=None, lines=(), tt_lim=(), splits=(1,)):
    """
        get_ind(tt::Vector, line::Vector;
                ind    = trues(length(tt)),
                lines  = (),
                tt_lim = (),
                splits = (1))

    Get BitVector of indices for further analysis from specified indices (subset),
    lines, and/or time range. Any or all of these may be used. Defaults to use all
    indices, lines, and times.

    **Arguments:**
    - `tt`:     time [s]
    - `line`:   line number(s)
    - `ind`:    (optional) selected data indices
    - `lines`:  (optional) selected line number(s)
    - `tt_lim`: (optional) end time limit or length-`2` start & end time limits (inclusive) [s]
    - `splits`: (optional) data splits, must sum to 1

    **Returns:**
    - `ind`: BitVector (or tuple of BitVector) of selected data indices
    """

    line = xyz.get('line')
    tt = xyz['tt']
    N = len(tt)
    if ind is None:
        ind = np.ones(N, dtype=bool)

    assert np.isclose(sum(splits), 1), f"sum of splits = {sum(splits)} ≠ 1"
    assert len(tt_lim) <= 2, f"length of tt_lim = {len(tt_lim)} > 2"
    assert len(splits) <= 3, f"number of splits = {len(splits)} > 3"

    if isinstance(ind, np.ndarray) and ind.dtype == bool:
        ind_ = ind.copy()
    elif ind is not None:
        ind_ = np.isin(np.arange(N), ind)
    else:
        ind_ = np.ones(N, dtype=bool)

    if len(lines) > 0:
        ind = np.zeros(N, dtype=bool)
        for l in lines:
            ind |= (line == l)
    else:
        ind = np.ones(N, dtype=bool)

    if len(tt_lim) == 1:
        ind &= (tt <= min(tt_lim))
    elif len(tt_lim) == 2:
        ind &= (tt >= min(tt_lim)) & (tt <= max(tt_lim))

    ind &= ind_

    if np.sum(ind) == 0:
        print("ind contains all falses")

    if len(splits) == 1:
        return ind
    elif len(splits) == 2:
        split1_end = round(np.sum(ind) * splits[0])
        ind1 = ind & (np.cumsum(ind) <= split1_end)
        ind2 = ind & ~ind1
        return ind1, ind2
    elif len(splits) == 3:
        split1_end = round(np.sum(ind) * splits[0])
        split2_end = round(np.sum(ind) * (splits[0] + splits[1]))
        ind1 = ind & (np.cumsum(ind) <= split1_end)
        ind2 = ind & (np.cumsum(ind) <= split2_end) & ~ind1
        ind3 = ind & ~(ind1 | ind2)
        return ind1, ind2, ind3


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
    b, a = butter(pole, cutoff, btype=btype)
    return b, a


def bpf_data(x, bpf=None):
    """
        bpf_data(x::AbstractMatrix; bpf=get_bpf())

    Bandpass (or low-pass or high-pass) filter columns of matrix.

    **Arguments:**
    - `x`:   data matrix (e.g., Tolles-Lawson `A` matrix)
    - `bpf`: (optional) filter object

    **Returns:**
    - `x_f`: data matrix, filtered
    """

    if bpf is None:
        bpf = get_bpf()  # Use the default bandpass filter if none is provided

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


def detrend(x, type="linear"):  # type{linear/constant}
    return scipy.signal.detrend(x, type=type)


def plot_mag(
        xyz,
        use_mags,
        ind=None,
        detrend_data=False,
        vec_terms=False,
        ylim=None,
        dpi=150,
        save_plot=False,
        plot_png="scalar_mags.png"
):
    """
    plot_mag(xyz::XYZ;
             ind                       = trues(xyz.traj.N),
             detrend_data::Bool        = false,
             use_mags::Vector{Symbol}  = [:all_mags],
             vec_terms::Vector{Symbol} = [:all],
             ylim::Tuple               = (),
             dpi::Int                  = 200,
             show_plot::Bool           = true,
             save_plot::Bool           = false,
             plot_png::String          = "scalar_mags.png")

    Plot scalar or vector (fluxgate) magnetometer data from a given flight test.

    **Arguments:**
    - `xyz`:          `XYZ` flight data struct
    - `ind`:          (optional) selected data indices
    - `detrend_data`: (optional) if true, detrend plot data
    - `use_mags`:     (optional) scalar or vector (fluxgate) magnetometers to plot {`:all_mags`, `:comp_mags` or `:mag_1_c`, `:mag_1_uc`, `:flux_a`, etc.}
        - `:all_mags`  = all provided scalar magnetometer fields (e.g., `:mag_1_c`, `:mag_1_uc`, etc.)
        - `:comp_mags` = provided compensation(s) between `:mag_1_uc` & `:mag_1_c`, etc.
    - `vec_terms`:    (optional) vector magnetometer (fluxgate) terms to plot {`:all` or `:x`,`:y`,`:z`,`:t`}
    - `ylim`:         (optional) length-`2` plot `y` limits (`ymin`,`ymax`) [nT]
    - `dpi`:          (optional) dots per inch (image resolution)
    - `show_plot`:    (optional) if true, show `p1`
    - `save_plot`:    (optional) if true, save `p1` as `plot_png`
    - `plot_png`:     (optional) plot file name to save (`.png` extension optional)

    **Returns:**
    - `p1`: plot of scalar or vector (fluxgate) magnetometer data
    """

    if ind is None:
        ind = np.ones(xyz['N'], dtype=bool)

    tt = (xyz['tt'][ind] - xyz['tt'][ind][0]) / 60
    xlab = "time [min]"

    fields = xyz.keys()
    list_c = [f"mag_{i}_c" for i in range(1, 6 + 1)]
    list_uc = [f"mag_{i}_uc" for i in range(1, 6 + 1)]
    mags_c = [field for field in list_c if field in fields]
    mags_uc = [field for field in list_uc if field in fields]
    mags_all = mags_c + mags_uc

    ylab = ""
    plt.figure(dpi=dpi)

    if ylim:
        plt.ylim(ylim)
    plt.xlabel(xlab)
    plt.ylabel(ylab)

    if "comp_mags" in use_mags:
        ylab = "magnetic field error [nT]"
        plt.ylabel(ylab)
        for i, mag_c in enumerate(mags_c):
            mag_uc = mags_uc[i]
            val = getattr(xyz, mag_uc)[ind] - getattr(xyz, mag_c)[ind]
            if detrend_data:
                val = detrend(val)
            plt.plot(tt, val, label=f"mag_{i + 1} comp")
            print(f"==== mag_{i + 1} comp ====")
            print(f"avg comp = {round(np.mean(val), 3)} nT")
            print(f"std dev  = {round(np.std(val), 3)} nT")
    elif vec_terms:
        vec = ["_x", "_y", "_z", "_t"]
        ylab = "magnetic field [nT]"
        if detrend_data:
            ylab = f"detrended {ylab}"
        plt.ylabel(ylab)

        for use_mag in use_mags:
            for vec_term in vec:
                val = xyz.get(use_mag + vec_term)[ind]
                if detrend_data:
                    val = detrend(val)
                plt.plot(tt, val, label=f"{use_mag} {vec_term}")
    else:
        ylab = "magnetic field [nT]"
        if detrend_data:
            ylab = f"detrended {ylab}"
        plt.ylabel(ylab)

        for mag in use_mags:
            val = xyz.get(mag)[ind]
            if detrend_data:
                val = detrend(val)
            plt.plot(tt, val, label=mag)

    plt.legend()
    if save_plot:
        plt.savefig(plot_png)

    return plt


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


def plot_model_vs_real(time_range, model_output, real_values, result_name, std, rmse, is_save=True):
    assert model_output.shape == real_values.shape, "模型输出和真实值的维度必须相同"

    start_time, end_time = time_range
    timestamps = np.linspace(0, 1, len(model_output))
    time_series = [start_time + (end_time - start_time) * t for t in timestamps]

    plt.figure(figsize=(12, 6))
    plt.plot(time_series, model_output,
             label='Model Output',
             color='blue',
             linewidth=2.5,
             linestyle='-',
             alpha=0.8)
    plt.plot(time_series, real_values,
             label='Real Values',
             color='red',
             linewidth=1.5,
             linestyle='--',
             alpha=0.9)

    plt.xticks(rotation=45)
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.title('STD:{:.2f}nT; RMSE:{:.2f}nT'.format(std, rmse), fontsize=14, pad=20)

    legend = plt.legend(fontsize=12,
                        framealpha=1,
                        edgecolor='black',
                        loc='upper right')

    # 设置网格和背景
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.gca().set_facecolor('#f5f5f5')  # 浅灰色背景

    # 调整边框
    for spine in plt.gca().spines.values():
        spine.set_visible(True)
        spine.set_edgecolor('gray')
        spine.set_linewidth(0.5)

    plt.tight_layout()

    if is_save:
        plt.savefig("./results/{}.png".format(result_name), dpi=300, bbox_inches='tight')
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


def inverse_transform(data, std):
    return data * std[1] + std[0]


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
