import matplotlib.pyplot as plt
import scipy
from scipy.signal import butter, filtfilt
from Math import *


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
    plt.figure()
    plt.xlabel("time [min]")
    plt.ylabel("magnetic field [nT]")
    if detrend_data:
        mag = detrend(mag, type=detrend_type)
    plt.plot(tt, mag)
    # plt.show()


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


def calculate_error(mag_comp, mag_true, detrend_type="linear"):
    mag_comp = detrend(mag_comp, type=detrend_type)
    mag_true = detrend(mag_true, type=detrend_type)
    return np.round(np.mean(abs(mag_comp - mag_true)), decimals=2)


def compute_std_delta_mag(mag_output, mag_1_c):  # Standard deviation of magnetic signal error
    errors = mag_output - mag_1_c
    mu_delta_mag = np.mean(errors)
    sigma_delta_mag = np.sqrt(np.mean((errors - mu_delta_mag) ** 2))
    sigma_delta_mag = round(sigma_delta_mag, 2)
    return sigma_delta_mag


def compute_improvement_ratio(sigma_uc, sigma_c):  # Improvement Ratio

    return sigma_uc / sigma_c


def compute_snr(sigma_magtruth, sigma_delta_mag):  # SNR, Signal-to-Noise Ratio
    return sigma_magtruth / sigma_delta_mag

# if __name__ == '__main__':
# flight = "Flt1006"
# df_flight_path = "datasets/dataframes/df_flight.csv"
# df_flight = pd.read_csv(df_flight_path)
# xyz = get_XYZ(flight, df_flight)
# print(xyz.keys())
#
# map_name = "Eastern_395"  # select map, full list in df_map
# df_options = [55770.0, 56609.0]
# line = "1006.08"  # select flight line (row) from df_options
# ind = get_ind(xyz, tt_lim=[df_options[0], df_options[1]])  # get Boolean indices
# print("ind:{}-{}".format(df_options[0], df_options[1]))
#
# TL_i = 5
# df_cal_path = "datasets/dataframes/df_cal.csv"
# df_cal = pd.read_csv(df_cal_path)
# TL_ind = get_ind(xyz, tt_lim=[df_cal['t_start'][TL_i], df_cal['t_end'][TL_i]])
# print("TL_ind:{}-{}".format(df_cal['t_start'][TL_i], df_cal['t_end'][TL_i]))
#
# p1 = plot_mag(xyz, ind=ind, use_mags=['mag_1_uc', 'mag_4_uc', 'mag_5_uc'], detrend_data=True)
# p2 = plot_mag(xyz, ind=ind, use_mags=['flux_d'], detrend_data=True, vec_terms=True)
# p2.show()
#
# lpf = get_bpf(pass1=0.0, pass2=0.2, fs=10.0)
# lpf_sig = -bpf_data(xyz['cur_strb'][TL_ind], bpf=lpf)  # apply low-pass filter, sign switched for easier comparison
# plt.figure()
# plt.plot(xyz['tt'][TL_ind], xyz['cur_strb'][TL_ind])
# plt.figure()
# plt.plot(xyz['tt'][TL_ind], lpf_sig)
# plt.show()
