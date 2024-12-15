import numpy as np
from utils import get_bpf, bpf_data


def fdm(x, scheme='central'):
    """
    Finite difference method (FDM) applied to `x`.

    Arguments:
    - x:      data vector (list or NumPy array)
    - scheme: (optional) finite difference method scheme used
        - 'backward':  1st derivative 1st-order backward difference
        - 'forward':   1st derivative 1st-order forward difference
        - 'central':   1st derivative 2nd-order central difference
        - 'backward2': 1st derivative 2nd-order backward difference
        - 'forward2':  1st derivative 2nd-order forward difference
        - 'fourth':    4th derivative central difference

    Returns:
    - dif: vector of finite differences (same length as `x`)
    """
    x = np.array(x)
    N = len(x)

    if scheme == 'backward' and N > 1:
        dif_1 = x[1] - x[0]
        dif_end = x[-1] - x[-2]
        dif_mid = x[1:-1] - x[:-2]
    elif scheme == 'forward' and N > 1:
        dif_1 = x[1] - x[0]
        dif_end = x[-1] - x[-2]
        dif_mid = x[2:] - x[1:-1]
    elif scheme in ['central', 'central2'] and N > 2:
        dif_1 = x[1] - x[0]
        dif_end = x[-1] - x[-2]
        dif_mid = (x[2:] - x[:-2]) / 2
    elif scheme == 'backward2' and N > 3:
        dif_1 = x[1:3] - x[0:2]
        dif_end = (3 * x[-1] - 4 * x[-2] + x[-3]) / 2
        dif_mid = (3 * x[2:-1] - 4 * x[1:-2] + x[:-3]) / 2
    elif scheme == 'forward2' and N > 3:
        dif_1 = (-x[2] + 4 * x[1] - 3 * x[0]) / 2
        dif_end = x[-2:] - x[-3:-1]
        dif_mid = (-x[3:] + 4 * x[2:-1] - 3 * x[1:-2]) / 2
    elif scheme in ['fourth', 'central4'] and N > 4:
        dif_1 = np.zeros(2)
        dif_end = np.zeros(2)
        dif_mid = (x[:-4] - 4 * x[1:-3] + 6 * x[2:-2] - 4 * x[3:-1] + x[4:]) / 16
    else:
        return np.zeros_like(x)

    dif = np.concatenate([np.atleast_1d(dif_1), dif_mid, np.atleast_1d(dif_end)])
    return dif


def create_TL_A(Bx, By, Bz,
                Bt=None, terms=("permanent", "induced", "eddy"),
                Bt_scale=50000, return_B=False):
    """
    Create Tolles-Lawson `A` matrix using vector magnetometer measurements.
Optionally returns the magnitude & derivatives of total field.

    Args:
        Bx, By, Bz: vector magnetometer measurements [nT].
        Bt: magnitude of vector magnetometer measurements or scalar magnetometer measurements for modified Tolles-Lawson [nT]
        terms: Tolles-Lawson terms to use {`:permanent`,`:induced`,`:eddy`,`:bias`}
        Bt_scale: scaling factor for induced & eddy current terms [nT]
        return_B: If True, also return Bt and B_dot.

    Returns:
        A: Tolles-Lawson A matrix.
        Bt: if `return_B = true`, magnitude of total field measurements [nT]
        B_dot: if `return_B = true`, finite differences of total field vector [nT]

    """
    # Calculate Bt if not provided
    if Bt is None:
        Bt = np.sqrt(Bx ** 2 + By ** 2 + Bz ** 2)

    # Normalize vectors
    Bx_hat, By_hat, Bz_hat = Bx / Bt, By / Bt, Bz / Bt

    # Finite differences
    Bx_dot, By_dot, Bz_dot = fdm(Bx), fdm(By), fdm(Bz)

    # Induced field terms
    Bx_hat_Bx = Bx_hat * Bx / Bt_scale
    Bx_hat_By = Bx_hat * By / Bt_scale
    Bx_hat_Bz = Bx_hat * Bz / Bt_scale
    By_hat_By = By_hat * By / Bt_scale
    By_hat_Bz = By_hat * Bz / Bt_scale
    Bz_hat_Bz = Bz_hat * Bz / Bt_scale

    # Eddy current terms
    Bx_hat_Bx_dot = Bx_hat * Bx_dot / Bt_scale
    Bx_hat_By_dot = Bx_hat * By_dot / Bt_scale
    Bx_hat_Bz_dot = Bx_hat * Bz_dot / Bt_scale
    By_hat_Bx_dot = By_hat * Bx_dot / Bt_scale
    By_hat_By_dot = By_hat * By_dot / Bt_scale
    By_hat_Bz_dot = By_hat * Bz_dot / Bt_scale
    Bz_hat_Bx_dot = Bz_hat * Bx_dot / Bt_scale
    Bz_hat_By_dot = Bz_hat * By_dot / Bt_scale
    Bz_hat_Bz_dot = Bz_hat * Bz_dot / Bt_scale

    # Build matrix A
    A = np.empty((len(Bt), 0))
    if "permanent" in terms:
        A = np.column_stack((A, Bx_hat, By_hat, Bz_hat))
    if "induced" in terms:
        A = np.column_stack((A, Bx_hat_Bx, Bx_hat_By, Bx_hat_Bz,
                             By_hat_By, By_hat_Bz, Bz_hat_Bz))
    if "eddy" in terms:
        A = np.column_stack((A, Bx_hat_Bx_dot, Bx_hat_By_dot, Bx_hat_Bz_dot,
                             By_hat_Bx_dot, By_hat_By_dot, By_hat_Bz_dot,
                             Bz_hat_Bx_dot, Bz_hat_By_dot, Bz_hat_Bz_dot))

    if return_B:
        B_dot = np.column_stack((Bx_dot, By_dot, Bz_dot))
        return A, Bt, B_dot
    else:
        return A


def linreg(y, x, lambd=0):
    """
        linreg(y, x; λ=0)

    Linear regression with data matrix.

    **Arguments:**
    - `y`: length-`N` observed data vector
    - `x`: `N` x `Nf` input data matrix (`Nf` is number of features)
    - `λ`: (optional) ridge parameter

    **Returns:**
    - `coef`: linear regression coefficients
    """
    I = np.identity(x.shape[1])  # Identity matrix of the same size as x's number of columns
    coef = np.linalg.solve(x.T @ x + lambd * I, x.T @ y)  # Solve for coefficients
    return coef


def create_TL_coef(Bx, By, Bz, Bt, B, lambd=0.0, terms=("permanent", "induced", "eddy"),
                   pass1=0.1, pass2=0.9, fs=10.0, pole=4, trim=20,
                   Bt_scale=50000, return_var=False):
    """
    Create Tolles-Lawson coefficients using vector & scalar magnetometer measurements with a bandpass, low-pass, or high-pass filter.

    Args:
        - `Bx`,`By`,`Bz`: vector magnetometer measurements [nT]
        - `B`:            scalar magnetometer measurements [nT]
        - `Bt`:           (optional) magnitude of vector magnetometer measurements or scalar magnetometer measurements for modified Tolles-Lawson [nT]
        - `lambd`:            (optional) ridge parameter
        - `terms`:        (optional) Tolles-Lawson terms to use {`:permanent`,`:induced`,`:eddy`,`:bias`}
        - `pass1`:        (optional) first passband frequency [Hz]
        - `pass2`:        (optional) second passband frequency [Hz]
        - `fs`:           (optional) sampling frequency [Hz]
        - `pole`:         (optional) number of poles for Butterworth filter
        - `trim`:         (optional) number of elements to trim after filtering
        - `Bt_scale`:     (optional) scaling factor for induced & eddy current terms [nT]
        - `return_var`:   (optional) if true, also return `B_var`

    Returns:
        - `coef`:  Tolles-Lawson coefficients
        - `B_var`: if `return_var = true`, fit error variance
    """

    # Create filter
    perform_filter = ((0 < pass1 < fs / 2) or (0 < pass2 < fs / 2))
    if perform_filter:
        bpf = get_bpf(pass1=pass1, pass2=pass2, fs=fs, pole=pole)
    else:
        print("Not filtering (or trimming) Tolles-Lawson data.")
        bpf = None

    # Create Tolles-Lawson `A` matrix
    A = create_TL_A(Bx, By, Bz, Bt=Bt, terms=terms, Bt_scale=Bt_scale)

    # Filter columns of A and B + trim edges
    if perform_filter:
        A = bpf_data(A, bpf=bpf)[trim:-trim, :]
        B = bpf_data(B, bpf=bpf)[trim:-trim]

    # Linear regression to get coefficients
    coef = linreg(B, A, lambd=lambd).flatten()

    if return_var:
        B_var = np.var(B - np.dot(A, coef))
        print(f"TL fit error variance: {B_var}")
        return coef, B_var
    else:
        return coef

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
# TL_i = 5  # 1006.04
# df_cal_path = "datasets/dataframes/df_cal.csv"
# df_cal = pd.read_csv(df_cal_path)
# TL_ind = get_ind(xyz, tt_lim=[df_cal['t_start'][TL_i], df_cal['t_end'][TL_i]])
# print("TL_ind:{}-{}".format(df_cal['t_start'][TL_i], df_cal['t_end'][TL_i]))
#
# lambd = 0.025  # ridge parameter for ridge regression
# use_vec = "flux_d"  # selected vector (flux) magnetometer
# use_sca = "mag_4_uc"
# terms_A = ["permanent", "induced", "eddy"]  # Tolles-Lawson terms to use
# Bx = xyz.get(use_vec + '_x')  # load Flux D data
# By = xyz.get(use_vec + '_y')
# Bz = xyz.get(use_vec + '_z')
# Bt = xyz.get(use_vec + '_t')
# TL_d_4 = create_TL_coef(Bx[TL_ind], By[TL_ind], Bz[TL_ind], Bt[TL_ind], xyz.get(use_sca)[TL_ind], lambd=lambd,
#                         terms=terms_A)  # coefficients with Flux D & Mag 4
# print(TL_d_4.shape)
# A = create_TL_A(Bx[ind], By[ind], Bz[ind], Bt=Bt[ind])  # Tolles-Lawson `A` matrix for Flux D
# print(A.shape)
# mag_1_sgl = xyz['mag_1_c'][ind]  # professionally compensated tail stinger, Mag 1
# mag_4_uc = xyz['mag_4_uc'][ind]  # uncompensated Mag 4
# mag_4_c = mag_4_uc - detrend(np.dot(A, TL_d_4), type='linear')  # compensated Mag 4
# print("mag error:{}nT".format(calculate_error(mag_4_c, mag_1_sgl)))
# tt = (xyz['tt'][ind] - xyz['tt'][ind][1]) / 60
# plot(tt, mag_1_sgl, detrend_data=True, detrend_type="linear")
# plot(tt, mag_4_uc, detrend_data=True, detrend_type="linear")
# plot(tt, mag_4_c, detrend_data=True, detrend_type="linear")
# plt.show()
