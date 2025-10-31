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
    # perform_filter = ((0 < pass1 < fs / 2) or (0 < pass2 < fs / 2))
    perform_filter = False
    if perform_filter:
        bpf = get_bpf(pass1=pass1, pass2=pass2, fs=fs, pole=pole)

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
