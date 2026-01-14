import numpy as np
from utils import apply_bpf, get_bpf, plot_signals


def get_tl_data(df, bpf, trim=20):
    B_vector = np.vstack([df['flux_b_x'].values, df['flux_b_y'].values, df['flux_b_z'].values]).T

    B_1 = np.asarray(df['mag_1_uc'].values).reshape(-1)
    B_1_f = apply_bpf(B_1, bpf)

    A_raw = build_A(B_vector, B_1)
    A = apply_bpf(A_raw, bpf)

    if trim:
        A = A[trim:-trim, :]
        B_1_f = B_1_f[trim:-trim]

    return A, B_1_f


def fdm(x, fs=10.0, scheme='central'):
    """
    Finite difference method (FDM) applied to `x`.

    Arguments:
    - x:      data vector (list or NumPy array)
    Returns:
    - dif: vector of finite differences (same length as `x`)
    """
    x = np.asarray(x)
    N = len(x)
    if N < 2:
        return np.zeros_like(x)
    dt = 1.0 / fs
    dif = np.zeros(N)
    if scheme == 'central' and N >= 3:
        dif[1:-1] = (x[2:] - x[:-2]) / (2.0 * dt)
        dif[0] = (x[1] - x[0]) / dt
        dif[-1] = (x[-1] - x[-2]) / dt
    else:
        dif[0] = (x[1] - x[0]) / dt
        dif[1:] = (x[1:] - x[:-1]) / dt
    return dif


def build_A(B_vector, B_scalar, fs=10.0):
    """
    1) 18:
         9 direction cosine basis terms + 9 of their first-order derivatives with respect to time:
           base(9):  cx, cy, cz, cx*cx, cx*cy, cx*cz, cy*cy, cy*cz, cz*cz
           deriv(9): d/dt[base(9)]（use fdm）
        cx = Bx/|B|, cy = By/|B|, cz = Bz/|B|
    """
    B = np.asarray(B_vector, dtype=float)
    if B.ndim != 2 or B.shape[1] < 3:
        raise ValueError("B_vector must be Nx3 or Nx>=3 (Bx, By, Bz[, ...]).")
    Bx, By, Bz = B[:, 0], B[:, 1], B[:, 2]
    Bt = np.sqrt(Bx * Bx + By * By + Bz * Bz)

    bpf = get_bpf(0.2, 0.8)
    H = apply_bpf(B_scalar, bpf)

    # Prevent division by zero
    eps = np.finfo(float).eps
    Bt_safe = np.where(Bt > eps, Bt, eps)

    cx = Bx / Bt_safe
    cy = By / Bt_safe
    cz = Bz / Bt_safe

    g1 = cx
    g2 = cy
    g3 = cz
    g4 = cx * cx * H
    g5 = 2 * cx * cy * H
    g6 = 2 * cx * cz * H
    g7 = cy * cy * H
    g8 = 2 * cy * cz * H
    g9 = cz * cz * H

    cx_ = fdm(g1, fs=fs)
    cy_ = fdm(g2, fs=fs)
    cz_ = fdm(g3, fs=fs)

    g10 = cx * cx_ * H
    g11 = cx * cy_ * H
    g12 = cx * cz_ * H
    g13 = cy * cx_ * H
    g14 = cy * cy_ * H
    g15 = cy * cz_ * H
    g16 = cz * cx_ * H
    g17 = cz * cy_ * H
    g18 = cz * cz_ * H

    base9 = [g1, g2, g3, g4, g5, g6, g7, g8, g9]
    deriv9 = [g10, g11, g12, g13, g14, g15, g16, g17, g18]

    A = np.column_stack(base9 + deriv9)
    return A


def caculate_coef(y, X, lambd=0):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    if X.shape[0] < X.shape[1]:
        raise ValueError("Insufficient data for fitting: rows < columns")
    XtX = np.dot(X.T, X)
    XtY = np.dot(X.T, y)
    if lambd > 0:
        XtX = XtX + lambd * np.eye(X.shape[1])
    coef = np.linalg.solve(XtX, XtY)
    return coef.reshape(-1)
