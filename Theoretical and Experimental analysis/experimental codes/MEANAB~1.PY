import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import chi2  # (not used now but you had it; safe to remove)
from scipy.linalg import expm  # for state transition

# -------------------------------
# Fixed model parameters
# -------------------------------
A = 1.0                 # amplitude
zeta = 0.2              # damping ratio
omega_n = 2 * np.pi     # natural frequency (rad/s)
omega_d = omega_n * np.sqrt(1 - zeta**2)  # damped natural frequency
phi_true = np.pi / 4    # true phase
sigma = 0.1             # noise std
T = 10.0
N = 1000
t = np.linspace(0, T, N, endpoint=False)  # 10s, 1k samples
dt = t[1] - t[0]

# Missing fractions to test (10% -> 90%)
missing_fractions = np.linspace(0.1, 0.9, 9)

# Monte Carlo runs per fraction (increase for stability)
MC = 600

# Reproducibility
rng = np.random.default_rng(42)

# -------------------------------
# Helpers
# -------------------------------
def damped_sinusoid(t, phi, A=A, zeta=zeta, omega_n=omega_n):
    """A * exp(-zeta*omega_n*t) * sin(omega_d*t + phi)."""
    omega_d = omega_n * np.sqrt(1 - zeta**2)
    return A * np.exp(-zeta * omega_n * t) * np.sin(omega_d * t + phi)

def generate_clean_signal(t):
    return damped_sinusoid(t, phi_true)

def remove_data_mar(y_ref, beta, rng):
    """
    MAR: remove floor(beta*N) points with probability ∝ |y_ref|.
    Returns a boolean mask 'is_missing' (True = missing).
    """
    Nn = y_ref.size
    k = int(np.floor(beta * Nn))
    if k <= 0:
        return np.zeros(Nn, dtype=bool)
    w = np.abs(y_ref) + 1e-9
    w = w / w.sum()
    miss_idx = rng.choice(Nn, size=k, replace=False, p=w)
    is_missing = np.zeros(Nn, dtype=bool)
    is_missing[miss_idx] = True
    return is_missing

def locf_impute(y):
    """Last Observation Carried Forward (with back-fill at start)."""
    y_imp = y.copy()
    last = np.nan
    for i in range(len(y_imp)):
        if np.isnan(y_imp[i]):
            y_imp[i] = last
        else:
            last = y_imp[i]
    if np.isnan(y_imp[0]):
        first_valid = np.flatnonzero(~np.isnan(y)).min()
        y_imp[:first_valid] = y_imp[first_valid]
    return y_imp

def linear_interpolation_impute(y):
    """Linear interpolation between known points (endpoints held constant)."""
    y_imp = y.copy()
    idx = np.arange(len(y_imp))
    mask = ~np.isnan(y_imp)
    if mask.all():
        return y_imp
    if (~mask).all():
        return np.zeros_like(y_imp)
    first_idx = idx[mask][0]
    last_idx = idx[mask][-1]
    y_imp[:first_idx] = y_imp[first_idx]
    y_imp[last_idx+1:] = y_imp[last_idx]
    y_imp[~mask] = np.interp(idx[~mask], idx[mask], y_imp[mask])
    return y_imp

def mle_phi_single_start(y_vec, t_vec, phi0):
    """Single-start MLE for phi using L-BFGS-B."""
    def obj(phi):
        y_hat = damped_sinusoid(t_vec, phi)
        r = y_vec - y_hat
        return np.dot(r, r)
    res = minimize(lambda x: obj(x[0]), x0=np.array([phi0]),
                   method='L-BFGS-B', bounds=[(-np.pi, np.pi)])
    return float(res.x[0]), float(res.fun)

def mle_phi_multistart(y_vec, t_vec, n_starts=12):
    """Multi-start wrapper to avoid poor local fits."""
    starts = np.linspace(-np.pi, np.pi, n_starts, endpoint=False)
    best_phi, best_val = None, np.inf
    for phi0 in starts:
        phi_hat, fval = mle_phi_single_start(y_vec, t_vec, phi0)
        if fval < best_val:
            best_phi, best_val = phi_hat, fval
    return best_phi

def circular_abs_error(phi_hat, phi_true):
    """
    Absolute circular error in radians: |wrapToPi(phi_hat - phi_true)|.
    """
    err = (phi_hat - phi_true + np.pi) % (2*np.pi) - np.pi
    return np.abs(err)

# -------------------------------
# Kalman Filter / RTS Smoother (for imputation)
# State x = [y; ydot], measurement y_k = [1 0] x_k
# -------------------------------
def _ss_matrices(zeta, omega_n, dt):
    Ac = np.array([[0.0, 1.0],
                   [-(omega_n**2), -2.0*zeta*omega_n]])
    F = expm(Ac * dt)
    H = np.array([[1.0, 0.0]])
    return F, H

def kalman_filter_impute(y_obs, sigma, zeta, omega_n, dt, q=1e-4):
    """
    Impute missing entries using the filtered state estimate.
    y_obs may contain np.nan where data are missing.
    q: process noise intensity (Q = q * I_2)
    """
    F, H = _ss_matrices(zeta, omega_n, dt)
    Q = q * np.eye(2)
    R = np.array([[sigma**2]])

    Nn = len(y_obs)
    x_f = np.zeros((Nn, 2))
    P_f = np.zeros((Nn, 2, 2))

    obs_idx = np.where(~np.isnan(y_obs))[0]
    if len(obs_idx) == 0:
        return np.zeros_like(y_obs)

    # init from first available measurement
    x0 = np.array([y_obs[obs_idx[0]], 0.0])
    P0 = np.eye(2) * 1e3

    x_pred = F @ x0
    P_pred = F @ P0 @ F.T + Q

    y_hat = np.zeros(Nn)
    for k in range(Nn):
        if not np.isnan(y_obs[k]):
            yk = np.array([[y_obs[k]]])
            S = H @ P_pred @ H.T + R
            K = P_pred @ H.T @ np.linalg.inv(S)
            innov = yk - H @ x_pred
            x_upd = x_pred + (K @ innov).ravel()
            P_upd = (np.eye(2) - K @ H) @ P_pred
        else:
            x_upd, P_upd = x_pred, P_pred

        x_f[k] = x_upd
        P_f[k] = P_upd
        y_hat[k] = (H @ x_upd.reshape(-1, 1)).item()

        # time update
        x_pred = F @ x_upd
        P_pred = F @ P_upd @ F.T + Q

    y_imp = y_obs.copy()
    nan_idx = np.isnan(y_imp)
    y_imp[nan_idx] = y_hat[nan_idx]
    return y_imp

def kalman_rts_smoother_impute(y_obs, sigma, zeta, omega_n, dt, q=1e-4):
    """
    Impute using Rauch–Tung–Striebel smoother (backward pass).
    """
    F, H = _ss_matrices(zeta, omega_n, dt)
    Q = q * np.eye(2)
    R = np.array([[sigma**2]])

    Nn = len(y_obs)
    x_f = np.zeros((Nn, 2))
    P_f = np.zeros((Nn, 2, 2))
    x_pred_hist = np.zeros((Nn, 2))
    P_pred_hist = np.zeros((Nn, 2, 2))

    obs_idx = np.where(~np.isnan(y_obs))[0]
    if len(obs_idx) == 0:
        return np.zeros_like(y_obs)

    # init
    x0 = np.array([y_obs[obs_idx[0]], 0.0])
    P0 = np.eye(2) * 1e3
    x_pred = F @ x0
    P_pred = F @ P0 @ F.T + Q

    # forward filter (store preds for smoother)
    for k in range(Nn):
        x_pred_hist[k] = x_pred
        P_pred_hist[k] = P_pred

        if not np.isnan(y_obs[k]):
            yk = np.array([[y_obs[k]]])
            S = H @ P_pred @ H.T + R
            K = P_pred @ H.T @ np.linalg.inv(S)
            innov = yk - H @ x_pred
            x_upd = x_pred + (K @ innov).ravel()
            P_upd = (np.eye(2) - K @ H) @ P_pred
        else:
            x_upd, P_upd = x_pred, P_pred

        x_f[k] = x_upd
        P_f[k] = P_upd

        x_pred = F @ x_upd
        P_pred = F @ P_upd @ F.T + Q

    # RTS backward pass
    x_s = np.zeros_like(x_f)
    P_s = np.zeros_like(P_f)
    x_s[-1] = x_f[-1]
    P_s[-1] = P_f[-1]

    for k in range(Nn - 2, -1, -1):
        P_pred_next = F @ P_f[k] @ F.T + Q
        Ck = P_f[k] @ F.T @ np.linalg.inv(P_pred_next)
        x_s[k] = x_f[k] + (Ck @ (x_s[k+1] - (F @ x_f[k]))).ravel()
        P_s[k] = P_f[k] + Ck @ (P_s[k+1] - P_pred_next) @ Ck.T

    y_hat_s = (np.array([[1.0, 0.0]]) @ x_s.T).ravel()
    y_imp = y_obs.copy()
    nan_idx = np.isnan(y_imp)
    y_imp[nan_idx] = y_hat_s[nan_idx]
    return y_imp

# -------------------------------
# Simulation
# -------------------------------
y_true = generate_clean_signal(t)

# Imputer registry (each takes y_miss and returns imputed vector)
def _locf(y): return locf_impute(y)
def _lin(y):  return linear_interpolation_impute(y)
def _kf(y):   return kalman_filter_impute(y, sigma=sigma, zeta=zeta, omega_n=omega_n, dt=dt, q=1e-4)
def _rts(y):  return kalman_rts_smoother_impute(y, sigma=sigma, zeta=zeta, omega_n=omega_n, dt=dt, q=1e-4)

methods = {
    "LOCF": _locf,
    "Linear Interpolation": _lin,
    "Kalman Filter": _kf,
    "RTS Smoother": _rts,
}

# Collect MAE (mean absolute circular error) and 95% bootstrap CIs vs missing fraction
names_all = list(methods.keys()) + ["MAR Observed Data"]
phi_mae    = {name: [] for name in names_all}
phi_mae_lo = {name: [] for name in names_all}
phi_mae_hi = {name: [] for name in names_all}

for beta in missing_fractions:
    estimates = {name: [] for name in names_all}

    for _ in range(MC):
        # Noisy observation
        noise = rng.normal(0.0, sigma, size=N)
        y_noisy = y_true + noise

        # MAR missingness based on CLEAN y_true (observed covariate proxy)
        is_missing = remove_data_mar(y_true, beta, rng)
        y_miss = y_noisy.copy()
        y_miss[is_missing] = np.nan

        # Observed-only baseline: fit using only available samples
        obs_mask = ~np.isnan(y_miss)
        t_obs = t[obs_mask]
        y_obs = y_miss[obs_mask]
        phi_hat_obs = mle_phi_multistart(y_obs, t_obs, n_starts=12)
        estimates["MAR Observed Data"].append(phi_hat_obs)

        # Imputation methods + MLE on full length after imputation
        for name, imputer in methods.items():
            y_imp = imputer(y_miss)
            phi_hat = mle_phi_multistart(y_imp, t, n_starts=12)
            estimates[name].append(phi_hat)

    # MAE and 95% bootstrap CI for each method
    B = 1000  # bootstrap replicates
    alpha = 0.05
    for name in estimates:
        phi_arr = np.asarray(estimates[name])

        # absolute circular errors for each MC replicate
        abs_err = circular_abs_error(phi_arr, phi_true)

        mae = abs_err.mean()

        # bootstrap percentile CI on MAE
        boot_means = np.empty(B)
        n = abs_err.size
        for b in range(B):
            idx = rng.integers(0, n, size=n)
            boot_means[b] = abs_err[idx].mean()
        lo = np.percentile(boot_means, 100*alpha/2)
        hi = np.percentile(boot_means, 100*(1 - alpha/2))

        phi_mae[name].append(mae)
        phi_mae_lo[name].append(lo)
        phi_mae_hi[name].append(hi)

# -------------------------------
# Plot — make LOCF clearly visible
# -------------------------------
beta_pct = missing_fractions
plt.figure(figsize=(8, 6))

# Distinct styles
styles = {
    "LOCF":                 dict(linestyle='-',  marker='o', linewidth=2.8, zorder=5),
    "Linear Interpolation": dict(linestyle='--', marker='s', linewidth=2.0, zorder=3),
    # "Kalman Filter":        dict(linestyle='-.', marker='^', linewidth=2.0, zorder=3),  # optional
    "RTS Smoother":         dict(linestyle=':',  marker='D', linewidth=2.0, zorder=3),
    "MAR Observed Data":    dict(linestyle='-',  marker='x', linewidth=2.0, zorder=2),
}

# Optional tiny x-jitter to separate nearly overlapping curves
USE_JITTER = False
x_offsets = {
    "LOCF": 0.0,
    "Linear Interpolation": 0.35,
    "Kalman Filter": -0.35,
    "RTS Smoother": 0.18,
    "MAR Observed Data": -0.18,
}

def x_for(name):
    if USE_JITTER:
        return beta_pct + x_offsets.get(name, 0.0)
    return beta_pct

# Plot all except LOCF first
plot_order = [n for n in names_all if n != "LOCF"]
for name in plot_order:
    # If you want to include Kalman Filter, remove this guard.
    if name == "Kalman Filter":
        continue
    vals = np.array(phi_mae[name])
    lo   = np.array(phi_mae_lo[name])
    hi   = np.array(phi_mae_hi[name])
    plt.plot(x_for(name), vals, label=name, **styles.get(name, {}))
    plt.fill_between(x_for(name), lo, hi, alpha=0.12, zorder=1)

# Plot LOCF last on top
name = "LOCF"
vals = np.array(phi_mae[name])
lo   = np.array(phi_mae_lo[name])
hi   = np.array(phi_mae_hi[name])
plt.plot(x_for(name), vals, label=name, **styles[name])
plt.fill_between(x_for(name), lo, hi, alpha=0.10, zorder=1)

plt.xlabel(r"Missing fraction $\beta$")
plt.ylabel(r"MAE of $\hat{\phi}$ (rad)")
# Keep linear scale for MAE
plt.grid(True, alpha=0.3)
plt.legend(title="Estimator", loc="best")
plt.tight_layout()
plt.savefig("phi_mae_vs_missing_fraction.pdf", format="pdf", bbox_inches="tight")
plt.show()

# # (Optional) quick sanity print:
# for name in names_all:
#     print(name, np.round(np.array(phi_mae[name])[:3], 4))
