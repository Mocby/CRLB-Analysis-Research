import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.linalg import expm

# ===========================
# Config — tweak to taste
# ===========================
SEED_BASE = 123
T = 5.0
N = 1000                    # smaller N makes methods differ more
sigma = 0.20              # a bit more noise
A = 1.0
zeta = 0.2
omega_n = 2 * np.pi * 1
omega_d = omega_n * np.sqrt(1 - zeta**2)
phi_true = np.pi / 4

missing_fractions = np.linspace(0.1, 0.9, 9)
missing_mode = "MCAR"     # "BLOCK" (contiguous gap), "MCAR"
n_iterations = 600         # MC runs per beta (increase for smoother curves)

# ===========================
# Time and clean signal
# ===========================
t = np.linspace(0, T, N, endpoint=False)
dt = t[1] - t[0]
y_true = A * np.exp(-zeta * omega_n * t) * np.sin(omega_d * t + phi_true)

# ===========================
# CRLB for φ (correct)
# ===========================
def crlb_phi(t, A, zeta, omega_n, omega_d, phi, sigma):
    g = A * np.exp(-zeta * omega_n * t) * np.cos(omega_d * t + phi)
    info = np.sum(g**2) / (sigma**2)
    return 1.0 / info

def crlb_phi_mcar(t, A, zeta, omega_n, omega_d, phi, sigma, beta):
    g = A * np.exp(-zeta * omega_n * t) * np.cos(omega_d * t + phi)
    info_full = np.sum(g**2) / (sigma**2)
    info_mcar = (1.0 - beta) * info_full
    return 1.0 / info_mcar

ideal_crlb_list = [crlb_phi(t, A, zeta, omega_n, omega_d, phi_true, sigma)
                   for _ in missing_fractions]
mcar_crlb_list  = [crlb_phi_mcar(t, A, zeta, omega_n, omega_d, phi_true, sigma, b)
                   for b in missing_fractions]

# ===========================
# Missingness generators
# ===========================
def remove_data_mcar(y, beta, rng):
    mask = rng.random(len(y)) > beta
    y_missing = y.copy()
    y_missing[~mask] = np.nan
    return y_missing, mask

def remove_data_block(y, beta, rng):
    """Remove ONE contiguous block of length ~beta*N."""
    L = len(y)
    block_len = max(1, int(np.round(beta * L)))
    start = rng.integers(0, L - block_len + 1)
    mask = np.ones(L, dtype=bool)
    mask[start:start+block_len] = False
    y_missing = y.copy()
    y_missing[~mask] = np.nan
    return y_missing, mask

# ===========================
# Simple imputers
# ===========================
def locf_impute(y):
    y = y.copy()
    valid_idx = np.where(~np.isnan(y))[0]
    if len(valid_idx) == 0:
        return np.zeros_like(y)
    for i in range(len(y)):
        if np.isnan(y[i]):
            y[i] = y[i - 1] if i > 0 else y[valid_idx[0]]
    return y

def linear_interpolation_impute(y):
    y = y.copy()
    nans = np.isnan(y)
    if np.all(~nans):
        return y
    idx = np.arange(len(y))
    y[nans] = np.interp(idx[nans], idx[~nans], y[~nans])
    return y

def simulation_impute(y_missing, mask, y_true, noise_std=0.1, rng=None):
    y_sim = y_missing.copy()
    noise = (rng.normal(0, 2 * noise_std, size=np.sum(~mask)) if rng is not None
             else np.random.normal(0, 2 * noise_std, size=np.sum(~mask)))
    y_sim[~mask] = y_true[~mask] + noise
    return y_sim

# ===========================
# Kalman / RTS for oscillator
# ===========================
def _ss_matrices(zeta, omega_n, dt):
    Ac = np.array([[0.0, 1.0],
                   [-(omega_n**2), -2.0*zeta*omega_n]])
    F = expm(Ac * dt)
    H = np.array([[1.0, 0.0]])
    return F, H

def kalman_filter_impute(y_obs, sigma, zeta, omega_n, dt, q=1e-4):
    F, H = _ss_matrices(zeta, omega_n, dt)
    Q = q * np.eye(2)
    R = np.array([[sigma**2]])
    Nn = len(y_obs)
    x_f = np.zeros((Nn, 2))
    P_f = np.zeros((Nn, 2, 2))

    first_idx = np.where(~np.isnan(y_obs))[0]
    if len(first_idx) == 0:
        return np.zeros_like(y_obs)
    x0 = np.array([y_obs[first_idx[0]], 0.0])
    P0 = np.diag([1.0, 1.0]) * 1e3

    for k in range(Nn):
        if k == 0:
            x_pred = F @ x0
            P_pred = F @ P0 @ F.T + Q
        else:
            x_pred = F @ x_f[k-1]
            P_pred = F @ P_f[k-1] @ F.T + Q

        if np.isnan(y_obs[k]):
            x_f[k] = x_pred
            P_f[k] = P_pred
        else:
            yk = np.array([[y_obs[k]]])
            S = H @ P_pred @ H.T + R
            K = P_pred @ H.T @ np.linalg.inv(S)
            innov = yk - H @ x_pred
            x_f[k] = x_pred + (K @ innov).ravel()
            P_f[k] = (np.eye(2) - K @ H) @ P_pred

    y_hat = (H @ x_f.transpose(1,0)).squeeze()
    y_imp = y_obs.copy()
    y_imp[np.isnan(y_imp)] = y_hat[np.isnan(y_imp)]
    return y_imp

def kalman_smoother_impute(y_obs, sigma, zeta, omega_n, dt, q=1e-4):
    F, H = _ss_matrices(zeta, omega_n, dt)
    Q = q * np.eye(2)
    R = np.array([[sigma**2]])
    Nn = len(y_obs)
    x_f = np.zeros((Nn, 2))
    P_f = np.zeros((Nn, 2, 2))

    first_idx = np.where(~np.isnan(y_obs))[0]
    if len(first_idx) == 0:
        return np.zeros_like(y_obs)
    x0 = np.array([y_obs[first_idx[0]], 0.0])
    P0 = np.diag([1.0, 1.0]) * 1e3

    for k in range(Nn):
        if k == 0:
            x_pred = F @ x0
            P_pred = F @ P0 @ F.T + Q
        else:
            x_pred = F @ x_f[k-1]
            P_pred = F @ P_f[k-1] @ F.T + Q

        if np.isnan(y_obs[k]):
            x_f[k] = x_pred
            P_f[k] = P_pred
        else:
            yk = np.array([[y_obs[k]]])
            S = H @ P_pred @ H.T + R
            K = P_pred @ H.T @ np.linalg.inv(S)
            innov = yk - H @ x_pred
            x_f[k] = x_pred + (K @ innov).ravel()
            P_f[k] = (np.eye(2) - K @ H) @ P_pred

    # RTS smoother
    x_s = np.zeros_like(x_f)
    P_s = np.zeros_like(P_f)
    x_s[-1], P_s[-1] = x_f[-1], P_f[-1]
    for k in range(Nn-2, -1, -1):
        P_pred = F @ P_f[k] @ F.T + Q
        Ck = P_f[k] @ F.T @ np.linalg.inv(P_pred)
        x_s[k] = x_f[k] + (Ck @ (x_s[k+1] - (F @ x_f[k]))).ravel()
        P_s[k] = P_f[k] + Ck @ (P_s[k+1] - P_pred) @ Ck.T

    y_hat_s = (H @ x_s.transpose(1,0)).squeeze()
    y_imp = y_obs.copy()
    y_imp[np.isnan(y_imp)] = y_hat_s[np.isnan(y_imp)]
    return y_imp

# ===========================
# Bayesian (MH over φ)
# ===========================
def _wrap_to_pi(phi):
    return (phi + np.pi) % (2*np.pi) - np.pi

def bayesian_impute_phi_mh(y_missing, t, sigma, A, zeta, omega_n, omega_d,
                           n_samples=600, burn=200, proposal_sd=0.06, seed=None):
    rng = np.random.default_rng(seed)
    mask = ~np.isnan(y_missing)
    if not np.any(mask):
        mu0 = A * np.exp(-zeta * omega_n * t) * np.sin(omega_d * t)
        return np.where(np.isnan(y_missing), mu0, y_missing)
    y_obs = y_missing[mask]; t_obs = t[mask]

    def mu_phi(phi, tt):
        return A * np.exp(-zeta * omega_n * tt) * np.sin(omega_d * tt + phi)

    def loglik(phi):
        resid = y_obs - mu_phi(phi, t_obs)
        return -0.5 * np.sum(resid**2) / (sigma**2)

    coarse = np.linspace(-np.pi, np.pi, 64, endpoint=False)
    ll_vals = np.array([loglik(ph) for ph in coarse])
    phi_curr = coarse[np.argmax(ll_vals)]; ll_curr = ll_vals.max()

    draws = []
    for _ in range(burn + n_samples):
        phi_prop = _wrap_to_pi(phi_curr + rng.normal(0.0, proposal_sd))
        ll_prop = loglik(phi_prop)
        if np.log(rng.uniform()) < (ll_prop - ll_curr):
            phi_curr, ll_curr = phi_prop, ll_prop
        draws.append(phi_curr)

    phi_samples = np.array(draws[burn:])
    mu_all = np.stack([mu_phi(ph, t) for ph in phi_samples], axis=0)
    y_pp_mean = mu_all.mean(axis=0)
    y_imp = y_missing.copy()
    y_imp[np.isnan(y_imp)] = y_pp_mean[np.isnan(y_imp)]
    return y_imp

# ===========================
# SAITS (preserve observed)
# ===========================
def saits_impute_missing_only(y_missing, epochs=30, batch_size=16, seed=123):
    try:
        from pypots.imputation import SAITS
        import torch, random
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception as e:
        raise ImportError(
            "SAITS requires 'pypots' and 'torch'. Install with: pip install pypots torch"
        ) from e

    X = y_missing[None, :, None]
    n_steps = X.shape[1]
    model = SAITS(
        n_steps=n_steps, n_features=1,
        n_layers=2, d_model=64, d_ffn=64, n_heads=2, d_k=32, d_v=32,
        dropout=0.2, epochs=epochs, batch_size=batch_size,
        saving_path=None, model_saving_strategy=None,
    )
    model.fit({"X": X})
    X_full = model.impute({"X": X})
    y_full = X_full[0, :, 0]
    y_out = y_missing.copy()
    nan_idx = np.isnan(y_out)
    y_out[nan_idx] = y_full[nan_idx]
    return y_out

# ===========================
# MLE for φ
# ===========================
def mle_phi(y, t):
    # Uses all provided samples in y and t
    best_phi = None
    best_loss = np.inf
    phi_grid = np.linspace(-np.pi, np.pi, 80)
    for phi_init in phi_grid:
        def nll(phi):
            y_model = A * np.exp(-zeta * omega_n * t) * np.sin(omega_d * t + phi)
            return np.sum((y - y_model) ** 2)
        res = minimize(nll, x0=np.array([phi_init]), bounds=[(-np.pi, np.pi)])
        if res.fun < best_loss:
            best_loss = res.fun
            best_phi = res.x[0]
    return best_phi

# ===========================
# Helper: bootstrap CI for variance
# ===========================
def bootstrap_var_ci(values, B=1000, alpha=0.05, seed=SEED_BASE):
    """Return (low, high) percentile CI for sample variance (ddof=1). NaNs are ignored."""
    vals = np.asarray(values)
    vals = vals[~np.isnan(vals)]
    if vals.size < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    boot = np.empty(B)
    n = vals.size
    for b in range(B):
        resample = vals[rng.integers(0, n, n)]
        boot[b] = np.var(resample, ddof=1)
    lo = np.percentile(boot, 100 * (alpha/2))
    hi = np.percentile(boot, 100 * (1 - alpha/2))
    return lo, hi

# ===========================
# Storage
# ===========================
phi_variance = {
    "missing_data": [], "locf": [], "linear": [], "sim": [],
    "kalman_filt": [], "kalman_smooth": [], "bayes": [], "saits": []
}

# Store raw φ-hat samples per beta for CI computation
phi_samples_per_beta = {
    "bayes": [],  # list of arrays per beta
    "saits": []
}

# CI arrays (95%) for Bayes and SAITS
ci95_lower = {"bayes": [], "saits": []}
ci95_upper = {"bayes": [], "saits": []}

# ===========================
# Main MC loop
# ===========================
for i_beta, beta in enumerate(missing_fractions):
    v_missing, v_locf, v_lin = [], [], []
    v_sim, v_kf, v_rts, v_bayes, v_saits = [], [], [], [], []

    for mc in range(n_iterations):
        rng = np.random.default_rng(SEED_BASE + mc)
        y_noisy = y_true + rng.normal(0, sigma, N)

        if missing_mode.upper() == "MCAR":
            y_missing, mask = remove_data_mcar(y_noisy, beta, rng)
        else:  # contiguous block gap
            y_missing, mask = remove_data_block(y_noisy, beta, rng)

        # 1) Missing-only (use only observed samples)
        phi_missing = mle_phi(y_missing[mask], t[mask]); v_missing.append(phi_missing)

        # 2) LOCF
        y_locf  = locf_impute(y_missing);                     v_locf.append(mle_phi(y_locf, t))

        # 3) Linear
        y_lin   = linear_interpolation_impute(y_missing);     v_lin.append(mle_phi(y_lin, t))

        # 4) Simulated imputation (oracle-ish mean + extra noise)
        y_sim   = simulation_impute(y_missing, mask, y_true, noise_std=sigma, rng=rng)
        v_sim.append(mle_phi(y_sim, t))

        # 5) Kalman filter
        y_kf    = kalman_filter_impute(y_missing, sigma, zeta, omega_n, dt)
        v_kf.append(mle_phi(y_kf, t))

        # 6) RTS smoother
        y_rts   = kalman_smoother_impute(y_missing, sigma, zeta, omega_n, dt)
        v_rts.append(mle_phi(y_rts, t))

        # 7) Bayesian (posterior predictive)
        y_bayes = bayesian_impute_phi_mh(
            y_missing, t, sigma, A, zeta, omega_n, omega_d,
            n_samples=600, burn=200, proposal_sd=0.06,
            seed=SEED_BASE + mc
        )
        v_bayes.append(mle_phi(y_bayes, t))

        # 8) SAITS (preserve observed) — optional; may be slow
        try:
            y_saits = saits_impute_missing_only(
                y_missing, epochs=40, batch_size=1, seed=SEED_BASE + mc
            )
            v_saits.append(mle_phi(y_saits, t))
        except ImportError:
            v_saits.append(np.nan)

    # Unbiased MC variance (ddof=1)
    phi_variance["missing_data"].append(np.var(v_missing, ddof=1))
    phi_variance["locf"].append(np.var(v_locf, ddof=1))
    phi_variance["linear"].append(np.var(v_lin, ddof=1))
    phi_variance["sim"].append(np.var(v_sim, ddof=1))
    phi_variance["kalman_filt"].append(np.var(v_kf, ddof=1))
    phi_variance["kalman_smooth"].append(np.var(v_rts, ddof=1))
    phi_variance["bayes"].append(np.var(v_bayes, ddof=1))
    phi_variance["saits"].append(np.nanvar(v_saits, ddof=1))

    # Save raw samples for CI and compute 95% bootstrap CI for variance
    phi_samples_per_beta["bayes"].append(np.array(v_bayes))
    phi_samples_per_beta["saits"].append(np.array(v_saits))

    lo_b, hi_b = bootstrap_var_ci(v_bayes, B=1000, alpha=0.05, seed=SEED_BASE + i_beta)
    ci95_lower["bayes"].append(lo_b)
    ci95_upper["bayes"].append(hi_b)

    lo_s, hi_s = bootstrap_var_ci(v_saits, B=1000, alpha=0.05, seed=SEED_BASE + 999 + i_beta)
    ci95_lower["saits"].append(lo_s)
    ci95_upper["saits"].append(hi_s)

# Convert lists to arrays
for k in phi_variance:
    phi_variance[k] = np.asarray(phi_variance[k])
bayes_lo = np.asarray(ci95_lower["bayes"])
bayes_hi = np.asarray(ci95_upper["bayes"])
saits_lo = np.asarray(ci95_lower["saits"])
saits_hi = np.asarray(ci95_upper["saits"])

# ===========================
# Plot
# ===========================

plt.rcParams.update({
    "font.size": 18,          # base font
    "axes.titlesize": 20,     # title font
    "axes.labelsize": 20,     # x/y label size
    "xtick.labelsize": 16,    # x ticks
    "ytick.labelsize": 16,    # y ticks
    "legend.fontsize": 16,    # legend text
    "legend.title_fontsize": 18
})

plt.figure(figsize=(16, 7))
plt.plot(missing_fractions, phi_variance["missing_data"], label="Missing Data (Observed data)", marker='o')
plt.plot(missing_fractions, phi_variance["locf"], label="LOCF Imputation", marker='s')
plt.plot(missing_fractions, phi_variance["linear"], label="Linear Interpolation", marker='^')
plt.plot(missing_fractions, phi_variance["sim"], label="Simulated Imputation", marker='d')
plt.plot(missing_fractions, phi_variance["kalman_filt"], label="Kalman Filter Imputation", marker='x')
plt.plot(missing_fractions, phi_variance["kalman_smooth"], label="RTS Smoother Imputation", marker='P')
plt.plot(missing_fractions, phi_variance["bayes"], label="Bayesian Imputation", marker='h')
plt.plot(missing_fractions, phi_variance["saits"], label="SAITS Imputation", marker='*')
plt.plot(missing_fractions, ideal_crlb_list, 'b--', label="CRLB (No missing data)")
plt.plot(missing_fractions, mcar_crlb_list, 'g--', label="MCAR CRLB")

# 95% CI shading for Bayes and SAITS (skip if NaNs)
# Bayesian CI
valid_bayes = ~np.isnan(bayes_lo) & ~np.isnan(bayes_hi)
if np.any(valid_bayes):
    plt.fill_between(missing_fractions[valid_bayes],
                     bayes_lo[valid_bayes],
                     bayes_hi[valid_bayes],
                     alpha=0.18, label="Bayesian 95% CI")

# SAITS CI
valid_saits = ~np.isnan(saits_lo) & ~np.isnan(saits_hi)
if np.any(valid_saits):
    plt.fill_between(missing_fractions[valid_saits],
                     saits_lo[valid_saits],
                     saits_hi[valid_saits],
                     alpha=0.18, label="SAITS 95% CI")

plt.xlabel(r"Missing Data Fraction")
plt.ylabel(r"Variance of Estimated Phase $\phi$ (rad$^2$)")
plt.yscale("log")
plt.grid(True, alpha=0.4)
plt.legend(ncol=2)
plt.tight_layout()
plt.savefig("phi_variance_plot_saits_with_CI.pdf", format="pdf", bbox_inches="tight")
plt.show()

