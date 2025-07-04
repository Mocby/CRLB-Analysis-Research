import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import shapiro

# Define parameters
A = 1.0
zeta = 0.2
omega_n = 2 * np.pi * 1  # Natural frequency (Hz)
omega_d = omega_n * np.sqrt(1 - zeta**2)
phi_true = np.pi / 4  # True phi
sigma = 0.1  # Noise standard deviation
N = 10000  # Number of points
t = np.linspace(0, 10, N)  # Time vector
missing_fractions = np.linspace(0.1, 0.9, 9)  # Missing fractions from 0.1 to 0.9

# Generate clean signal
y_true = A * np.exp(-zeta * omega_n * t) * np.sin(omega_d * t + phi_true)

# Closed-form CRLB expressions
def calculate_crlb_fisher(noise_std, A, zeta, omega_n, omega_d, t, phi):
    # CRLBideal = 2 * sigma^2 / (A^2 * sum_{k=1}^N e^{-2 zeta omega_n t_k})
    sum_exp = np.sum(np.exp(-2 * zeta * omega_n * t))
    return 2 * noise_std**2 / (A**2 * sum_exp)

def calculate_mcar_crlb(noise_std, A, zeta, omega_n, omega_d, t, phi, missing_fraction):
    # CRLBmissing = 2 * sigma^2 / (A^2 * (1 - beta) * sum_exp)
    sum_exp = np.sum(np.exp(-2 * zeta * omega_n * t))
    return 2 * noise_std**2 / (A**2 * (1 - missing_fraction) * sum_exp)

# Compute CRLB values
ideal_crlb_list = []
mcar_crlb_list = []

for fraction in missing_fractions:
    ideal_crlb_list.append(
        calculate_crlb_fisher(sigma, A, zeta, omega_n, omega_d, t, phi_true)
    )
    mcar_crlb_list.append(
        calculate_mcar_crlb(sigma, A, zeta, omega_n, omega_d, t, phi_true, fraction)
    )

# Function to remove data at a given missing fraction
def remove_data(y, missing_fraction):
    mask = np.random.rand(N) > missing_fraction
    y_missing = np.copy(y)
    y_missing[~mask] = np.nan
    return y_missing, mask

# Imputation methods
def locf_impute(y):
    valid_idx = np.where(~np.isnan(y))[0]
    for i in range(len(y)):
        if np.isnan(y[i]):
            y[i] = y[i - 1] if i > 0 else y[valid_idx[0]]
    return y

def linear_interpolation_impute(y):
    nans, x = np.isnan(y), lambda z: z.nonzero()[0]
    y[nans] = np.interp(x(nans), x(~nans), y[~nans])
    return y

def simulation_impute(y_missing, mask, y_true, noise_std=0.1):
    y_sim = y_missing.copy()
    y_sim[~mask] = y_true[~mask] + np.random.normal(0, 2 * noise_std, size=np.sum(~mask))
    return y_sim

# Maximum likelihood estimation of phi (grid search + refine)
def mle_phi(y, t):
    best_phi = None
    best_loss = np.inf
    phi_grid = np.linspace(-np.pi, np.pi, 100)
    for phi_init in phi_grid:
        def negative_log_likelihood(phi):
            y_model = A * np.exp(-zeta * omega_n * t) * np.sin(omega_d * t + phi)
            return np.sum((y - y_model) ** 2)
        result = minimize(negative_log_likelihood,
                          x0=np.array([phi_init]),
                          bounds=[(-np.pi, np.pi)])
        if result.fun < best_loss:
            best_loss = result.fun
            best_phi = result.x[0]
    return best_phi

# Simulation parameters
n_iterations = 600  # Increased for better variance estimate

# Store variances and estimates
phi_variance = { "missing_data": [], "locf": [], "linear": [], "sim": [] }
phi_estimates_all = { "missing_data": [], "locf": [], "linear": [], "sim": [] }

for missing_fraction in missing_fractions:
    phi_missing_list = []
    phi_locf_list = []
    phi_linear_list = []
    phi_sim_list = []

    for _ in range(n_iterations):
        y_noisy = y_true + np.random.normal(0, sigma, N)
        y_missing, mask = remove_data(y_noisy, missing_fraction)

        # 1) Missing-only
        phi_missing = mle_phi(y_missing[mask], t[mask])
        phi_missing_list.append(phi_missing)

        # 2) LOCF
        y_locf = locf_impute(y_missing.copy())
        phi_locf_list.append(mle_phi(y_locf, t))

        # 3) Linear interpolation
        y_linear = linear_interpolation_impute(y_missing.copy())
        phi_linear_list.append(mle_phi(y_linear, t))

        # 4) Simulation-based imputation
        y_sim = simulation_impute(y_missing.copy(), mask, y_true, sigma)
        phi_sim_list.append(mle_phi(y_sim, t))

    phi_variance["missing_data"].append(np.var(phi_missing_list))
    phi_variance["locf"].append(np.var(phi_locf_list))
    phi_variance["linear"].append(np.var(phi_linear_list))
    phi_variance["sim"].append(np.var(phi_sim_list))

    phi_estimates_all["missing_data"].append(phi_missing_list)
    phi_estimates_all["locf"].append(phi_locf_list)
    phi_estimates_all["linear"].append(phi_linear_list)
    phi_estimates_all["sim"].append(phi_sim_list)

# Plot results
plt.figure(figsize=(10, 5))
plt.plot(missing_fractions, phi_variance["missing_data"], label="Missing Data", marker='o')
plt.plot(missing_fractions, phi_variance["locf"], label="LOCF Imputation", marker='s')
plt.plot(missing_fractions, phi_variance["linear"], label="Linear Interpolation", marker='^')
plt.plot(missing_fractions, phi_variance["sim"], label="Simulated Imputation", marker='d')
plt.plot(missing_fractions, ideal_crlb_list, 'b--', label="CRLB (No missing data)")
plt.plot(missing_fractions, mcar_crlb_list, 'g--', label="MCAR CRLB")
plt.xlabel("Missing Data Fraction")
plt.ylabel("Variance of Estimated Phi")
plt.legend()
plt.title("Variance of Estimated Phi for Different Missing Data Fractions")
plt.grid(True)
plt.tight_layout()
plt.show()


