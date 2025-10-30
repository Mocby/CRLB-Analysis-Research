import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import norm

# Define parameters (same as in your code)
A = 1.0
zeta = 0.2
omega_n = 2 * np.pi * 1  # Natural frequency (Hz)
omega_d = omega_n * np.sqrt(1 - zeta**2)
phi_true = np.pi / 4  # True phi
sigma = 0.1  # Noise standard deviation
N = 10000  # Number of points
t = np.linspace(0, 10, N)  # Time vector

# Generate clean signal
y_true = A * np.exp(-zeta * omega_n * t) * np.sin(omega_d * t + phi_true)

# MLE function (grid search + refine)
def mle_phi(y, t):
    best_phi = None
    best_loss = np.inf
    phi_grid = np.linspace(-np.pi, np.pi, 100)
    for phi_init in phi_grid:
        def negative_log_likelihood(phi):
            y_model = A * np.exp(-zeta * omega_n * t) * np.sin(omega_d * t + phi)
            return np.sum((y - y_model) ** 2)
        result = minimize(negative_log_likelihood, x0=np.array([phi_init]), bounds=[(-np.pi, np.pi)])
        if result.fun < best_loss:
            best_loss = result.fun
            best_phi = result.x[0]
    return best_phi

# Data removal function
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

# Simulation parameters
n_iterations = 600

# Containers for phi estimates
phi_full = []       # full dataset (no missing data)
phi_locf_05 = []    # LOCF with 50% missing
phi_linear_05 = []  # Linear interpolation with 50% missing
phi_sim_05 = []     # Simulation imputation with 50% missing

phi_locf_09 = []    # LOCF with 90% missing
phi_linear_09 = []  # Linear interpolation with 90% missing
phi_sim_09 = []     # Simulation imputation with 90% missing

# Run simulation
for _ in range(n_iterations):
    y_noisy = y_true + np.random.normal(0, sigma, N)

    # Full dataset estimation
    phi_full.append(mle_phi(y_noisy, t))

    # For missing probability 0.5 and 0.9, get imputed estimates
    for missing_fraction in [0.5, 0.9]:
        y_missing, mask = remove_data(y_noisy, missing_fraction)

        # LOCF Imputation
        y_locf = locf_impute(y_missing.copy())
        phi_est = mle_phi(y_locf, t)
        if missing_fraction == 0.5:
            phi_locf_05.append(phi_est)
        else:
            phi_locf_09.append(phi_est)

        # Linear Interpolation Imputation
        y_linear = linear_interpolation_impute(y_missing.copy())
        phi_est = mle_phi(y_linear, t)
        if missing_fraction == 0.5:
            phi_linear_05.append(phi_est)
        else:
            phi_linear_09.append(phi_est)

        # Simulation Imputation
        y_sim = simulation_impute(y_missing.copy(), mask, y_true, sigma)
        phi_est = mle_phi(y_sim, t)
        if missing_fraction == 0.5:
            phi_sim_05.append(phi_est)
        else:
            phi_sim_09.append(phi_est)

# Define a function to plot histogram and fitted normal density
def plot_histogram(data, title, ax):
    mu, std = np.mean(data), np.std(data)
    ax.hist(data, bins=30, density=True, alpha=0.6, color='g')

    xmin, xmax = ax.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    ax.plot(x, p, 'k', linewidth=2)
    ax.set_title(f"{title}\nμ = {mu:.4f}, σ = {std:.4f}")

# Create plots for missing fraction = 0.5 and 0.9
fig, axs = plt.subplots(2, 4, figsize=(20, 10))

# Column labels for each technique
techniques = [
    ("Full Data", phi_full),
    ("LOCF", phi_locf_05),
    ("Linear Interp.", phi_linear_05),
    ("Simulated Imputation", phi_sim_05)
]

for col, (tech, data) in enumerate(techniques):
    plot_histogram(data, f"{tech} (Missing 0%)" if tech=="Full Data" else f"{tech} (Missing=0.5)", axs[0, col])

techniques_09 = [
    ("LOCF", phi_locf_09),
    ("Linear Interp.", phi_linear_09),
    ("Simulated Imputaion", phi_sim_09)
]

# For the full dataset, we reuse the same distribution on both rows for reference
plot_histogram(phi_full, "Full Data (Missing 0%)", axs[1, 0])

for col, (tech, data) in enumerate(techniques_09, start=1):
    plot_histogram(data, f"{tech} (Missing=0.9)", axs[1, col])

plt.tight_layout()
plt.savefig("phi_histograms_imputation_MCAR.pdf", format="pdf", bbox_inches="tight")
plt.show()
