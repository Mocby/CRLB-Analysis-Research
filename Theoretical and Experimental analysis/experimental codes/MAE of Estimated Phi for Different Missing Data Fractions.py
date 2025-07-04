import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Define parameters
A = 1.0
zeta = 0.2
omega_n = 2 * np.pi * 1  # Natural frequency (Hz)
omega_d = omega_n * np.sqrt(1 - zeta**2)
phi_true = np.pi / 4  # True phi
sigma = 0.1  # Noise standard deviation
N = 10000 # Number of points
t = np.linspace(0, 10, N)  # Time vector
missing_fractions = np.linspace(0.1, 0.9, 9)  # Missing fractions from 0.1 to 0.9

# Generate clean signal
y_true = A * np.exp(-zeta * omega_n * t) * np.sin(omega_d * t + phi_true)

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

# Maximum likelihood estimation of phi
def mle_phi(y, t):
    def negative_log_likelihood(phi):
        y_model = A * np.exp(-zeta * omega_n * t) * np.sin(omega_d * t + phi)
        return np.sum((y - y_model) ** 2)
    result = minimize(negative_log_likelihood, x0=np.array([phi_true]), method='L-BFGS-B', bounds=[(-np.pi, np.pi)])
    return result.x[0]

# Simulation parameters
n_iterations = 600

# Store MAE values
phi_mae = {
    "missing_data": [],
    "locf": [],
    "linear": [],
    "sim": []
}

for missing_fraction in missing_fractions:
    phi_missing_list = []
    phi_locf_list = []
    phi_linear_list = []
    phi_sim_list = []

    for _ in range(n_iterations):
        y_noisy = y_true + np.random.normal(0, sigma, N)
        y_missing, mask = remove_data(y_noisy, missing_fraction)

        phi_missing = mle_phi(y_missing[mask], t[mask])
        phi_missing_list.append(abs(phi_missing - phi_true))

        y_locf = locf_impute(y_missing.copy())
        phi_locf = mle_phi(y_locf, t)
        phi_locf_list.append(abs(phi_locf - phi_true))

        y_linear = linear_interpolation_impute(y_missing.copy())
        phi_linear = mle_phi(y_linear, t)
        phi_linear_list.append(abs(phi_linear - phi_true))

        y_sim = simulation_impute(y_missing.copy(), mask, y_true, sigma)
        phi_sim = mle_phi(y_sim, t)
        phi_sim_list.append(abs(phi_sim - phi_true))

    phi_mae["missing_data"].append(np.mean(phi_missing_list))
    phi_mae["locf"].append(np.mean(phi_locf_list))
    phi_mae["linear"].append(np.mean(phi_linear_list))
    phi_mae["sim"].append(np.mean(phi_sim_list))

# Plot MAE results
plt.figure(figsize=(10, 5))
plt.plot(missing_fractions, phi_mae["missing_data"], label="Missing Data", marker='o')
plt.plot(missing_fractions, phi_mae["locf"], label="LOCF Imputation", marker='s')
plt.plot(missing_fractions, phi_mae["linear"], label="Linear Interpolation", marker='^')
plt.plot(missing_fractions, phi_mae["sim"], label="Simulated Imputation", marker='d')
plt.xlabel("Missing Data Fraction")
plt.ylabel("Mean Absolute Error of Estimated Phi")
plt.title("MAE of Estimated Phi for Different Missing Data Fractions")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()