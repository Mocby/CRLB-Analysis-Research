import numpy as np
import matplotlib.pyplot as plt

def compute_crlb(sigma2, N, A, zeta, omega_n, beta, alpha2):
    beta_frac = beta / N
    t_vec = np.arange(1, N + 1)
    decay_sum = np.sum(np.exp(-2 * zeta * omega_n * t_vec))
    
    CRLBIdeal = 2 * sigma2 / (A**2 * decay_sum)
    CRLBmissing = 2 * sigma2 / (A**2 * (1 - beta_frac) * decay_sum) if beta_frac < 1 else np.inf
    CRLBimputed = 2 * sigma2 / (A**2 * ((1 - beta_frac) + alpha2 * beta_frac) * decay_sum) \
                  if ((1 - beta_frac) + alpha2 * beta_frac) > 0 else np.inf
    
    return CRLBIdeal, CRLBmissing, CRLBimputed

# Parameters
sigma2 = 1.0
A = 2.0
zeta = 0.1
omega_n = 0.1
beta = 20
alpha2 = 0.5

N_values = np.arange(10, 110, 10)

CRLBIdeal_list = []
CRLBmissing_list = []
CRLBimputed_list = []

for N in N_values:
    crlb_id, crlb_miss, crlb_imp = compute_crlb(sigma2, N, A, zeta, omega_n, beta, alpha2)
    CRLBIdeal_list.append(crlb_id)
    CRLBmissing_list.append(crlb_miss)
    CRLBimputed_list.append(crlb_imp)

# Plot
plt.figure()
plt.plot(N_values, CRLBIdeal_list, linestyle='--', marker='o', label="CRLB (no missing data)")
plt.plot(N_values, CRLBmissing_list, linestyle='-', marker='s', label="CRLB (MCAR missing data)")
plt.plot(N_values, CRLBimputed_list, linestyle='-.', marker='^', label="CRLB (imputed)")
plt.xlabel("Total Observations (N)")
plt.ylabel("CRLB")
plt.title("Comparison of CRLBs with and without Imputation")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
