import numpy as np
import matplotlib.pyplot as plt

def compute_crlb(sigma2, N, A, zeta, omega_n, beta):
    """
    Compute CRLBIdeal and CRLBmissing using the summed decay term

        Σ_{k=1}^{N} e^{-2 ζ ω_n t_k},   with t_k = k (Δt = 1)

    Parameters
    ----------
    sigma2 : float
        Noise variance σ².
    N : int
        Total number of observations.
    A : float
        Amplitude factor.
    zeta : float
        Damping ratio ζ.
    omega_n : float
        Natural (undamped) frequency ω_n (rad ⋅ s⁻¹).
    beta : int
        Number of missing observations.

    Returns
    -------
    CRLBIdeal, CRLBmissing : tuple of float
        Ideal and missing-data Cramér–Rao lower bounds.
    """
    beta_frac = beta / N                       # fraction of data that is missing
    t_vec = np.arange(1, N + 1)                # t_k = 1,2,…,N  (assume Δt = 1)
    decay_sum = np.sum(np.exp(-2 * zeta * omega_n * t_vec))

    CRLBIdeal   = 2 * sigma2 / (A**2 * decay_sum)

    if beta_frac < 1:                          # at least one sample observed
        CRLBmissing = 2 * sigma2 / (A**2 * (1 - beta_frac) * decay_sum)
    else:                                      # all data missing ⇒ uninformative
        CRLBmissing = np.inf

    return CRLBIdeal, CRLBmissing


# -------------------------------------------------------------------------
# Example usage
sigma2   = 1.0        # variance
A        = 2.0        # amplitude factor
zeta     = 0.1        # damping ratio
omega_n  = 0.02        # natural frequency (rad/s)
beta     = 40         # missing-sample count

N_values = np.arange(10, 110, 10)   # total-observation grid

CRLBIdeal_list   = []
CRLBmissing_list = []

for N in N_values:
    CRLBIdeal, CRLBmissing = compute_crlb(sigma2, N, A, zeta, omega_n, beta)
    CRLBIdeal_list.append(CRLBIdeal)
    CRLBmissing_list.append(CRLBmissing)

# Plot the two bounds versus N
plt.figure(figsize=(8, 6))
plt.plot(N_values, CRLBIdeal_list,   '--o', label='CRLB$_{\\text{ideal}}$')
plt.plot(N_values, CRLBmissing_list, '-s',  label='CRLB$_{\\text{missing}}$')
plt.xlabel('Total observations $N$')
plt.ylabel('CRLB')
#plt.title('CRLB vs. sample size with fixed missing count $\\beta = 40$')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Print the first pair as a sanity check
CRLBIdeal, CRLBmissing = compute_crlb(sigma2, N_values[0], A, zeta, omega_n, beta)
print(f"CRLBIdeal   (N={N_values[0]}): {CRLBIdeal:.4e}")
print(f"CRLBmissing (N={N_values[0]}): {CRLBmissing}")
