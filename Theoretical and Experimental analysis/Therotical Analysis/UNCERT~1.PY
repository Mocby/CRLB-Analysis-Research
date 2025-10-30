import numpy as np
import matplotlib.pyplot as plt

def compute_crlb(sigma2, N, A, zeta, omega_n, t, beta, alpha):
    """
    Compute CRLBIdeal, CRLBmissing, and CRLBimputed using
    sum_{t=1}^N e^{-2 ζ ω_n t}.
    """
    # sum_{t=1}^N e^{-2 ζ ω_n t}
    time_idx = np.arange(1, N + 1)
    sum_decay = np.sum(np.exp(-2 * zeta * omega_n * time_idx))

    beta_frac = beta / N

    # ideal CRLB
    CRLBIdeal = 2 * sigma2 / (A**2 * sum_decay)

    # missing-data CRLB
    CRLBmissing = (2 * sigma2 / (A**2 * (1 - beta_frac) * sum_decay)
                   if beta_frac < 1 else np.inf)

    # imputed-data CRLB
    denom = (1 - beta_frac) + alpha**2 * beta_frac
    CRLBImputed = (2 * sigma2 / (A**2 * denom * sum_decay)
                   if denom > 0 else np.inf)

    return CRLBIdeal, CRLBmissing, CRLBImputed

# Parameters
sigma2 = 1.0
N =1010
A = 2.0
zeta = 0.1
omega_n = 0.2
t = 1.0
beta_values = [10, 30, 40]
alpha_values = np.linspace(0, 1.1, 15)

imputed_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

plt.figure(figsize=(9, 7))

for idx, beta in enumerate(beta_values):
    CRLBmissing_list = []
    CRLBImputed_list = []

    for alpha in alpha_values:
        _, crlb_miss, crlb_imp = compute_crlb(
            sigma2, N, A, zeta, omega_n, t, beta, alpha
        )
        CRLBmissing_list.append(crlb_miss)
        CRLBImputed_list.append(crlb_imp)

    # plot imputed curves
    plt.plot(alpha_values, CRLBImputed_list,
             linestyle='-.', marker='o',
             color=imputed_colors[idx],
             label=rf"Imputed ($\beta={beta}$)")

    # missing-data horizontal line
    plt.axhline(y=CRLBmissing_list[0],
                linestyle="--", color="r",
                label=rf"Missing CRLB ($\beta={beta}$)")

# plot no‐missing‐data line
plt.axhline(y=compute_crlb(sigma2, N, A, zeta, omega_n, t, 0, 0)[0],
            linestyle=":", color="black",
            label="CRLB (No Missing Data)")

plt.xlabel(r"Imputation Uncertainty Factor $\alpha$")
plt.ylabel("CRLB of $\phi$ (rad$^2$)")
#plt.title("CRLB Phase Estimation of Mass–Spring–Damper Model")
plt.legend()
plt.grid(True)
plt.yscale("log")
plt.savefig("Uncertaintyimpuimpact.pdf", format="pdf", bbox_inches="tight")
plt.show()
