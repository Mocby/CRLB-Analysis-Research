# CRLB-Based Analysis of Missing Data and Imputation in Phase Estimation  

This repository contains theoretical derivations and Python simulations for studying how **missing data** and **imputation** affect **phase estimation** in a damped sinusoidal (mass–spring–damper) system using the **Cramér–Rao Lower Bound (CRLB)** framework.

---

## 📂 Repository Structure

### 🔬 Theoretical Analysis (MCAR + Imputation)
- **`Comparison CRLB with and without missing data.py`**  
  - Ideal CRLB (no missing data) and MCAR-adjusted CRLB.
- **`Comparison of CRLBs with and without Imputation.py`**  
  - Imputation-adjusted CRLB with uncertainty factor α.
- **`Uncertainty impact on different levels of CRLB.py`**  
  - Effect of missingness β and imputation uncertainty α on CRLB.

**Results (MCAR/Imputation):**
[CRLB_with_and_without_missing.pdf](https://github.com/user-attachments/files/23922566/CRLB_with_and_without_missing.pdf)
, [Uncertaintyalpha.pdf](https://github.com/user-attachments/files/23922607/Uncertaintyalpha.pdf), 
[CRLB_with_imputation_vs_missing.pdf](https://github.com/user-attachments/files/23922606/CRLB_with_imputation_vs_missing.pdf)

### 📉 MAR Variance Analysis (Preliminary)
Scripts and figures corresponding to Section VI-B of the manuscript analyze:
- Phase-estimator variance under **MAR** (Missing At Random) patterns.
- Comparison between:
  - Observed-only estimator (no imputation)
  - LOCF, Linear, Kalman, RTS, etc.
- Reproduces the MAR variance behavior shown in **Fig. 9** of the paper, where variance saturates at high missingness and observed-only consistently outperforms imputation.

---

## 🧪 Experimental Analysis (MCAR & MAR)

- **`MAE of Estimated Phi for Different Missing Data Fractions.py`**  
  Empirical MAE vs missing-data fraction for multiple imputations.
- **`Variance of Estimated Phi for Different Missing Data Fractions.py`**  
  Empirical variance vs CRLB under MCAR.
- **`Normality check.py`**  
  Shapiro–Wilk tests + Q–Q plots (approximate normality at 50% & 90% missing).
- **`normal distribution code over 0.5 and 0.9 missing data probability.py`**  
  Histogram + KDE plots to visualize bias and spread of phase estimates.


  **Results included:**
[phi_variance_plot_saits_with_CI.pdf](https://github.com/user-attachments/files/23922731/phi_variance_plot_saits_with_CI.pdf)
[phi_MAE_plot_no_CRLB.pdf](https://github.com/user-attachments/files/23922732/phi_MAE_plot_no_CRLB.pdf)
<img width="1989" height="990" alt="Distribution of Estimated parameter" src="https://github.com/user-attachments/assets/36e6cabe-d20c-401b-96a1-1e8577edc67e" />
[phi_variance_vs_missing_fraction_MAR.pdf](https://github.com/user-attachments/files/23922785/phi_variance_vs_missing_fraction_MAR.pdf)

**Implemented Imputation Methods (MCAR & MAR experiments):**  
LOCF, Linear Interpolation, Kalman Filter, RTS Smoother, Bayesian Imputation, SAITS (Transformer-based), and **Ideal Simulated Imputation (oracle)**.

---

## 📈 Methods & Techniques

- Maximum Likelihood Estimation (MLE) for the phase parameter ϕ  
- Closed-form CRLB derivations for:
  - Ideal (complete data)  
  - MCAR missingness  
  - Imputed data with uncertainty factor α  
- Monte-Carlo simulations (typically 600 runs per setting)  
- Metrics:
  - Variance vs CRLB  
  - Mean Absolute Error (MAE)  
  - Bootstrap confidence intervals  
  - Shapiro–Wilk normality tests  

---

## 🔍 Key Findings

- **MCAR (Theory + Experiments)**
  - Missing data increases CRLB roughly in proportion to (1 − β).
  - Imputation can partially recover information but **never reaches** the ideal CRLB.
  - RTS and Bayesian methods perform best; SAITS is stable but biased.
  - Oracle simulated imputation matches the theoretical CRLB, validating the derivations.

- **MAR (Preliminary Experiments)**
  - Variance under MAR is higher and saturates at large missing fractions.
  - Across all tested methods, **observed-only estimation (no imputation) outperforms** imputation once missingness is moderate or high.
  - Results motivate a dedicated MAR-specific CRLB in future work.

---

## 🚧 Future Extensions

- CRLB derivations for **MAR** and **MNAR** missingness  
- Quantization-aware CRLB for low-resolution ADCs  
- Joint estimation of multiple parameters (e.g., ϕ and ω_d) with missing + quantized data  
- Systematic benchmarking of deep-learning imputers (SAITS and others)  
- Validation on real industrial sensor datasets  

---

