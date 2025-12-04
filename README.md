# CRLB-Based Analysis of Missing Data and Imputation in Phase Estimation

This repository contains theoretical and experimental Python scripts for analyzing the **Cramér-Rao Lower Bound (CRLB)** in the context of **missing data** and **data imputation**. The target application is phase estimation in damped sinusoidal signals, a common model in sensor and signal processing systems.

## 📂 Structure

### 🔬 Theoretical Analysis
- `Comparison CRLB with and without missing data.py`: Compares the CRLB in ideal vs. missing-data scenarios.
- `Comparison of CRLBs with and without Imputation.py`: Introduces an imputation-adjusted CRLB and compares three variants.
- `Uncertainty impact on different levels of CRLB.py`: Studies how imputation uncertainty (α) influences CRLB under various missing data ratios.

Results:
[CRLB_with_and_without_missing.pdf](https://github.com/user-attachments/files/23922566/CRLB_with_and_without_missing.pdf)

### 🧪 Experimental Analysis
- `MAE of Estimated Phi for Different Missing Data Fractions.py`: Computes Mean Absolute Error of estimated phase using LOCF, Linear, and Simulated imputation across missing data levels.
- `Variance of Estimated Phi for Different Missing Data Fractions.py`: Evaluates variance of estimated phase and compares it to theoretical CRLBs.
- `Variance of Estimated Phi for Different Missing Data Fractions and Normality check.py`: Includes Shapiro-Wilk normality checks for imputed distributions.
- `normal distribution code over 0.5 and 0.9 missing data probability.py`: Plots histograms and fitted Gaussians for phase estimates under 50% and 90% missingness.

## 📈 Techniques Used
- Maximum Likelihood Estimation (MLE) for phase parameter.
- Imputation methods: Last Observation Carried Forward (LOCF), Linear Interpolation,RTS Smoother,SAITS (Transformer-based), Bayesian Imputation, and Simulation-based.
- Statistical validation using variance, MAE, and Shapiro-Wilk test.
- Theoretical derivation and empirical confirmation of CRLBs.

## 🚧 Future Work
This repository will be extended to include:
- MAR and MNAR missingness models
- Quantization-aware CRLBs
- Deep learning-based imputation evaluation

---
