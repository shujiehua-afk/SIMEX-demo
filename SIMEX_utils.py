


# %%
# Purpose:
#   1) Let user specify (beta0, beta1, sig_u, sig_e, sig_w, n, etc.).
#   2) Generate many independent datasets (X,Y) under the measurement-error model.
#   3) Choose one dataset as the "observed" dataset; keep its X fixed.
#   4) Parametric replicate Y|X (M times), run SIMEX (nonlinear) per replicate,
#      obtain point estimate and SE via sample SD across replicates.
#   5) On the SAME observed dataset, compute corrected-estimate and its delta-method SE.
#   6) Compare results.
#
# Notes:
# - Nonlinear extrapolation model: f(l) = a + b/(c + l). We use scipy.optimize.curve_fit.
# - Requires scipy and numpy, pandas.
# - This script keeps things self-contained and focuses on numeric outputs.

# ======================================================================= #

import argparse
from typing import Union
import numpy as np
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
from numpy.random import default_rng
from scipy.optimize import curve_fit

DEFAULT_SAVE_DIR: Union[str, Path] = "results"  
# DEFAULT_SAVE_DIR: Union[str, Path] = "/Users/huashujie/Desktop/simex_pipeline"


# ======================================================================= #

rng = default_rng(9)

@dataclass
class Params:
    n: int = 1000
    beta0: float = 2.0
    beta1: float = 3.0
    mu_u: float = 0.0
    sig_u: float = 1.0
    sig_e: float = 1.0   # residual sd
    sig_w: float = 0.7   # measurement-error sd (omega)
    B: int = 100         # SIMEX replicates per lambda
    M: int = 100          # number of parametric Y replicates (imputations)
    n_datasets: int = 50 # how many independent (X,Y) pairs to generate in step 2.1
    lambda_start: float = 0.0
    lambda_end: float = 2.0
    lambda_step: float = 0.05
        
# ======================================================================= #

def gen_dataset(p: Params, seed=None):
    r = default_rng(seed)
    U = r.normal(loc=p.mu_u, scale=p.sig_u, size=p.n)
    eta = r.normal(loc=0.0, scale=p.sig_w, size=p.n)
    X = U + eta
    eps = r.normal(loc=0.0, scale=p.sig_e, size=p.n)
    Y = p.beta0 + p.beta1 * U + eps
    return X, Y, U

def ols_y_on_x(y, x):
    x = np.asarray(x)
    y = np.asarray(y)
    Xmat = np.column_stack([np.ones_like(x), x])
    XtX = Xmat.T @ Xmat
    XtY = Xmat.T @ y
    beta_hat = np.linalg.solve(XtX, XtY)  # [intercept, slope]
    resid = y - Xmat @ beta_hat
    dof = x.size - 2
    sigma2_hat = resid.T @ resid / dof
    # var(beta_hat) = sigma2_hat * (XtX)^{-1}
    cov_beta = sigma2_hat * np.linalg.inv(XtX)
    return beta_hat, sigma2_hat, cov_beta

def simex_nonlin_estimate(X, Y, lambdas, B, sig_w, seed=None):
    r = default_rng(seed)
    X = np.asarray(X)
    Y = np.asarray(Y)
    betas = []
    for lam in lambdas:
        slopes = []
        if lam < 0:
            raise ValueError("Lambda grid must be >= 0 for SIMEX simulation stage.")
        for _ in range(B):
            Z = r.normal(0.0, 1.0, size=X.size)
            X_lam = X + np.sqrt(lam) * sig_w * Z
            # OLS Y ~ X_lam
            # _, _, slope = np.polyfit(X_lam, Y, 1, full=False, cov=False) if False else (None, None, None)
            # We'll just reuse our OLS to stay explicit:
            b_hat, _, _ = ols_y_on_x(Y, X_lam)
            slopes.append(b_hat[1])
        betas.append(np.mean(slopes))
    betas = np.asarray(betas)

    # Nonlinear: f(l) = a + b/(c + l)
    def f(l, a, b, c):
        return a + b / (c + l)

    # Smart-ish initial guesses:
    l0, lmid, lmax = lambdas[0], lambdas[len(lambdas)//2], lambdas[-1]
    y0, ymid, ymax = betas[0], betas[len(betas)//2], betas[-1]
    # crude guess
    a0 = ymax
    b0 = (y0 - ymax) * (1 + l0)
    c0 = 1.0
    p0 = [a0, b0, c0]

    popt, _ = curve_fit(f, lambdas, betas, p0=p0, maxfev=20000)
    a, b, c = popt
    beta_at_minus1 = f(-1.0, a, b, c)
    return {
        "lambdas": np.array(lambdas),
        "beta_lambda": betas,
        "nl_params": (a, b, c),
        "simex_est": float(beta_at_minus1),
    }

def parametric_replicates_Y_given_X(X, Y, M, seed=None):
    """
    Generate M parametric replicates of Y given observed X (naive SLR is correctly specified).
    """
    r = default_rng(seed)
    # Fit naive SLR Y ~ X
    beta_hat, sigma2_hat, cov_beta = ols_y_on_x(Y, X)
    Xmat = np.column_stack([np.ones_like(X), X])
    dof = X.size - 2
    reps = []
    for _ in range(M):
        # Sigma^2_m via scaled chi-square
        sigma2_m = sigma2_hat * (r.chisquare(dof) / dof)
        # Beta_m via Normal
        beta_m = r.multivariate_normal(mean=beta_hat, cov=sigma2_m * np.linalg.inv(Xmat.T @ Xmat))
        # Errors
        eps_m = r.normal(0.0, np.sqrt(sigma2_m), size=X.size)
        Y_m = Xmat @ beta_m + eps_m
        reps.append(Y_m)
    return np.array(reps)  # shape (M, n)

def corrected_estimator_and_se(X, Y, sig_w):
    """
    Compute corrected-estimate and its delta-method SE on the SAME observed (X,Y).
    """
    beta_hat, sigma2_hat, _ = ols_y_on_x(Y, X)
    slope_naive = beta_hat[1]
    n = X.size
    Sxx = np.sum((X - np.mean(X))**2)
    var_slope_naive = sigma2_hat / Sxx

    # sample variance of X with ddof=1
    s2_x = Sxx / (n - 1)
    # correction factor (sigma_X^2 / (sigma_X^2 - sigma_w^2)) = 1 / (1 - sig_w^2 / s2_x)
    denom = 1.0 - (sig_w**2) / s2_x
    beta_corr = slope_naive / denom

    # Delta-method variance
    # beta_corr = slope_naive / omega_hat, where omega_hat = 1 - sig_w^2 / s2_x
    # Var(beta_corr) ≈ (1/omega_hat)^2 Var(slope_naive) + (slope_naive / omega_hat^2)^2 Var(omega_hat)
    # Var(omega_hat) when sig_w^2 known: Var(sig_w^2 / s2_x) = sig_w^4 * Var(1/s2_x)
    # For Normal X, Var(s2_x) = 2*sigma_X^4/(n-1). Using delta for 1/s2_x gives Var(1/s2_x) ≈ Var(s2_x)/sigma_X^8
    # Plug-in with s2_x for sigma_X^2:
    var_s2x = 2 * (s2_x**2) / (n - 1)
    var_inv_s2x = var_s2x / (s2_x**4)
    var_omega = (sig_w**4) * var_inv_s2x
    var_beta_corr = (1/denom)**2 * var_slope_naive + (slope_naive / (denom**2))**2 * var_omega

    return {
        "beta_naive": float(slope_naive),
        "var_beta_naive": float(var_slope_naive),
        "beta_corrected": float(beta_corr),
        "var_beta_corrected": float(var_beta_corr),
    }

# def simex_variance_decomposition(df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Given a summary DataFrame from multiple runs, return a one-row variance decomposition table.
#     """
#     true_var = df["simex_point"].var(ddof=1)
#     E_cond_var = (df["simex_se"] ** 2).mean()
#     between_X_var = max(0.0, true_var - E_cond_var)
#     row = {
#         "true_var": true_var,
#         "E_cond_var": E_cond_var,
#         "between_X_var": between_X_var,
#         "n_repeats": len(df),
#         "n": int(df["n"].iloc[0]) if "n" in df.columns else None,
#         "B": int(df["B"].iloc[0]) if "B" in df.columns else None,
#         "M": int(df["M"].iloc[0]) if "M" in df.columns else None,
#         "sig_e": float(df["sig_e"].iloc[0]) if "sig_e" in df.columns else None,
#         "sig_w": float(df["sig_w"].iloc[0]) if "sig_w" in df.columns else None,
#         "sig_u": float(df["sig_u"].iloc[0]) if "sig_u" in df.columns else None,
#         "mu_u": float(df["mu_u"].iloc[0]) if "mu_u" in df.columns else None,
#         "true_beta1": float(df["true_beta1"].iloc[0]) if "true_beta1" in df.columns else None,
#     }
#     return pd.DataFrame([row])


def run_pipeline(p: Params, seed=42, save_dir: Union[str, Path] = DEFAULT_SAVE_DIR):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 2.1 Generate many datasets; we'll store the first as "observed"
    datasets = []
    for i in range(p.n_datasets):
        X, Y, U = gen_dataset(p, seed=seed+i)
        datasets.append((X, Y, U))
    X_obs, Y_obs, U_obs = datasets[0]

    # 2.2 From the observed (X,Y), parametric replicate M times and run SIMEX per replicate
    lambdas = np.arange(p.lambda_start, p.lambda_end + 1e-12, p.lambda_step)
    Y_reps = parametric_replicates_Y_given_X(X_obs, Y_obs, M=p.M, seed=seed+999)
    simex_estimates = []
    for m in range(p.M):
        res = simex_nonlin_estimate(X_obs, Y_reps[m], lambdas=lambdas, B=p.B, sig_w=p.sig_w, seed=seed+2000+m)
        simex_estimates.append(res["simex_est"])
    simex_estimates = np.asarray(simex_estimates)

    # Also run SIMEX once on the original observed data for the point estimate
    res_obs = simex_nonlin_estimate(X_obs, Y_obs, lambdas=lambdas, B=p.B, sig_w=p.sig_w, seed=seed+12345)
    simex_point = res_obs["simex_est"]
    simex_se = simex_estimates.std(ddof=1)
    z = 1.959963984540054
    simex_ci = (simex_point - z*simex_se, simex_point + z*simex_se)

    # 2.3 Corrected-estimate on the SAME observed data
    corr = corrected_estimator_and_se(X_obs, Y_obs, sig_w=p.sig_w)
    corr_point = corr["beta_corrected"]
    corr_se = np.sqrt(corr["var_beta_corrected"])
    corr_ci = (corr_point - z*corr_se, corr_point + z*corr_se)

    # Save summaries
    summary = {
        "true_beta1": p.beta1,
        "simex_point": float(simex_point),
        "simex_se": float(simex_se),
        "simex_ci_low": float(simex_ci[0]),
        "simex_ci_high": float(simex_ci[1]),
        "corrected_point": float(corr_point),
        "corrected_se": float(corr_se),
        "corrected_ci_low": float(corr_ci[0]),
        "corrected_ci_high": float(corr_ci[1]),
        "n": p.n,
        "B": p.B,
        "M": p.M,
        "sig_e": p.sig_e,
        "sig_w": p.sig_w,
        "sig_u": p.sig_u,
        "mu_u": p.mu_u,
    }
    df_summary = pd.DataFrame([summary])
    df_summary.to_csv(save_dir / "summary.csv", index=False)

    # Save the replicate SIMEX estimates
    pd.DataFrame({"simex_nonlin": simex_estimates}).to_csv(save_dir / "simex_replicate_estimates.csv", index=False)

    return {
        "save_dir": str(save_dir),
        "summary_path": str(save_dir / "summary.csv"),
        "replicate_path": str(save_dir / "simex_replicate_estimates.csv"),
        "summary": summary,
    }




def run_test():
    p = Params(
        n=1000, beta0=2, beta1=3,
        mu_u=0, sig_u=1, sig_e=1, sig_w=1.0,
        B=100, M=100, n_datasets=50,
        lambda_start=0, lambda_end=2, lambda_step=0.05
    )
    return(run_pipeline(p)["summary"])




if __name__ == "__main__":
    print("Running quick test...")
    run_test()






