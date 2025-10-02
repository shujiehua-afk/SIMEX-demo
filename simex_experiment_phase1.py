#!/usr/bin/env python
# coding: utf-8

# In[2]:


import argparse
from typing import Union
import numpy as np
import pandas as pd
from dataclasses import dataclass
from pathlib import Path

from numpy.random import default_rng

try:
    from scipy.optimize import curve_fit
except Exception as e:
    raise RuntimeError("scipy is required (curve_fit). Please ensure scipy is installed.") from e

rng = default_rng()

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
    M: int = 100         # number of parametric Y replicates (imputations)
    n_datasets: int = 50 # how many independent (X,Y) pairs to generate in step 2.1
    lambda_start: float = 0.0
    lambda_end: float = 2.0
    lambda_step: float = 0.05

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
    sigma2_hat = float(resid.T @ resid / dof)
    cov_beta = sigma2_hat * np.linalg.inv(XtX)  # var(beta_hat) = sigma2_hat * (XtX)^{-1}
    return beta_hat, sigma2_hat, cov_beta

def nonlinear_model(lam, a, b, c):
    return a + b / (c + lam)

def get_p0(lams, ybar):
    x1, x2, x3 = lams[0], np.median(lams), lams[-1]
    y1, y2, y3 = ybar[0], np.median(ybar), ybar[-1]
    # 三点法估初值，退化时给一个保守初值
    denom = (y1 - y2) / (y2 - y3) * (x3 - x2) - (x2 - x1)
    c = ((x2 - x1) * x3 - (y1 - y2) / (y2 - y3) * (x3 - x2) * x1) / denom
    b = (y1 - y2) * (c + x1) * (c + x2) / (x2 - x1)
    a = y1 - b / (c + x1)
    return [a, b, c]

def simex_nonlin_estimate(X, Y, lambdas, B, sig_w, seed=None):
    r = default_rng(seed)
    X = np.asarray(X)
    Y = np.asarray(Y)
    betas = []
    for lam in lambdas:
        if lam < 0:
            raise ValueError("Lambda grid must be >= 0 for SIMEX simulation stage.")
        slopes = []
        for _ in range(B):
            Z = r.normal(0.0, 1.0, size=X.size)
            X_lam = X + np.sqrt(lam) * sig_w * Z
            b_hat, _, _ = ols_y_on_x(Y, X_lam)
            slopes.append(b_hat[1])
        betas.append(np.mean(slopes))
    betas = np.asarray(betas)

    # 初值：用更稳的 get_p0
    try:
        p0 = get_p0(np.asarray(lambdas), betas)
    except Exception:
        # 兜底（极端退化时）
        a0 = float(betas[-1])
        b0 = float((betas[0] - betas[-1]) * (1 + float(lambdas[0])))
        c0 = 1.0
        p0 = [a0, b0, c0]

    a = b = c = np.nan
    beta_at_minus1 = np.nan
    try:
        # 示例边界：限制 c 远离 -1（按数据可调），a,b 放宽
        bounds = ([-np.inf, -np.inf, -10.0], [np.inf, np.inf, -0.2])
        popt, _ = curve_fit(nonlinear_model, lambdas, betas, p0=p0, bounds=bounds, maxfev=20000)
        a, b, c = popt
        beta_at_minus1 = float(nonlinear_model(-1.0, a, b, c))
    except Exception:
        # 不抛出，返回 NaN 让上层统计失败
        pass

    return {
        "lambdas": np.array(lambdas),
        "beta_lambda": betas,
        "nl_params": (a, b, c),
        "simex_est": beta_at_minus1,
    }

def parametric_replicates_Y_given_X(X, Y, M, seed=None):
    """Generate M parametric replicates of Y given observed X (naive SLR is correctly specified)."""
    r = default_rng(seed)
    beta_hat, sigma2_hat, _ = ols_y_on_x(Y, X)  # naive SLR Y~X
    Xmat = np.column_stack([np.ones_like(X), X])
    dof = X.size - 2
    reps = []
    XtX_inv = np.linalg.inv(Xmat.T @ Xmat)
    for _ in range(M):
        # Sigma^2_m via scaled chi-square
        sigma2_m = sigma2_hat * (r.chisquare(dof) / dof)
        # Beta_m via Normal
        beta_m = r.multivariate_normal(mean=beta_hat, cov=sigma2_m * XtX_inv)
        # Errors
        eps_m = r.normal(0.0, np.sqrt(sigma2_m), size=X.size)
        Y_m = Xmat @ beta_m + eps_m
        reps.append(Y_m)
    return np.array(reps)  # shape (M, n)

def corrected_estimator_and_se(X, Y, sig_w):
    """Compute corrected-estimate and its delta-method SE on the SAME observed (X,Y)."""
    beta_hat, sigma2_hat, _ = ols_y_on_x(Y, X)
    slope_naive = float(beta_hat[1])
    n = X.size
    Sxx = float(np.sum((X - np.mean(X))**2))
    var_slope_naive = sigma2_hat / Sxx

    # sample variance of X with ddof=1
    s2_x = Sxx / (n - 1)
    # correction factor (sigma_X^2 / (sigma_X^2 - sigma_w^2)) = 1 / (1 - sig_w^2 / s2_x)
    denom = 1.0 - (sig_w**2) / s2_x
    beta_corr = slope_naive / denom

    # Delta-method variance (sig_w^2 known)
    var_s2x = 2 * (s2_x**2) / (n - 1)
    var_inv_s2x = var_s2x / (s2_x**4)
    var_omega = (sig_w**4) * var_inv_s2x
    var_beta_corr = (1/denom)**2 * var_slope_naive + (slope_naive / (denom**2))**2 * var_omega

    return {
        "beta_naive": slope_naive,
        "var_beta_naive": var_slope_naive,
        "beta_corrected": float(beta_corr),
        "var_beta_corrected": float(var_beta_corr),
    }

def simex_variance_decomposition(df: pd.DataFrame) -> pd.DataFrame:
    """给定多次 run 的 summary DataFrame，返回一行的方差分解表。"""
    true_var = df["simex_point"].var(ddof=1)
    E_cond_var = (df["simex_se"] ** 2).mean()
    between_X_var = max(0.0, true_var - E_cond_var)
    row = {
        "true_var": true_var,
        "E_cond_var": E_cond_var,
        "between_X_var": between_X_var,
        "n_repeats": len(df),
        "n": int(df["n"].iloc[0]) if "n" in df.columns else None,
        "B": int(df["B"].iloc[0]) if "B" in df.columns else None,
        "M": int(df["M"].iloc[0]) if "M" in df.columns else None,
        "sig_e": float(df["sig_e"].iloc[0]) if "sig_e" in df.columns else None,
        "sig_w": float(df["sig_w"].iloc[0]) if "sig_w" in df.columns else None,
        "sig_u": float(df["sig_u"].iloc[0]) if "sig_u" in df.columns else None,
        "mu_u": float(df["mu_u"].iloc[0]) if "mu_u" in df.columns else None,
        "true_beta1": float(df["true_beta1"].iloc[0]) if "true_beta1" in df.columns else None,
    }
    return pd.DataFrame([row])

def run_pipeline(p: Params, seed: int = 42,
                 save_dir: Union[str, Path] = "/Users/huashujie/Desktop/simex_pipeline"):
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
    n_fail = 0
    for m in range(p.M):
        res = simex_nonlin_estimate(X_obs, Y_reps[m],
                                    lambdas=lambdas, B=p.B, sig_w=p.sig_w, seed=seed+2000+m)
        if np.isfinite(res["simex_est"]):     
            simex_estimates.append(res["simex_est"])
        else:
            n_fail += 1
    simex_estimates = np.asarray(simex_estimates)

    # SIMEX point estimate on the observed data itself
    res_obs = simex_nonlin_estimate(X_obs, Y_obs,
                                    lambdas=lambdas, B=p.B, sig_w=p.sig_w, seed=seed+12345)
    simex_point = res_obs["simex_est"]
    simex_se = float(simex_estimates.std(ddof=1))
    z = 1.959963984540054
    simex_ci = (simex_point - z*simex_se, simex_point + z*simex_se)

    # 2.3 Corrected-estimate on the SAME observed data
    corr = corrected_estimator_and_se(X_obs, Y_obs, sig_w=p.sig_w)
    corr_point = corr["beta_corrected"]
    corr_se = float(np.sqrt(corr["var_beta_corrected"]))
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
        "n_fail":int(n_fail)
    }
    df_summary = pd.DataFrame([summary])
    df_summary.to_csv(save_dir / "summary.csv", index=False)

    # Save the replicate SIMEX estimates
    pd.DataFrame({"simex_nonlin": simex_estimates}).to_csv(
        save_dir / "simex_replicate_estimates.csv", index=False
    )

    return {
        "save_dir": str(save_dir),
        "summary_path": str(save_dir / "summary.csv"),
        "replicate_path": str(save_dir / "simex_replicate_estimates.csv"),
        "summary": summary,
    }


# In[23]:


from pathlib import Path

# 使用桌面路径
save_dir = Path("/Users/huashujie/Desktop/simex_pipeline_demo")

# 构造参数
p = Params(
    n=1000,           # 样本量
    beta0=2,
    beta1=3,
    mu_u=1.0,
    sig_u=1.0,
    sig_e=1.0,
    sig_w=0.7,
    B=100,            # 每个 λ 下做 50 次加噪 OLS
    M=100,            # 参数化重复次数
    n_datasets=1,    # 只生成 1 份 (X,Y)
    lambda_start=0.0,
    lambda_end=2.0,
    lambda_step=0.05  # λ 网格: 0.0, 0.2, 0.4, 0.6, 0.8, 1.0 共6个
)


# In[24]:


result = run_pipeline(p, seed=42, save_dir=save_dir)

print("保存路径:", result["save_dir"])
print("Summary CSV:", result["summary_path"])
print("Replicate Estimates CSV:", result["replicate_path"])
print("Summary 内容:", result["summary"])


# In[3]:


# %%
# File 2: simex_vs_corrected_pipeline.py
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
#
import argparse
from typing import Union
import numpy as np
import pandas as pd
from dataclasses import dataclass
from pathlib import Path

from numpy.random import default_rng

try:
    from scipy.optimize import curve_fit
except Exception as e:
    raise RuntimeError("scipy is required (curve_fit). Please ensure scipy is installed.") from e

rng = default_rng()

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
            _, _, slope = np.polyfit(X_lam, Y, 1, full=False, cov=False) if False else (None, None, None)
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
    """Generate M parametric replicates of Y given observed X (naive SLR is correctly specified)."""
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
    """Compute corrected-estimate and its delta-method SE on the SAME observed (X,Y)."""
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

def simex_variance_decomposition(df: pd.DataFrame) -> pd.DataFrame:
    """给定多次 run 的 summary DataFrame，返回一行的方差分解表。"""
    true_var = df["simex_point"].var(ddof=1)
    E_cond_var = (df["simex_se"] ** 2).mean()
    between_X_var = max(0.0, true_var - E_cond_var)
    row = {
        "true_var": true_var,
        "E_cond_var": E_cond_var,
        "between_X_var": between_X_var,
        "n_repeats": len(df),
        "n": int(df["n"].iloc[0]) if "n" in df.columns else None,
        "B": int(df["B"].iloc[0]) if "B" in df.columns else None,
        "M": int(df["M"].iloc[0]) if "M" in df.columns else None,
        "sig_e": float(df["sig_e"].iloc[0]) if "sig_e" in df.columns else None,
        "sig_w": float(df["sig_w"].iloc[0]) if "sig_w" in df.columns else None,
        "sig_u": float(df["sig_u"].iloc[0]) if "sig_u" in df.columns else None,
        "mu_u": float(df["mu_u"].iloc[0]) if "mu_u" in df.columns else None,
        "true_beta1": float(df["true_beta1"].iloc[0]) if "true_beta1" in df.columns else None,
    }
    return pd.DataFrame([row])


def run_pipeline(p: Params, seed=42, save_dir: Union[str, Path] = "/Users/huashujie/Desktop/simex_pipeline"):
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


# In[4]:


# 假设已经 import Params 和 run_pipeline
p = Params(
    n=1000, beta0=2, beta1=3,
    mu_u=0, sig_u=1, sig_e=1, sig_w=1.0,
    B=100, M=100, n_datasets=50,
    lambda_start=0, lambda_end=2, lambda_step=0.05
)

# out = run_pipeline(p, save_dir="/Users/huashujie/Desktop/simex_pipeline_demo")
# print(out["summary"])
run_pipeline(p)["summary"]


# In[21]:


# 批量运行：对不同 sig_w（1.0→0.6），每档跑 100 组，分别落盘并记录汇总
# 假设已 from your_module import Params, run_pipeline

from pathlib import Path
import pandas as pd
import numpy as np

# 进度条（若无 tqdm 则退化为普通 range）
try:
    from tqdm import trange
except Exception:
    trange = range

# ===== 配置 =====
BASE_DIR = Path("/Users/huashujie/Desktop/simex_pipeline_demo")  # 总目录
BASE_DIR.mkdir(parents=True, exist_ok=True)

N_RUNS_PER_GROUP = 100
SIG_W_GRID = [1.0, 0.9, 0.8, 0.7, 0.6]  # 从 1.0 到 0.6

# 其余参数模板（除 sig_w 外相同）
P_TEMPLATE = Params(
    n=1000, beta0=2, beta1=3,
    mu_u=0, sig_u=1, sig_e=1, sig_w=0.7,   # sig_w 会被覆盖
    B=100, M=100, n_datasets=50,
    lambda_start=0, lambda_end=2, lambda_step=0.05
)

all_groups_index = []  # 记录每组汇总文件路径

for sigw in SIG_W_GRID:
    group_dir = BASE_DIR / f"sigw_{sigw:.1f}"
    group_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for r in trange(N_RUNS_PER_GROUP, desc=f"sig_w={sigw:.1f}"):
        # 构造本次运行的参数（仅替换 sig_w）
        p = Params(
            n=P_TEMPLATE.n, beta0=P_TEMPLATE.beta0, beta1=P_TEMPLATE.beta1,
            mu_u=P_TEMPLATE.mu_u, sig_u=P_TEMPLATE.sig_u, sig_e=P_TEMPLATE.sig_e,
            sig_w=sigw,
            B=P_TEMPLATE.B, M=P_TEMPLATE.M, n_datasets=P_TEMPLATE.n_datasets,
            lambda_start=P_TEMPLATE.lambda_start,
            lambda_end=P_TEMPLATE.lambda_end,
            lambda_step=P_TEMPLATE.lambda_step
        )

        run_dir = group_dir / f"run_{r:03d}"
        out = run_pipeline(p, seed=42 + r, save_dir=run_dir)
        summaries.append(out["summary"])  # 一行 dict

    # 本组的 100 行汇总
    df_group = pd.DataFrame(summaries)
    group_summary_csv = group_dir / "summary_all.csv"
    df_group.to_csv(group_summary_csv, index=False)

    all_groups_index.append({
        "sig_w": sigw,
        "summary_csv": str(group_summary_csv)
    })

# 在总目录下写一个索引，指向每个组的汇总
pd.DataFrame(all_groups_index).to_csv(BASE_DIR / "INDEX_groups.csv", index=False)

print(f"完成。结果保存在：{BASE_DIR}")


# In[ ]:


## 注：第二段是运行代码，第一段有nan问题。


# In[28]:


# SIMEX 拟合诊断（多数据集、多 σ_w 汇总版）
# 依赖：你的原函数已在会话中（Params, gen_dataset, ols_y_on_x, simex_nonlin_estimate）
# 不修改原代码；仅复用其接口。

import numpy as np
import pandas as pd
from numpy.random import default_rng
from scipy.optimize import curve_fit

# --- 与原模型一致的非线性形式 ---
def nonlinear_model(lam, a, b, c):
    return a + b / (c + lam)

# --- 计算单个 (X,Y) 的逐点 MC-SE，用于标准化残差 ---
def simex_pointwise_mc_se(X, Y, lambdas, B_mc, sig_w, seed):
    r = default_rng(seed)
    se_means = []
    for lam in lambdas:
        slopes = []
        for _ in range(B_mc):
            Z = r.normal(0.0, 1.0, size=X.size)
            X_lam = X + np.sqrt(lam) * sig_w * Z
            b_hat, _, _ = ols_y_on_x(Y, X_lam)
            slopes.append(b_hat[1])
        slopes = np.asarray(slopes)
        se_means.append(slopes.std(ddof=1) / np.sqrt(B_mc))
    return np.asarray(se_means)

# --- 对单个 (X,Y) 做一次完整诊断 ---
def simex_fit_diagnostics_single(X, Y, sig_w, lambdas, B, B_mc=300, seed=42):
    res = simex_nonlin_estimate(X, Y, lambdas=lambdas, B=B, sig_w=sig_w, seed=seed)
    betas = res["beta_lambda"]; a, b, c = res["nl_params"]
    beta_fit = nonlinear_model(lambdas, a, b, c)
    resid = betas - beta_fit

    se_beta = simex_pointwise_mc_se(X, Y, lambdas, B_mc=B_mc, sig_w=sig_w, seed=seed+777)
    z = resid / np.maximum(se_beta, 1e-12)

    ss_res = np.sum(resid**2)
    ss_tot = np.sum((betas - betas.mean())**2) + 1e-12
    r2 = 1 - ss_res/ss_tot
    rmse = np.sqrt(np.mean(resid**2))
    max_abs_z = np.max(np.abs(z))
    frac_within_2 = float(np.mean(np.abs(z) <= 2.0))

    # 留一拟合
    loo_preds = []
    for i in range(len(lambdas)):
        mask = np.ones(len(lambdas), dtype=bool); mask[i] = False
        try:
            popt, _ = curve_fit(nonlinear_model, lambdas[mask], betas[mask], p0=(a,b,c), maxfev=5000)
            ai,bi,ci = popt
            loo_preds.append(nonlinear_model(lambdas[i], ai,bi,ci))
        except Exception:
            loo_preds.append(np.nan)
    loo_preds = np.asarray(loo_preds)
    loo_rmse = float(np.sqrt(np.nanmean((betas - loo_preds)**2)))

    return dict(
        rmse=float(rmse), r2=float(r2),
        max_abs_z=float(max_abs_z), frac_within_2=float(frac_within_2),
        loo_rmse=loo_rmse
    )

# --- 主函数：每个 σ_w 生成 K 份数据，逐份诊断并汇总 ---
def simex_fit_diagnostics_multi(p_template, sigw_grid, K=30, B_mc=300, seed=2025):
    rng = default_rng(seed)
    lambdas = np.arange(p_template.lambda_start, p_template.lambda_end + 1e-12, p_template.lambda_step)
    out_rows = []
    per_sigma_tables = {}  # 每个 σ_w 的逐数据集结果表

    for sigw in sigw_grid:
        rows = []
        for k in range(K):
            p = Params(
                n=p_template.n, beta0=p_template.beta0, beta1=p_template.beta1,
                mu_u=p_template.mu_u, sig_u=p_template.sig_u, sig_e=p_template.sig_e,
                sig_w=sigw, B=p_template.B, M=p_template.M, n_datasets=1,
                lambda_start=p_template.lambda_start,
                lambda_end=p_template.lambda_end,
                lambda_step=p_template.lambda_step
            )
            X, Y, U = gen_dataset(p, seed=seed + int(1e4*sigw) + k)
            d = simex_fit_diagnostics_single(
                X, Y, sig_w=sigw, lambdas=lambdas, B=p.B, B_mc=B_mc, seed=seed + 99 + k
            )
            d.update(dict(sig_w=float(sigw), run=k))
            rows.append(d)

        df = pd.DataFrame(rows).sort_values("run")
        per_sigma_tables[sigw] = df

        # 汇总：均值、标准差、分位数
        summ = df.describe(percentiles=[0.1,0.25,0.5,0.75,0.9]).T.reset_index().rename(columns={"index":"metric"})
        summ.insert(0, "sig_w", sigw)
        out_rows.append(summ)

    summary_long = pd.concat(out_rows, ignore_index=True)
    return summary_long, per_sigma_tables

# ===== 用法示例 =====
# p_template = Params(n=1000, beta0=2, beta1=3, mu_u=0, sig_u=1, sig_e=1, sig_w=0.7,
#                     B=100, M=100, n_datasets=1, lambda_start=0, lambda_end=2, lambda_step=0.05)
# sigw_grid = [0.6, 0.8, 1.0]
# summary_long, per_sigma = simex_fit_diagnostics_multi(p_template, sigw_grid, K=30, B_mc=400, seed=1234)
# # summary_long 为每个指标的均值/Std/分位数；per_sigma[sigw] 为该 σ_w 下每份数据的逐项指标。


# In[29]:


p_template = Params(n=1000, beta0=2, beta1=3, mu_u=0, sig_u=1, sig_e=1, sig_w=0.7,
                    B=100, M=100, n_datasets=1, lambda_start=0, lambda_end=2, lambda_step=0.05)
sigw_grid = [0.6, 0.8, 1.0]
summary_long, per_sigma = simex_fit_diagnostics_multi(p_template, sigw_grid, K=30, B_mc=400, seed=1234)
# summary_long 为每个指标的均值/Std/分位数；per_sigma[sigw] 为该 σ_w 下每份数据的逐项指标。


# In[30]:


summary_long, per_sigma


# In[32]:


summary_long.to_csv("/Users/huashujie/Desktop/summary_long.csv", index=False)


# In[33]:


# 外层 bootstrap 估算 SIMEX 的方差分解（给定一组 X,Y 与 sig_w）
# 依赖：你已有的函数已 import：
#   - Params, simex_nonlin_estimate, parametric_replicates_Y_given_X
#   - 以及 numpy, pandas
from numpy.random import default_rng
import numpy as np
import pandas as pd

def bootstrap_simex_variance_decomposition(
    X, Y, p, R=200, seed=1234
):
    """
    输入
    ----
    X, Y : 给定的一组观测向量（不改动）
    p    : 你的 Params 对象（使用其中的 sig_w, B, M, lambda_*）
    R    : 外层 bootstrap 次数（建议 200+）
    seed : 随机种子

    输出
    ----
    summary : dict，含 true_var, E_cond_var, between_X_var 等
    df_boot : DataFrame，逐个 bootstrap 样本的 simex_point 与 simex_se
    """
    r = default_rng(seed)
    n = len(X)
    lambdas = np.arange(p.lambda_start, p.lambda_end + 1e-12, p.lambda_step)

    rows = []
    for b in range(R):
        # 1) 外层 bootstrap：对索引放回抽样，制造“不同的 X 样本”
        idx = r.integers(0, n, size=n)
        Xb = np.asarray(X)[idx]
        Yb = np.asarray(Y)[idx]

        # 2) 条件 SE（与你主流程一致）：固定 Xb，
        #    先在 (Xb, Yb) 上做 M 次 Y|X 的参数化复制，再对每个复制跑 SIMEX，取 std 作为 SE
        Y_reps = parametric_replicates_Y_given_X(Xb, Yb, M=p.M, seed=int(seed + 9999 + b))
        ests = []
        for m in range(p.M):
            res_m = simex_nonlin_estimate(Xb, Y_reps[m], lambdas=lambdas, B=p.B, sig_w=p.sig_w, seed=int(seed + 2000 + b*1000 + m))
            ests.append(res_m["simex_est"])
        simex_se = float(np.std(ests, ddof=1))

        # 3) 点估计（与你主流程一致）：在 (Xb, Yb) 上跑一次 SIMEX 得到 simex_point
        res_b = simex_nonlin_estimate(Xb, Yb, lambdas=lambdas, B=p.B, sig_w=p.sig_w, seed=int(seed + 12345 + b))
        simex_point = float(res_b["simex_est"])

        rows.append({"run": b, "simex_point": simex_point, "simex_se": simex_se})

    df_boot = pd.DataFrame(rows)

    # 4) 方差分解（与你的函数同口径）
    true_var = df_boot["simex_point"].var(ddof=1)                # 总方差 Var(θ)
    E_cond_var = (df_boot["simex_se"] ** 2).mean()               # E[Var(θ|X)]
    between_X_var = max(0.0, true_var - E_cond_var)              # Var(E[θ|X])

    summary = {
        "true_var": float(true_var),
        "E_cond_var": float(E_cond_var),
        "between_X_var": float(between_X_var),
        "R_boot": R,
        "n": len(X),
        "B": p.B,
        "M": p.M,
        "sig_w": p.sig_w,
        "lambda_start": p.lambda_start,
        "lambda_end": p.lambda_end,
        "lambda_step": p.lambda_step,
    }
    return summary, df_boot


# In[46]:


p


# In[47]:


X_obs, Y_obs, U_obs = gen_dataset(p, seed=142)


# In[49]:


def estimate_between_X_variance(X, Y, p, R=200, seed=42):
    summary, _ = bootstrap_simex_variance_decomposition(X, Y, p, R=R, seed=seed)
    return summary["between_X_var"]


# In[50]:


between_X_var = estimate_between_X_variance(X_obs, Y_obs, p, R=200, seed=42)


# In[ ]:





# In[ ]:





# In[ ]:





# In[48]:


# 假设你已有一组 X_obs, Y_obs，以及 Params p（其中 p.sig_w 已设定）
summary, df_boot = bootstrap_simex_variance_decomposition(X_obs, Y_obs, p, R=200, seed=42)

print(summary)        # 查看 true_var / E_cond_var / between_X_var
df_boot.head()        # 每个 bootstrap 样本的 simex_point 与 simex_se


# In[ ]:




