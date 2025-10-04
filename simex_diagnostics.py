


import numpy as np
import pandas as pd
from numpy.random import default_rng
from scipy.optimize import curve_fit
from simex_utils import Params, gen_dataset, ols_y_on_x, simex_nonlin_estimate


# ---------------------------------------------------------------------
# Nonlinear form used by the SIMEX fit (must match the model used upstream)
# β(λ) ≈ a + b / (c + λ)
# ---------------------------------------------------------------------
def nonlinear_model(lam, a, b, c):
    return a + b / (c + lam)


# ---------------------------------------------------------------------
# Compute pointwise Monte Carlo SE for β(λ) at each λ, on a single (X, Y)
# This is used to standardize residuals (resid / SE) in diagnostics.
#
# Args
#   X, Y        : arrays for a single dataset
#   lambdas     : 1D array of λ values used in SIMEX
#   B_mc        : MC reps per λ to estimate variability of the slope
#   sig_w       : measurement error SD (σ_w)
#   seed        : RNG seed for reproducibility
#
# Returns
#   se_means : 1D array, same length as lambdas; each entry is
#              SD(slope estimates across B_mc) / sqrt(B_mc)
# ---------------------------------------------------------------------
def simex_pointwise_mc_se(X, Y, lambdas, B_mc, sig_w, seed):
    r = default_rng(seed)
    se_means = []
    for lam in lambdas:
        slopes = []
        for _ in range(B_mc):
            # simulate extra measurement error for SIMEX at level λ
            Z = r.normal(0.0, 1.0, size=X.size)
            X_lam = X + np.sqrt(lam) * sig_w * Z
            # slope estimate from OLS Y ~ X_λ
            b_hat, _, _ = ols_y_on_x(Y, X_lam)
            slopes.append(b_hat[1])
        slopes = np.asarray(slopes)
        # MC SE of the mean slope at this λ (SD / sqrt(B_mc))
        se_means.append(slopes.std(ddof=1) / np.sqrt(B_mc))
    return np.asarray(se_means)


# ---------------------------------------------------------------------
# Run a full SIMEX fit + diagnostics on a single dataset (X, Y).
# It evaluates how well the nonlinear model matches β(λ) values,
# and how robust the fit is via leave-one-out prediction.
#
# Args
#   X, Y        : arrays for one dataset
#   sig_w       : measurement error SD (σ_w) for this dataset
#   lambdas     : 1D λ grid used by SIMEX
#   B           : SIMEX MC reps per λ to compute β(λ)
#   B_mc        : extra MC reps for pointwise SE (standardizing residuals)
#   seed        : RNG seed for reproducibility
#
# Returns
#   dict with diagnostics:
#     - rmse          : root mean squared error of fit on β(λ)
#     - r2            : coefficient of determination for the fit
#     - max_abs_z     : max standardized residual |resid|/SE(β)
#     - frac_within_2 : fraction of standardized residuals within ±2
#     - loo_rmse      : leave-one-out prediction RMSE across λ points
# ---------------------------------------------------------------------
def simex_fit_diagnostics_single(X, Y, sig_w, lambdas, B, B_mc=300, seed=42):
    # Run SIMEX to get β(λ) and fitted nonlinear parameters (a, b, c)
    res = simex_nonlin_estimate(X, Y, lambdas=lambdas, B=B, sig_w=sig_w, seed=seed)
    betas = res["beta_lambda"]; a, b, c = res["nl_params"]

    # Residuals of β(λ) vs. nonlinear fit
    beta_fit = nonlinear_model(lambdas, a, b, c)
    resid = betas - beta_fit

    # Standardize residuals using MC SE at each λ
    se_beta = simex_pointwise_mc_se(X, Y, lambdas, B_mc=B_mc, sig_w=sig_w, seed=seed+777)
    z = resid / np.maximum(se_beta, 1e-12)

    # In-sample fit quality metrics
    ss_res = np.sum(resid**2)
    ss_tot = np.sum((betas - betas.mean())**2) + 1e-12
    r2 = 1 - ss_res/ss_tot
    rmse = np.sqrt(np.mean(resid**2))
    max_abs_z = np.max(np.abs(z))
    frac_within_2 = float(np.mean(np.abs(z) <= 2.0))

    # Leave-one-out (LOO) across λ-points: drop one λ, refit (a, b, c), predict the left-out β
    loo_preds = []
    for i in range(len(lambdas)):
        mask = np.ones(len(lambdas), dtype=bool); mask[i] = False
        try:
            popt, _ = curve_fit(nonlinear_model, lambdas[mask], betas[mask], p0=(a,b,c), maxfev=5000)
            ai,bi,ci = popt
            loo_preds.append(nonlinear_model(lambdas[i], ai,bi,ci))
        except Exception:
            # if fit fails for a subset, mark as NaN to avoid crashing
            loo_preds.append(np.nan)
    loo_preds = np.asarray(loo_preds)
    loo_rmse = float(np.sqrt(np.nanmean((betas - loo_preds)**2)))

    return dict(
        rmse=float(rmse), r2=float(r2),
        max_abs_z=float(max_abs_z), frac_within_2=float(frac_within_2),
        loo_rmse=loo_rmse
    )


# ---------------------------------------------------------------------
# Multi-dataset, multi-σ_w driver:
# For each σ_w in sigw_grid, generate K datasets from p_template (with that σ_w),
# run the single-dataset diagnostics, and aggregate results.
#
# Args
#   p_template : Params object (template). Its sig_w is overridden by sigw_grid.
#   sigw_grid  : iterable of σ_w values to evaluate (e.g., [0.6, 0.8, 1.0])
#   K          : number of datasets per σ_w
#   B_mc       : MC reps for pointwise SE (for standardized residuals)
#   seed       : RNG seed for dataset generation
#
# Returns
#   summary_long     : long-form summary (mean/std/quantiles) per metric × σ_w
#   per_sigma_tables : dict {σ_w -> DataFrame of K rows with raw diagnostics}
# ---------------------------------------------------------------------
def simex_fit_diagnostics_multi(p_template, sigw_grid, K=30, B_mc=300, seed=2025):
    rng = default_rng(seed)  # (kept for symmetry; not used explicitly)
    lambdas = np.arange(p_template.lambda_start, p_template.lambda_end + 1e-12, p_template.lambda_step)
    out_rows = []
    per_sigma_tables = {}  # raw diagnostics per σ_w (K rows)

    for sigw in sigw_grid:
        rows = []
        for k in range(K):
            # Build Params for this dataset (same as template except sig_w and n_datasets=1)
            p = Params(
                n=p_template.n, beta0=p_template.beta0, beta1=p_template.beta1,
                mu_u=p_template.mu_u, sig_u=p_template.sig_u, sig_e=p_template.sig_e,
                sig_w=sigw, B=p_template.B, M=p_template.M, n_datasets=1,
                lambda_start=p_template.lambda_start,
                lambda_end=p_template.lambda_end,
                lambda_step=p_template.lambda_step
            )
            # Generate one dataset and run single-dataset diagnostics
            X, Y, U = gen_dataset(p, seed=seed + int(1e4*sigw) + k)
            d = simex_fit_diagnostics_single(
                X, Y, sig_w=sigw, lambdas=lambdas, B=p.B, B_mc=B_mc, seed=seed + 99 + k
            )
            d.update(dict(sig_w=float(sigw), run=k))
            rows.append(d)

        # Raw diagnostics table for this σ_w (K rows)
        df = pd.DataFrame(rows).sort_values("run")
        per_sigma_tables[sigw] = df

        # Group-level summary (mean/std/quantiles) for each metric
        summ = df.describe(percentiles=[0.1,0.25,0.5,0.75,0.9]).T.reset_index().rename(columns={"index":"metric"})
        summ.insert(0, "sig_w", sigw)
        out_rows.append(summ)

    # Concatenate summaries across σ_w values
    summary_long = pd.concat(out_rows, ignore_index=True)
    return summary_long, per_sigma_tables


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    # ---------------------------------------------------------------
    # CLI: users can override any setting; otherwise defaults are used
    # ---------------------------------------------------------------
    parser = argparse.ArgumentParser(
        prog="simex_fit_diagnostics",
        description="Run SIMEX fit diagnostics with optional overrides."
    )

    # ---- I/O & run controls (all optional) ----
    parser.add_argument("--base-dir", type=str, default="./simex_fit_diag_out",
                        help="Output directory. If not given, uses ./simex_fit_diag_out.")
    parser.add_argument("--sigw-grid", type=str, default="0.6,0.8,1.0",
                        help="Comma-separated sig_w grid, e.g. '0.6,0.8,1.0'.")
    parser.add_argument("--K", type=int, default=30,
                        help="Number of datasets per sig_w.")
    parser.add_argument("--B-mc", dest="B_mc", type=int, default=400,
                        help="MC reps for pointwise SE (used for standardized residuals).")
    parser.add_argument("--seed", type=int, default=1234,
                        help="Global seed for reproducibility.")

    # ---- Template params (optional; defaults = your current code) ----
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--beta0", type=float, default=2.0)
    parser.add_argument("--beta1", type=float, default=3.0)
    parser.add_argument("--mu-u", dest="mu_u", type=float, default=0.0)
    parser.add_argument("--sig-u", dest="sig_u", type=float, default=1.0)
    parser.add_argument("--sig-e", dest="sig_e", type=float, default=1.0)
    parser.add_argument("--sig-w-template", dest="sig_w_template", type=float, default=0.7,
                        help="Template-only value (real sig_w comes from --sigw-grid).")
    parser.add_argument("--B", type=int, default=100,
                        help="SIMEX reps per lambda (passed into single-run).")
    parser.add_argument("--M", type=int, default=100)
    parser.add_argument("--lambda-start", type=float, default=0.0)
    parser.add_argument("--lambda-end", type=float, default=2.0)
    parser.add_argument("--lambda-step", type=float, default=0.05)

    args = parser.parse_args()

    # ---- Resolve paths & grids ----
    BASE_DIR = Path(args.base_dir)
    BASE_DIR.mkdir(parents=True, exist_ok=True)

    # Parse the σ_w grid string, or fall back to the default
    sigw_grid = [float(x) for x in args.sigw_grid.split(",")] if args.sigw_grid else [0.6, 0.8, 1.0]

    # ---- Build template (sig_w in template is a placeholder; real values come from --sigw-grid) ----
    p_template = Params(
        n=args.n, beta0=args.beta0, beta1=args.beta1,
        mu_u=args.mu_u, sig_u=args.sig_u, sig_e=args.sig_e, sig_w=args.sig_w_template,
        B=args.B, M=args.M, n_datasets=1,
        lambda_start=args.lambda_start, lambda_end=args.lambda_end, lambda_step=args.lambda_step
    )

    # ---- Run diagnostics ----
    summary_long, per_sigma = simex_fit_diagnostics_multi(
        p_template, sigw_grid, K=args.K, B_mc=args.B_mc, seed=args.seed
    )

    # ---- Save outputs under BASE_DIR ----
    summary_path = BASE_DIR / "simex_fit_diagnostics_summary.csv"
    summary_long.to_csv(summary_path, index=False)

    # Optional: dump per-σ_w raw tables (one file per σ_w)
    for sw, df in per_sigma.items():
        df.to_csv(BASE_DIR / f"per_sigma_{sw:.1f}.csv", index=False)

    print("[OK] Diagnostics finished.")
    print(" base_dir     :", BASE_DIR)
    print(" summary_long :", summary_path)
    print(" per_sigma_*  :", f"{len(per_sigma)} files (one per sig_w)")

