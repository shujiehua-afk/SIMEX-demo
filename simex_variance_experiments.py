


import numpy as np
import pandas as pd
from numpy.random import default_rng
from simex_utils import (
    Params,
    gen_dataset,
    simex_nonlin_estimate,
    corrected_estimator_and_se,
    parametric_replicates_Y_given_X,
)

# =====================================================================
# Step 1: Independent datasets
# Generate R independent (X,Y) datasets, compute SIMEX and corrected each time.
# Returns arrays of estimates and keeps the first dataset for later steps.
# =====================================================================
def exp_step1_independent_datasets(p: Params, R: int = 200, seed: int = 123):
    rng = default_rng(seed)
    lambdas = np.arange(p.lambda_start, p.lambda_end + 1e-12, p.lambda_step)
    simex_vals, corr_vals = [], []
    X0 = Y0 = None

    for r in range(R):
        X, Y, _ = gen_dataset(p, seed=seed + r)
        if r == 0:
            X0, Y0 = X.copy(), Y.copy()

        res = simex_nonlin_estimate(X, Y, lambdas=lambdas, B=p.B, sig_w=p.sig_w, seed=seed + 10_000 + r)
        simex_vals.append(res["simex_est"])

        corr = corrected_estimator_and_se(X, Y, sig_w=p.sig_w)
        corr_vals.append(corr["beta_corrected"])

    return {
        "simex": np.array(simex_vals, dtype=float),
        "corrected": np.array(corr_vals, dtype=float),
        "first_dataset": (X0, Y0),
    }


# =====================================================================
# Step 2: Fixed X, parametric replicates for Y
# Given a fixed X0, generate M parametric Y replicates and compute SIMEX/corrected.
# =====================================================================
def exp_step2_fixedX_paramY(X0, Y0, p: Params, M: int = 200, seed: int = 456):
    lambdas = np.arange(p.lambda_start, p.lambda_end + 1e-12, p.lambda_step)
    Y_reps = parametric_replicates_Y_given_X(X0, Y0, M=M, seed=seed + 9999)
    simex_vals, corr_vals = [], []

    for m in range(M):
        Ym = Y_reps[m]
        res = simex_nonlin_estimate(X0, Ym, lambdas=lambdas, B=p.B, sig_w=p.sig_w, seed=seed + 2000 + m)
        simex_vals.append(res["simex_est"])

        corr = corrected_estimator_and_se(X0, Ym, sig_w=p.sig_w)
        corr_vals.append(corr["beta_corrected"])

    return {"simex": np.array(simex_vals, dtype=float),
            "corrected": np.array(corr_vals, dtype=float)}


# =====================================================================
# Step 3: Bootstrap from first dataset
# Resample (X0,Y0) with replacement Rb times and compute SIMEX/corrected.
# =====================================================================
def exp_step3_bootstrap_from_first(X0, Y0, p: Params, Rb: int = 200, seed: int = 789):
    rng = default_rng(seed)
    n = len(X0)
    lambdas = np.arange(p.lambda_start, p.lambda_end + 1e-12, p.lambda_step)
    simex_vals, corr_vals = [], []

    for r in range(Rb):
        idx = rng.integers(0, n, size=n)
        Xb, Yb = np.asarray(X0)[idx], np.asarray(Y0)[idx]

        res = simex_nonlin_estimate(Xb, Yb, lambdas=lambdas, B=p.B, sig_w=p.sig_w, seed=seed + 2000 + r)
        simex_vals.append(res["simex_est"])

        corr = corrected_estimator_and_se(Xb, Yb, sig_w=p.sig_w)
        corr_vals.append(corr["beta_corrected"])

    return {"simex": np.array(simex_vals, dtype=float),
            "corrected": np.array(corr_vals, dtype=float)}


# =====================================================================
# Step 4: Repeat SIMEX on the same dataset
# For fixed (X,Y), repeat SIMEX Rsim times to capture Monte Carlo noise.
# =====================================================================
def exp_step4_repeat_simex(X, Y, p, Rsim=100, seed=777):
    rng = default_rng(seed)
    lambdas = np.arange(p.lambda_start, p.lambda_end + 1e-12, p.lambda_step)
    simex_vals = []
    for r in range(Rsim):
        res = simex_nonlin_estimate(X, Y, lambdas=lambdas, B=p.B, sig_w=p.sig_w, seed=seed + r)
        simex_vals.append(res["simex_est"])
    return {"simex": np.array(simex_vals)}


# =====================================================================
# Main execution block
# Loops over a grid of σ_w values, runs all four experiments, and
# summarizes the variances of SIMEX and corrected estimates.
# =====================================================================
if __name__ == '__main__':
    sigw_list = [0.6, 0.7, 0.8, 0.9, 1.0]
    all_results = {}

    for sigw in sigw_list:
        p = Params()
        p.sig_w = sigw

        test_1 = exp_step1_independent_datasets(p, R=100)
        X_obs, Y_obs = test_1['first_dataset']
        test_2 = exp_step2_fixedX_paramY(X_obs, Y_obs, p, M=100)
        test_3 = exp_step3_bootstrap_from_first(X_obs, Y_obs, p, Rb=100)
        test_4 = exp_step4_repeat_simex(X_obs, Y_obs, p, Rsim=100)

        all_results[sigw] = {"test1": test_1, "test2": test_2,
                             "test3": test_3, "test4": test_4}

    # --- Summarize variances across tests per σ_w ---
    rows = []
    for sigw, res in all_results.items():
        for k, df in res.items():
            if k == "test4":  # no corrected estimator in test4
                rows.append({"sig_w": sigw, "test": k,
                             "simex_var": np.var(df["simex"], ddof=1),
                             "corrected_var": np.nan})
            else:
                rows.append({"sig_w": sigw, "test": k,
                             "simex_var": np.var(df["simex"], ddof=1),
                             "corrected_var": np.var(df["corrected"], ddof=1)})

    summary_df = pd.DataFrame(rows)
    print(summary_df)
    # Optional: summary_df.to_csv("simex_variance_summary.csv", index=False)






