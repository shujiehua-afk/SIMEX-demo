


# =========================================== #
# Batch SIMEX pipeline runner
#
# This script performs multiple SIMEX simulations for different levels of 
# measurement error variance (sig_w). For each value in SIG_W_GRID, it will:
#   - Run N_RUNS_PER_GROUP independent simulations.
#   - Save the results into individual run folders.
#   - Aggregate results of each group into a summary_all.csv.
#   - Finally, create an INDEX_groups.csv pointing to all group summaries.
#
# Usage:
#   - Adjust USER_BASE_DIR to the directory where you want results stored.
#   - Run the script via: python batch_simex.py
#   - Output: USER_BASE_DIR will contain one folder per sig_w group and 
#     a global index CSV.
# =========================================== #

from pathlib import Path
import pandas as pd
import numpy as np

# Optional dependency: tqdm for progress bar (falls back to range if missing)
try:
    from tqdm import trange
except Exception:
    trange = range

# Import your own SIMEX utilities (make sure simex_utils.py is accessible)
from simex_utils import Params, run_pipeline

# =========================================== #

# User can change this: root directory for all results
USER_BASE_DIR = "./simex_pipeline_demo"

# Number of runs per sig_w group
N_RUNS_PER_GROUP = 100

# Grid of measurement error variance levels to test
SIG_W_GRID = [1.0, 0.9, 0.8, 0.7, 0.6]

# Template of other parameters (sig_w will be overwritten inside the loop)
P_TEMPLATE = Params(
    n=1000, beta0=2, beta1=3,
    mu_u=0, sig_u=1, sig_e=1, sig_w=0.7,   # sig_w will be replaced
    B=100, M=100, n_datasets=50,
    lambda_start=0, lambda_end=2, lambda_step=0.05
)

# Collects index entries for all group-level summaries
all_groups_index = []

# =========================================== #


def main():
    for sigw in SIG_W_GRID:
        # Create directory for this sig_w group
        group_dir = BASE_DIR / f"sigw_{sigw:.1f}"
        group_dir.mkdir(parents=True, exist_ok=True)

        summaries = []
        for r in trange(N_RUNS_PER_GROUP, desc=f"sig_w={sigw:.1f}"):
            # Build parameters for this run (only sig_w differs from template)
            p = Params(
                n=P_TEMPLATE.n, beta0=P_TEMPLATE.beta0, beta1=P_TEMPLATE.beta1,
                mu_u=P_TEMPLATE.mu_u, sig_u=P_TEMPLATE.sig_u, sig_e=P_TEMPLATE.sig_e,
                sig_w=sigw,
                B=P_TEMPLATE.B, M=P_TEMPLATE.M, n_datasets=P_TEMPLATE.n_datasets,
                lambda_start=P_TEMPLATE.lambda_start,
                lambda_end=P_TEMPLATE.lambda_end,
                lambda_step=P_TEMPLATE.lambda_step
            )

            # Directory for this single run
            run_dir = group_dir / f"run_{r:03d}"

            # Execute pipeline and collect the summary dict
            out = run_pipeline(p, seed=42 + r, save_dir=run_dir)
            summaries.append(out["summary"])

        # Aggregate the 100 runs for this group into one CSV
        df_group = pd.DataFrame(summaries)
        group_summary_csv = group_dir / "summary_all.csv"
        df_group.to_csv(group_summary_csv, index=False)

        # Add to global index list
        all_groups_index.append({
            "sig_w": sigw,
            "summary_csv": str(group_summary_csv)
        })

    # Write global index CSV pointing to all group summaries
    pd.DataFrame(all_groups_index).to_csv(BASE_DIR / "INDEX_groups.csv", index=False)

    print(f"Done. Results saved under: {BASE_DIR}")


if __name__ == "__main__":
    # Initialize base directory before running
    BASE_DIR = Path(USER_BASE_DIR)
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    main()

