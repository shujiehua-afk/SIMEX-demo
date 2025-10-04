# SIMEX Project

## Background
*(To be completed: describe the problem context, motivation, and statistical background here.)*

## Problem Statement
*(To be completed: specify the goals of the project, hypotheses, or research questions here.)*

---

## Project Structure

- **`simex_utils.py`**  
  Core utility functions used across the project.  
  Includes:
  - `Params` class (experiment configuration)  
  - Data generation (`gen_dataset`)  
  - Estimation functions (`ols_y_on_x`, `simex_nonlin_estimate`, `corrected_estimator_and_se`)  
  - Parametric replicates (`parametric_replicates_Y_given_X`)  

- **`simex_batch.py`**  
  Batch execution script for running SIMEX pipelines across multiple measurement error variances (`σ_w`).  
  - Iterates over a grid of `σ_w` values (e.g., 1.0 → 0.6)  
  - Runs multiple replications per group  
  - Saves group-level summaries (`summary_all.csv`) and an index file (`INDEX_groups.csv`)  

- **`simex_fit_diagnostics.py`**  
  Diagnostics for evaluating the SIMEX nonlinear fit.  
  - Computes residual-based metrics (RMSE, R², standardized residual checks)  
  - Supports leave-one-out (LOO) diagnostics  
  - Runs across multiple datasets and `σ_w` values, producing summary tables  
  - Command-line arguments allow customization of base directory, grid, seeds, etc.  

- **`simex_variance_experiments.py`**  
  Experimental script to compare SIMEX vs. corrected estimators under different resampling schemes.  
  - Step 1: Multiple independent datasets  
  - Step 2: Fixed X with parametric Y replicates  
  - Step 3: Bootstrap from one dataset  
  - Step 4: Repeated SIMEX runs on the same dataset (Monte Carlo noise)  
  - Outputs a summary DataFrame (or CSV) with variances of SIMEX and corrected estimators.  

---

## Installation
*(To be completed: add instructions for environment setup, dependencies, and installation here.)*

## Usage
*(To be completed: provide examples of how to run each script, including command-line options if applicable.)*

## Outputs
*(To be completed: describe output formats and how to interpret results.)*

## License
*(To be completed: add license information here, e.g., MIT, GPL, etc.)*
