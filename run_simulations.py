"""Work 2, 3, 4: Monte Carlo Simulation Suite for Bias, RMSE, 95% CI Coverage, Post-Treatment Bias, and Learnability Boundaries."""
from __future__ import annotations

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import argparse
import json
import time
from pathlib import Path
from joblib import Parallel, delayed
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from dgp import (
    generate_dgp_baseline,
    generate_dgp_identifiability,
    generate_dgp_misspecification,
    generate_dgp_post_treatment,
    generate_dgp_sure_things_vs_persuadables,
)
from estimators import (
    estimate_aipw_insample,
    estimate_dml_irm,
    estimate_naive,
    estimate_oracle,
    evaluate_policy,
    ipw_score,
    select_top_k_fraction,
    summary_from_scores,
)


def summarize_mc_runs(records: list[dict], group_keys: list[str]) -> pd.DataFrame:
    """Aggregate replication-level records into Bias, RMSE, Empirical SD, Mean SE, 95% Coverage, CI Width, and Power."""
    df = pd.DataFrame(records)
    rows = []
    for keys, sub in df.groupby(group_keys, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_keys, keys))
        r_count = len(sub)
        pop_ate = float(sub["pop_ate"].iloc[0])
        est = sub["estimate"].to_numpy(dtype=float)
        se = sub["se"].to_numpy(dtype=float)
        low = sub["ci_lower"].to_numpy(dtype=float)
        high = sub["ci_upper"].to_numpy(dtype=float)
        sample_ate = sub["sample_ate"].to_numpy(dtype=float)

        err_pop = est - pop_ate
        cov_pop = (low <= pop_ate) & (pop_ate <= high)
        cov_sample = (low <= sample_ate) & (sample_ate <= high)
        cov_rate = float(np.mean(cov_pop))

        row.update({
            "replications": int(r_count),
            "true_pop_ate": pop_ate,
            "mean_sample_ate": float(np.mean(sample_ate)),
            "mean_estimate": float(np.mean(est)),
            "bias": float(np.mean(err_pop)),
            "abs_bias": float(abs(np.mean(err_pop))),
            "rmse": float(np.sqrt(np.mean(err_pop ** 2))),
            "empirical_sd": float(np.std(est, ddof=1)),
            "mean_se": float(np.mean(se)),
            "se_calibration_ratio": float(np.mean(se) / np.std(est, ddof=1)) if np.std(est, ddof=1) > 0 else np.nan,
            "coverage_95": cov_rate,
            "coverage_mc_se": float(np.sqrt(cov_rate * (1.0 - cov_rate) / r_count)),
            "coverage_sample_ate_95": float(np.mean(cov_sample)),
            "mean_ci_width": float(np.mean(high - low)),
            "power_gt_zero": float(np.mean(low > 0.0)),
            "reject_zero_rate": float(np.mean((low > 0.0) | (high < 0.0))),
        })
        for extra_col in [
            "true_extreme_e_fraction",
            "actual_clipped_fraction",
            "outcome_auc",
            "propensity_auc",
            "ess_control",
            "ess_treated",
            "logistic_beta_t",
        ]:
            if extra_col in sub.columns:
                row[f"mean_{extra_col}"] = float(sub[extra_col].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def _single_baseline_missing_cells_rep(
    rep: int,
    exp_name: str,
    setting_label: str,
    n: int,
    target_ate: float,
    overlap_scale: float,
    base_seed: int,
) -> list[dict]:
    """
    Compute the 2 newly added cells of the 2x2 H1 factorial design on the exact same seed & DGP draw:
      - `AIPW No Cross-Fit (Regular Tree)`
      - `Cross-Fitted DML-IRM (High-Capacity Tree)`
    plus exact `true_extreme_e_fraction` and Oracle `actual_clipped_fraction`.
    """
    with threadpool_limits(limits=1):
        seed = base_seed + rep * 97
        data = generate_dgp_baseline(
            n=n, target_ate=target_ate, overlap_scale=overlap_scale, seed=seed
        )
        x, t, y = data["x"], data["t"], data["y"]
        true_extreme_e = float(np.mean((data["e_true"] < 0.02) | (data["e_true"] > 0.98)))
        oracle_clipped = float(np.mean((data["e_true"] < 0.001) | (data["e_true"] > 0.999)))

        res_insample_reg = estimate_aipw_insample(
            x, t, y, e_learner="gbdt", m_learner="gbdt", seed=seed
        )
        res_dml_hi = estimate_dml_irm(
            x, t, y, n_splits=3, e_learner="gbdt_overfit", m_learner="gbdt_overfit", seed=seed
        )
        return [
            {
                "experiment": exp_name,
                "setting": setting_label,
                "n": n,
                "overlap_scale": overlap_scale,
                "method": "AIPW No Cross-Fit (Regular Tree)",
                "pop_ate": data["pop_ate"],
                "sample_ate": data["sample_ate"],
                "logistic_beta_t": data["logistic_beta_t"],
                "estimate": res_insample_reg["estimate"],
                "se": res_insample_reg["se"],
                "ci_lower": res_insample_reg["ci_lower"],
                "ci_upper": res_insample_reg["ci_upper"],
                "true_extreme_e_fraction": true_extreme_e,
                "actual_clipped_fraction": res_insample_reg["diagnostics"]["clipped_fraction"],
                "oracle_clipped_fraction": oracle_clipped,
            },
            {
                "experiment": exp_name,
                "setting": setting_label,
                "n": n,
                "overlap_scale": overlap_scale,
                "method": "Cross-Fitted DML-IRM (High-Capacity Tree)",
                "pop_ate": data["pop_ate"],
                "sample_ate": data["sample_ate"],
                "logistic_beta_t": data["logistic_beta_t"],
                "estimate": res_dml_hi["estimate"],
                "se": res_dml_hi["se"],
                "ci_lower": res_dml_hi["ci_lower"],
                "ci_upper": res_dml_hi["ci_upper"],
                "true_extreme_e_fraction": true_extreme_e,
                "actual_clipped_fraction": res_dml_hi["diagnostics"]["clipped_fraction"],
                "oracle_clipped_fraction": oracle_clipped,
            },
        ]


def _single_baseline_rep(
    rep: int,
    exp_name: str,
    setting_label: str,
    n: int,
    target_ate: float,
    overlap_scale: float,
    base_seed: int,
) -> list[dict]:
    """
    Run one Monte Carlo replication with a complete 2x2 factorial design for H1:
      - Cross-Fitting (No vs 3-Fold) x Tree Capacity (Regular `gbdt` vs High-Capacity `gbdt_overfit`)
    plus Naive Diff-in-Means and Oracle AIPW.
    """
    with threadpool_limits(limits=1):
        seed = base_seed + rep * 97
        data = generate_dgp_baseline(
            n=n, target_ate=target_ate, overlap_scale=overlap_scale, seed=seed
        )
        x, t, y = data["x"], data["t"], data["y"]
        true_extreme_e = float(np.mean((data["e_true"] < 0.02) | (data["e_true"] > 0.98)))
        out = []

        # 1. Naive Diff-in-Means
        res_naive = estimate_naive(t, y)
        out.append({
            "experiment": exp_name,
            "setting": setting_label,
            "n": n,
            "overlap_scale": overlap_scale,
            "method": "Naive Diff-in-Means",
            "pop_ate": data["pop_ate"],
            "sample_ate": data["sample_ate"],
            "logistic_beta_t": data["logistic_beta_t"],
            "estimate": res_naive["estimate"],
            "se": res_naive["se"],
            "ci_lower": res_naive["ci_lower"],
            "ci_upper": res_naive["ci_upper"],
            "true_extreme_e_fraction": true_extreme_e,
            "actual_clipped_fraction": 0.0,
        })

        # 2a. In-sample AIPW WITHOUT cross-fitting (Regular Tree `gbdt`)
        res_insample_reg = estimate_aipw_insample(
            x, t, y, e_learner="gbdt", m_learner="gbdt", seed=seed
        )
        out.append({
            "experiment": exp_name,
            "setting": setting_label,
            "n": n,
            "overlap_scale": overlap_scale,
            "method": "AIPW No Cross-Fit (Regular Tree)",
            "pop_ate": data["pop_ate"],
            "sample_ate": data["sample_ate"],
            "logistic_beta_t": data["logistic_beta_t"],
            "estimate": res_insample_reg["estimate"],
            "se": res_insample_reg["se"],
            "ci_lower": res_insample_reg["ci_lower"],
            "ci_upper": res_insample_reg["ci_upper"],
            "true_extreme_e_fraction": true_extreme_e,
            "actual_clipped_fraction": res_insample_reg["diagnostics"]["clipped_fraction"],
        })

        # 2b. In-sample AIPW WITHOUT cross-fitting (High-Capacity Tree `gbdt_overfit`)
        res_insample_hi = estimate_aipw_insample(
            x, t, y, e_learner="gbdt_overfit", m_learner="gbdt_overfit", seed=seed
        )
        out.append({
            "experiment": exp_name,
            "setting": setting_label,
            "n": n,
            "overlap_scale": overlap_scale,
            "method": "AIPW No Cross-Fit (High-Capacity Tree)",
            "pop_ate": data["pop_ate"],
            "sample_ate": data["sample_ate"],
            "logistic_beta_t": data["logistic_beta_t"],
            "estimate": res_insample_hi["estimate"],
            "se": res_insample_hi["se"],
            "ci_lower": res_insample_hi["ci_lower"],
            "ci_upper": res_insample_hi["ci_upper"],
            "true_extreme_e_fraction": true_extreme_e,
            "actual_clipped_fraction": res_insample_hi["diagnostics"]["clipped_fraction"],
        })

        # 3a. 3-Fold Cross-Fitted AIPW / DML-IRM (Regular Tree `gbdt` - Main Estimator)
        res_dml_reg = estimate_dml_irm(
            x, t, y, n_splits=3, e_learner="gbdt", m_learner="gbdt", seed=seed
        )
        out.append({
            "experiment": exp_name,
            "setting": setting_label,
            "n": n,
            "overlap_scale": overlap_scale,
            "method": "Cross-Fitted DML-IRM (Regular Tree)",
            "pop_ate": data["pop_ate"],
            "sample_ate": data["sample_ate"],
            "logistic_beta_t": data["logistic_beta_t"],
            "estimate": res_dml_reg["estimate"],
            "se": res_dml_reg["se"],
            "ci_lower": res_dml_reg["ci_lower"],
            "ci_upper": res_dml_reg["ci_upper"],
            "true_extreme_e_fraction": true_extreme_e,
            "actual_clipped_fraction": res_dml_reg["diagnostics"]["clipped_fraction"],
        })

        # 3b. 3-Fold Cross-Fitted AIPW / DML-IRM (High-Capacity Tree `gbdt_overfit`)
        res_dml_hi = estimate_dml_irm(
            x, t, y, n_splits=3, e_learner="gbdt_overfit", m_learner="gbdt_overfit", seed=seed
        )
        out.append({
            "experiment": exp_name,
            "setting": setting_label,
            "n": n,
            "overlap_scale": overlap_scale,
            "method": "Cross-Fitted DML-IRM (High-Capacity Tree)",
            "pop_ate": data["pop_ate"],
            "sample_ate": data["sample_ate"],
            "logistic_beta_t": data["logistic_beta_t"],
            "estimate": res_dml_hi["estimate"],
            "se": res_dml_hi["se"],
            "ci_lower": res_dml_hi["ci_lower"],
            "ci_upper": res_dml_hi["ci_upper"],
            "true_extreme_e_fraction": true_extreme_e,
            "actual_clipped_fraction": res_dml_hi["diagnostics"]["clipped_fraction"],
        })

        # 4. Ground-Truth Oracle AIPW (clip_bounds=(0.001, 0.999))
        res_oracle = estimate_oracle(t, y, data["p0_true"], data["p1_true"], data["e_true"], clip_bounds=(0.001, 0.999))
        out.append({
            "experiment": exp_name,
            "setting": setting_label,
            "n": n,
            "overlap_scale": overlap_scale,
            "method": "Oracle AIPW (True Nuances)",
            "pop_ate": data["pop_ate"],
            "sample_ate": data["sample_ate"],
            "logistic_beta_t": data["logistic_beta_t"],
            "estimate": res_oracle["estimate"],
            "se": res_oracle["se"],
            "ci_lower": res_oracle["ci_lower"],
            "ci_upper": res_oracle["ci_upper"],
            "true_extreme_e_fraction": true_extreme_e,
            "actual_clipped_fraction": float(np.mean((data["e_true"] < 0.001) | (data["e_true"] > 0.999))),
        })
        return out


def _single_misspec_rep(rep: int, n: int, base_seed: int) -> list[dict]:
    """Run one replication of Experiment 4: 4-Way Model Misspecification & Double Robustness."""
    with threadpool_limits(limits=1):
        seed = base_seed + rep * 101
        data = generate_dgp_misspecification(n=n, target_ate=0.03, seed=seed)
        df, t, y = data["df"], data["t"], data["y"]
        true_cols = data["true_cols"]
        raw_cols = data["raw_cols"]

        specs = [
            ("1. Both Correct", true_cols, true_cols),
            ("2. Only e(X) Correct (m Misspecified)", true_cols, raw_cols),
            ("3. Only m(X) Correct (e Misspecified)", raw_cols, true_cols),
            ("4. Both Misspecified", raw_cols, raw_cols),
        ]
        out = []
        for spec_name, e_cols, m_cols in specs:
            res = estimate_dml_irm(
                df, t, y, n_splits=3,
                e_learner="logistic", m_learner="logistic",
                e_features=e_cols, m_features=m_cols,
                seed=seed,
            )
            # Also record pure Outcome Regression (plug-in m1 - m0) and pure IPW under the exact same fold predictions
            or_scores = res["m1"] - res["m0"]
            or_sum = summary_from_scores(or_scores)
            ipw_scores = ipw_score(y, t, res["e"])
            ipw_sum = summary_from_scores(ipw_scores)

            for est_name, s_dict in [
                ("Cross-Fitted AIPW / DML-IRM", res),
                ("Plug-in Outcome Regression (Naive SE, Ignores 1st-Stage Var)", or_sum),
                ("IPW (Propensity Only)", ipw_sum),
            ]:
                out.append({
                    "experiment": "misspecification",
                    "setting": spec_name,
                    "method": est_name,
                    "pop_ate": data["pop_ate"],
                    "sample_ate": data["sample_ate"],
                    "estimate": s_dict["estimate"],
                    "se": s_dict["se"],
                    "ci_lower": s_dict["ci_lower"],
                    "ci_upper": s_dict["ci_upper"],
                })
        return out


def _single_post_treatment_rep(rep: int, scenario: str, n: int, base_seed: int) -> list[dict]:
    """Run one replication of Work 3: Post-Treatment Variable Adjustment Bias."""
    with threadpool_limits(limits=1):
        seed = base_seed + rep * 103
        data = generate_dgp_post_treatment(n=n, scenario=scenario, seed=seed)
        df, t, y = data["df"], data["t"], data["y"]
        pre_cols = data["pre_cols"]
        all_cols = data["all_cols"]

        out = []
        for feat_label, cols in [
            ("Pre-Treatment Features Only (Correct)", pre_cols),
            ("Pre + Post-Treatment Visit (Incorrect Adjustment)", all_cols),
        ]:
            res = estimate_dml_irm(
                df, t, y, n_splits=3,
                e_learner="gbdt", m_learner="gbdt",
                e_features=cols, m_features=cols,
                seed=seed,
            )
            out.append({
                "scenario": scenario,
                "adjustment_set": feat_label,
                "pop_ate": data["pop_ate"],
                "sample_ate": data["sample_ate"],
                "estimate": res["estimate"],
                "se": res["se"],
                "ci_lower": res["ci_lower"],
                "ci_upper": res["ci_upper"],
                "outcome_auc": res["diagnostics"]["outcome_auc"],
                "propensity_auc": res["diagnostics"]["propensity_auc"],
            })
        return out


def _single_identifiability_rep(rep: int, mode: str, n: int, base_seed: int) -> dict:
    """Run one replication of Work 4: Identifiability & Uplift Learnability Boundaries."""
    with threadpool_limits(limits=1):
        seed = base_seed + rep * 107
        data = generate_dgp_identifiability(n=n, mode=mode, seed=seed)
        x, t, y = data["x"], data["t"], data["y"]

        res_dml = estimate_dml_irm(
            x, t, y, n_splits=3, e_learner="gbdt", m_learner="gbdt", seed=seed
        )
        # Train a DR-Learner uplift model on fold 1..2 and evaluate targeting policy on fold 0
        # or fit on first half and evaluate on second half to test individual uplift learnability!
        half = n // 2
        res_tr = estimate_dml_irm(
            x.iloc[:half], t[:half], y[:half], n_splits=2, e_learner="gbdt", m_learner="gbdt", seed=seed
        )
        cate_reg = HistGradientBoostingRegressor(
            max_iter=60, learning_rate=0.05, max_leaf_nodes=7, min_samples_leaf=30, random_state=seed
        ).fit(x.iloc[:half], res_tr["psi"])
        pred_tau_te = cate_reg.predict(x.iloc[half:])
        true_tau_te = data["tau_true"][half:]
        psi_te = res_dml["psi"][half:]

        # Policy: target users with predicted uplift > 0.01
        pi_pos = (pred_tau_te > 0.01).astype(float)
        true_policy_gain = float(np.mean(pi_pos * true_tau_te))
        est_policy_gain = float(np.mean(pi_pos * psi_te))
        pehe = float(np.sqrt(np.mean((pred_tau_te - true_tau_te) ** 2)))

        return {
            "mode": mode,
            "n": n,
            "method": "Cross-Fitted DML-IRM (Regular Tree)",
            "pop_ate": data["pop_ate"],
            "sample_ate": data["sample_ate"],
            "estimate": res_dml["estimate"],
            "se": res_dml["se"],
            "ci_lower": res_dml["ci_lower"],
            "ci_upper": res_dml["ci_upper"],
            "true_extreme_e_fraction": float(np.mean((data["e_true"] < 0.02) | (data["e_true"] > 0.98))),
            "actual_clipped_fraction": res_dml["diagnostics"]["clipped_fraction"],
            "ess_control": res_dml["diagnostics"]["ess_control"],
            "true_cate_std": float(np.std(data["tau_true"])),
            "cate_pehe": pehe,
            "true_policy_gain": true_policy_gain,
            "est_policy_gain": est_policy_gain,
            "target_fraction": float(np.mean(pi_pos)),
        }


def _single_sure_things_policy_rep(rep: int, n: int, base_seed: int) -> list[dict]:
    """
    Run one replication of Question 10 Bottleneck Verification under a UNIFIED Train-Test Protocol:
      1. ALL targeting models (Baseline Prob mu0, T-Learner, and DR-Learner) are fitted
         STRICTLY on `train_data`.
      2. Locked models score `test_data["x"]` to determine targeting decisions pi(X_test).
      3. `test_data` outcomes (T_test, Y_test) are used ONLY to compute cross-fitted DR evaluation
         scores psi_te and AIPW baseline conversion scores psi0_te.
    """
    with threadpool_limits(limits=1):
        seed = base_seed + rep * 131
        train_data = generate_dgp_sure_things_vs_persuadables(n=n, seed=seed)
        test_data = generate_dgp_sure_things_vs_persuadables(n=n, seed=seed + 1)

        # 1. Fit 3-fold OOF DR pseudo-outcomes on train_data for DR-Learner,
        #    and fit full-train mu0/mu1 models on train_data for Baseline Prob and T-Learner
        res_tr = estimate_dml_irm(
            train_data["x"], train_data["t"], train_data["y"],
            n_splits=3, e_learner="gbdt", m_learner="gbdt", seed=seed,
        )
        dr_reg = HistGradientBoostingRegressor(
            max_iter=50, max_bins=64, learning_rate=0.06, max_leaf_nodes=10, min_samples_leaf=25, random_state=seed
        ).fit(train_data["x"], res_tr["psi"])

        # Fit mu0 and mu1 on train_data and predict on test_data["x"] (strictly out-of-sample)
        res_tr_full = estimate_aipw_insample(
            train_data["x"], train_data["t"], train_data["y"],
            e_learner="gbdt", m_learner="gbdt", seed=seed,
        )
        # Re-fit nuisance on train_data and predict on test_data["x"]
        from estimators import _fit_and_predict_nuisance
        x_tr_arr = train_data["x"].to_numpy(dtype=float)
        x_te_arr = test_data["x"].to_numpy(dtype=float)
        mu0_pred_from_train, mu1_pred_from_train, _ = _fit_and_predict_nuisance(
            x_tr_arr, x_tr_arr, train_data["t"], train_data["y"],
            x_te_arr, x_te_arr,
            e_learner="gbdt", m_learner="gbdt", seed=seed,
        )
        up_t_from_train = mu1_pred_from_train - mu0_pred_from_train
        up_dr_from_train = dr_reg.predict(test_data["x"])

        # 2. Evaluate on test set via 3-fold cross-fitted DR scores (test T, Y used ONLY for evaluation)
        res_te = estimate_dml_irm(
            test_data["x"], test_data["t"], test_data["y"],
            n_splits=3, e_learner="gbdt", m_learner="gbdt", seed=seed + 1,
        )
        psi_te = res_te["psi"]
        t_te = test_data["t"].astype(float)
        y_te = test_data["y"].astype(float)
        mu0_te_eval = res_te["m0"]
        e_te_eval = res_te["e"]
        # AIPW-corrected baseline conversion score psi0_te on test set
        psi0_te = mu0_te_eval + (1.0 - t_te) * (y_te - mu0_te_eval) / (1.0 - e_te_eval)

        true_tau_te = test_data["tau_true"]
        true_p0_te = test_data["p0_true"]

        # 3. Lock policies using ONLY train-fitted predictions on X_test
        q = 0.20
        pi_rand = np.full(len(psi_te), q, dtype=float)
        pi_mu0 = select_top_k_fraction(mu0_pred_from_train, q, seed=seed)
        pi_t = select_top_k_fraction(up_t_from_train, q, seed=seed)
        pi_dr = select_top_k_fraction(up_dr_from_train, q, seed=seed)

        margin = 100.0
        contact_cost = 1.50
        coupon_subsidy = 15.0

        out = []
        for pol_name, pi_vec in [
            ("Random 20% Targeting", pi_rand),
            ("Baseline Purchase Prob mu0 Top 20% (Train-Fitted)", pi_mu0),
            ("T-Learner Uplift Top 20% (Train-Fitted)", pi_t),
            ("DR-Learner Uplift Top 20% (Train-Fitted)", pi_dr),
        ]:
            est_pol = evaluate_policy(psi_te, pi_vec, cost=contact_cost, margin=margin, model_name=pol_name, policy_id="top_20pct")
            true_inc = float(np.mean(pi_vec * true_tau_te))
            true_net_contact = float(np.mean(pi_vec * (margin * true_tau_te - contact_cost)))
            true_baseline_subsidy = float(np.mean(pi_vec * coupon_subsidy * true_p0_te))
            est_baseline_subsidy_aipw = float(np.mean(pi_vec * coupon_subsidy * psi0_te))
            true_net_coupon = float(np.mean(pi_vec * ((margin - coupon_subsidy) * true_tau_te - coupon_subsidy * true_p0_te - 0.50)))
            est_net_coupon_aipw = float(np.mean(pi_vec * ((margin - coupon_subsidy) * psi_te - coupon_subsidy * psi0_te - 0.50)))

            out.append({
                "policy": pol_name,
                "treatment_fraction": q,
                "true_incremental_effect": true_inc,
                "est_incremental_effect": est_pol["estimated_incremental_effect"],
                "est_se": est_pol["standard_error"],
                "ci_lower": est_pol["ci_lower"],
                "ci_upper": est_pol["ci_upper"],
                "covered_true_policy_effect": bool(est_pol["ci_lower"] <= true_inc <= est_pol["ci_upper"]),
                "true_net_gain_contact_cost": true_net_contact,
                "est_net_gain_contact_cost": est_pol["estimated_net_gain"],
                "true_baseline_subsidy": true_baseline_subsidy,
                "est_baseline_subsidy_aipw": est_baseline_subsidy_aipw,
                "true_net_gain_with_coupon_subsidy": true_net_coupon,
                "est_net_gain_with_coupon_subsidy": est_net_coupon_aipw,
            })
        return out


def plot_simulation_figures(
    core_df: pd.DataFrame,
    misspec_df: pd.DataFrame,
    post_df: pd.DataFrame,
    ident_df: pd.DataFrame,
    fig_dir: Path,
):
    """Create publication-grade multi-panel figures for all simulation experiments."""
    fig_dir.mkdir(parents=True, exist_ok=True)

    # --- Figure 1: 4 Core Reliability Experiments (Effect Size, Sample Size, Overlap, Model Misspecification) ---
    fig, axes = plt.subplots(2, 2, figsize=(13.8, 9.8))
    method_styles = {
        "Naive Diff-in-Means": ("#d62728", "o", "--"),
        "AIPW No Cross-Fit (Regular Tree)": ("#e377c2", "v", "--"),
        "AIPW No Cross-Fit (High-Capacity Tree)": ("#ff7f0e", "s", "-."),
        "Cross-Fitted DML-IRM (Regular Tree)": ("#1f77b4", "^", "-"),
        "Cross-Fitted DML-IRM (High-Capacity Tree)": ("#17becf", "P", "-"),
        "Oracle AIPW (True Nuances)": ("#2ca02c", "d", ":"),
    }

    # Panel A: Effect Size vs 95% Coverage & Power to Exclude Zero
    ax = axes[0, 0]
    sub_eff = core_df[core_df["experiment"] == "effect_size"]
    for m_name, (col, mk, ls) in method_styles.items():
        s = sub_eff[sub_eff["method"] == m_name].sort_values("true_pop_ate")
        ax.plot(s["true_pop_ate"] * 100.0, s["coverage_95"] * 100.0, label=f"{m_name} (Cov)", color=col, marker=mk, linestyle=ls, linewidth=1.6)
    s_dml = sub_eff[sub_eff["method"] == "Cross-Fitted DML-IRM (Regular Tree)"].sort_values("true_pop_ate")
    ax.plot(s_dml["true_pop_ate"] * 100.0, s_dml["power_gt_zero"] * 100.0, label="DML-IRM Positive Detection (CI Low > 0)", color="#9467bd", marker="*", linestyle="-", linewidth=2.0)
    ax.axhline(95.0, color="black", linestyle=":", linewidth=1.0, label="Nominal 95% Target")
    ax.set_xlabel("True Population ATE (Percentage Points, %)")
    ax.set_ylabel("Percentage (%)")
    ax.set_title("(A) 2x2 H1 Design & Small Effect: Coverage vs Positive Detection")
    ax.set_ylim(-5, 105)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=6.5, loc="lower right")

    # Panel B: Sample Size vs RMSE & CI Width
    ax = axes[0, 1]
    sub_n = core_df[core_df["experiment"] == "sample_size"]
    for m_name, (col, mk, ls) in method_styles.items():
        s = sub_n[sub_n["method"] == m_name].sort_values("n")
        ax.plot(s["n"], s["rmse"] * 100.0, label=f"{m_name} (RMSE)", color=col, marker=mk, linestyle=ls, linewidth=1.6)
    s_dml_n = sub_n[sub_n["method"] == "Cross-Fitted DML-IRM (Regular Tree)"].sort_values("n")
    ax.plot(s_dml_n["n"], s_dml_n["mean_ci_width"] * 100.0, label="DML-IRM Mean 95% CI Width", color="#8c564b", marker="X", linestyle=":", linewidth=1.8)
    ax.set_xscale("log")
    ax.set_xlabel("Sample Size n (log scale)")
    ax.set_ylabel("Error / CI Width (Percentage Points, %)")
    ax.set_title("(B) Sample Size Scaling: RMSE & Uncertainty Reduction")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=6.5, loc="upper right")

    # Panel C: Overlap Degradation vs Coverage
    ax = axes[1, 0]
    sub_ov = core_df[core_df["experiment"] == "overlap"]
    for m_name, (col, mk, ls) in method_styles.items():
        s = sub_ov[sub_ov["method"] == m_name].sort_values("overlap_scale")
        ax.plot(s["overlap_scale"], s["coverage_95"] * 100.0, label=f"{m_name}", color=col, marker=mk, linestyle=ls, linewidth=1.6)
    ax.axhline(95.0, color="black", linestyle=":", linewidth=1.0)
    ax.set_xlabel("Propensity Steepness Scale (Larger = Worse Overlap)")
    ax.set_ylabel("95% CI Coverage (%)")
    ax.set_title("(C) Lack of Overlap: Coverage Degradation Under Extreme Propensities")
    ax.set_ylim(-5, 105)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=6.5, loc="lower left")

    # Panel D: 4-Way Model Misspecification (Double Robustness Verification)
    ax = axes[1, 1]
    settings_order = [
        "1. Both Correct",
        "2. Only e(X) Correct (m Misspecified)",
        "3. Only m(X) Correct (e Misspecified)",
        "4. Both Misspecified",
    ]
    x_idx = np.arange(len(settings_order))
    width = 0.25
    for idx_m, (m_label, col) in enumerate([
        ("Cross-Fitted AIPW / DML-IRM", "#1f77b4"),
        ("Plug-in Outcome Regression (Naive SE, Ignores 1st-Stage Var)", "#ff7f0e"),
        ("IPW (Propensity Only)", "#2ca02c"),
    ]):
        sub_m = misspec_df[misspec_df["method"] == m_label].set_index("setting").loc[settings_order]
        ax.bar(x_idx + (idx_m - 1) * width, sub_m["abs_bias"] * 100.0, width=width, label=m_label, color=col, alpha=0.88)
    ax.set_xticks(x_idx)
    ax.set_xticklabels(["Both\nCorrect", "Only e(X)\nCorrect", "Only m(X)\nCorrect", "Both\nWrong"], fontsize=8.5)
    ax.set_ylabel("Absolute Bias |E[hat_tau] - tau| (Percentage Points, %)")
    ax.set_title("(D) Double Robustness Verification Across 4 Misspecification Regimes")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=7.5, loc="upper left")

    fig.tight_layout()
    fig.savefig(fig_dir / "simulation_4experiments_summary.png", dpi=180)
    plt.close(fig)

    # --- Figure 2: Post-Treatment Feature Bias (Predictive AUC vs Causal Bias & Coverage) ---
    fig2, axes2 = plt.subplots(1, 2, figsize=(13.2, 4.8))
    scen_labels = {
        "mediator_only": "Mediator Only\n(T -> Visit -> Y)",
        "mediator_and_collider": "Mediator + Collider\n(T -> Visit <- U -> Y)",
        "reverse_amplification": "Reverse Collider\n(Inflation Case)",
    }
    scen_keys = ["mediator_only", "mediator_and_collider", "reverse_amplification"]
    x_pos = np.arange(len(scen_keys))
    w = 0.35

    pre_sub = post_df[post_df["adjustment_set"].str.startswith("Pre-Treatment")].set_index("scenario").loc[scen_keys]
    post_sub = post_df[post_df["adjustment_set"].str.startswith("Pre + Post")].set_index("scenario").loc[scen_keys]

    # Left: True ATE vs Pre-Only Estimate vs Post-Included Estimate
    axes2[0].bar(x_pos - w, pre_sub["true_pop_ate"] * 100.0, width=w * 0.85, label="True Total ATE", color="#2ca02c", alpha=0.85)
    axes2[0].bar(x_pos, pre_sub["mean_estimate"] * 100.0, width=w * 0.85, label="DML-IRM (Pre-Treatment X Only)", color="#1f77b4", alpha=0.9)
    axes2[0].bar(x_pos + w, post_sub["mean_estimate"] * 100.0, width=w * 0.85, label="DML-IRM (Pre X + Post-Treatment Visit)", color="#d62728", alpha=0.9)
    axes2[0].axhline(0.0, color="black", linewidth=0.8)
    axes2[0].set_xticks(x_pos)
    axes2[0].set_xticklabels([scen_labels[k] for k in scen_keys], fontsize=8.5)
    axes2[0].set_ylabel("ATE Estimate (Percentage Points, %)")
    axes2[0].set_title("Causal Effect Distortion from Conditioning on Post-Treatment Variable")
    axes2[0].grid(axis="y", alpha=0.3)
    axes2[0].legend(fontsize=8)

    # Right: Paradox between Outcome Predictive AUC Improvement and 95% Causal Coverage Collapse
    ax_r = axes2[1]
    ax_r.bar(x_pos - w / 2, pre_sub["mean_outcome_auc"], width=w, label="Outcome AUC (Pre-Treatment X Only)", color="#aec7e8")
    ax_r.bar(x_pos + w / 2, post_sub["mean_outcome_auc"], width=w, label="Outcome AUC (With Post-Treatment Visit)", color="#ff9896")
    ax_r.set_ylim(0.5, 1.0)
    ax_r.set_xticks(x_pos)
    ax_r.set_xticklabels([scen_labels[k] for k in scen_keys], fontsize=8.5)
    ax_r.set_ylabel("Out-of-Fold Outcome Prediction ROC-AUC")

    ax_r2 = ax_r.twinx()
    ax_r2.plot(x_pos - w / 2, pre_sub["coverage_95"] * 100.0, "o-", color="#1f77b4", linewidth=2, markersize=8, label="95% CI Coverage (Pre-Only)")
    ax_r2.plot(x_pos + w / 2, post_sub["coverage_95"] * 100.0, "s--", color="#d62728", linewidth=2, markersize=8, label="95% CI Coverage (With Post-Visit)")
    ax_r2.set_ylim(-5, 108)
    ax_r2.set_ylabel("95% Causal Confidence Interval Coverage (%)")
    ax_r.set_title("Prediction vs Causality Paradox: Higher AUC != Valid Causal Inference")
    lines1, labels1 = ax_r.get_legend_handles_labels()
    lines2, labels2 = ax_r2.get_legend_handles_labels()
    ax_r2.legend(lines1 + lines2, labels1 + labels2, fontsize=7.5, loc="center right")

    fig2.tight_layout()
    fig2.savefig(fig_dir / "post_treatment_bias_paradox.png", dpi=180)
    plt.close(fig2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replications", type=int, default=200, help="Number of Monte Carlo replications per scenario")
    parser.add_argument("--jobs", type=int, default=4, help="Parallel worker processes")
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--incremental", action="store_true", help="Reuse existing 4 H1 cells and only run the 2 new H1 cells + updated policy sim")
    args = parser.parse_args()

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    R = args.replications
    t0 = time.time()

    print(f"Starting Monte Carlo Simulation Suite with R={R} replications per setting (jobs={args.jobs})...", flush=True)

    # 1. Core Suite: Effect Size, Sample Size, Overlap
    core_tasks = []
    # Exp 1: Effect size (n=2000, overlap=1.0)
    for ate_val in [0.050, 0.020, 0.005, 0.001]:
        for r in range(R):
            core_tasks.append(("effect_size", f"ATE={ate_val*100:.2f}%", 2000, ate_val, 1.0, r, args.seed + 1000))
    # Exp 2: Sample size (target_ate=0.015, overlap=1.0)
    for n_val in [500, 1500, 3000, 6000]:
        for r in range(R):
            core_tasks.append(("sample_size", f"n={n_val}", n_val, 0.015, 1.0, r, args.seed + 20000))
    # Exp 3: Overlap (n=2000, target_ate=0.020)
    for ov_val in [0.5, 1.0, 2.2, 3.5]:
        for r in range(R):
            core_tasks.append(("overlap", f"overlap_scale={ov_val}", 2000, 0.020, ov_val, r, args.seed + 40000))

    if args.incremental and (out_dir / "simulation_core_metrics.csv").exists():
        old_core = pd.read_csv(out_dir / "simulation_core_metrics.csv")
        method_order = [
            "Naive Diff-in-Means",
            "AIPW No Cross-Fit (Regular Tree)",
            "AIPW No Cross-Fit (High-Capacity Tree)",
            "Cross-Fitted DML-IRM (Regular Tree)",
            "Cross-Fitted DML-IRM (High-Capacity Tree)",
            "Oracle AIPW (True Nuances)",
        ]
        if (
            set(method_order).issubset(set(old_core["method"].unique()))
            and len(old_core) == 12 * len(method_order)
            and "mean_actual_clipped_fraction" in old_core.columns
            and "mean_true_extreme_e_fraction" in old_core.columns
        ):
            print("Incremental mode: all 72 2x2 H1 core cells and clipping diagnostics already complete; reusing.", flush=True)
            core_summary = old_core.copy()
        else:
            print(f"Incremental mode: computing/verifying 2x2 H1 cells across {len(core_tasks)} tasks...", flush=True)
            missing_nested = Parallel(n_jobs=args.jobs)(
                delayed(_single_baseline_missing_cells_rep)(r, exp, label, n_v, ate_v, ov_v, b_seed)
                for (exp, label, n_v, ate_v, ov_v, r, b_seed) in core_tasks
            )
            missing_records = [item for sublist in missing_nested for item in sublist]
            new_cells_summary = summarize_mc_runs(
                missing_records, ["experiment", "setting", "n", "overlap_scale", "method"]
            )
            # Map true_extreme_e_fraction and oracle_clipped_fraction per setting over all R replications
            miss_df = pd.DataFrame(missing_records)
            setting_diag = miss_df.groupby(["experiment", "setting"], sort=False)[
                ["true_extreme_e_fraction", "oracle_clipped_fraction"]
            ].mean().reset_index()

            # Idempotently normalize method names and remove any pre-existing rows of new_cells_summary
            old_core["method"] = old_core["method"].replace({
                "AIPW (No Cross-Fitting)": "AIPW No Cross-Fit (High-Capacity Tree)",
                "Cross-Fitted AIPW / DML-IRM": "Cross-Fitted DML-IRM (Regular Tree)",
            })
            old_core = old_core[~old_core["method"].isin(new_cells_summary["method"].unique())].copy()

            # Preserve existing clipping column before merging updated setting_diag
            if "mean_actual_clipped_fraction" in old_core.columns:
                existing_clip = old_core["mean_actual_clipped_fraction"].to_numpy(dtype=float)
            elif "mean_clipped_fraction" in old_core.columns:
                existing_clip = old_core["mean_clipped_fraction"].to_numpy(dtype=float)
            else:
                existing_clip = np.zeros(len(old_core), dtype=float)
            old_core["_existing_clip"] = existing_clip

            drop_pre = [
                c for c in ["mean_true_extreme_e_fraction", "mean_actual_clipped_fraction", "mean_clipped_fraction", "true_extreme_e_fraction", "oracle_clipped_fraction"]
                if c in old_core.columns
            ]
            old_core = old_core.drop(columns=drop_pre)
            old_core = old_core.merge(setting_diag, on=["experiment", "setting"], how="left")
            old_core["mean_true_extreme_e_fraction"] = old_core["true_extreme_e_fraction"]
            old_core["mean_actual_clipped_fraction"] = np.where(
                old_core["method"] == "Oracle AIPW (True Nuances)",
                old_core["oracle_clipped_fraction"],
                old_core["_existing_clip"],
            )
            old_core = old_core.drop(columns=["_existing_clip", "true_extreme_e_fraction", "oracle_clipped_fraction"])

            combined_core = pd.concat([old_core, new_cells_summary[old_core.columns]], ignore_index=True)
            combined_core["method"] = pd.Categorical(combined_core["method"], categories=method_order, ordered=True)
            setting_order = old_core[["experiment", "setting"]].drop_duplicates().reset_index(drop=True)
            setting_order["setting_rank"] = np.arange(len(setting_order))
            combined_core = combined_core.merge(setting_order, on=["experiment", "setting"], how="left")
            core_summary = combined_core.sort_values(["setting_rank", "method"]).drop(columns=["setting_rank"]).reset_index(drop=True)
            core_summary["method"] = core_summary["method"].astype(str)
    else:
        print(f"Running {len(core_tasks)} core tasks (Effect Size, Sample Size, Overlap)...", flush=True)
        core_nested = Parallel(n_jobs=args.jobs)(
            delayed(_single_baseline_rep)(r, exp, label, n_v, ate_v, ov_v, b_seed)
            for (exp, label, n_v, ate_v, ov_v, r, b_seed) in core_tasks
        )
        core_records = [item for sublist in core_nested for item in sublist]
        core_summary = summarize_mc_runs(core_records, ["experiment", "setting", "n", "overlap_scale", "method"])

    core_summary.to_csv(out_dir / "simulation_core_metrics.csv", index=False)
    print(f"Core experiments done in {time.time() - t0:.1f}s", flush=True)

    # 2. Experiment 4: 4-Way Model Misspecification (Double Robustness)
    if args.incremental and (out_dir / "simulation_misspecification_metrics.csv").exists():
        misspec_summary = pd.read_csv(out_dir / "simulation_misspecification_metrics.csv")
        misspec_summary["method"] = misspec_summary["method"].replace({
            "Plug-in Outcome Regression (m1 - m0)": "Plug-in Outcome Regression (Naive SE, Ignores 1st-Stage Var)"
        })
        misspec_summary.to_csv(out_dir / "simulation_misspecification_metrics.csv", index=False)
    else:
        print(f"Running {R} replications for 4-Way Model Misspecification...", flush=True)
        misspec_nested = Parallel(n_jobs=args.jobs)(
            delayed(_single_misspec_rep)(r, 2500, args.seed + 60000) for r in range(R)
        )
        misspec_records = [item for sublist in misspec_nested for item in sublist]
        misspec_summary = summarize_mc_runs(misspec_records, ["experiment", "setting", "method"])
        misspec_summary.to_csv(out_dir / "simulation_misspecification_metrics.csv", index=False)

    # 3. Work 3: Post-Treatment Variable Bias
    if args.incremental and (out_dir / "post_treatment_metrics.csv").exists():
        post_summary = pd.read_csv(out_dir / "post_treatment_metrics.csv")
    else:
        print(f"Running {R} replications x 3 scenarios for Post-Treatment Feature Bias...", flush=True)
        post_tasks = [
            (r, scen, 2000, args.seed + 80000 + i * 5000)
            for i, scen in enumerate(["mediator_only", "mediator_and_collider", "reverse_amplification"])
            for r in range(R)
        ]
        post_nested = Parallel(n_jobs=args.jobs)(
            delayed(_single_post_treatment_rep)(r, scen, n_v, b_seed) for (r, scen, n_v, b_seed) in post_tasks
        )
        post_records = [item for sublist in post_nested for item in sublist]
        post_summary = summarize_mc_runs(post_records, ["scenario", "adjustment_set"])
        post_summary.to_csv(out_dir / "post_treatment_metrics.csv", index=False)

    # 4. Work 4: Identifiability & Unlearnability Boundaries
    if args.incremental and (out_dir / "identifiability_metrics.csv").exists():
        ident_summary = pd.read_csv(out_dir / "identifiability_metrics.csv")
        if "mean_clipped_fraction" in ident_summary.columns:
            ident_summary = ident_summary.rename(columns={"mean_clipped_fraction": "mean_actual_clipped_fraction"})
        ident_summary["method"] = ident_summary["method"].replace({
            "Cross-Fitted AIPW / DML-IRM": "Cross-Fitted DML-IRM (Regular Tree)"
        })
        # Always compute true_extreme_e_fraction over the full R replications
        true_ext_map = {}
        for idx, (mode, n_v) in enumerate([
            ("unmeasured_confounding", 1000),
            ("unmeasured_confounding", 3500),
            ("lack_of_overlap", 2000),
            ("rare_events_small_effect", 3000),
            ("zero_ate_uniform", 2000),
            ("zero_ate_cancellation", 2000),
        ]):
            b_seed = args.seed + 100000 + idx * 5000
            vals = []
            for r in range(R):
                e_vec = generate_dgp_identifiability(n=n_v, mode=mode, seed=b_seed + r * 107)["e_true"]
                vals.append(float(np.mean((e_vec < 0.02) | (e_vec > 0.98))))
            true_ext_map[(mode, n_v)] = float(np.mean(vals))
        ident_summary["mean_true_extreme_e_fraction"] = [
            true_ext_map.get((r["mode"], int(r["n"])), 0.0) for _, r in ident_summary.iterrows()
        ]
        # Place mean_true_extreme_e_fraction right before mean_actual_clipped_fraction for schema consistency
        cols = list(ident_summary.columns)
        if "mean_true_extreme_e_fraction" in cols and "mean_actual_clipped_fraction" in cols:
            cols.remove("mean_true_extreme_e_fraction")
            clip_idx = cols.index("mean_actual_clipped_fraction")
            cols.insert(clip_idx, "mean_true_extreme_e_fraction")
            ident_summary = ident_summary[cols]
        ident_summary.to_csv(out_dir / "identifiability_metrics.csv", index=False)
    else:
        print(f"Running {R} replications for Identifiability & Effect Cancellation...", flush=True)
        ident_configs = [
            ("unmeasured_confounding", 1000),
            ("unmeasured_confounding", 3500),
            ("lack_of_overlap", 2000),
            ("rare_events_small_effect", 3000),
            ("zero_ate_uniform", 2000),
            ("zero_ate_cancellation", 2000),
        ]
        ident_tasks = [
            (r, mode, n_v, args.seed + 100000 + idx * 5000)
            for idx, (mode, n_v) in enumerate(ident_configs)
            for r in range(R)
        ]
        ident_records = Parallel(n_jobs=args.jobs)(
            delayed(_single_identifiability_rep)(r, mode, n_v, b_seed) for (r, mode, n_v, b_seed) in ident_tasks
        )
        ident_summary = summarize_mc_runs(ident_records, ["mode", "n", "method"])
        # Append average uplift learnability metrics per mode
        ident_df_raw = pd.DataFrame(ident_records)
        extra_ident = ident_df_raw.groupby(["mode", "n"], sort=False)[
            ["true_cate_std", "cate_pehe", "true_policy_gain", "est_policy_gain", "target_fraction"]
        ].mean().reset_index()
        ident_summary = ident_summary.merge(extra_ident, on=["mode", "n"], how="left")
        ident_summary.to_csv(out_dir / "identifiability_metrics.csv", index=False)

    # 5. Question 10 Bottleneck Verification: Sure Things vs Persuadables Policy Evaluation
    if (
        args.incremental
        and (out_dir / "sure_things_vs_persuadables_metrics.csv").exists()
        and "mean_est_baseline_subsidy_aipw" in pd.read_csv(out_dir / "sure_things_vs_persuadables_metrics.csv").columns
    ):
        st_summary = pd.read_csv(out_dir / "sure_things_vs_persuadables_metrics.csv")
    else:
        print(f"Running {R} replications for Sure Things vs Persuadables Policy Evaluation...", flush=True)
        st_nested = Parallel(n_jobs=args.jobs)(
            delayed(_single_sure_things_policy_rep)(r, 2500, args.seed + 150000) for r in range(R)
        )
        st_records = [item for sublist in st_nested for item in sublist]
        st_df = pd.DataFrame(st_records)
        st_summary = st_df.groupby("policy", sort=False).agg(
            replications=("true_incremental_effect", "count"),
            mean_true_incremental_effect=("true_incremental_effect", "mean"),
            mean_est_incremental_effect=("est_incremental_effect", "mean"),
            policy_bias=("est_incremental_effect", lambda s: float(np.mean(s - st_df.loc[s.index, "true_incremental_effect"]))),
            policy_coverage_95=("covered_true_policy_effect", "mean"),
            mean_true_net_gain_contact=("true_net_gain_contact_cost", "mean"),
            mean_est_net_gain_contact=("est_net_gain_contact_cost", "mean"),
            mean_true_baseline_subsidy=("true_baseline_subsidy", "mean"),
            mean_est_baseline_subsidy_aipw=("est_baseline_subsidy_aipw", "mean"),
            mean_true_net_gain_coupon=("true_net_gain_with_coupon_subsidy", "mean"),
            mean_est_net_gain_coupon=("est_net_gain_with_coupon_subsidy", "mean"),
        ).reset_index()
        st_summary.to_csv(out_dir / "sure_things_vs_persuadables_metrics.csv", index=False)

    print("Generating simulation figures...", flush=True)
    plot_simulation_figures(core_summary, misspec_summary, post_summary, ident_summary, fig_dir)

    all_payload = {
        "replications": R,
        "elapsed_seconds": round(time.time() - t0, 2),
        "core_experiments": core_summary.to_dict(orient="records"),
        "misspecification": misspec_summary.to_dict(orient="records"),
        "post_treatment_bias": post_summary.to_dict(orient="records"),
        "identifiability": ident_summary.to_dict(orient="records"),
        "sure_things_vs_persuadables": st_summary.to_dict(orient="records"),
    }
    (out_dir / "simulation_all_summary.json").write_text(
        json.dumps(all_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"All simulations finished in {time.time() - t0:.1f}s!", flush=True)


if __name__ == "__main__":
    main()
