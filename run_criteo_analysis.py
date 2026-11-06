"""Work 1 & Work 6: Real Criteo Data ATE Estimation Comparison & Independent Evaluation of Teammate's Uplift Policies."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from estimators import (
    estimate_naive,
    evaluate_policy,
    ipw_score,
    select_top_k_fraction,
    summary_from_scores,
)

DEFAULT_TEAMMATE_DIR = Path(
    r"D:\WX_files\xwechat_files\wxid_4n8a0yrbh44r22_f387\msg\file\2026-11\uplift_training\uplift_training\results"
)


def evaluate_criteo_ate(test_df: pd.DataFrame, train_oof_df: pd.DataFrame) -> pd.DataFrame:
    """Compare Naive Diff-in-Means, IPW, Held-out AIPW/DR, Training 3-Fold Cross-Fitted DML, and Pooled Cross-Fitted DML."""
    t_test = test_df["treatment"].to_numpy(dtype=int)
    y_test = test_df["conversion"].to_numpy(dtype=float)
    e_test = test_df["propensity"].to_numpy(dtype=float)
    psi_test = test_df["dr_pseudo"].to_numpy(dtype=float)
    psi_train_oof = train_oof_df["dr_pseudo_oof"].to_numpy(dtype=float)

    rows = []

    # 1. Naive difference in means on Test Set (N=200,000)
    naive_res = estimate_naive(t_test, y_test)
    rows.append({
        "dataset_scope": "Test Set (N=200,000)",
        "method": "Naive Diff-in-Means (Unadjusted)",
        "estimate": naive_res["estimate"],
        "se": naive_res["se"],
        "ci_lower": naive_res["ci_lower"],
        "ci_upper": naive_res["ci_upper"],
        "ci_width": naive_res["ci_width"],
        "p_value": naive_res["p_value"],
        "estimate_pct_pt": naive_res["estimate"] * 100.0,
        "ci_lower_pct_pt": naive_res["ci_lower"] * 100.0,
        "ci_upper_pct_pt": naive_res["ci_upper"] * 100.0,
        "notes": "Direct difference E[Y|T=1] - E[Y|T=0]; ignores baseline covariate imbalance",
    })

    # 2. Horvitz-Thompson IPW on Test Set (N=200,000)
    ipw_scores = ipw_score(y_test, t_test, e_test)
    ipw_res = summary_from_scores(ipw_scores)
    rows.append({
        "dataset_scope": "Test Set (N=200,000)",
        "method": "IPW (Inverse Propensity Weighting)",
        "estimate": ipw_res["estimate"],
        "se": ipw_res["se"],
        "ci_lower": ipw_res["ci_lower"],
        "ci_upper": ipw_res["ci_upper"],
        "ci_width": ipw_res["ci_width"],
        "p_value": ipw_res["p_value"],
        "estimate_pct_pt": ipw_res["estimate"] * 100.0,
        "ci_lower_pct_pt": ipw_res["ci_lower"] * 100.0,
        "ci_upper_pct_pt": ipw_res["ci_upper"] * 100.0,
        "notes": "Weights by 1/e and 1/(1-e) without outcome regression variance reduction",
    })

    # 3. Held-out Out-of-Sample AIPW / DR on Test Set (N=200,000)
    dr_test_res = summary_from_scores(psi_test)
    rows.append({
        "dataset_scope": "Test Set (N=200,000)",
        "method": "Out-of-Sample AIPW / DR (Train-Fitted Nuisance)",
        "estimate": dr_test_res["estimate"],
        "se": dr_test_res["se"],
        "ci_lower": dr_test_res["ci_lower"],
        "ci_upper": dr_test_res["ci_upper"],
        "ci_width": dr_test_res["ci_width"],
        "p_value": dr_test_res["p_value"],
        "estimate_pct_pt": dr_test_res["estimate"] * 100.0,
        "ci_lower_pct_pt": dr_test_res["ci_lower"] * 100.0,
        "ci_upper_pct_pt": dr_test_res["ci_upper"] * 100.0,
        "notes": "Nuisance (mu0, mu1, e) fitted on 600k train set, evaluated out-of-sample on 200k test set",
    })

    # 4. 3-Fold Cross-Fitted AIPW / DML-IRM on Training Set (N=600,000)
    dr_train_res = summary_from_scores(psi_train_oof)
    rows.append({
        "dataset_scope": "Train Set (N=600,000)",
        "method": "3-Fold Cross-Fitted AIPW / DML-IRM",
        "estimate": dr_train_res["estimate"],
        "se": dr_train_res["se"],
        "ci_lower": dr_train_res["ci_lower"],
        "ci_upper": dr_train_res["ci_upper"],
        "ci_width": dr_train_res["ci_width"],
        "p_value": dr_train_res["p_value"],
        "estimate_pct_pt": dr_train_res["estimate"] * 100.0,
        "ci_lower_pct_pt": dr_train_res["ci_lower"] * 100.0,
        "ci_upper_pct_pt": dr_train_res["ci_upper"] * 100.0,
        "notes": "Strict 3-fold out-of-fold nuisance predictions across all 600,000 training rows",
    })

    # 5. Pooled Cross-Sample AIPW / DML-IRM (N=800,000)
    psi_pooled = np.concatenate([psi_train_oof, psi_test])
    dr_pooled_res = summary_from_scores(psi_pooled)
    rows.append({
        "dataset_scope": "Train OOF + Test (N=800,000)",
        "method": "Pooled Out-of-Fold AIPW / DML-IRM",
        "estimate": dr_pooled_res["estimate"],
        "se": dr_pooled_res["se"],
        "ci_lower": dr_pooled_res["ci_lower"],
        "ci_upper": dr_pooled_res["ci_upper"],
        "ci_width": dr_pooled_res["ci_width"],
        "p_value": dr_pooled_res["p_value"],
        "estimate_pct_pt": dr_pooled_res["estimate"] * 100.0,
        "ci_lower_pct_pt": dr_pooled_res["ci_lower"] * 100.0,
        "ci_upper_pct_pt": dr_pooled_res["ci_upper"] * 100.0,
        "notes": "All 800,000 rows scored strictly out-of-fold; higher sample size narrows CI width by ~50%",
    })

    return pd.DataFrame(rows)


def evaluate_teammate_policies(
    test_df: pd.DataFrame,
    cost: float = 0.05,
    margin: float = 100.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Evaluate teammate's targeting policies and baseline comparisons on the 200,000 independent test set.
    Returns:
      - policy_df: Full policy comparison table (at 20% fixed budget, threshold budget, and 100% treat-all)
      - paired_df: Paired difference tests between DR-Learner policy and competing strategies
      - cost_sens_df: Cost sensitivity analysis across varying per-user targeting costs and conversion subsidies
    """
    t_test = test_df["treatment"].to_numpy(dtype=float)
    y_test = test_df["conversion"].to_numpy(dtype=float)
    e_test = test_df["propensity"].to_numpy(dtype=float)
    psi = test_df["dr_pseudo"].to_numpy(dtype=float)
    mu0 = test_df["mu0"].to_numpy(dtype=float)
    mu1 = test_df["mu1"].to_numpy(dtype=float)
    # AIPW-corrected baseline outcome score psi_0 = mu0 + (1 - T)*(Y - mu0)/(1 - e)
    psi0 = mu0 + (1.0 - t_test) * (y_test - mu0) / (1.0 - e_test)
    up_const = test_df["uplift_constant"].to_numpy(dtype=float)
    up_t = test_df["uplift_t"].to_numpy(dtype=float)
    up_dr = test_df["uplift_dr"].to_numpy(dtype=float)
    n = len(psi)

    threshold = cost / margin
    thresh_str = f"{threshold:g}"

    policies = []

    # --- Group 1: Fixed 20% Treatment Fraction (Prompt Section 6 Required Table) ---
    q20 = 0.20
    # 1a. Random 20% (Expected value pi(X) = 0.20)
    pi_rand20_exp = np.full(n, q20, dtype=float)
    policies.append(("Random Targeting (Expected)", "top_20pct_expected", pi_rand20_exp))
    # 1b. Random 20% (Single seeded realization)
    pi_rand20_draw = select_top_k_fraction(np.zeros(n), q20, seed=42)
    policies.append(("Random Targeting (Seeded Draw)", "top_20pct_random_draw", pi_rand20_draw))
    # 1c. Baseline Purchase Probability Sorting mu0(X) Top 20%
    pi_mu0_20 = select_top_k_fraction(mu0, q20, seed=42)
    policies.append(("Baseline Conversion Prob (mu0)", "top_20pct_by_mu0", pi_mu0_20))
    # 1d. Treated Purchase Probability Sorting mu1(X) Top 20%
    pi_mu1_20 = select_top_k_fraction(mu1, q20, seed=42)
    policies.append(("Treated Conversion Prob (mu1)", "top_20pct_by_mu1", pi_mu1_20))
    # 1e. Teammate T-Learner Uplift Sorting Top 20%
    pi_t_20 = select_top_k_fraction(up_t, q20, seed=42)
    policies.append(("T-Learner Uplift (Teammate)", "top_20pct_by_uplift", pi_t_20))
    # 1f. Teammate DR-Learner Uplift Sorting Top 20%
    pi_dr_20 = select_top_k_fraction(up_dr, q20, seed=42)
    policies.append(("DR-Learner Uplift (Teammate Selected)", "top_20pct_by_uplift", pi_dr_20))

    # --- Group 2: Pre-Fixed Economic Threshold Rule (uplift > c/m) & Independent Benchmarks ---
    pi_treat_all = np.ones(n, dtype=float)
    pi_dr_thresh = (up_dr > threshold).astype(float)
    q_dr = float(np.mean(pi_dr_thresh))
    q_dr_pct_str = f"{q_dr * 100:.2f}%"
    pi_t_thresh = (up_t > threshold).astype(float)
    q_t_pct_str = f"{float(np.mean(pi_t_thresh)) * 100:.2f}%"
    pi_const_thresh = (up_const > threshold).astype(float)
    q_const_pct_str = f"{float(np.mean(pi_const_thresh)) * 100:.2f}%"
    pi_rand_qdr = np.full(n, q_dr, dtype=float)
    pi_mu0_qdr = select_top_k_fraction(mu0, q_dr, seed=42)
    pi_t_qdr = select_top_k_fraction(up_t, q_dr, seed=42)

    policies.extend([
        ("Treat None (0%)", "benchmark_0pct", np.zeros(n, dtype=float)),
        ("Treat All (100%)", "benchmark_100pct", pi_treat_all),
        (f"Constant Model (Threshold > {thresh_str}, {q_const_pct_str})", f"threshold_{thresh_str}", pi_const_thresh),
        (f"Random Targeting ({q_dr_pct_str} Expected)", f"matched_{q_dr_pct_str}_expected", pi_rand_qdr),
        (f"Baseline Conversion Prob mu0 (Top {q_dr_pct_str})", f"matched_{q_dr_pct_str}_by_mu0", pi_mu0_qdr),
        (f"T-Learner Uplift (Top {q_dr_pct_str} Matched)", f"matched_{q_dr_pct_str}_by_uplift", pi_t_qdr),
        (f"T-Learner Uplift (Threshold > {thresh_str}, {q_t_pct_str})", f"threshold_{thresh_str}", pi_t_thresh),
        (f"DR-Learner Uplift (Threshold > {thresh_str}, {q_dr_pct_str}, Selected)", f"threshold_{thresh_str}", pi_dr_thresh),
    ])

    records = []
    for model_name, policy_id, pi in policies:
        res = evaluate_policy(
            psi, pi, cost=cost, margin=margin, model_name=model_name, policy_id=policy_id
        )
        records.append(res)

    policy_df = pd.DataFrame(records)

    # Paired comparisons on the exact same 200,000 test users
    paired_comparisons = [
        ("DR Top 20% vs Random 20% (Expected)", pi_dr_20, pi_rand20_exp),
        ("DR Top 20% vs Baseline Prob mu0 Top 20%", pi_dr_20, pi_mu0_20),
        ("DR Top 20% vs T-Learner Top 20%", pi_dr_20, pi_t_20),
        (f"DR Threshold ({q_dr_pct_str}) vs Treat All (100%)", pi_dr_thresh, pi_treat_all),
        (f"DR Threshold ({q_dr_pct_str}) vs Random ({q_dr_pct_str})", pi_dr_thresh, pi_rand_qdr),
        (f"DR Threshold ({q_dr_pct_str}) vs Baseline mu0 ({q_dr_pct_str})", pi_dr_thresh, pi_mu0_qdr),
        (f"DR Threshold ({q_dr_pct_str}) vs T-Learner Threshold ({q_t_pct_str})", pi_dr_thresh, pi_t_thresh),
        (f"DR Threshold ({q_dr_pct_str}) vs T-Learner Matched ({q_dr_pct_str})", pi_dr_thresh, pi_t_qdr),
    ]
    paired_rows = []
    for comp_name, pi_a, pi_b in paired_comparisons:
        diff_inc = (pi_a - pi_b) * psi
        diff_net = (pi_a - pi_b) * (margin * psi - cost)
        inc_s = summary_from_scores(diff_inc)
        net_s = summary_from_scores(diff_net)
        paired_rows.append({
            "comparison": comp_name,
            "frac_a": float(np.mean(pi_a)),
            "frac_b": float(np.mean(pi_b)),
            "diff_incremental_effect": inc_s["estimate"],
            "diff_inc_se": inc_s["se"],
            "diff_inc_ci_lower": inc_s["ci_lower"],
            "diff_inc_ci_upper": inc_s["ci_upper"],
            "diff_inc_p_value": inc_s["p_value"],
            "diff_net_gain": net_s["estimate"],
            "diff_net_se": net_s["se"],
            "diff_net_ci_lower": net_s["ci_lower"],
            "diff_net_ci_upper": net_s["ci_upper"],
            "diff_net_p_value": net_s["p_value"],
        })
    paired_df = pd.DataFrame(paired_rows)

    # Cost sensitivity across varying per-contact cost c AND coupon redemption subsidy s
    cost_rows = []
    for c_val in [0.01, 0.03, 0.05, 0.08, 0.10, 0.12, 0.15]:
        for label, pi_vec in [
            ("Random 20%", pi_rand20_exp),
            ("Baseline mu0 Top 20%", pi_mu0_20),
            ("T-Learner Top 20%", pi_t_20),
            ("DR-Learner Top 20%", pi_dr_20),
            (f"DR-Learner Fixed Threshold ({q_dr_pct_str})", pi_dr_thresh),
            ("Treat All (100%)", pi_treat_all),
        ]:
            net_s = summary_from_scores(pi_vec * (margin * psi - c_val))
            cost_rows.append({
                "cost_model": "per_targeted_contact_cost",
                "cost_per_targeted": c_val,
                "coupon_subsidy_per_conversion": 0.0,
                "margin_per_conversion": margin,
                "breakeven_uplift": c_val / margin,
                "strategy": label,
                "treatment_fraction": float(np.mean(pi_vec)),
                "estimated_baseline_subsidy_aipw": 0.0,
                "estimated_baseline_subsidy_plugin": 0.0,
                "net_gain_per_user": net_s["estimate"],
                "net_gain_se": net_s["se"],
                "net_gain_ci_lower": net_s["ci_lower"],
                "net_gain_ci_upper": net_s["ci_upper"],
                "plugin_net_gain_unadjusted_mu0": net_s["estimate"],
            })

    # Coupon / Redemption Subsidy Model (with full AIPW correction on baseline outcome Y(0)):
    # Each conversion among targeted users costs a subsidy `s` (e.g., coupon discount = 20 out of 100 margin),
    # plus contact cost c0 = 0.03.
    # Using AIPW-corrected baseline conversion score psi0_i = mu0(X_i) + (1-T_i)*(Y_i-mu0(X_i))/(1-e_i),
    # the doubly robust net value score is:
    #   pi(X_i) * [(margin - s) * psi_i - s * psi0_i - c0] = pi(X_i) * [margin * psi_i - s * psi1_i - c0].
    for s_val in [10.0, 20.0, 30.0]:
        c0 = 0.03
        # Decision rule uses pre-treatment predictions (up_dr, mu0) available at targeting time
        pi_dr_profit_opt = (((margin - s_val) * up_dr - s_val * mu0 - c0) > 0).astype(float)
        for label, pi_vec in [
            ("Random 20%", pi_rand20_exp),
            ("Baseline mu0 Top 20%", pi_mu0_20),
            ("T-Learner Uplift Top 20%", pi_t_20),
            ("DR-Learner Uplift Top 20%", pi_dr_20),
            ("DR-Learner Subsidy-Aware Policy", pi_dr_profit_opt),
            ("Treat All (100%)", pi_treat_all),
        ]:
            baseline_sub_aipw = float(np.mean(pi_vec * s_val * psi0))
            baseline_sub_plugin = float(np.mean(pi_vec * s_val * mu0))
            net_scores_aipw = pi_vec * ((margin - s_val) * psi - s_val * psi0 - c0)
            net_s = summary_from_scores(net_scores_aipw)
            plugin_net = float(np.mean(pi_vec * ((margin - s_val) * psi - s_val * mu0 - c0)))
            cost_rows.append({
                "cost_model": "coupon_redemption_subsidy",
                "cost_per_targeted": c0,
                "coupon_subsidy_per_conversion": s_val,
                "margin_per_conversion": margin,
                "breakeven_uplift": c0 / (margin - s_val),
                "strategy": label,
                "treatment_fraction": float(np.mean(pi_vec)),
                "estimated_baseline_subsidy_aipw": baseline_sub_aipw,
                "estimated_baseline_subsidy_plugin": baseline_sub_plugin,
                "net_gain_per_user": net_s["estimate"],
                "net_gain_se": net_s["se"],
                "net_gain_ci_lower": net_s["ci_lower"],
                "net_gain_ci_upper": net_s["ci_upper"],
                "plugin_net_gain_unadjusted_mu0": plugin_net,
            })
    cost_sens_df = pd.DataFrame(cost_rows)
    return policy_df, paired_df, cost_sens_df


def plot_criteo_and_policy_figures(
    ate_df: pd.DataFrame,
    policy_df: pd.DataFrame,
    cost_sens_df: pd.DataFrame,
    test_df: pd.DataFrame,
    fig_dir: Path,
):
    """Generate diagnostic plots for Criteo ATE comparison and Teammate Policy evaluation."""
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Figure 1: ATE Comparison Forest Plot + Policy Evaluation Comparison
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))

    # Left panel: ATE Forest Plot (in percentage points)
    ax = axes[0]
    y_pos = np.arange(len(ate_df))
    ests = ate_df["estimate_pct_pt"].to_numpy()
    lows = ate_df["ci_lower_pct_pt"].to_numpy()
    highs = ate_df["ci_upper_pct_pt"].to_numpy()
    labels = [f"{r['method']}\n({r['dataset_scope']})" for _, r in ate_df.iterrows()]

    ax.errorbar(
        ests, y_pos, xerr=[ests - lows, highs - ests],
        fmt="o", color="#1f77b4", ecolor="#1f77b4", elinewidth=2, capsize=4, markersize=7
    )
    ax.axvline(0.0, color="red", linestyle="--", linewidth=1, label="Zero Effect")
    ax.axvline(ests[2], color="gray", linestyle=":", linewidth=1, label="Test DR ATE (+0.1005 pp)")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("Estimated ATE (Percentage Points, %)")
    ax.set_title("Criteo ATE Estimates & 95% Confidence Intervals")
    ax.grid(axis="x", alpha=0.3)
    ax.legend(loc="lower right", fontsize=8.5)

    # Right panel: Top 20% Policy Incremental Effect & Net Gain Comparison
    ax2 = axes[1]
    sub20 = policy_df[policy_df["policy_id"].isin([
        "top_20pct_expected", "top_20pct_by_mu0", "top_20pct_by_mu1", "top_20pct_by_uplift"
    ])].copy()
    short_names = [
        "Random\n(20%)",
        "Baseline Prob\nmu0 (20%)",
        "Treated Prob\nmu1 (20%)",
        "T-Learner\nUplift (20%)",
        "DR-Learner\nUplift (20%)",
    ]
    x_pos = np.arange(len(sub20))
    inc_pp = sub20["estimated_incremental_effect"].to_numpy() * 100.0
    inc_low = sub20["ci_lower"].to_numpy() * 100.0
    inc_high = sub20["ci_upper"].to_numpy() * 100.0
    colors = ["#7f7f7f", "#ff7f0e", "#d62728", "#2ca02c", "#1f77b4"]

    bars = ax2.bar(x_pos, inc_pp, yerr=[inc_pp - inc_low, inc_high - inc_pp], capsize=4, color=colors, alpha=0.85)
    ax2.axhline(0.0, color="black", linewidth=0.8)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(short_names, fontsize=8.5)
    ax2.set_ylabel("Population Incremental Conversion (Percentage Points, %)")
    ax2.set_title("20% Targeting Budget: Incremental Effect E[pi(X)*tau(X)]")
    ax2.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, inc_pp):
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            max(val, 0) + 0.004,
            f"{val:.4f}%",
            ha="center", va="bottom", fontsize=8, fontweight="bold"
        )

    fig.tight_layout()
    fig.savefig(fig_dir / "criteo_ate_and_policy20.png", dpi=180)
    plt.close(fig)

    # Figure 2: Cumulative Incremental Gain Curve (including Baseline Risk mu0 vs Uplift models) & Cost Sensitivity
    psi = test_df["dr_pseudo"].to_numpy(dtype=float)
    mu0 = test_df["mu0"].to_numpy(dtype=float)
    up_t = test_df["uplift_t"].to_numpy(dtype=float)
    up_dr = test_df["uplift_dr"].to_numpy(dtype=float)
    n = len(psi)

    fig2, axes2 = plt.subplots(1, 2, figsize=(13.5, 5.0))
    fracs = np.linspace(0.0, 1.0, 21)

    for label, scores, color, ls in [
        ("DR-Learner Uplift (Teammate Selected)", up_dr, "#1f77b4", "-"),
        ("T-Learner Uplift (Teammate)", up_t, "#2ca02c", "--"),
        ("Baseline Conversion Prob mu0(X)", mu0, "#ff7f0e", "-."),
    ]:
        gains = [0.0]
        rng = np.random.default_rng(42)
        order = np.lexsort((rng.random(n), -scores))
        cum_psi = np.cumsum(psi[order]) / n
        for q in fracs[1:]:
            k = int(np.round(q * n))
            gains.append(float(cum_psi[k - 1]) * 100.0)
        axes2[0].plot(fracs * 100.0, gains, label=label, color=color, linestyle=ls, linewidth=2)

    axes2[0].plot([0, 100], [0, float(np.mean(psi)) * 100.0], "k:", linewidth=1.5, label="Random Targeting")
    axes2[0].axvline(20.0, color="gray", linestyle="--", alpha=0.6, label="20% Budget")
    axes2[0].axvline(11.6935, color="#1f77b4", linestyle=":", alpha=0.6, label="DR Threshold (11.69%)")
    axes2[0].set_xlabel("Targeted Population Percentage (%)")
    axes2[0].set_ylabel("Cumulative Incremental Conversions (% of Total Population)")
    axes2[0].set_title("Targeting Gain Curves: Uplift vs Baseline Purchase Probability")
    axes2[0].grid(alpha=0.3)
    axes2[0].legend(fontsize=8.5)

    # Right panel: Cost sensitivity (Per-Targeted Contact Cost Model)
    contact_sub = cost_sens_df[cost_sens_df["cost_model"] == "per_targeted_contact_cost"]
    for strategy, color, marker in [
        ("DR-Learner Fixed Threshold (11.69%)", "#1f77b4", "o"),
        ("DR-Learner Top 20%", "#17becf", "s"),
        ("T-Learner Top 20%", "#2ca02c", "^"),
        ("Baseline mu0 Top 20%", "#ff7f0e", "v"),
        ("Random 20%", "#7f7f7f", "x"),
        ("Treat All (100%)", "#d62728", "d"),
    ]:
        sub = contact_sub[contact_sub["strategy"] == strategy]
        axes2[1].plot(
            sub["cost_per_targeted"], sub["net_gain_per_user"],
            label=strategy, color=color, marker=marker, linewidth=1.8
        )
    axes2[1].axhline(0.0, color="black", linestyle="--", linewidth=0.9)
    axes2[1].axvline(0.05, color="gray", linestyle=":", label="Default Cost c=0.05")
    axes2[1].set_xlabel("Marketing Cost per Targeted User (c, with Margin m=100)")
    axes2[1].set_ylabel("Estimated Net Gain per Candidate User")
    axes2[1].set_title("Net Economic Value Across Cost Assumptions")
    axes2[1].grid(alpha=0.3)
    axes2[1].legend(fontsize=8)

    fig2.tight_layout()
    fig2.savefig(fig_dir / "policy_gain_and_cost_sensitivity.png", dpi=180)
    plt.close(fig2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teammate-dir", type=Path, default=DEFAULT_TEAMMATE_DIR)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--cost", type=float, default=0.05)
    parser.add_argument("--margin", type=float, default=100.0)
    args = parser.parse_args()

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"

    print(f"Loading teammate artifacts from: {args.teammate_dir}", flush=True)
    test_df = pd.read_csv(args.teammate_dir / "test_predictions.csv")
    train_oof_df = pd.read_csv(args.teammate_dir / "train_oof.csv")
    teammate_metrics = json.loads((args.teammate_dir / "metrics.json").read_text(encoding="utf-8"))

    print("Evaluating Criteo ATE across methods...", flush=True)
    ate_df = evaluate_criteo_ate(test_df, train_oof_df)
    ate_df.to_csv(out_dir / "criteo_ate_comparison.csv", index=False)

    print("Evaluating teammate targeting policies and baselines...", flush=True)
    policy_df, paired_df, cost_sens_df = evaluate_teammate_policies(
        test_df, cost=args.cost, margin=args.margin
    )
    policy_df.to_csv(out_dir / "policy_evaluation_table.csv", index=False)
    paired_df.to_csv(out_dir / "policy_paired_comparisons.csv", index=False)
    cost_sens_df.to_csv(out_dir / "policy_cost_sensitivity.csv", index=False)

    # Also save the exact minimal interface table requested by Section 8 of the prompt
    minimal_cols = [
        "model_name",
        "policy_id",
        "treatment_fraction",
        "estimated_incremental_effect",
        "standard_error",
        "ci_lower",
        "ci_upper",
        "estimated_net_gain",
    ]
    policy_df[minimal_cols].to_csv(out_dir / "teammate_minimal_evaluation_interface.csv", index=False)

    print("Generating Criteo & policy evaluation figures...", flush=True)
    plot_criteo_and_policy_figures(ate_df, policy_df, cost_sens_df, test_df, fig_dir)

    summary_payload = {
        "teammate_selected_model": teammate_metrics["selection"]["model"],
        "teammate_policy_threshold": teammate_metrics["selection"]["policy_threshold"],
        "ate_comparison": ate_df.to_dict(orient="records"),
        "policy_evaluation": policy_df.to_dict(orient="records"),
        "paired_comparisons": paired_df.to_dict(orient="records"),
    }
    (out_dir / "criteo_and_policy_summary.json").write_text(
        json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("Criteo & Policy Evaluation Complete!", flush=True)


if __name__ == "__main__":
    main()
