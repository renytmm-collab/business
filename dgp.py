"""Data Generating Processes (DGPs) with Known Ground-Truth Causal Effects for Simulation Experiments."""
from __future__ import annotations

from functools import lru_cache
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit


def _nonlinear_index(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute latent confounding, baseline outcome, and heterogeneity indices from X.
    Combines linear, threshold/step, and smooth nonlinear terms so both tree-based
    and polynomial learners can capture the confounding structure in finite samples.
    """
    z_step = np.where(x[:, 0] > 0.0, 0.6, -0.6) + np.where(x[:, 1] > 0.0, 0.5, -0.5)
    z_quad = 0.25 * (np.clip(x[:, 0], -2.5, 2.5) ** 2 - 1.0)
    # Propensity index g_e(X): positive values mean higher probability of receiving marketing T=1
    g_e = 0.55 * x[:, 0] + 0.45 * x[:, 1] + 0.45 * z_step + z_quad - 0.20 * x[:, 2]
    # Baseline buying propensity index g_m0(X): positively correlated with g_e(X) -> positive confounding!
    g_m0 = 0.65 * x[:, 0] + 0.55 * x[:, 1] + 0.55 * z_step + z_quad + 0.25 * x[:, 3]
    # CATE heterogeneity index g_tau(X) with mean ~ 0
    g_tau = 0.30 * x[:, 0] - 0.20 * x[:, 2] + 0.20 * np.where(x[:, 1] > 0.0, 1.0, -1.0)
    return g_e, g_m0, g_tau


@lru_cache(maxsize=64)
def calibrate_logistic_shift(
    target_ate: float,
    base_intercept: float = -1.4,
    heterogeneity_scale: float = 0.2,
) -> tuple[float, float]:
    """
    Solve for the logistic treatment coefficient `beta_t` such that the population
    average treatment effect E_X[p1(X) - p0(X)] equals `target_ate` up to high precision,
    using a fixed Monte Carlo reference population of N=500,000 draws.

    Returns (beta_t, exact_population_ate).
    """
    rng = np.random.default_rng(20261105)
    x_ref = rng.normal(size=(100_000, 5))
    _, g_m0, g_tau = _nonlinear_index(x_ref)
    eta0 = base_intercept + g_m0
    p0 = expit(eta0)

    def objective(beta_t: float) -> float:
        p1 = expit(eta0 + beta_t + heterogeneity_scale * g_tau)
        return float(np.mean(p1 - p0) - target_ate)

    beta_t = float(brentq(objective, -5.0, 5.0, xtol=1e-7))
    p1 = expit(eta0 + beta_t + heterogeneity_scale * g_tau)
    pop_ate = float(np.mean(p1 - p0))
    return beta_t, pop_ate


def generate_dgp_baseline(
    n: int = 3000,
    target_ate: float = 0.02,
    overlap_scale: float = 1.0,
    base_intercept: float = -1.4,
    heterogeneity_scale: float = 0.2,
    seed: int = 42,
) -> dict:
    """
    Standard Confounded Binary Outcome DGP for:
      - Experiment 1 (Effect Size: varying `target_ate`)
      - Experiment 2 (Sample Size: varying `n`)
      - Experiment 3 (Overlap / Positivity: varying `overlap_scale`)
      - In-sample AIPW vs Cross-fitted DML-IRM comparison

    Since high-propensity users also have higher baseline conversion probability p0(X),
    Naive Diff-in-Means suffers from upward selection bias.
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 5))
    g_e, g_m0, g_tau = _nonlinear_index(x)

    beta_t, pop_ate = calibrate_logistic_shift(
        round(float(target_ate), 6),
        round(float(base_intercept), 4),
        round(float(heterogeneity_scale), 4),
    )

    e_true = expit(overlap_scale * g_e)
    eta0 = base_intercept + g_m0
    p0_true = expit(eta0)
    p1_true = expit(eta0 + beta_t + heterogeneity_scale * g_tau)
    tau_true = p1_true - p0_true

    t = rng.binomial(1, e_true)
    p_obs = np.where(t == 1, p1_true, p0_true)
    y = rng.binomial(1, p_obs)

    feature_names = [f"f{i}" for i in range(x.shape[1])]
    df_x = pd.DataFrame(x, columns=feature_names)
    return {
        "x": df_x,
        "t": t,
        "y": y,
        "p0_true": p0_true,
        "p1_true": p1_true,
        "e_true": e_true,
        "tau_true": tau_true,
        "pop_ate": pop_ate,
        "sample_ate": float(np.mean(tau_true)),
        "logistic_beta_t": beta_t,
        "feature_names": feature_names,
    }


@lru_cache(maxsize=16)
def _calibrate_misspec_ate(target_ate: float = 0.03) -> tuple[float, float]:
    """Calibrate population ATE for the 4-way Model Misspecification DGP."""
    rng = np.random.default_rng(20261106)
    x_ref = rng.normal(size=(100_000, 4))
    z0 = (x_ref[:, 0] ** 2 - 1.0) / np.sqrt(2.0)
    z1 = x_ref[:, 0] * x_ref[:, 1]
    z2 = np.sin(np.pi * x_ref[:, 2]) / np.sqrt(0.5)
    z3 = x_ref[:, 3]
    eta0 = -1.3 + 0.75 * z0 + 0.65 * z1 - 0.45 * z2 + 0.30 * z3
    p0 = expit(eta0)

    def obj(beta_t: float) -> float:
        p1 = expit(eta0 + beta_t + 0.25 * z0 - 0.20 * z1)
        return float(np.mean(p1 - p0) - target_ate)

    beta_t = float(brentq(obj, -4.0, 4.0, xtol=1e-7))
    p1 = expit(eta0 + beta_t + 0.25 * z0 - 0.20 * z1)
    return beta_t, float(np.mean(p1 - p0))


def generate_dgp_misspecification(
    n: int = 3000,
    target_ate: float = 0.03,
    seed: int = 42,
) -> dict:
    """
    DGP for Experiment 4 (4-Way Model Misspecification & Double Robustness Verification).

    True confounding operates through nonlinear orthogonal basis features:
      z0 = (x0^2 - 1)/sqrt(2), z1 = x0*x1, z2 = sin(pi*x2)/sqrt(0.5), z3 = x3.
    Because E[x_j * z0] = 0 and E[x_j * z1] = 0 for Gaussian X, a misspecified linear model
    using only raw [x0, x1, x2, x3] fails to capture the confounding between e(X) and m(X),
    whereas a model given the true specification [z0, z1, z2, z3] is exact.
    """
    rng = np.random.default_rng(seed)
    x_raw = rng.normal(size=(n, 4))
    z0 = (x_raw[:, 0] ** 2 - 1.0) / np.sqrt(2.0)
    z1 = x_raw[:, 0] * x_raw[:, 1]
    z2 = np.sin(np.pi * x_raw[:, 2]) / np.sqrt(0.5)
    z3 = x_raw[:, 3]

    beta_t, pop_ate = _calibrate_misspec_ate(round(float(target_ate), 6))

    # True propensity and outcome depend on [z0, z1, z2, z3]
    e_true = expit(0.80 * z0 + 0.70 * z1 - 0.45 * z2 + 0.20 * z3)
    eta0 = -1.3 + 0.75 * z0 + 0.65 * z1 - 0.45 * z2 + 0.30 * z3
    p0_true = expit(eta0)
    p1_true = expit(eta0 + beta_t + 0.25 * z0 - 0.20 * z1)
    tau_true = p1_true - p0_true

    t = rng.binomial(1, e_true)
    y = rng.binomial(1, np.where(t == 1, p1_true, p0_true))

    df = pd.DataFrame({
        "x0": x_raw[:, 0],
        "x1": x_raw[:, 1],
        "x2": x_raw[:, 2],
        "x3": x_raw[:, 3],
        "z0": z0,
        "z1": z1,
        "z2": z2,
        "z3": z3,
    })
    return {
        "df": df,
        "raw_cols": ["x0", "x1", "x2", "x3"],
        "true_cols": ["z0", "z1", "z2", "z3"],
        "t": t,
        "y": y,
        "p0_true": p0_true,
        "p1_true": p1_true,
        "e_true": e_true,
        "tau_true": tau_true,
        "pop_ate": pop_ate,
        "sample_ate": float(np.mean(tau_true)),
    }


@lru_cache(maxsize=16)
def _compute_post_treatment_pop_ate(scenario: str) -> float:
    """Compute exact population total ATE for each post-treatment DGP scenario."""
    rng = np.random.default_rng(20261107)
    n_ref = 100_000
    x = rng.normal(size=(n_ref, 4))
    u = rng.normal(size=n_ref)
    base_x = 0.45 * x[:, 0] + 0.35 * x[:, 1] - 0.25 * x[:, 2] + 0.20 * x[:, 3]

    if scenario == "mediator_only":
        # T -> Visit -> Conversion (most of advertising effect goes through increasing site visits)
        p_v0 = expit(-0.8 + 0.5 * base_x)
        p_v1 = expit(-0.8 + 0.5 * base_x + 1.6)  # Ad strongly increases visit probability
        # Conversion probability given (X, T, V)
        p_y_t0_v0 = expit(-2.4 + 0.6 * base_x)
        p_y_t0_v1 = expit(-2.4 + 0.6 * base_x + 1.8)
        p_y_t1_v0 = expit(-2.4 + 0.6 * base_x + 0.15)
        p_y_t1_v1 = expit(-2.4 + 0.6 * base_x + 0.15 + 1.8)
        p0 = (1.0 - p_v0) * p_y_t0_v0 + p_v0 * p_y_t0_v1
        p1 = (1.0 - p_v1) * p_y_t1_v0 + p_v1 * p_y_t1_v1
        return float(np.mean(p1 - p0))

    if scenario == "mediator_and_collider":
        # T -> Visit <- U_intent -> Conversion
        # Unobserved immediate buying intent U strongly drives both Visit and Conversion
        p_v0 = expit(-0.9 + 0.4 * base_x + 1.5 * u)
        p_v1 = expit(-0.9 + 0.4 * base_x + 1.7 + 1.5 * u)
        p_y_t0 = (1.0 - p_v0) * expit(-2.5 + 0.5 * base_x + 1.6 * u) + p_v0 * expit(-2.5 + 0.5 * base_x + 1.2 + 1.6 * u)
        p_y_t1 = (1.0 - p_v1) * expit(-2.5 + 0.5 * base_x + 0.1 + 1.6 * u) + p_v1 * expit(-2.5 + 0.5 * base_x + 0.1 + 1.2 + 1.6 * u)
        return float(np.mean(p_y_t1 - p_y_t0))

    if scenario == "reverse_amplification":
        # Case where conditioning on a post-treatment variable OVERESTIMATES the effect:
        # Ad T has a small positive effect on Y, but T negatively affects a post-treatment cost/complaint
        # indicator Z_post while U_loyalty positively affects Z_post and Y -> conditioning on Z_post inflates ATE!
        p_v0 = expit(-0.2 + 0.4 * base_x + 1.5 * u)
        p_v1 = expit(-0.2 + 0.4 * base_x - 1.4 + 1.5 * u)
        p0 = expit(-1.8 + 0.5 * base_x + 1.5 * u)
        p1 = expit(-1.8 + 0.5 * base_x + 0.25 + 1.5 * u)
        return float(np.mean(p1 - p0))

    raise ValueError(f"Unknown post-treatment scenario: {scenario}")


def generate_dgp_post_treatment(
    n: int = 3000,
    scenario: str = "mediator_only",
    seed: int = 42,
) -> dict:
    """
    DGP for Work 3 (Section 4): Post-Treatment Feature Bias Experiment.
    Supports three causal mechanisms:
      1. `mediator_only`: T -> Visit -> Y (blocking mediator path underestimates total effect)
      2. `mediator_and_collider`: T -> Visit <- U -> Y (blocks mediator AND opens collider, can flip sign!)
      3. `reverse_amplification`: T -> Z_post <- U -> Y with negative T->Z_post (conditioning inflates/overestimates ATE!)
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 4))
    u = rng.normal(size=n)
    base_x = 0.45 * x[:, 0] + 0.35 * x[:, 1] - 0.25 * x[:, 2] + 0.20 * x[:, 3]
    e_true = expit(0.55 * x[:, 0] + 0.40 * x[:, 1] - 0.30 * x[:, 2])
    t = rng.binomial(1, e_true)

    if scenario == "mediator_only":
        p_v0 = expit(-0.8 + 0.5 * base_x)
        p_v1 = expit(-0.8 + 0.5 * base_x + 1.6)
        v_post = rng.binomial(1, np.where(t == 1, p_v1, p_v0))
        p_y = expit(-2.4 + 0.6 * base_x + 0.15 * t + 1.8 * v_post)
        y = rng.binomial(1, p_y)
        p0_true = (1.0 - p_v0) * expit(-2.4 + 0.6 * base_x) + p_v0 * expit(-2.4 + 0.6 * base_x + 1.8)
        p1_true = (1.0 - p_v1) * expit(-2.4 + 0.6 * base_x + 0.15) + p_v1 * expit(-2.4 + 0.6 * base_x + 0.15 + 1.8)

    elif scenario == "mediator_and_collider":
        p_v0 = expit(-0.9 + 0.4 * base_x + 1.5 * u)
        p_v1 = expit(-0.9 + 0.4 * base_x + 1.7 + 1.5 * u)
        v_post = rng.binomial(1, np.where(t == 1, p_v1, p_v0))
        p_y = expit(-2.5 + 0.5 * base_x + 0.10 * t + 1.2 * v_post + 1.6 * u)
        y = rng.binomial(1, p_y)
        # Marginal over U for sample-level benchmark
        p0_true = (1.0 - p_v0) * expit(-2.5 + 0.5 * base_x + 1.6 * u) + p_v0 * expit(-2.5 + 0.5 * base_x + 1.2 + 1.6 * u)
        p1_true = (1.0 - p_v1) * expit(-2.5 + 0.5 * base_x + 0.1 + 1.6 * u) + p_v1 * expit(-2.5 + 0.5 * base_x + 0.1 + 1.2 + 1.6 * u)

    elif scenario == "reverse_amplification":
        p_v0 = expit(-0.2 + 0.4 * base_x + 1.5 * u)
        p_v1 = expit(-0.2 + 0.4 * base_x - 1.4 + 1.5 * u)
        v_post = rng.binomial(1, np.where(t == 1, p_v1, p_v0))
        p_y = expit(-1.8 + 0.5 * base_x + 0.25 * t + 1.5 * u)
        y = rng.binomial(1, p_y)
        p0_true = expit(-1.8 + 0.5 * base_x + 1.5 * u)
        p1_true = expit(-1.8 + 0.5 * base_x + 0.25 + 1.5 * u)
    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    pop_ate = _compute_post_treatment_pop_ate(scenario)
    pre_cols = ["f0", "f1", "f2", "f3"]
    df = pd.DataFrame(x, columns=pre_cols)
    df["post_visit"] = v_post.astype(float)

    return {
        "df": df,
        "pre_cols": pre_cols,
        "all_cols": pre_cols + ["post_visit"],
        "t": t,
        "y": y,
        "v_post": v_post,
        "p0_true": p0_true,
        "p1_true": p1_true,
        "e_true": e_true,
        "pop_ate": pop_ate,
        "sample_ate": float(np.mean(p1_true - p0_true)),
    }


def generate_dgp_identifiability(
    n: int = 3000,
    mode: str = "unmeasured_confounding",
    seed: int = 42,
) -> dict:
    """
    DGP for Work 4 (Section 5): When Causal / Uplift Conclusions Cannot Be Reliably Drawn.
    Modes:
      - `unmeasured_confounding`: Unobserved U drives both T and Y -> non-identifiable at any sample size.
      - `lack_of_overlap`: Deterministic/near-deterministic assignment in region X0 > 0.5 -> extrapolation failure.
      - `rare_events_small_effect`: Valid identification & overlap, but tiny effect (0.1%) + rare outcome (~0.3%).
      - `zero_ate_uniform`: True tau(X) = 0 everywhere -> ATE = 0 and no uplift learnable.
      - `zero_ate_cancellation`: True tau(X) = +0.06 for X0 > 0 and -0.06 for X0 <= 0 -> ATE = 0,
        yet strong individual CATE is highly learnable!
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 4))
    u = rng.normal(size=n)  # Unobserved confounder

    if mode == "unmeasured_confounding":
        e_true = expit(0.4 * x[:, 0] + 1.3 * u)
        p0_true = expit(-1.5 + 0.4 * x[:, 0] + 0.3 * x[:, 1] + 1.4 * u)
        p1_true = np.clip(p0_true + 0.02, 0.001, 0.999)
        pop_ate = 0.02

    elif mode == "lack_of_overlap":
        # Severe positivity violation: users with x0 > 0 almost always get T=1 (e ~ 0.997)
        # and have a distinct negative/different response than x0 <= 0
        e_true = expit(4.5 * x[:, 0] + 1.5 * x[:, 1])
        p0_true = expit(-1.2 + 0.8 * x[:, 0] + 0.5 * (x[:, 0] ** 2 - 1.0))
        tau_vec = np.where(x[:, 0] > 0.0, -0.04, 0.08)
        p1_true = np.clip(p0_true + tau_vec, 0.001, 0.999)
        pop_ate = 0.02

    elif mode == "rare_events_small_effect":
        # Mimics Criteo conversion: e ~ 0.85, baseline conversion ~ 0.002, true ATE = 0.001 (0.1 percentage point)
        e_true = np.full(n, 0.85)
        p0_true = np.clip(expit(-6.2 + 0.3 * x[:, 0] + 0.2 * x[:, 1]), 0.0002, 0.05)
        p1_true = np.clip(p0_true + 0.001, 0.0002, 0.06)
        pop_ate = 0.001

    elif mode == "zero_ate_uniform":
        e_true = expit(0.5 * x[:, 0] + 0.3 * x[:, 1])
        p0_true = expit(-1.5 + 0.5 * x[:, 0] + 0.4 * x[:, 1])
        p1_true = p0_true.copy()
        pop_ate = 0.0

    elif mode == "zero_ate_cancellation":
        # Half of population has +0.06 effect, half has -0.06 effect -> Population ATE is exactly 0.0!
        e_true = expit(0.5 * x[:, 0] + 0.3 * x[:, 1])
        p0_true = np.clip(expit(-1.5 + 0.4 * x[:, 1] - 0.3 * x[:, 2]), 0.08, 0.90)
        tau_vec = np.where(x[:, 0] > 0.0, 0.06, -0.06)
        p1_true = np.clip(p0_true + tau_vec, 0.01, 0.99)
        pop_ate = 0.0

    else:
        raise ValueError(f"Unknown identifiability mode: {mode}")

    t = rng.binomial(1, e_true)
    y = rng.binomial(1, np.where(t == 1, p1_true, p0_true))
    tau_true = p1_true - p0_true
    feature_names = ["f0", "f1", "f2", "f3"]
    df_x = pd.DataFrame(x, columns=feature_names)

    return {
        "x": df_x,
        "t": t,
        "y": y,
        "u": u,
        "p0_true": p0_true,
        "p1_true": p1_true,
        "e_true": e_true,
        "tau_true": tau_true,
        "pop_ate": pop_ate,
        "sample_ate": float(np.mean(tau_true)),
        "feature_names": feature_names,
    }


def generate_dgp_sure_things_vs_persuadables(
    n: int = 4000,
    seed: int = 42,
) -> dict:
    """
    DGP for Question 10 Bottleneck Verification:
    '优惠券/营销发给本来就会买的人（Sure Things）造成预算浪费 vs 发给可说服人群（Persuadables）'.

    User segments determined by pre-treatment features X:
      - Sure Things (x0 > 0.65, ~26% of users): High baseline buying prob p0 ~ 0.65, near-zero uplift tau = +0.005.
      - Persuadables (-0.5 <= x0 <= 0.65 and x1 > -0.2, ~25% of users): Moderate baseline p0 ~ 0.15, strong uplift tau = +0.14.
      - Lost Causes / Sleeping Dogs (remaining ~49% of users): Low baseline p0 ~ 0.08, negative/tiny uplift tau = -0.02.
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 4))
    e_true = expit(0.50 * x[:, 0] + 0.35 * x[:, 1] - 0.20 * x[:, 2])

    is_sure_thing = x[:, 0] > 0.65
    is_persuadable = (~is_sure_thing) & (x[:, 0] >= -0.50) & (x[:, 1] > -0.20)

    p0_true = np.where(
        is_sure_thing,
        np.clip(0.65 + 0.08 * expit(x[:, 1]), 0.55, 0.80),
        np.where(
            is_persuadable,
            np.clip(0.14 + 0.04 * expit(x[:, 2]), 0.08, 0.25),
            np.clip(0.07 + 0.03 * expit(x[:, 3]), 0.02, 0.15),
        ),
    )
    tau_true = np.where(
        is_sure_thing,
        0.005,
        np.where(is_persuadable, 0.140, -0.020),
    )
    p1_true = np.clip(p0_true + tau_true, 0.001, 0.999)

    t = rng.binomial(1, e_true)
    y = rng.binomial(1, np.where(t == 1, p1_true, p0_true))
    feature_names = ["f0", "f1", "f2", "f3"]
    df_x = pd.DataFrame(x, columns=feature_names)
    return {
        "x": df_x,
        "t": t,
        "y": y,
        "p0_true": p0_true,
        "p1_true": p1_true,
        "e_true": e_true,
        "tau_true": tau_true,
        "pop_ate": float(np.mean(tau_true)),
        "sample_ate": float(np.mean(tau_true)),
        "feature_names": feature_names,
    }

