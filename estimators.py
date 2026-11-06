"""Core Causal Effect Estimators: Naive Diff-in-Means, In-Sample AIPW, Cross-Fitted DML-IRM / AIPW, Oracle, and Policy Value Evaluator."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import brier_score_loss, log_loss, r2_score, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


def summary_from_scores(psi: np.ndarray, alpha: float = 0.05) -> dict:
    """Compute point estimate, asymptotic standard error, and (1-alpha) Wald CI from influence scores."""
    psi = np.asarray(psi, dtype=float)
    n = len(psi)
    if n < 2:
        raise ValueError("Require at least 2 observations to compute standard error.")
    est = float(np.mean(psi))
    se = float(np.std(psi, ddof=1) / np.sqrt(n))
    z_crit = float(stats.norm.ppf(1.0 - alpha / 2.0))
    ci_lower = est - z_crit * se
    ci_upper = est + z_crit * se
    p_val = float(2.0 * (1.0 - stats.norm.cdf(abs(est / se)))) if se > 0 else 1.0
    return {
        "estimate": est,
        "se": se,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "ci_width": ci_upper - ci_lower,
        "p_value": p_val,
        "n": int(n),
    }


def aipw_score(
    y: np.ndarray,
    t: np.ndarray,
    m0: np.ndarray,
    m1: np.ndarray,
    e: np.ndarray,
) -> np.ndarray:
    """Compute Augmented Inverse Propensity Weighted (AIPW / DML-IRM) orthogonal score."""
    y = np.asarray(y, dtype=float)
    t = np.asarray(t, dtype=float)
    m0 = np.asarray(m0, dtype=float)
    m1 = np.asarray(m1, dtype=float)
    e = np.asarray(e, dtype=float)
    return (m1 - m0) + t * (y - m1) / e - (1.0 - t) * (y - m0) / (1.0 - e)


def ipw_score(y: np.ndarray, t: np.ndarray, e: np.ndarray) -> np.ndarray:
    """Compute Horvitz-Thompson Inverse Propensity Weighted (IPW) score."""
    y = np.asarray(y, dtype=float)
    t = np.asarray(t, dtype=float)
    e = np.asarray(e, dtype=float)
    return t * y / e - (1.0 - t) * y / (1.0 - e)


def estimate_naive(t: np.ndarray, y: np.ndarray, alpha: float = 0.05) -> dict:
    """Step 1 Baseline: Naive difference in means E[Y|T=1] - E[Y|T=0] with Neyman robust SE."""
    t = np.asarray(t, dtype=int)
    y = np.asarray(y, dtype=float)
    mask1 = t == 1
    mask0 = t == 0
    n1 = int(np.sum(mask1))
    n0 = int(np.sum(mask0))
    if n1 < 2 or n0 < 2:
        raise ValueError(f"Require >=2 samples in both treatment and control; got n1={n1}, n0={n0}")
    y1 = y[mask1]
    y0 = y[mask0]
    mean1 = float(np.mean(y1))
    mean0 = float(np.mean(y0))
    est = mean1 - mean0
    var1 = float(np.var(y1, ddof=1))
    var0 = float(np.var(y0, ddof=1))
    se = float(np.sqrt(var1 / n1 + var0 / n0))
    z_crit = float(stats.norm.ppf(1.0 - alpha / 2.0))
    ci_lower = est - z_crit * se
    ci_upper = est + z_crit * se
    p_val = float(2.0 * (1.0 - stats.norm.cdf(abs(est / se)))) if se > 0 else 1.0

    # Construct sample-level influence scores whose sample mean is est
    p_treat = n1 / (n1 + n0)
    psi = t * (y - mean1) / p_treat - (1 - t) * (y - mean0) / (1.0 - p_treat) + est
    return {
        "method": "naive_diff_in_means",
        "estimate": est,
        "se": se,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "ci_width": ci_upper - ci_lower,
        "p_value": p_val,
        "mean_treated": mean1,
        "mean_control": mean0,
        "n_treated": n1,
        "n_control": n0,
        "n": n1 + n0,
        "psi": psi,
    }


def _build_classifier(kind: str, seed: int = 42, **kwargs):
    """Build nuisance binary classifier for propensity e(X) or binary outcome m_t(X)."""
    if kind == "gbdt":
        return HistGradientBoostingClassifier(
            max_iter=kwargs.get("max_iter", 30),
            max_bins=kwargs.get("max_bins", 64),
            learning_rate=kwargs.get("learning_rate", 0.10),
            max_leaf_nodes=kwargs.get("max_leaf_nodes", 15),
            min_samples_leaf=kwargs.get("min_samples_leaf", 20),
            l2_regularization=kwargs.get("l2_regularization", 1.0),
            early_stopping=False,
            random_state=seed,
        )
    if kind == "gbdt_overfit":
        # Deliberately high-capacity tree to illustrate in-sample overfitting vs cross-fitting
        return HistGradientBoostingClassifier(
            max_iter=kwargs.get("max_iter", 45),
            max_bins=kwargs.get("max_bins", 64),
            learning_rate=kwargs.get("learning_rate", 0.18),
            max_leaf_nodes=kwargs.get("max_leaf_nodes", 63),
            min_samples_leaf=kwargs.get("min_samples_leaf", 5),
            l2_regularization=kwargs.get("l2_regularization", 0.0),
            early_stopping=False,
            random_state=seed,
        )
    if kind == "logistic":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(C=kwargs.get("C", 1.0), max_iter=500, random_state=seed),
        )
    if kind == "poly_logistic":
        return make_pipeline(
            PolynomialFeatures(degree=2, include_bias=False),
            StandardScaler(),
            LogisticRegression(C=kwargs.get("C", 1.0), max_iter=500, random_state=seed),
        )
    if hasattr(kind, "fit"):
        return clone(kind)
    raise ValueError(f"Unknown learner kind: {kind}")


def _build_regressor(kind: str, seed: int = 42, **kwargs):
    """Build nuisance regressor for continuous outcomes if needed."""
    if kind in ("gbdt", "gbdt_overfit"):
        is_overfit = kind == "gbdt_overfit"
        return HistGradientBoostingRegressor(
            max_iter=kwargs.get("max_iter", 200 if is_overfit else 100),
            learning_rate=kwargs.get("learning_rate", 0.12 if is_overfit else 0.05),
            max_leaf_nodes=kwargs.get("max_leaf_nodes", 63 if is_overfit else 15),
            min_samples_leaf=kwargs.get("min_samples_leaf", 5 if is_overfit else 30),
            l2_regularization=kwargs.get("l2_regularization", 0.0 if is_overfit else 5.0),
            early_stopping=False,
            random_state=seed,
        )
    if kind == "linear":
        return make_pipeline(StandardScaler(), Ridge(alpha=kwargs.get("alpha", 1.0)))
    if kind == "poly_linear":
        return make_pipeline(
            PolynomialFeatures(degree=2, include_bias=False),
            StandardScaler(),
            Ridge(alpha=kwargs.get("alpha", 1.0)),
        )
    if hasattr(kind, "fit"):
        return clone(kind)
    raise ValueError(f"Unknown regressor kind: {kind}")


def _fit_and_predict_nuisance(
    x_train_e: np.ndarray,
    x_train_m: np.ndarray,
    t_train: np.ndarray,
    y_train: np.ndarray,
    x_eval_e: np.ndarray,
    x_eval_m: np.ndarray,
    e_learner: str = "gbdt",
    m_learner: str = "gbdt",
    propensity_mode: str = "estimated",
    known_propensity: float | None = None,
    is_binary_y: bool = True,
    seed: int = 42,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit nuisance models (e, m0, m1) on training split and predict on evaluation split."""
    idx0 = np.flatnonzero(t_train == 0)
    idx1 = np.flatnonzero(t_train == 1)
    if len(idx0) == 0 or len(idx1) == 0:
        raise ValueError("Training fold has empty treatment or control arm.")

    # Fit outcome models m0(X) and m1(X)
    if is_binary_y:
        if len(np.unique(y_train[idx0])) < 2:
            m0_pred = np.full(len(x_eval_m), float(np.mean(y_train[idx0])))
        else:
            clf0 = _build_classifier(m_learner, seed=seed, **kwargs)
            clf0.fit(x_train_m[idx0], y_train[idx0])
            m0_pred = clf0.predict_proba(x_eval_m)[:, 1]

        if len(np.unique(y_train[idx1])) < 2:
            m1_pred = np.full(len(x_eval_m), float(np.mean(y_train[idx1])))
        else:
            clf1 = _build_classifier(m_learner, seed=seed + 1, **kwargs)
            clf1.fit(x_train_m[idx1], y_train[idx1])
            m1_pred = clf1.predict_proba(x_eval_m)[:, 1]
    else:
        reg0 = _build_regressor(m_learner, seed=seed, **kwargs)
        reg0.fit(x_train_m[idx0], y_train[idx0])
        m0_pred = reg0.predict(x_eval_m)

        reg1 = _build_regressor(m_learner, seed=seed + 1, **kwargs)
        reg1.fit(x_train_m[idx1], y_train[idx1])
        m1_pred = reg1.predict(x_eval_m)

    # Fit or assign propensity model e(X)
    if propensity_mode == "constant":
        e_val = float(known_propensity if known_propensity is not None else np.mean(t_train))
        e_raw = np.full(len(x_eval_e), e_val, dtype=float)
    elif propensity_mode == "estimated":
        clf_e = _build_classifier(e_learner, seed=seed + 2, **kwargs)
        clf_e.fit(x_train_e, t_train)
        e_raw = clf_e.predict_proba(x_eval_e)[:, 1]
    else:
        raise ValueError(f"Unknown propensity_mode: {propensity_mode}")

    return m0_pred, m1_pred, e_raw


def _compute_diagnostics(
    y: np.ndarray,
    t: np.ndarray,
    m0: np.ndarray,
    m1: np.ndarray,
    e_raw: np.ndarray,
    e_clipped: np.ndarray,
    clip_bounds: tuple[float, float],
    is_binary_y: bool,
) -> dict:
    """Compute overlap diagnostics and auxiliary model prediction quality metrics."""
    low, high = clip_bounds
    y_pred_obs = np.where(t == 1, m1, m0)
    diag = {
        "e_raw_min": float(np.min(e_raw)),
        "e_raw_max": float(np.max(e_raw)),
        "e_raw_std": float(np.std(e_raw)),
        "clipped_low_fraction": float(np.mean(e_raw < low)),
        "clipped_high_fraction": float(np.mean(e_raw > high)),
        "clipped_fraction": float(np.mean((e_raw < low) | (e_raw > high))),
    }
    # Effective sample size (ESS) of IPW weights
    w1 = t / e_clipped
    w0 = (1.0 - t) / (1.0 - e_clipped)
    diag["ess_treated"] = float((np.sum(w1) ** 2) / np.sum(w1 ** 2)) if np.sum(w1 ** 2) > 0 else 0.0
    diag["ess_control"] = float((np.sum(w0) ** 2) / np.sum(w0 ** 2)) if np.sum(w0 ** 2) > 0 else 0.0

    # Outcome predictive performance
    if is_binary_y and len(np.unique(y)) == 2:
        y_prob = np.clip(y_pred_obs, 1e-7, 1.0 - 1e-7)
        diag["outcome_auc"] = float(roc_auc_score(y, y_prob))
        diag["outcome_brier"] = float(brier_score_loss(y, y_prob))
        diag["outcome_logloss"] = float(log_loss(y, y_prob))
    else:
        diag["outcome_r2"] = float(r2_score(y, y_pred_obs))
        diag["outcome_mse"] = float(np.mean((y - y_pred_obs) ** 2))

    if len(np.unique(t)) == 2 and np.std(e_raw) > 1e-8:
        diag["propensity_auc"] = float(roc_auc_score(t, e_raw))
    else:
        diag["propensity_auc"] = 0.5
    return diag


def _extract_feature_arrays(
    x: pd.DataFrame | np.ndarray,
    e_features: list | None = None,
    m_features: list | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(x, pd.DataFrame):
        xe = x[e_features].to_numpy(dtype=float) if e_features is not None else x.to_numpy(dtype=float)
        xm = x[m_features].to_numpy(dtype=float) if m_features is not None else x.to_numpy(dtype=float)
    else:
        x_arr = np.asarray(x, dtype=float)
        if x_arr.ndim == 1:
            x_arr = x_arr.reshape(-1, 1)
        xe = x_arr[:, e_features] if e_features is not None else x_arr
        xm = x_arr[:, m_features] if m_features is not None else x_arr
    return xe, xm


def estimate_aipw_insample(
    x: pd.DataFrame | np.ndarray,
    t: np.ndarray,
    y: np.ndarray,
    e_learner: str = "gbdt",
    m_learner: str = "gbdt",
    e_features: list | None = None,
    m_features: list | None = None,
    clip_bounds: tuple[float, float] = (0.02, 0.98),
    propensity_mode: str = "estimated",
    known_propensity: float | None = None,
    alpha: float = 0.05,
    seed: int = 42,
    **kwargs,
) -> dict:
    """Step 2 Baseline: In-sample AIPW WITHOUT cross-fitting (to study overfitting impact)."""
    t = np.asarray(t, dtype=int)
    y = np.asarray(y, dtype=float)
    is_binary_y = bool(np.isin(y, [0, 1]).all())
    xe, xm = _extract_feature_arrays(x, e_features, m_features)

    m0, m1, e_raw = _fit_and_predict_nuisance(
        xe, xm, t, y, xe, xm,
        e_learner=e_learner,
        m_learner=m_learner,
        propensity_mode=propensity_mode,
        known_propensity=known_propensity,
        is_binary_y=is_binary_y,
        seed=seed,
        **kwargs,
    )
    e_clipped = np.clip(e_raw, clip_bounds[0], clip_bounds[1])
    psi = aipw_score(y, t, m0, m1, e_clipped)
    res = summary_from_scores(psi, alpha=alpha)
    res.update({
        "method": "aipw_no_crossfit",
        "m0": m0,
        "m1": m1,
        "e_raw": e_raw,
        "e": e_clipped,
        "psi": psi,
        "diagnostics": _compute_diagnostics(y, t, m0, m1, e_raw, e_clipped, clip_bounds, is_binary_y),
    })
    return res


def estimate_dml_irm(
    x: pd.DataFrame | np.ndarray,
    t: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    e_learner: str = "gbdt",
    m_learner: str = "gbdt",
    e_features: list | None = None,
    m_features: list | None = None,
    clip_bounds: tuple[float, float] = (0.02, 0.98),
    propensity_mode: str = "estimated",
    known_propensity: float | None = None,
    alpha: float = 0.05,
    seed: int = 42,
    **kwargs,
) -> dict:
    """Step 3 Main Estimator: K-Fold Cross-Fitted AIPW / DoubleML IRM for ATE."""
    t = np.asarray(t, dtype=int)
    y = np.asarray(y, dtype=float)
    n = len(y)
    is_binary_y = bool(np.isin(y, [0, 1]).all())
    xe, xm = _extract_feature_arrays(x, e_features, m_features)

    m0 = np.full(n, np.nan, dtype=float)
    m1 = np.full(n, np.nan, dtype=float)
    e_raw = np.full(n, np.nan, dtype=float)

    # Choose stratified splitting on (T, Y) when binary Y has enough samples per cell
    if is_binary_y:
        strata = 2 * t + y.astype(int)
        counts = np.bincount(strata, minlength=4)
        if counts.min() >= n_splits:
            splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            split_iter = splitter.split(xe, strata)
        else:
            splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            split_iter = splitter.split(xe, t)
    else:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        split_iter = splitter.split(xe, t)

    for fold_idx, (train_idx, eval_idx) in enumerate(split_iter):
        m0_k, m1_k, e_raw_k = _fit_and_predict_nuisance(
            xe[train_idx], xm[train_idx], t[train_idx], y[train_idx],
            xe[eval_idx], xm[eval_idx],
            e_learner=e_learner,
            m_learner=m_learner,
            propensity_mode=propensity_mode,
            known_propensity=known_propensity,
            is_binary_y=is_binary_y,
            seed=seed + 10 * fold_idx,
            **kwargs,
        )
        m0[eval_idx] = m0_k
        m1[eval_idx] = m1_k
        e_raw[eval_idx] = e_raw_k

    e_clipped = np.clip(e_raw, clip_bounds[0], clip_bounds[1])
    psi = aipw_score(y, t, m0, m1, e_clipped)
    res = summary_from_scores(psi, alpha=alpha)
    res.update({
        "method": f"dml_irm_{n_splits}fold",
        "m0": m0,
        "m1": m1,
        "e_raw": e_raw,
        "e": e_clipped,
        "psi": psi,
        "diagnostics": _compute_diagnostics(y, t, m0, m1, e_raw, e_clipped, clip_bounds, is_binary_y),
    })
    return res


def estimate_oracle(
    t: np.ndarray,
    y: np.ndarray,
    p0_true: np.ndarray,
    p1_true: np.ndarray,
    e_true: np.ndarray,
    clip_bounds: tuple[float, float] = (0.001, 0.999),
    alpha: float = 0.05,
) -> dict:
    """Ideal Benchmark (Simulation Only): AIPW score using exact ground-truth (p0, p1, e)."""
    t = np.asarray(t, dtype=int)
    y = np.asarray(y, dtype=float)
    p0_true = np.asarray(p0_true, dtype=float)
    p1_true = np.asarray(p1_true, dtype=float)
    e_true = np.asarray(e_true, dtype=float)
    e_clipped = np.clip(e_true, clip_bounds[0], clip_bounds[1])
    psi = aipw_score(y, t, p0_true, p1_true, e_clipped)
    res = summary_from_scores(psi, alpha=alpha)
    res.update({
        "method": "oracle_aipw",
        "psi": psi,
    })
    return res


def evaluate_policy(
    psi: np.ndarray,
    selected: np.ndarray,
    cost: float = 0.05,
    margin: float = 100.0,
    alpha: float = 0.05,
    model_name: str = "model",
    policy_id: str = "policy",
) -> dict:
    """
    Evaluate a targeting policy pi(X) in [0, 1] using DR pseudo-outcomes psi_i.

    Computes population-level incremental effect E[pi(X) * tau(X)], conditional effect among
    targeted users E[tau(X) | pi(X)=1], and net economic gain E[pi(X) * (margin * tau(X) - cost)].
    """
    psi = np.asarray(psi, dtype=float)
    pi = np.asarray(selected, dtype=float)
    if len(psi) != len(pi):
        raise ValueError("Length mismatch between psi and selected policy indicator.")

    treat_frac = float(np.mean(pi))
    inc_scores = pi * psi
    inc_stats = summary_from_scores(inc_scores, alpha=alpha)

    net_scores = pi * (margin * psi - cost)
    net_stats = summary_from_scores(net_scores, alpha=alpha)

    # Conditional average treatment effect on the targeted subpopulation
    if treat_frac > 0:
        cond_ate = inc_stats["estimate"] / treat_frac
        cond_se = inc_stats["se"] / treat_frac
    else:
        cond_ate = 0.0
        cond_se = 0.0

    return {
        "model_name": model_name,
        "policy_id": policy_id,
        "treatment_fraction": treat_frac,
        "targeted_count": int(np.round(np.sum(pi))),
        "total_count": int(len(psi)),
        "estimated_incremental_effect": inc_stats["estimate"],
        "standard_error": inc_stats["se"],
        "ci_lower": inc_stats["ci_lower"],
        "ci_upper": inc_stats["ci_upper"],
        "conditional_effect_on_targeted": cond_ate,
        "conditional_se_on_targeted": cond_se,
        "estimated_net_gain": net_stats["estimate"],
        "net_gain_se": net_stats["se"],
        "net_gain_ci_lower": net_stats["ci_lower"],
        "net_gain_ci_upper": net_stats["ci_upper"],
        "assumed_cost": float(cost),
        "assumed_margin": float(margin),
    }


def select_top_k_fraction(
    scores: np.ndarray,
    fraction: float,
    seed: int = 42,
) -> np.ndarray:
    """
    Create binary targeting indicator for top `fraction` of users sorted by `scores`.
    Breaks ties randomly using a fixed seed so constant/tied models do not depend on file order.
    """
    scores = np.asarray(scores, dtype=float)
    n = len(scores)
    k = int(np.round(fraction * n))
    if k <= 0:
        return np.zeros(n, dtype=float)
    if k >= n:
        return np.ones(n, dtype=float)
    rng = np.random.default_rng(seed)
    jitter = rng.random(n)
    order = np.lexsort((jitter, -scores))
    selected = np.zeros(n, dtype=float)
    selected[order[:k]] = 1.0
    return selected
