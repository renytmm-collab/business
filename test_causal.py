"""Unit tests for Causal Effect Estimators, DGPs, Cross-Fitting, Double Robustness, and Policy Evaluation."""
import unittest
import numpy as np
import pandas as pd

from dgp import (
    generate_dgp_baseline,
    generate_dgp_identifiability,
    generate_dgp_misspecification,
    generate_dgp_post_treatment,
)
from estimators import (
    aipw_score,
    estimate_aipw_insample,
    estimate_dml_irm,
    estimate_naive,
    estimate_oracle,
    evaluate_policy,
    select_top_k_fraction,
)


class CausalEstimatorTests(unittest.TestCase):
    def test_aipw_double_robustness_algebraic_identity(self):
        """Verify exact expectation of AIPW score equals p1 - p0 when either e or (m0, m1) is true."""
        e_true, p0_true, p1_true = 0.35, 0.12, 0.20
        t = np.array([0, 0, 1, 1])
        y = np.array([0, 1, 0, 1])
        weights = np.array([
            (1.0 - e_true) * (1.0 - p0_true),
            (1.0 - e_true) * p0_true,
            e_true * (1.0 - p1_true),
            e_true * p1_true,
        ])
        # Case 1: Correct outcome model, wrong propensity model
        psi_m_correct = aipw_score(y, t, p0_true, p1_true, 0.75)
        self.assertAlmostEqual(float(np.sum(weights * psi_m_correct)), p1_true - p0_true, places=12)

        # Case 2: Wrong outcome model, correct propensity model
        psi_e_correct = aipw_score(y, t, 0.50, 0.80, e_true)
        self.assertAlmostEqual(float(np.sum(weights * psi_e_correct)), p1_true - p0_true, places=12)

    def test_dgp_probabilities_valid_and_calibrated(self):
        """Ensure all generated probabilities lie strictly in (0, 1) and population ATE matches target."""
        for target in (0.05, 0.01, 0.001):
            data = generate_dgp_baseline(n=2000, target_ate=target, seed=42)
            self.assertTrue(np.all((data["p0_true"] > 0.0) & (data["p0_true"] < 1.0)))
            self.assertTrue(np.all((data["p1_true"] > 0.0) & (data["p1_true"] < 1.0)))
            self.assertTrue(np.all((data["e_true"] > 0.0) & (data["e_true"] < 1.0)))
            self.assertAlmostEqual(data["pop_ate"], target, places=5)
            # Logistic beta_t should differ from probability-scale ATE
            self.assertNotAlmostEqual(data["logistic_beta_t"], data["pop_ate"], places=2)

    def test_cross_fitted_dml_beats_naive_on_confounded_data(self):
        """Verify DML-IRM removes confounding bias compared to Naive difference-in-means."""
        data = generate_dgp_baseline(n=4000, target_ate=0.02, seed=10)
        res_naive = estimate_naive(data["t"], data["y"])
        res_dml = estimate_dml_irm(data["x"], data["t"], data["y"], n_splits=3, seed=10)
        res_oracle = estimate_oracle(data["t"], data["y"], data["p0_true"], data["p1_true"], data["e_true"])

        err_naive = abs(res_naive["estimate"] - data["pop_ate"])
        err_dml = abs(res_dml["estimate"] - data["pop_ate"])
        self.assertLess(err_dml, err_naive)
        self.assertAlmostEqual(res_dml["estimate"], res_oracle["estimate"], delta=0.015)

    def test_post_treatment_feature_distorts_ate_while_improving_auc(self):
        """Verify conditioning on post-treatment visit increases outcome AUC but biases causal ATE."""
        data = generate_dgp_post_treatment(n=4000, scenario="mediator_and_collider", seed=7)
        res_pre = estimate_dml_irm(
            data["df"], data["t"], data["y"],
            e_features=data["pre_cols"], m_features=data["pre_cols"], n_splits=3, seed=7,
        )
        res_post = estimate_dml_irm(
            data["df"], data["t"], data["y"],
            e_features=data["all_cols"], m_features=data["all_cols"], n_splits=3, seed=7,
        )
        # Predictive AUC for Y improves when post_visit is included
        self.assertGreater(res_post["diagnostics"]["outcome_auc"], res_pre["diagnostics"]["outcome_auc"])
        # Yet causal bias relative to true total ATE is much worse when post_visit is included
        bias_pre = abs(res_pre["estimate"] - data["pop_ate"])
        bias_post = abs(res_post["estimate"] - data["pop_ate"])
        self.assertGreater(bias_post, bias_pre + 0.03)

    def test_policy_evaluation_population_vs_targeted_scale(self):
        """Verify population incremental effect = treatment_fraction * conditional_effect_on_targeted."""
        psi = np.array([0.05, 0.03, -0.01, -0.02, 0.04, 0.01, -0.03, 0.02, 0.00, 0.01])
        scores = np.array([0.9, 0.8, 0.1, 0.05, 0.85, 0.4, 0.01, 0.7, 0.3, 0.5])
        sel = select_top_k_fraction(scores, fraction=0.2, seed=42)
        self.assertAlmostEqual(float(np.mean(sel)), 0.2)
        pol = evaluate_policy(psi, sel, cost=0.05, margin=100.0, model_name="uplift", policy_id="top20")
        self.assertAlmostEqual(
            pol["estimated_incremental_effect"],
            pol["treatment_fraction"] * pol["conditional_effect_on_targeted"],
            places=12,
        )

    def test_baseline_outcome_aipw_score_double_robustness(self):
        """Verify AIPW baseline outcome score psi0 = mu0 + (1-T)*(Y-mu0)/(1-e) is doubly robust for E[Y(0)]."""
        e_true, p0_true, p1_true = 0.40, 0.18, 0.28
        t = np.array([0, 0, 1, 1])
        y = np.array([0, 1, 0, 1])
        weights = np.array([
            (1.0 - e_true) * (1.0 - p0_true),
            (1.0 - e_true) * p0_true,
            e_true * (1.0 - p1_true),
            e_true * p1_true,
        ])
        for mu0_guess, e_guess in [(p0_true, 0.75), (0.60, e_true)]:
            psi0 = mu0_guess + (1.0 - t) * (y - mu0_guess) / (1.0 - e_guess)
            self.assertAlmostEqual(float(np.sum(weights * psi0)), p0_true, places=12)



if __name__ == "__main__":
    unittest.main()
