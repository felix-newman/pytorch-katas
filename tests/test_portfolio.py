"""Regression tests for Markowitz window-shift instability."""

from __future__ import annotations

import unittest

import numpy as np

from pytorch_katas.portfolio import (
    TRADING_DAYS,
    TWO_WEEKS,
    allocate_window_pair,
    cov_condition_number,
    equal_weights,
    factor_precision,
    factor_tangency_weights,
    frobenius_rel,
    global_min_variance_weights,
    l1_turnover,
    mean_shift_std,
    reconstruct_factor_moments,
    rolling_weights,
    sample_moments,
    shrink_tangency_weights,
    simulate_equity_universe,
    structured_covariance,
    tangency_weights,
)


class TestAllocations(unittest.TestCase):
    def test_weights_sum_to_one(self) -> None:
        universe = simulate_equity_universe(n_days=300, n_assets=8, seed=1)
        mu, cov = sample_moments(universe.returns)
        for w in (
            tangency_weights(mu, cov),
            global_min_variance_weights(cov),
            shrink_tangency_weights(universe.returns),
            factor_tangency_weights(universe.returns),
            equal_weights(universe.n_assets),
        ):
            self.assertAlmostEqual(float(w.sum()), 1.0, places=10)

    def test_gmv_is_long_the_low_vol_direction(self) -> None:
        cov = np.diag([0.01, 0.04, 0.09]).astype(np.float64)
        w = global_min_variance_weights(cov)
        self.assertGreater(w[0], w[1])
        self.assertGreater(w[1], w[2])


class TestMeanShiftMath(unittest.TestCase):
    def test_mean_shift_std_matches_monte_carlo(self) -> None:
        daily_vol = 0.015
        window = TRADING_DAYS
        shift = TWO_WEEKS
        theory = mean_shift_std(daily_vol, window, shift)
        rng = np.random.default_rng(0)
        n_paths = 4000
        n_days = window + shift
        paths = rng.normal(0.0, daily_vol, size=(n_paths, n_days))
        mu_a = paths[:, :window].mean(axis=1)
        mu_b = paths[:, shift : shift + window].mean(axis=1)
        empirical = float(np.std(mu_b - mu_a))
        self.assertAlmostEqual(empirical, theory, delta=0.15 * theory)

    def test_annualized_shift_is_economically_large(self) -> None:
        # ~24% annual vol, one-year window, two-week slide → several percent of
        # annualized premium, comparable to the spread you are trying to estimate.
        daily_vol = 0.015
        annualized = mean_shift_std(daily_vol, TRADING_DAYS, TWO_WEEKS) * TRADING_DAYS
        self.assertGreater(annualized, 0.04)
        self.assertLess(annualized, 0.12)


class TestWindowSensitivity(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.universe = simulate_equity_universe(n_days=750, n_assets=12, seed=0)
        cls.pair = allocate_window_pair(cls.universe.returns, start=200)

    def test_covariance_is_ill_conditioned(self) -> None:
        self.assertGreater(cov_condition_number(self.pair.cov_a), 50)

    def test_sample_means_move_when_window_slides(self) -> None:
        d_mu_ann = (self.pair.mu_b - self.pair.mu_a) * TRADING_DAYS
        self.assertGreater(float(np.max(np.abs(d_mu_ann))), 0.02)

    def test_unconstrained_tangency_turns_over_violently(self) -> None:
        w_a, w_b = self.pair.weights["tangency"]
        self.assertGreater(l1_turnover(w_a, w_b), 0.8)
        # Error-maximization: the book is leveraged, not a 1/N tilt.
        self.assertGreater(float(np.max(np.abs(w_a))), 0.8)

    def test_shrinkage_is_much_more_stable(self) -> None:
        raw = l1_turnover(*self.pair.weights["tangency"])
        shrunk = l1_turnover(*self.pair.weights["shrink"])
        equal = l1_turnover(*self.pair.weights["equal"])
        self.assertEqual(equal, 0.0)
        self.assertLess(shrunk, 0.4 * raw)
        self.assertLess(shrunk, 0.35)

    def test_ridge_and_gmv_are_calmer_than_raw_tangency(self) -> None:
        raw = l1_turnover(*self.pair.weights["tangency"])
        self.assertLess(l1_turnover(*self.pair.weights["ridge"]), raw)
        self.assertLess(l1_turnover(*self.pair.weights["gmv"]), raw)

    def test_rolling_tangency_keeps_relevering(self) -> None:
        path = rolling_weights(self.universe.returns, method="tangency", step=TWO_WEEKS)
        turnovers = [l1_turnover(path[i], path[i + 1]) for i in range(len(path) - 1)]
        self.assertGreater(float(np.median(turnovers)), 0.5)
        shrink_path = rolling_weights(self.universe.returns, method="shrink", step=TWO_WEEKS)
        shrink_turnovers = [l1_turnover(shrink_path[i], shrink_path[i + 1]) for i in range(len(shrink_path) - 1)]
        self.assertLess(float(np.median(shrink_turnovers)), 0.5 * float(np.median(turnovers)))

    def test_factor_book_barely_moves(self) -> None:
        raw = l1_turnover(*self.pair.weights["tangency"])
        structured = l1_turnover(*self.pair.weights["factor"])
        self.assertLess(structured, 0.15)
        self.assertLess(structured, 0.1 * raw)
        w_a, w_b = self.pair.weights["factor"]
        # CAPM book is a mild long-only tilt along beta, not a leveraged long/short.
        self.assertTrue(np.all(w_a > 0))
        self.assertTrue(np.all(w_b > 0))
        self.assertLess(float(np.max(w_a)), 0.25)


class TestFactorReconstruction(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.universe = simulate_equity_universe(n_days=750, n_assets=12, seed=0)
        cls.window = cls.universe.returns[200:452]
        cls.fit = reconstruct_factor_moments(cls.window)
        cls.sample_mu, cls.sample_cov = sample_moments(cls.window)
        cls.true_mu, cls.true_cov = cls.universe.true_moments()
        cls.capm_mu, _ = cls.universe.capm_moments()

    def test_dgp_covariance_is_rank_one_plus_isotropic_noise(self) -> None:
        cov = structured_covariance(self.universe.betas, self.universe.factor_vol, self.universe.idio_vol)
        np.testing.assert_allclose(cov, self.universe.structured_covariance())
        spike = cov - np.diag(np.square(self.universe.idio_vol))
        singular = np.linalg.svd(spike, compute_uv=False)
        self.assertLess(singular[1] / singular[0], 1e-10)

    def test_woodbury_matches_dense_inverse(self) -> None:
        prec = factor_precision(self.fit.beta, self.fit.factor_var, self.fit.idio_var)
        np.testing.assert_allclose(prec, np.linalg.inv(self.fit.cov), rtol=1e-10, atol=1e-12)

    def test_reconstructed_covariance_is_rank_one_plus_isotropic_noise(self) -> None:
        spike = self.fit.cov - np.diag(self.fit.idio_var)
        singular = np.linalg.svd(spike, compute_uv=False)
        self.assertLess(singular[1] / singular[0], 1e-10)

    def test_reconstruction_beats_sample_on_true_covariance(self) -> None:
        sample_err = frobenius_rel(self.sample_cov, self.true_cov)
        factor_err = frobenius_rel(self.fit.cov, self.true_cov)
        self.assertLess(factor_err, sample_err)
        self.assertLess(factor_err, 0.15)

    def test_reconstruction_beats_sample_on_the_precision(self) -> None:
        true_prec = np.linalg.inv(self.true_cov)
        sample_err = frobenius_rel(np.linalg.inv(self.sample_cov), true_prec)
        factor_err = frobenius_rel(self.fit.precision(), true_prec)
        self.assertLess(factor_err, 0.6 * sample_err)

    def test_factor_weights_track_the_oracle_capm_book(self) -> None:
        oracle = tangency_weights(*self.universe.capm_moments())
        sample_w = tangency_weights(self.sample_mu, self.sample_cov)
        factor_w = factor_tangency_weights(self.window)
        self.assertLess(float(np.linalg.norm(factor_w - oracle)), float(np.linalg.norm(sample_w - oracle)))

    def test_capm_means_track_true_beta_not_sample_noise(self) -> None:
        sample_err = float(np.linalg.norm(self.sample_mu - self.capm_mu))
        factor_err = float(np.linalg.norm(self.fit.mu - self.capm_mu))
        self.assertLess(factor_err, 0.5 * sample_err)
        # Reconstructed premia are a monotone tilt along estimated beta.
        self.assertGreater(float(np.corrcoef(self.fit.mu, self.fit.beta)[0, 1]), 0.999)

    def test_two_week_shift_barely_moves_reconstructed_means(self) -> None:
        pair = allocate_window_pair(self.universe.returns, start=200)
        d_sample = np.max(np.abs(pair.mu_b - pair.mu_a)) * TRADING_DAYS
        d_factor = np.max(np.abs(pair.factor_b.mu - pair.factor_a.mu)) * TRADING_DAYS
        self.assertGreater(d_sample, 0.05)
        self.assertLess(d_factor, 0.4 * d_sample)
        self.assertLess(d_factor, 0.03)

    def test_rolling_factor_is_stable(self) -> None:
        raw = rolling_weights(self.universe.returns, method="tangency", step=TWO_WEEKS)
        structured = rolling_weights(self.universe.returns, method="factor", step=TWO_WEEKS)
        raw_to = [l1_turnover(raw[i], raw[i + 1]) for i in range(len(raw) - 1)]
        fac_to = [l1_turnover(structured[i], structured[i + 1]) for i in range(len(structured) - 1)]
        self.assertLess(float(np.median(fac_to)), 0.15)
        self.assertLess(float(np.median(fac_to)), 0.15 * float(np.median(raw_to)))


if __name__ == "__main__":
    unittest.main()
