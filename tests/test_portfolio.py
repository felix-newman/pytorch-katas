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
    global_min_variance_weights,
    l1_turnover,
    mean_shift_std,
    rolling_weights,
    sample_moments,
    shrink_tangency_weights,
    simulate_equity_universe,
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


if __name__ == "__main__":
    unittest.main()
