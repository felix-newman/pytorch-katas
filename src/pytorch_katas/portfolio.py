"""Classical mean-variance allocation and why a two-week window shift wrecks it.

Markowitz / tangency weights are

    w*  ∝  Σ^{-1} μ

Both μ and Σ are estimated from a finite window of returns. The mapping
(μ, Σ) → w* is badly conditioned: sample means move by economically large
amounts when the window slides a few days, and Σ^{-1} amplifies whatever
noise sits in the low-eigenvalue (near-arbitrage) directions.

This module builds a one-factor equity universe, estimates rolling
Markowitz portfolios, and compares them to shrinkage / equal-weight rules
that do not treat every wiggle in μ̂ as signal.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from sklearn.covariance import LedoitWolf

Array = NDArray[np.float64]

TRADING_DAYS = 252
TWO_WEEKS = 10


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EquityUniverse:
    """Daily excess returns from a one-factor model, plus the truths used to generate them."""

    returns: Array  # (T, N)
    names: tuple[str, ...]
    betas: Array
    idio_vol: Array
    annual_premia: Array
    factor_vol: float

    @property
    def n_days(self) -> int:
        return int(self.returns.shape[0])

    @property
    def n_assets(self) -> int:
        return int(self.returns.shape[1])


def simulate_equity_universe(
    n_days: int = 750,
    n_assets: int = 12,
    factor_vol: float = 0.012,
    idio_vol: float = 0.008,
    annual_market_premium: float = 0.08,
    seed: int = 0,
) -> EquityUniverse:
    """Simulate a crowded equity book: high β-correlation, similar expected returns.

    Assets differ only by a bit of beta and a few percent of idiosyncratic
    premium — exactly the setting where sample means cannot tell them apart,
    but an unconstrained optimizer will happily take huge long/short bets.
    """
    rng = np.random.default_rng(seed)
    betas = np.linspace(0.75, 1.25, n_assets)
    idio = np.full(n_assets, idio_vol)
    # True premia hug the CAPM line plus a tiny residual alpha (±1.5%/year).
    residual_alpha = np.linspace(-0.015, 0.015, n_assets)
    annual_premia = annual_market_premium * betas + residual_alpha
    daily_mu = annual_premia / TRADING_DAYS

    factor = rng.normal(0.0, factor_vol, size=n_days)
    noise = rng.normal(0.0, 1.0, size=(n_days, n_assets)) * idio
    returns = daily_mu + np.outer(factor, betas) + noise
    names = tuple(f"A{i + 1:02d}" for i in range(n_assets))
    return EquityUniverse(returns, names, betas, idio, annual_premia, factor_vol)


# ---------------------------------------------------------------------------
# Moments and allocations
# ---------------------------------------------------------------------------


def sample_moments(returns: Array) -> tuple[Array, Array]:
    """Sample mean and covariance of a (T, N) return window."""
    if returns.ndim != 2 or returns.shape[0] < 3:
        raise ValueError("returns must be a (T, N) array with T >= 3")
    mu = returns.mean(axis=0)
    # np.cov uses Bessel correction; that is what a practitioner estimates.
    cov = np.cov(returns, rowvar=False)
    if cov.ndim == 0:
        cov = np.array([[float(cov)]])
    return mu, cov


def _normalize_budget(raw: Array) -> Array:
    total = raw.sum()
    if abs(total) < 1e-18:
        raise ValueError("weights are numerically degenerate (sum ~ 0)")
    return raw / total


def tangency_weights(mu: Array, cov: Array) -> Array:
    """Unconstrained maximum-Sharpe (tangency) portfolio, fully invested.

    w = Σ^{-1} μ / 1' Σ^{-1} μ
    """
    raw = np.linalg.solve(cov, mu)
    return _normalize_budget(raw)


def global_min_variance_weights(cov: Array) -> Array:
    """Global minimum-variance portfolio: w = Σ^{-1} 1 / 1' Σ^{-1} 1."""
    ones = np.ones(cov.shape[0], dtype=np.float64)
    raw = np.linalg.solve(cov, ones)
    return _normalize_budget(raw)


def equal_weights(n_assets: int) -> Array:
    return np.full(n_assets, 1.0 / n_assets, dtype=np.float64)


def ridge_tangency_weights(mu: Array, cov: Array, ridge: float) -> Array:
    """Tangency portfolio on a diagonally regularized covariance."""
    if ridge < 0:
        raise ValueError("ridge must be non-negative")
    damped = cov + ridge * np.eye(cov.shape[0])
    return tangency_weights(mu, damped)


def shrink_moments(
    returns: Array,
    mu_shrink: float = 0.95,
    extra_ridge: float = 0.1,
) -> tuple[Array, Array]:
    """Ledoit–Wolf covariance + shrink the mean toward the grand mean.

    Sample means are the dominant estimation-error source (Merton 1980).
    Pulling every asset toward the cross-sectional average is a cheap
    James–Stein step. Ledoit–Wolf alone is not enough on a crowded
    equity book — a small extra ridge (fraction of average variance)
    kills the leftover 1/λ_min bets.
    """
    if not 0.0 <= mu_shrink <= 1.0:
        raise ValueError("mu_shrink must be in [0, 1]")
    if extra_ridge < 0:
        raise ValueError("extra_ridge must be non-negative")
    mu, _ = sample_moments(returns)
    grand = np.full_like(mu, mu.mean())
    mu_s = (1.0 - mu_shrink) * mu + mu_shrink * grand
    cov_s = LedoitWolf().fit(returns).covariance_.astype(np.float64)
    ridge = extra_ridge * float(np.mean(np.diag(cov_s)))
    cov_s = cov_s + ridge * np.eye(cov_s.shape[0])
    return mu_s, cov_s


def shrink_tangency_weights(
    returns: Array,
    mu_shrink: float = 0.95,
    extra_ridge: float = 0.1,
) -> Array:
    mu, cov = shrink_moments(returns, mu_shrink=mu_shrink, extra_ridge=extra_ridge)
    return tangency_weights(mu, cov)


def long_only_mv_weights(mu: Array, cov: Array, risk_aversion: float = 5.0) -> Array:
    """Long-only mean-variance: min ½ w'Σw − (1/λ) μ'w  s.t. w ≥ 0, 1'w = 1."""
    from scipy.optimize import minimize

    n = mu.shape[0]
    if risk_aversion <= 0:
        raise ValueError("risk_aversion must be positive")

    def objective(w: Array) -> float:
        return 0.5 * float(w @ cov @ w) - float(mu @ w) / risk_aversion

    cons = {"type": "eq", "fun": lambda w: w.sum() - 1.0}
    bounds = [(0.0, 1.0)] * n
    w0 = equal_weights(n)
    result = minimize(
        objective,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
        options={"ftol": 1e-12, "maxiter": 500},
    )
    if not result.success:
        # Fall back to a projected closed form rather than crash a demo.
        raw = np.maximum(np.linalg.solve(cov, mu), 0.0)
        return _normalize_budget(raw)
    return result.x.astype(np.float64)


# ---------------------------------------------------------------------------
# Sensitivity diagnostics
# ---------------------------------------------------------------------------


def mean_shift_std(daily_vol: float, window: int, shift: int = TWO_WEEKS) -> float:
    """Std of the change in a sample mean when a window slides by `shift` days.

    For iid returns the two endpoint blocks of length `shift` are independent,
    so

        μ̂_B − μ̂_A  =  (1/T) (Σ_{in} r − Σ_{out} r)
        std         =  σ √(2 · shift) / T

    Annualize by multiplying by TRADING_DAYS.
    """
    if daily_vol < 0 or window <= 0 or shift <= 0:
        raise ValueError("daily_vol, window and shift must be positive")
    return daily_vol * np.sqrt(2.0 * shift) / window


def cov_condition_number(cov: Array) -> float:
    eig = np.linalg.eigvalsh(cov)
    eig = eig[eig > 0]
    return float(eig.max() / eig.min())


def l1_turnover(w_a: Array, w_b: Array) -> float:
    """Gross turnover to go from w_a to w_b: ½ ‖w_b − w_a‖₁."""
    return 0.5 * float(np.abs(w_b - w_a).sum())


@dataclass(frozen=True)
class WindowPair:
    """Two overlapping estimation windows and the allocations they produce."""

    start_a: int
    start_b: int
    window: int
    mu_a: Array
    mu_b: Array
    cov_a: Array
    cov_b: Array
    weights: dict[str, tuple[Array, Array]]

    @property
    def shift(self) -> int:
        return self.start_b - self.start_a


def allocate_window_pair(
    returns: Array,
    window: int = TRADING_DAYS,
    shift: int = TWO_WEEKS,
    start: int = 0,
    ridge: float = 5e-4,
    mu_shrink: float = 0.95,
) -> WindowPair:
    """Estimate allocations on [start, start+W) and on the window shifted by `shift` days."""
    end_b = start + shift + window
    if end_b > returns.shape[0]:
        raise ValueError("not enough returns for this window/shift/start")
    a = returns[start : start + window]
    b = returns[start + shift : start + shift + window]
    mu_a, cov_a = sample_moments(a)
    mu_b, cov_b = sample_moments(b)
    weights = {
        "tangency": (tangency_weights(mu_a, cov_a), tangency_weights(mu_b, cov_b)),
        "gmv": (global_min_variance_weights(cov_a), global_min_variance_weights(cov_b)),
        "ridge": (ridge_tangency_weights(mu_a, cov_a, ridge), ridge_tangency_weights(mu_b, cov_b, ridge)),
        "shrink": (shrink_tangency_weights(a, mu_shrink), shrink_tangency_weights(b, mu_shrink)),
        "long_only": (long_only_mv_weights(mu_a, cov_a), long_only_mv_weights(mu_b, cov_b)),
        "equal": (equal_weights(a.shape[1]), equal_weights(b.shape[1])),
    }
    return WindowPair(start, start + shift, window, mu_a, mu_b, cov_a, cov_b, weights)


def rolling_weights(
    returns: Array,
    window: int = TRADING_DAYS,
    step: int = TWO_WEEKS,
    method: str = "tangency",
    ridge: float = 5e-4,
    mu_shrink: float = 0.95,
) -> Array:
    """Rolling allocations. Returns an array of shape (n_windows, N)."""
    t, n = returns.shape
    starts = range(0, t - window + 1, step)
    rows: list[Array] = []
    for s in starts:
        chunk = returns[s : s + window]
        mu, cov = sample_moments(chunk)
        if method == "tangency":
            w = tangency_weights(mu, cov)
        elif method == "gmv":
            w = global_min_variance_weights(cov)
        elif method == "ridge":
            w = ridge_tangency_weights(mu, cov, ridge)
        elif method == "shrink":
            w = shrink_tangency_weights(chunk, mu_shrink)
        elif method == "long_only":
            w = long_only_mv_weights(mu, cov)
        elif method == "equal":
            w = equal_weights(n)
        else:
            raise ValueError(f"unknown method {method!r}")
        rows.append(w)
    return np.stack(rows, axis=0)


# ---------------------------------------------------------------------------
# Reporting / figures
# ---------------------------------------------------------------------------


def summarize_pair(pair: WindowPair, names: tuple[str, ...] | None = None) -> str:
    """Human-readable report of how much a two-week slide moved the book."""
    names = names or tuple(f"A{i + 1:02d}" for i in range(pair.mu_a.shape[0]))
    d_mu_ann = (pair.mu_b - pair.mu_a) * TRADING_DAYS
    lines = [
        f"Window {pair.window} days, shifted by {pair.shift} trading days (~{pair.shift / 5:.0f} weeks).",
        f"Condition number of Σ̂ (window A): {cov_condition_number(pair.cov_a):.1f}",
        f"Largest |Δμ| (annualized): {np.max(np.abs(d_mu_ann)):.2%}",
        f"Cross-sectional std of Δμ (annualized): {np.std(d_mu_ann):.2%}",
        "",
        f"{'rule':<12} {'turnover':>10} {'max |w| A':>10} {'max |w| B':>10} {'max |Δw|':>10}",
    ]
    for name, (w_a, w_b) in pair.weights.items():
        lines.append(
            f"{name:<12} {l1_turnover(w_a, w_b):>9.1%} {np.max(np.abs(w_a)):>10.1%} "
            f"{np.max(np.abs(w_b)):>10.1%} {np.max(np.abs(w_b - w_a)):>10.1%}"
        )
    lines += ["", "Annualized sample means, window A vs B:"]
    for i, name in enumerate(names):
        lines.append(
            f"  {name}: {pair.mu_a[i] * TRADING_DAYS:>7.2%}  →  {pair.mu_b[i] * TRADING_DAYS:>7.2%}  "
            f"(Δ {d_mu_ann[i]:>+7.2%})"
        )
    return "\n".join(lines)


def save_sensitivity_figures(
    universe: EquityUniverse,
    out_dir: str | Path,
    window: int = TRADING_DAYS,
    shift: int = TWO_WEEKS,
    start: int = 200,
) -> dict[str, Path]:
    """Write the three plots that make the instability visible."""
    import matplotlib.pyplot as plt

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pair = allocate_window_pair(universe.returns, window=window, shift=shift, start=start)
    names = list(universe.names)
    x = np.arange(len(names))
    paths: dict[str, Path] = {}

    # 1. Side-by-side tangency vs shrinkage weights
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), sharey=False)
    for ax, rule, title in (
        (axes[0], "tangency", "Unconstrained tangency"),
        (axes[1], "shrink", "Ledoit–Wolf + shrunk means"),
    ):
        w_a, w_b = pair.weights[rule]
        width = 0.38
        ax.bar(x - width / 2, w_a, width, label=f"days [{pair.start_a}, {pair.start_a + window})", color="#1f4e79")
        ax.bar(x + width / 2, w_b, width, label=f"days [{pair.start_b}, {pair.start_b + window})", color="#c45911")
        ax.axhline(0.0, color="black", linewidth=0.6)
        ax.set_xticks(x, names, rotation=45, ha="right")
        ax.set_ylabel("weight")
        ax.set_title(title)
        ax.legend(frameon=False, fontsize=8)
        ax.set_ylim(min(-1.5, w_a.min(), w_b.min()) - 0.1, max(1.5, w_a.max(), w_b.max()) + 0.1)
    fig.suptitle(
        f"Same book, window slid by {shift} trading days  ·  "
        f"turnover {l1_turnover(*pair.weights['tangency']):.0%} vs "
        f"{l1_turnover(*pair.weights['shrink']):.0%}",
        fontsize=12,
    )
    fig.tight_layout()
    paths["weights"] = out / "window_shift_weights.png"
    fig.savefig(paths["weights"], dpi=140)
    plt.close(fig)

    # 2. Rolling heatmaps
    raw = rolling_weights(universe.returns, window=window, step=shift, method="tangency")
    shrunk = rolling_weights(universe.returns, window=window, step=shift, method="shrink")
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 6.6), sharex=True)
    vmax = max(np.percentile(np.abs(raw), 98), 1.0)
    for ax, data, title in (
        (axes[0], raw, "Rolling unconstrained tangency — weights jump every two weeks"),
        (axes[1], shrunk, "Rolling shrink / Ledoit–Wolf — same data, stable book"),
    ):
        im = ax.imshow(data.T, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest")
        ax.set_yticks(np.arange(len(names)), names)
        ax.set_ylabel("asset")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02, label="weight")
    axes[1].set_xlabel("rolling window index (step = 10 trading days)")
    fig.tight_layout()
    paths["rolling"] = out / "rolling_allocation_heatmap.png"
    fig.savefig(paths["rolling"], dpi=140)
    plt.close(fig)

    # 3. Mechanism: noisy Δμ vs. Σ^{-1} amplification
    eigvals = np.sort(np.linalg.eigvalsh(pair.cov_a))[::-1]
    d_mu_ann = (pair.mu_b - pair.mu_a) * TRADING_DAYS
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    axes[0].bar(x, d_mu_ann, color="#7b2d8e")
    axes[0].axhline(0.0, color="black", linewidth=0.6)
    axes[0].set_xticks(x, names, rotation=45, ha="right")
    axes[0].set_ylabel("Δμ̂ (annualized)")
    axes[0].set_title("A two-week slide already moves estimated premia by several %")
    axes[1].semilogy(np.arange(1, len(eigvals) + 1), eigvals, marker="o", color="#1f4e79")
    axes[1].set_xlabel("eigenvalue rank")
    axes[1].set_ylabel("λ(Σ̂)")
    axes[1].set_title(f"Σ̂ is ill-conditioned  (κ = {cov_condition_number(pair.cov_a):.0f})")
    fig.tight_layout()
    paths["mechanism"] = out / "mean_shift_and_eigenvalues.png"
    fig.savefig(paths["mechanism"], dpi=140)
    plt.close(fig)

    return paths
