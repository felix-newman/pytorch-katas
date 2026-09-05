"""Classical mean-variance allocation and why a two-week window shift wrecks it.

Markowitz / tangency weights are

    w*  ∝  Σ^{-1} μ

The sample matrices (μ̂, Σ̂) are too noisy to invert. A first-principles
reconstruction assumes a one-factor world

    r_t  =  μ + β f_t + ε_t
    Σ    =  σ_f² ββᵀ + D
    μ    =  λ β

and rebuilds both matrices from (β, σ_f, D, λ) instead of filling every
entry of Σ̂ from the window. That is Sharpe's single-index model: N betas,
N residual variances, one factor variance, one price of risk — not
N(N+1)/2 covariances and N sample means.

This module builds a one-factor equity universe, contrasts the sample
matrices with the reconstructed ones, and shows that only the sample
book jumps when the window slides two weeks.
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

    def structured_covariance(self) -> Array:
        """Exact DGP covariance: Σ = σ_f² ββᵀ + diag(σ_ε²)."""
        return structured_covariance(self.betas, self.factor_vol, self.idio_vol)

    def true_moments(self) -> tuple[Array, Array]:
        """True mean (including residual alpha) and structured covariance."""
        return self.annual_premia / TRADING_DAYS, self.structured_covariance()

    def capm_moments(self, annual_market_premium: float = 0.08) -> tuple[Array, Array]:
        """Equilibrium moments: μ = λβ, same structured covariance. Residual alpha is dropped."""
        mu = self.betas * (annual_market_premium / TRADING_DAYS)
        return mu, self.structured_covariance()


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


def structured_covariance(betas: Array, factor_vol: float, idio_vol: Array | float) -> Array:
    """Σ = σ_f² ββᵀ + D, D = diag(σ_ε²)."""
    betas = np.asarray(betas, dtype=np.float64)
    idio = np.broadcast_to(np.asarray(idio_vol, dtype=np.float64), betas.shape)
    return (factor_vol**2) * np.outer(betas, betas) + np.diag(np.square(idio))


def frobenius_rel(estimate: Array, truth: Array) -> float:
    """‖E − T‖_F / ‖T‖_F."""
    denom = float(np.linalg.norm(truth, ord="fro"))
    if denom < 1e-18:
        raise ValueError("truth matrix is numerically zero")
    return float(np.linalg.norm(estimate - truth, ord="fro") / denom)


@dataclass(frozen=True)
class FactorMoments:
    """One-factor reconstruction of the Markowitz ingredients.

    Σ = σ_f² ββᵀ + D,   μ = λ β
    """

    mu: Array
    cov: Array
    beta: Array
    factor_var: float
    idio_var: Array
    price_of_risk: float

    def precision(self) -> Array:
        """Woodbury inverse: Σ^{-1} = D^{-1} − D^{-1}ββᵀD^{-1} / (1/σ_f² + βᵀD^{-1}β)."""
        return factor_precision(self.beta, self.factor_var, self.idio_var)


def factor_precision(beta: Array, factor_var: float, idio_var: Array) -> Array:
    """Closed-form Σ^{-1} for a rank-one-plus-diagonal covariance."""
    if factor_var <= 0:
        raise ValueError("factor_var must be positive")
    d_inv = 1.0 / np.asarray(idio_var, dtype=np.float64)
    v = d_inv * beta
    denom = 1.0 / factor_var + float(beta @ v)
    return np.diag(d_inv) - np.outer(v, v) / denom


def reconstruct_factor_moments(returns: Array) -> FactorMoments:
    """Rebuild μ and Σ from the spiked one-factor model.

    The first-principles matrices are

        Σ  =  (λ₁ − σ̄²) q₁q₁ᵀ  +  σ̄² I
        μ  =  λ β,   β = √(λ₁ − σ̄²) q₁,   λ = (βᵀ μ̂) / (βᵀ β)

    λ₁, q₁ are the market eigenpair of the sample covariance; σ̄² is the
    average of the leftover eigenvalues (the idiosyncratic bulk). Residual
    alphas and residual covariances are treated as estimation error, not
    as an investment opportunity.

    Identification: the latent factor is scaled to unit variance, so
    Σ = ββᵀ + σ̄² I and Woodbury applies with factor_var = 1.
    """
    mu_hat, cov_hat = sample_moments(returns)
    n = cov_hat.shape[0]
    evals, evecs = np.linalg.eigh(cov_hat)
    lam1 = float(evals[-1])
    q1 = evecs[:, -1]
    if lam1 <= 0:
        raise ValueError("leading covariance eigenvalue is not positive")
    if float(q1.sum()) < 0:
        q1 = -q1
    bulk = float(np.mean(evals[:-1])) if n > 1 else 0.0
    spike = max(lam1 - bulk, 0.0)
    beta = q1 * np.sqrt(spike) if spike > 0 else q1
    idio_var = np.full(n, max(bulk, 1e-12), dtype=np.float64)
    cov = np.outer(beta, beta) + np.diag(idio_var)
    denom = float(beta @ beta)
    price = float(beta @ mu_hat) / denom if denom > 0 else 0.0
    mu = price * beta
    return FactorMoments(mu, cov, beta, 1.0, idio_var, price)


def factor_tangency_weights(returns: Array) -> Array:
    """Tangency portfolio on the reconstructed (μ, Σ), inverted with Woodbury."""
    fit = reconstruct_factor_moments(returns)
    raw = fit.precision() @ fit.mu
    return _normalize_budget(raw)


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
    factor_a: FactorMoments
    factor_b: FactorMoments

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
    factor_a = reconstruct_factor_moments(a)
    factor_b = reconstruct_factor_moments(b)
    weights = {
        "tangency": (tangency_weights(mu_a, cov_a), tangency_weights(mu_b, cov_b)),
        "gmv": (global_min_variance_weights(cov_a), global_min_variance_weights(cov_b)),
        "ridge": (ridge_tangency_weights(mu_a, cov_a, ridge), ridge_tangency_weights(mu_b, cov_b, ridge)),
        "shrink": (shrink_tangency_weights(a, mu_shrink), shrink_tangency_weights(b, mu_shrink)),
        "factor": (factor_tangency_weights(a), factor_tangency_weights(b)),
        "long_only": (long_only_mv_weights(mu_a, cov_a), long_only_mv_weights(mu_b, cov_b)),
        "equal": (equal_weights(a.shape[1]), equal_weights(b.shape[1])),
    }
    return WindowPair(start, start + shift, window, mu_a, mu_b, cov_a, cov_b, weights, factor_a, factor_b)


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
        elif method == "factor":
            w = factor_tangency_weights(chunk)
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


def summarize_pair(
    pair: WindowPair,
    names: tuple[str, ...] | None = None,
    universe: EquityUniverse | None = None,
) -> str:
    """Human-readable report of how much a two-week slide moved the book."""
    names = names or tuple(f"A{i + 1:02d}" for i in range(pair.mu_a.shape[0]))
    d_mu_ann = (pair.mu_b - pair.mu_a) * TRADING_DAYS
    d_mu_fac = (pair.factor_b.mu - pair.factor_a.mu) * TRADING_DAYS
    lines = [
        f"Window {pair.window} days, shifted by {pair.shift} trading days (~{pair.shift / 5:.0f} weeks).",
        f"Condition number of sample Σ̂:        {cov_condition_number(pair.cov_a):.1f}",
        f"Condition number of reconstructed Σ: {cov_condition_number(pair.factor_a.cov):.1f}",
        f"Largest |Δμ| sample / factor (ann.): "
        f"{np.max(np.abs(d_mu_ann)):.2%} / {np.max(np.abs(d_mu_fac)):.2%}",
        f"Cross-sectional std of Δμ sample / factor (ann.): "
        f"{np.std(d_mu_ann):.2%} / {np.std(d_mu_fac):.2%}",
    ]
    if universe is not None:
        truth_cov = universe.structured_covariance()
        lines += [
            f"Relative Frobenius error vs true Σ, sample:        {frobenius_rel(pair.cov_a, truth_cov):.2%}",
            f"Relative Frobenius error vs true Σ, reconstructed: {frobenius_rel(pair.factor_a.cov, truth_cov):.2%}",
        ]
    lines += [
        "",
        f"{'rule':<12} {'turnover':>10} {'max |w| A':>10} {'max |w| B':>10} {'max |Δw|':>10}",
    ]
    for name, (w_a, w_b) in pair.weights.items():
        lines.append(
            f"{name:<12} {l1_turnover(w_a, w_b):>9.1%} {np.max(np.abs(w_a)):>10.1%} "
            f"{np.max(np.abs(w_b)):>10.1%} {np.max(np.abs(w_b - w_a)):>10.1%}"
        )
    lines += ["", "Annualized means, sample vs reconstructed (window A → B):"]
    for i, name in enumerate(names):
        lines.append(
            f"  {name}: sample {pair.mu_a[i] * TRADING_DAYS:>7.2%} → {pair.mu_b[i] * TRADING_DAYS:>7.2%} "
            f"(Δ {d_mu_ann[i]:>+7.2%})   "
            f"factor {pair.factor_a.mu[i] * TRADING_DAYS:>7.2%} → {pair.factor_b.mu[i] * TRADING_DAYS:>7.2%} "
            f"(Δ {d_mu_fac[i]:>+7.2%})"
        )
    return "\n".join(lines)


def save_sensitivity_figures(
    universe: EquityUniverse,
    out_dir: str | Path,
    window: int = TRADING_DAYS,
    shift: int = TWO_WEEKS,
    start: int = 200,
) -> dict[str, Path]:
    """Write the plots that contrast sample matrices with the factor reconstruction."""
    import matplotlib.pyplot as plt

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pair = allocate_window_pair(universe.returns, window=window, shift=shift, start=start)
    names = list(universe.names)
    x = np.arange(len(names))
    paths: dict[str, Path] = {}
    truth_cov = universe.structured_covariance()
    truth_mu, _ = universe.capm_moments()

    # 1. Covariance: sample vs reconstructed vs truth
    matrices = (
        (pair.cov_a, "Sample Σ̂"),
        (pair.factor_a.cov, "Reconstructed Σ = σ_f² ββᵀ + D"),
        (truth_cov, "True Σ from the DGP"),
    )
    vmax = max(float(np.max(np.abs(m))) for m, _ in matrices)
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.2), constrained_layout=True)
    for ax, (mat, title) in zip(axes, matrices, strict=True):
        im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest")
        ax.set_xticks(x, names, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(x, names, fontsize=7)
        err = frobenius_rel(mat, truth_cov)
        ax.set_title(f"{title}\n‖· − Σ‖_F / ‖Σ‖_F = {err:.1%}", fontsize=10)
    fig.colorbar(im, ax=list(axes), fraction=0.02, pad=0.02)
    fig.suptitle("Off-diagonals of Σ̂ are residual noise. The factor rebuild keeps only ββᵀ + D.", fontsize=11)
    paths["covariance"] = out / "covariance_reconstruction.png"
    fig.savefig(paths["covariance"], dpi=140)
    plt.close(fig)

    # 2. Spectra + means
    sample_eigs = np.sort(np.linalg.eigvalsh(pair.cov_a))[::-1]
    factor_eigs = np.sort(np.linalg.eigvalsh(pair.factor_a.cov))[::-1]
    truth_eigs = np.sort(np.linalg.eigvalsh(truth_cov))[::-1]
    ranks = np.arange(1, len(sample_eigs) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    axes[0].semilogy(ranks, sample_eigs, "o-", color="#c45911", label="sample")
    axes[0].semilogy(ranks, factor_eigs, "s-", color="#1f4e79", label="reconstructed")
    axes[0].semilogy(ranks, truth_eigs, "k--", label="true DGP")
    axes[0].set_xlabel("eigenvalue rank")
    axes[0].set_ylabel("λ")
    axes[0].set_title("Keep the market spike, flatten the noisy bulk")
    axes[0].legend(frameon=False, fontsize=8)
    width = 0.28
    axes[1].bar(x - width, pair.mu_a * TRADING_DAYS, width, label="sample μ̂", color="#c45911")
    axes[1].bar(x, pair.factor_a.mu * TRADING_DAYS, width, label="μ = λβ", color="#1f4e79")
    axes[1].bar(x + width, truth_mu * TRADING_DAYS, width, label="true CAPM μ", color="#7f7f7f")
    axes[1].axhline(0.0, color="black", linewidth=0.6)
    axes[1].set_xticks(x, names, rotation=45, ha="right")
    axes[1].set_ylabel("annualized mean")
    axes[1].set_title("One price of risk. Residual alphas are discarded.")
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    paths["moments"] = out / "factor_moment_reconstruction.png"
    fig.savefig(paths["moments"], dpi=140)
    plt.close(fig)

    # 3. Weights: sample tangency vs factor tangency
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), sharey=False)
    for ax, rule, title in (
        (axes[0], "tangency", "Invert the sample matrices"),
        (axes[1], "factor", "Invert the reconstructed matrices"),
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
        lo = min(-0.05, float(w_a.min()), float(w_b.min())) - 0.05
        hi = max(0.4, float(w_a.max()), float(w_b.max())) + 0.05
        if rule == "tangency":
            lo = min(-1.5, float(w_a.min()), float(w_b.min())) - 0.1
            hi = max(1.5, float(w_a.max()), float(w_b.max())) + 0.1
        ax.set_ylim(lo, hi)
    fig.suptitle(
        f"Same book, window slid by {shift} trading days  ·  "
        f"turnover {l1_turnover(*pair.weights['tangency']):.0%} vs "
        f"{l1_turnover(*pair.weights['factor']):.0%}",
        fontsize=12,
    )
    fig.tight_layout()
    paths["weights"] = out / "window_shift_weights.png"
    fig.savefig(paths["weights"], dpi=140)
    plt.close(fig)

    # 4. Rolling heatmaps: sample vs factor
    raw = rolling_weights(universe.returns, window=window, step=shift, method="tangency")
    structured = rolling_weights(universe.returns, window=window, step=shift, method="factor")
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 6.6), sharex=True)
    for ax, data, title in (
        (axes[0], raw, "Rolling sample tangency — inverts noise every two weeks"),
        (axes[1], structured, "Rolling factor reconstruction — same data, structural book"),
    ):
        vmax_w = max(float(np.percentile(np.abs(data), 98)), 0.25)
        im = ax.imshow(data.T, aspect="auto", cmap="RdBu_r", vmin=-vmax_w, vmax=vmax_w, interpolation="nearest")
        ax.set_yticks(np.arange(len(names)), names)
        ax.set_ylabel("asset")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02, label="weight")
    axes[1].set_xlabel("rolling window index (step = 10 trading days)")
    fig.tight_layout()
    paths["rolling"] = out / "rolling_allocation_heatmap.png"
    fig.savefig(paths["rolling"], dpi=140)
    plt.close(fig)

    paths.update(save_equation_plates(out))
    return paths


def save_equation_plates(out_dir: str | Path) -> dict[str, Path]:
    """Rasterize the identities. Cursor and GitHub do not run MathJax on $...$."""
    import matplotlib.pyplot as plt

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    plates = {
        "eq_model": (
            [
                r"$r_t = \mu + \beta f_t + \varepsilon_t$",
                r"$\Sigma = (\lambda_1 - \bar\sigma^2)\, q_1 q_1^\top + \bar\sigma^2 I$",
                r"$\mu = \lambda\beta$",
                r"$w^\star \propto \Sigma^{-1}\mu$",
            ],
            10.6,
            3.2,
        ),
        "eq_woodbury": (
            [r"$\Sigma^{-1} = \bar\sigma^{-2} I \;-\; \dfrac{\bar\sigma^{-4}\beta\beta^\top}{1 + \bar\sigma^{-2}\beta^\top\beta}$"],
            10.6,
            1.55,
        ),
        "eq_mean_shift": (
            [r"$\mathrm{std}(\hat\mu_B - \hat\mu_A) = \dfrac{\sigma\sqrt{2h}}{T}$"],
            10.6,
            1.55,
        ),
    }
    paths: dict[str, Path] = {}
    for name, (lines, width, height) in plates.items():
        fig, ax = plt.subplots(figsize=(width, height))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        ax.set_axis_off()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        n = len(lines)
        for i, line in enumerate(lines):
            y = 1.0 - (i + 0.55) / (n + 0.1)
            ax.text(0.5, y, line, fontsize=20 if n > 1 else 22, ha="center", va="center")
        path = out / f"{name}.png"
        fig.savefig(path, dpi=160, facecolor="white", bbox_inches="tight", pad_inches=0.28)
        plt.close(fig)
        paths[name] = path
    return paths
