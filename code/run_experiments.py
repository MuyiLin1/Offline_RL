#!/usr/bin/env python3
"""CLI: default runs Exp 1 + Exp 2 + learning curves; use --exp1-only / --exp2-only / --learning-curves-only to restrict."""

from __future__ import annotations

import argparse
import json
import os
import sys

# stderr is unbuffered on most platforms; stdout may wait until first newline
print("run_experiments: importing numpy/matplotlib…", file=sys.stderr, flush=True)
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from hybrid_fmucb import (
    OfflineRewardStats,
    build_rule_F,
    compute_c_man,
    hybrid_fmucb_pick,
    manipulation_contrast,
    offline_candidate_manipulation,
    pooled_mean,
    regression_confidence_radius,
    theorem1_offline_transfer_check,
    true_best_manipulation,
)

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
for _style in ("seaborn-v0_8-whitegrid", "seaborn-whitegrid", "ggplot"):
    try:
        plt.style.use(_style)
        break
    except OSError:
        continue

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 220,
    "font.size": 14, "axes.titlesize": 16, "axes.labelsize": 15,
    "legend.fontsize": 13, "xtick.labelsize": 13, "ytick.labelsize": 13,
    "axes.grid": True, "grid.alpha": 0.35, "grid.linestyle": "--",
    "axes.facecolor": "#fafafa", "figure.facecolor": "white",
    "axes.edgecolor": "#333333", "axes.linewidth": 1.0,
})

COLORS = {
    "baseline": "#C73E1D", "hybrid": "#2E86AB", "gated": "#9B59B6",
    "good": "#3A7D44", "poor": "#A23B72", "neutral": "#7D8590",
    "tabular": "#6B4E71", "contextual": "#2A9D8F",
}


def _style_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

@dataclass
class StackelbergBandit:
    """Tabular rewards mu_l(a,b), mu_f(a,b) ~ Uniform(0,1)."""

    mu_leader: np.ndarray    # shape (n_a, n_b)
    mu_follower: np.ndarray  # shape (n_a, n_b)

    @classmethod
    def sample(cls, n_a: int, n_b: int, rng: np.random.Generator) -> "StackelbergBandit":
        return cls(
            rng.uniform(0.0, 1.0, (n_a, n_b)),
            rng.uniform(0.0, 1.0, (n_a, n_b)),
        )

    @classmethod
    def fixed_4x4(cls) -> "StackelbergBandit":
        """A hand-designed 4x4 game with clean manipulation structure.

        Properties (verified analytically):
          - Stackelberg equilibrium: (a=0, b=0) — leader gets 0.85, follower gets 0.40
          - Best manipulation: target a=1, F_fm(1)=b1 — follower gets 0.90
            F_fm = [3, 1, 3, 3] (worst responses at non-target actions)
          - Under manipulation, leader sees:
              a=0: 0.10, a=1: 0.65, a=2: 0.10, a=3: 0.05
            so leader uniquely prefers a=1.
          - Delta_3 = 0.55  (manipulation contrast)
          - Delta_5 = 0.20  (min worst-response identification gap)
          - Follower gain from manipulation: 0.90 vs 0.40 (Stackelberg)
        """
        mu_l = np.array([
            [0.85, 0.70, 0.30, 0.10],   # a=0: SE target, wr=b3(0.10)
            [0.50, 0.65, 0.30, 0.10],   # a=1: manipulation target, wr=b3(0.10)
            [0.45, 0.35, 0.55, 0.10],   # a=2: wr=b3(0.10)
            [0.40, 0.30, 0.50, 0.05],   # a=3: wr=b3(0.05)
        ])
        mu_f = np.array([
            [0.40, 0.30, 0.20, 0.10],   # BR(0)=b0, follower gets 0.40 (SE payoff)
            [0.25, 0.90, 0.40, 0.35],   # BR(1)=b1, follower gets 0.90 (manipulation payoff!)
            [0.50, 0.60, 0.30, 0.45],   # BR(2)=b1
            [0.35, 0.55, 0.65, 0.50],   # BR(3)=b2
        ])
        return cls(mu_l, mu_f)

    @property
    def n_a(self) -> int:
        return self.mu_leader.shape[0]

    @property
    def n_b(self) -> int:
        return self.mu_leader.shape[1]

    def follower_br(self, a: int) -> int:
        return int(np.argmax(self.mu_follower[a]))

    def stackelberg_leader_action(self) -> int:
        br = np.argmax(self.mu_follower, axis=1)
        vals = np.array([self.mu_leader[a, br[a]] for a in range(self.n_a)])
        return int(np.argmax(vals))

    def leader_reward_at_br(self, a: int) -> float:
        return float(self.mu_leader[a, self.follower_br(a)])


# ---------------------------------------------------------------------------
# Bernoulli sampling helper (Deviation 1)
# ---------------------------------------------------------------------------

def bernoulli_sample(mu: float, rng: np.random.Generator) -> float:
    """r ~ Ber(mu) as in paper Sec. 2.  Returns 0.0 or 1.0."""
    return float(rng.random() < np.clip(mu, 0.0, 1.0))


# ---------------------------------------------------------------------------
# EXP3
# ---------------------------------------------------------------------------

def exp3_sample(weights: np.ndarray, gamma: float, rng: np.random.Generator) -> int:
    n = len(weights)
    w = np.maximum(weights, 1e-18)
    p = (1.0 - gamma) * (w / w.sum()) + gamma / n
    return int(rng.choice(n, p=p))


def exp3_update(weights: np.ndarray, a: int, reward: float, gamma: float) -> None:
    n = len(weights)
    w = np.maximum(weights, 1e-18)
    p = (1.0 - gamma) * (w / w.sum()) + gamma / n
    r_hat = reward / max(p[a], 1e-18)
    weights[a] *= np.exp(gamma * r_hat / n)


# ---------------------------------------------------------------------------
# Offline dataset builders -- all use Bernoulli rewards (Deviation 1)
# ---------------------------------------------------------------------------

def _sample_bernoulli_pair(
    mu_l: float, mu_f: float, rng: np.random.Generator
) -> Tuple[float, float]:
    return bernoulli_sample(mu_l, rng), bernoulli_sample(mu_f, rng)


def build_offline_uniform(
    env: StackelbergBandit, n_off: int, rng: np.random.Generator
) -> OfflineRewardStats:
    n_a, n_b = env.n_a, env.n_b
    nv = np.zeros((n_a, n_b), dtype=np.int64)
    sf = np.zeros((n_a, n_b)); sl = np.zeros((n_a, n_b))
    for _ in range(n_off):
        a = int(rng.integers(0, n_a)); b = int(rng.integers(0, n_b))
        rl, rf = _sample_bernoulli_pair(env.mu_leader[a, b], env.mu_follower[a, b], rng)
        nv[a, b] += 1; sf[a, b] += rf; sl[a, b] += rl
    return OfflineRewardStats(nv, sf, sl)


def build_offline_good_coverage(
    env: StackelbergBandit, n_off: int, rng: np.random.Generator
) -> OfflineRewardStats:
    """Good coverage: concentrate on manipulation-relevant cells.

    The follower needs coverage on the cells that appear in the manipulation
    contrast Delta_{F,a*}(mu_l), i.e. (a_fm, F_fm(a_fm)) and (a, F_fm(a))
    for a != a_fm.  This is what low C_man means.

    Split: 60% on manipulation-relevant cells (a, F_fm(a)),
           20% on follower BR cells, 20% uniform.
    """
    n_a, n_b = env.n_a, env.n_b
    nv = np.zeros((n_a, n_b), dtype=np.int64)
    sf = np.zeros((n_a, n_b)); sl = np.zeros((n_a, n_b))
    F_fm, a_fm = true_best_manipulation(env.mu_leader, env.mu_follower)
    # Manipulation-relevant cells: (a, F_fm(a)) for each a
    manip_cells = [(a, int(F_fm[a])) for a in range(n_a)]
    for _ in range(n_off):
        u = rng.random()
        if u < 0.6:
            a, b = manip_cells[int(rng.integers(0, len(manip_cells)))]
        elif u < 0.8:
            a = int(rng.integers(0, n_a)); b = env.follower_br(a)
        else:
            a = int(rng.integers(0, n_a)); b = int(rng.integers(0, n_b))
        rl, rf = _sample_bernoulli_pair(env.mu_leader[a, b], env.mu_follower[a, b], rng)
        nv[a, b] += 1; sf[a, b] += rf; sl[a, b] += rl
    return OfflineRewardStats(nv, sf, sl)


def build_offline_poor_coverage(
    env: StackelbergBandit, n_off: int, rng: np.random.Generator
) -> OfflineRewardStats:
    n_a, n_b = env.n_a, env.n_b
    nv = np.zeros((n_a, n_b), dtype=np.int64)
    sf = np.zeros((n_a, n_b)); sl = np.zeros((n_a, n_b))
    a_star = env.stackelberg_leader_action(); b_star = env.follower_br(a_star)
    for _ in range(n_off):
        a = int(rng.integers(0, n_a))
        if a == a_star:
            others = [b for b in range(n_b) if b != b_star]
            if others:
                b = int(rng.choice(others))
            else:
                while a == a_star:
                    a = int(rng.integers(0, n_a))
                b = int(rng.integers(0, n_b))
        else:
            b = int(rng.integers(0, n_b))
        rl, rf = _sample_bernoulli_pair(env.mu_leader[a, b], env.mu_follower[a, b], rng)
        nv[a, b] += 1; sf[a, b] += rf; sl[a, b] += rl
    return OfflineRewardStats(nv, sf, sl)


def build_offline_adversarial(
    env: StackelbergBandit, n_off: int, rng: np.random.Generator
) -> OfflineRewardStats:
    """Adversarial coverage: concentrates data on non-manipulation cells.

    Specifically avoids the manipulation-relevant cells (a, F_fm(a)) which the
    Gated's certification gate checks.  This means:
      - Gated: confidence sets on manip cells stay wide → certification FAILS
                → falls back to safe UCB exploration → performance ≈ baseline
      - Hybrid/ETC: use biased estimates → commit to wrong responses
                → performance degrades as more bad data cements wrong beliefs
    """
    n_a, n_b = env.n_a, env.n_b
    nv = np.zeros((n_a, n_b), dtype=np.int64)
    sf = np.zeros((n_a, n_b)); sl = np.zeros((n_a, n_b))
    F_fm, _ = true_best_manipulation(env.mu_leader, env.mu_follower)
    manip_cells = set((a, int(F_fm[a])) for a in range(n_a))

    for _ in range(n_off):
        # 90% on non-manipulation cells, 10% sparse on manipulation cells
        if rng.random() < 0.9:
            # Pick a non-manipulation cell
            for _attempt in range(20):
                a = int(rng.integers(0, n_a))
                b = int(rng.integers(0, n_b))
                if (a, b) not in manip_cells:
                    break
        else:
            # Sparse: uniform random (some will hit manip cells)
            a = int(rng.integers(0, n_a))
            b = int(rng.integers(0, n_b))
        rl, rf = _sample_bernoulli_pair(env.mu_leader[a, b], env.mu_follower[a, b], rng)
        nv[a, b] += 1; sf[a, b] += rf; sl[a, b] += rl
    return OfflineRewardStats(nv, sf, sl)


# ---------------------------------------------------------------------------
# Main simulation loop (Bernoulli rewards, regression confidence sets)
# ---------------------------------------------------------------------------

def _unpack_offline(
    offline_init: Optional[OfflineRewardStats],
    n_a: int,
    n_b: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if offline_init is None:
        return (
            np.zeros((n_a, n_b), dtype=np.int64),
            np.zeros((n_a, n_b)),
            np.zeros((n_a, n_b)),
        )
    return offline_init.n_visits.copy(), offline_init.sum_r_f.copy(), offline_init.sum_r_l.copy()


def simulate_gated_run(
    env: StackelbergBandit,
    horizon: int,
    rng: np.random.Generator,
    *,
    gamma_exp3: float,
    offline_init: Optional[OfflineRewardStats] = None,
    delta: float = 0.05,
    progress_rounds: bool = False,
    progress_desc: str = "round",
) -> Dict[str, np.ndarray]:
    """
    One trajectory: Gated offline-online method (certify-or-discard).
    If offline data passes the transfer theorem, commit to offline candidate.
    If it fails, discard all offline data and explore from scratch with FMUCB.

    Rewards are Bernoulli: r_l ~ Ber(mu_l(a,b)), r_f ~ Ber(mu_f(a,b)).
    Confidence sets use regression_confidence_radius (Deviation 2).
    T_{f,w} = sum_t 1{b_t != F^fm(a_t)}.

    Set ``progress_rounds=True`` for a per-round tqdm bar (e.g. learning-curve runs).
    """
    n_a, n_b = env.n_a, env.n_b
    n_visits, sum_r_f, sum_r_l = _unpack_offline(offline_init, n_a, n_b)
    weights = np.ones(n_a)
    F_true, _ = true_best_manipulation(env.mu_leader, env.mu_follower)
    opt_payoff = env.leader_reward_at_br(env.stackelberg_leader_action())

    offline_m0_nonempty = np.array([np.nan])
    theorem1_transfer_ok = np.array([np.nan])
    theorem1_transfer_lhs = np.array([np.nan])
    theorem1_transfer_threshold = np.array([np.nan])
    theorem1_delta3 = np.array([np.nan])
    certified_F = None  # Will be set if offline transfer is certified
    if offline_init is not None:
        cand_F, _, _, _ = offline_candidate_manipulation(
            sum_r_l, sum_r_f, n_visits, n_a, n_b, delta
        )
        offline_m0_nonempty = np.array([1.0 if cand_F is not None else 0.0])
        ok, lhs, thr, d3 = theorem1_offline_transfer_check(
            env.mu_leader, env.mu_follower, n_visits, sum_r_l, n_a, n_b
        )
        theorem1_transfer_ok = np.array([1.0 if ok else 0.0])
        theorem1_transfer_lhs = np.array([lhs])
        theorem1_transfer_threshold = np.array([thr])
        theorem1_delta3 = np.array([d3])

        if ok and cand_F is not None:
            # Certification PASSED: commit to offline candidate (like ETC but verified)
            certified_F = cand_F
        else:
            # Certification FAILED: discard offline data entirely and explore fresh.
            # Bad offline data can bias UCB confidence sets and hurt exploration.
            n_visits = np.zeros((n_a, n_b), dtype=np.int64)
            sum_r_f = np.zeros((n_a, n_b))
            sum_r_l = np.zeros((n_a, n_b))

    a_hist = np.zeros(horizon, dtype=np.int32)
    b_hist = np.zeros(horizon, dtype=np.int32)
    subopt = np.zeros(horizon, dtype=np.bool_)
    leader_regret = np.zeros(horizon)
    fallback_count = 0

    if progress_rounds:
        round_iter = tqdm(
            range(1, horizon + 1),
            desc=progress_desc,
            leave=False,
            unit="t",
            total=horizon,
            mininterval=0.25,
        )
    else:
        round_iter = range(1, horizon + 1)
    for t in round_iter:
        a = exp3_sample(weights, gamma_exp3, rng)
        if certified_F is not None:
            # Certified: commit to offline manipulation rule
            b = int(certified_F[a])
            used_fallback = False
        else:
            b, used_fallback = hybrid_fmucb_pick(
                a, n_visits, sum_r_f, sum_r_l, t, n_a, n_b, rng, delta=delta
            )
        if used_fallback:
            fallback_count += 1

        # Bernoulli rewards (Deviation 1)
        r_l = bernoulli_sample(env.mu_leader[a, b], rng)
        r_f = bernoulli_sample(env.mu_follower[a, b], rng)

        exp3_update(weights, a, r_l, gamma_exp3)
        n_visits[a, b] += 1
        sum_r_f[a, b] += r_f
        sum_r_l[a, b] += r_l

        subopt[t - 1] = (b != F_true[a])
        a_hist[t - 1] = a
        b_hist[t - 1] = b
        leader_regret[t - 1] = opt_payoff - env.mu_leader[a, b]

    return {
        "subopt": subopt,
        "a_hist": a_hist,
        "b_hist": b_hist,
        "leader_regret": leader_regret,
        "fallback_count": fallback_count,
        "offline_m0_nonempty": offline_m0_nonempty,
        "theorem1_transfer_ok": theorem1_transfer_ok,
        "theorem1_transfer_lhs": theorem1_transfer_lhs,
        "theorem1_transfer_threshold": theorem1_transfer_threshold,
        "theorem1_delta3": theorem1_delta3,
    }


# ---------------------------------------------------------------------------
# Convergence metric and CI helpers
# ---------------------------------------------------------------------------

def convergence_round(
    subopt: np.ndarray,
    window: int = 200,
    threshold: float = 0.2,
    k_sustain: int = 3,
) -> int:
    h = len(subopt)
    min_t = window + k_sustain - 1
    if h < min_t:
        return h
    for t in range(min_t, h + 1):
        if all(
            float(subopt[t - s - window: t - s].mean()) <= threshold
            for s in range(k_sustain)
        ):
            return t
    return h


def ci_mean(
    data: np.ndarray, axis: int = 0, z: float = 1.96
) -> Tuple[np.ndarray, np.ndarray]:
    m = data.mean(axis=axis)
    s = data.std(axis=axis, ddof=1) / np.sqrt(max(1, data.shape[axis]))
    return m, z * s


def ci_mean_nan_1d(data: np.ndarray, z: float = 1.96) -> Tuple[float, float]:
    x = data[np.isfinite(data)]
    if x.size == 0:
        return float("nan"), float("nan")
    m = float(x.mean())
    if x.size <= 1:
        return m, 0.0
    return m, z * float(x.std(ddof=1)) / np.sqrt(x.size)


def ci_mean_nan(
    data: np.ndarray, axis: int = 0, z: float = 1.96
) -> Tuple[np.ndarray, np.ndarray]:
    """Mean and ~95% CI row-wise, ignoring NaNs (effective n varies)."""
    m = np.nanmean(data, axis=axis)
    valid = np.sum(np.isfinite(data), axis=axis).astype(np.float64)
    valid = np.maximum(valid, 1.0)
    std = np.nanstd(data, axis=axis, ddof=1)
    std = np.where(np.isfinite(std), std, 0.0)
    se = std / np.sqrt(valid)
    return m, z * se


def rolling_subopt_rate_at_a_star(
    subopt: np.ndarray,
    a_hist: np.ndarray,
    a_star: int,
    window: int,
) -> np.ndarray:
    """Rolling mean subopt over rounds in each window with `a_t=a^*` (NaN if none)."""
    n = len(subopt)
    if n < window:
        return np.array([], dtype=np.float64)
    out = np.empty(n - window + 1, dtype=np.float64)
    for i in range(n - window + 1):
        sub = subopt[i : i + window]
        m = a_hist[i : i + window] == a_star
        out[i] = np.nan if not np.any(m) else float(sub[m].mean())
    return out


def follower_subopt_rate_at_a_star(
    tr: Dict[str, np.ndarray], env: StackelbergBandit
) -> float:
    a_star = env.stackelberg_leader_action()
    mask = tr["a_hist"] == a_star
    return float(tr["subopt"][mask].mean()) if np.any(mask) else float("nan")


# ---------------------------------------------------------------------------
# Experiment 1: Offline dataset size vs T_{f,w}
# ---------------------------------------------------------------------------

def simulate_hybrid_run(
    env: StackelbergBandit,
    horizon: int,
    rng: np.random.Generator,
    *,
    gamma_exp3: float,
    offline_init: Optional[OfflineRewardStats] = None,
    delta: float = 0.05,
) -> Dict[str, np.ndarray]:
    """
    Hybrid-FMUCB (Algorithm 1 from the paper).
    Initialize confidence sets using offline data, then run FMUCB every round
    using D_off ∪ D_{t-1} to build shrinking confidence sets. No certification
    gate — offline data always contributes to the confidence widths.
    """
    n_a, n_b = env.n_a, env.n_b
    n_visits, sum_r_f, sum_r_l = _unpack_offline(offline_init, n_a, n_b)
    weights = np.ones(n_a)
    F_true, _ = true_best_manipulation(env.mu_leader, env.mu_follower)

    a_hist = np.zeros(horizon, dtype=np.int32)
    b_hist = np.zeros(horizon, dtype=np.int32)
    subopt = np.zeros(horizon, dtype=np.bool_)

    for t in range(1, horizon + 1):
        a = exp3_sample(weights, gamma_exp3, rng)
        b, _ = hybrid_fmucb_pick(
            a, n_visits, sum_r_f, sum_r_l, t, n_a, n_b, rng, delta=delta
        )
        r_l = bernoulli_sample(env.mu_leader[a, b], rng)
        r_f = bernoulli_sample(env.mu_follower[a, b], rng)
        exp3_update(weights, a, r_l, gamma_exp3)
        n_visits[a, b] += 1
        sum_r_l[a, b] += r_l
        sum_r_f[a, b] += r_f
        a_hist[t - 1] = a
        b_hist[t - 1] = b
        subopt[t - 1] = (b != F_true[a])

    return {"subopt": subopt, "a_hist": a_hist, "b_hist": b_hist}


def simulate_explore_then_commit_run(
    env: StackelbergBandit,
    horizon: int,
    rng: np.random.Generator,
    *,
    gamma_exp3: float,
    offline_init: Optional[OfflineRewardStats] = None,
    delta: float = 0.05,
) -> Dict[str, np.ndarray]:
    """
    Explore-then-commit baseline: use offline data to identify the best manipulation
    candidate, then commit to that response rule for the entire horizon.
    No online refinement. If offline data is insufficient to certify any manipulation,
    fall back to follower best-response (argmax mu_f_hat).
    """
    n_a, n_b = env.n_a, env.n_b
    n_visits, sum_r_f, sum_r_l = _unpack_offline(offline_init, n_a, n_b)
    F_true, _ = true_best_manipulation(env.mu_leader, env.mu_follower)

    # Find best offline candidate
    cand_F, _, _, _ = offline_candidate_manipulation(
        sum_r_l, sum_r_f, n_visits, n_a, n_b, delta
    )

    # If no certified candidate, use follower BR under offline estimates
    if cand_F is None:
        mu_f_hat = pooled_mean(sum_r_f, n_visits)
        cand_F = np.argmax(mu_f_hat, axis=1).astype(np.int32)

    weights = np.ones(n_a)
    subopt = np.zeros(horizon, dtype=np.bool_)

    for t in range(1, horizon + 1):
        a = exp3_sample(weights, gamma_exp3, rng)
        b = int(cand_F[a])
        r_l = bernoulli_sample(env.mu_leader[a, b], rng)
        exp3_update(weights, a, r_l, gamma_exp3)
        subopt[t - 1] = (b != F_true[a])

    return {"subopt": subopt}


def _plot_exp1_averaged(out_dir, all_results, n_off_list, horizon, cum_target, data_quality):
    """Produce averaged Exp 1 plots from multiple per-game results."""
    x_labels = [str(n) for n in n_off_list]
    x_plot = np.array(n_off_list, dtype=float) + 1.0
    num_games = len(all_results)

    # Stack per-game means into arrays [num_games, len(n_off)]
    base_all = np.array([r["tfw_baseline_mean"] for r in all_results])
    hyb_all = np.array([r["tfw_hybrid_mean"] for r in all_results])
    gated_all = np.array([r["tfw_gated_mean"] for r in all_results])
    etc_all = np.array([r["tfw_etc_mean"] for r in all_results])

    mb = base_all.mean(0); eb = 1.96 * base_all.std(0, ddof=1) / np.sqrt(num_games) if num_games > 1 else np.zeros_like(mb)
    mh = hyb_all.mean(0); eh = 1.96 * hyb_all.std(0, ddof=1) / np.sqrt(num_games) if num_games > 1 else np.zeros_like(mh)
    mg = gated_all.mean(0); eg = 1.96 * gated_all.std(0, ddof=1) / np.sqrt(num_games) if num_games > 1 else np.zeros_like(mg)
    metc = etc_all.mean(0); eetc = 1.96 * etc_all.std(0, ddof=1) / np.sqrt(num_games) if num_games > 1 else np.zeros_like(metc)

    quality_label = {"good": "good-coverage", "neutral": "uniform", "poor": "poor-coverage", "adversarial": "adversarial"}[data_quality]

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    ax.set_xscale("log"); ax.grid(True, which="both", alpha=0.3)
    ax.plot(x_plot, mb, "o-", color=COLORS["baseline"], lw=3, ms=8, label="FMUCB (online only)")
    ax.fill_between(x_plot, mb - eb, mb + eb, color=COLORS["baseline"], alpha=0.18)
    ax.plot(x_plot, mg, "^--", color=COLORS["gated"], lw=2.5, ms=7, label="Gated-FMUCB")
    ax.fill_between(x_plot, mg - eg, mg + eg, color=COLORS["gated"], alpha=0.15)
    ax.plot(x_plot, metc, "v:", color="#F4A261", lw=2.5, ms=7, label="Explore-then-commit")
    ax.fill_between(x_plot, metc - eetc, metc + eetc, color="#F4A261", alpha=0.15)
    ax.plot(x_plot, mh, "s-", color=COLORS["hybrid"], lw=3, ms=8, label="Hybrid-FMUCB")
    ax.fill_between(x_plot, mh - eh, mh + eh, color=COLORS["hybrid"], alpha=0.18)
    ax.set_xticks(x_plot); ax.set_xticklabels(x_labels, rotation=30, ha="right")
    ax.set_xlabel(r"Offline dataset size $N_{\mathrm{off}}$")
    ax.set_ylabel(r"$T_{f,w}$ (mistakes vs.\ true best manipulation $F^{fm}$)")
    ax.set_title(rf"Experiment 1: Offline data size vs $T_{{f,w}}$ ({quality_label} data, {num_games} games avg)")
    ax.legend(frameon=True, fancybox=True, shadow=True)
    _style_axis(ax); fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "exp1_noff_vs_tfw.png"), bbox_inches="tight")
    plt.close(fig)


def experiment1(
    out_dir: str, seeds: Sequence[int], n_a: int, n_b: int,
    horizon: int, gamma_exp3: float, n_off_grid: Sequence[int],
    cum_n_off: Optional[int] = None, progress: bool = True,
    delta: float = 0.05, game_seed: Optional[int] = None,
    data_quality: str = "poor",
    game_seeds: Optional[Sequence[int]] = None,
) -> None:
    n_off_list = list(n_off_grid)
    cum_target = (max(n_off_list) if cum_n_off is None else cum_n_off)
    if cum_target not in n_off_list:
        cum_target = max(n_off_list)
    conv_win, conv_thr, conv_k = 200, 0.2, 3

    # Select offline data builder based on quality setting
    _offline_builders = {
        "good": build_offline_good_coverage,
        "neutral": build_offline_uniform,
        "poor": build_offline_poor_coverage,
        "adversarial": build_offline_adversarial,
        "mixed": None,  # special case: randomly pick per seed
    }
    build_offline = _offline_builders[data_quality]

    # Multi-game-seed: if game_seeds provided, run each game and average
    if game_seeds is not None and len(game_seeds) > 1:
        all_results = []
        for gs in game_seeds:
            sub_dir = os.path.join(out_dir, f"game_{gs}")
            os.makedirs(sub_dir, exist_ok=True)
            experiment1(
                sub_dir, seeds, n_a, n_b, horizon, gamma_exp3, n_off_grid,
                cum_n_off=cum_n_off, progress=progress, delta=delta,
                game_seed=gs, data_quality=data_quality, game_seeds=None,
            )
            with open(os.path.join(sub_dir, "exp1_summary.json")) as f:
                all_results.append(json.load(f))
        # Average across games
        avg = {"n_off": all_results[0]["n_off"], "game_seeds": list(game_seeds),
               "data_quality": data_quality, "num_games": len(game_seeds)}
        for key in ["tfw_baseline_mean", "tfw_hybrid_mean", "tfw_gated_mean", "tfw_etc_mean",
                    "convergence_baseline_mean", "convergence_hybrid_mean",
                    "theorem1_offline_transfer_rate_gated_mean"]:
            vals = [r[key] for r in all_results if key in r]
            if vals:
                avg[key] = np.mean(vals, axis=0).tolist()
                avg[key + "_per_game"] = [v if isinstance(v, list) else [v] for v in vals]
        with open(os.path.join(out_dir, "exp1_summary.json"), "w") as f:
            json.dump(avg, f, indent=2)
        # Also produce averaged plot
        _plot_exp1_averaged(out_dir, all_results, n_off_list, horizon, cum_target, data_quality)
        return

    t_fw_base = np.zeros((len(seeds), len(n_off_list)))
    t_fw_hyb = np.zeros((len(seeds), len(n_off_list)))
    t_fw_gated = np.zeros((len(seeds), len(n_off_list)))
    t_fw_etc = np.zeros((len(seeds), len(n_off_list)))
    conv_base = np.zeros((len(seeds), len(n_off_list)))
    conv_hyb = np.zeros((len(seeds), len(n_off_list)))
    cum_base_rows: List[np.ndarray] = []
    cum_hyb_rows: List[np.ndarray] = []
    thm1_transfer_gated = np.full((len(seeds), len(n_off_list)), np.nan)
    offline_m0_gated = np.full((len(seeds), len(n_off_list)), np.nan)

    # Use fixed game for controlled gaps; all seeds see the same game,
    # variance comes only from stochastic rewards / EXP3 randomness.
    if game_seed is not None:
        game_rng = np.random.default_rng(game_seed)
        fixed_env = StackelbergBandit.sample(n_a, n_b, game_rng)
    elif n_a == 4 and n_b == 4:
        fixed_env = StackelbergBandit.fixed_4x4()
    else:
        fixed_env = None

    # Compute C_man for this game instance
    c_man_value = float("nan")
    c_man_threshold_noff = float("nan")
    if fixed_env is not None:
        F_fm, a_fm = true_best_manipulation(fixed_env.mu_leader, fixed_env.mu_follower)
        delta3 = manipulation_contrast(fixed_env.mu_leader, F_fm, a_fm)
        # Estimate the theoretical N_off threshold from C_man
        # For each N_off, compute expected MSE under uniform nu and check transfer condition
        c_man_per_noff = {}
        for n_off_val in n_off_list:
            if n_off_val == 0:
                continue
            # Expected visits per cell under uniform: n_off / (n_a * n_b)
            expected_nv = np.full((n_a, n_b), n_off_val / (n_a * n_b))
            c_man_val = compute_c_man(F_fm, a_fm, n_a, n_b, expected_nv)
            # Expected MSE under uniform: sum of sigma^2/n per cell weighted by nu
            # For Bernoulli, Var(X) = p(1-p) <= 0.25, E_nu[MSE] ~ 0.25 * n_a * n_b / n_off
            expected_mse = 0.25 * n_a * n_b / n_off_val
            lhs = c_man_val * float(np.sqrt(expected_mse))
            c_man_per_noff[n_off_val] = {"c_man": c_man_val, "lhs": lhs, "threshold": delta3 / 4.0}
        c_man_value = c_man_val if n_off_list[-1] > 0 else float("nan")

    for si, seed in enumerate(
        tqdm(seeds, desc="Exp 1", unit="seed", disable=not progress)
    ):
        rng = np.random.default_rng(seed)
        env = fixed_env if fixed_env is not None else StackelbergBandit.sample(n_a, n_b, rng)

        # Run baseline ONCE per seed (it doesn't use offline data, so the
        # result must be identical across N_off values).
        rng_baseline = np.random.default_rng(seed * 100_000 + 7)
        tr_baseline = simulate_hybrid_run(
            env, horizon, rng_baseline, gamma_exp3=gamma_exp3, delta=delta
        )
        baseline_tfw = tr_baseline["subopt"].sum()
        baseline_conv = convergence_round(
            tr_baseline["subopt"], conv_win, conv_thr, conv_k
        )
        baseline_cum = np.cumsum(tr_baseline["subopt"].astype(float))

        n_off_indices = range(len(n_off_list))
        if progress:
            n_off_indices = tqdm(
                n_off_indices,
                desc=f"Exp1 seed {seed} N_off",
                leave=False,
                unit="cell",
            )
        for j in n_off_indices:
            n_off = n_off_list[j]
            rng_off = np.random.default_rng(seed * 100_000 + j + 11)
            rng_h = np.random.default_rng(seed * 100_000 + j + 13)
            if data_quality == "mixed":
                # Randomly pick data quality per seed (simulates unknown quality)
                _mixed_builders = [build_offline_good_coverage, build_offline_uniform, build_offline_adversarial]
                _pick = int(rng_off.integers(0, len(_mixed_builders)))
                off = _mixed_builders[_pick](env, n_off, rng_off) if n_off > 0 else None
            else:
                off = build_offline(env, n_off, rng_off) if n_off > 0 else None
            # Hybrid-FMUCB (paper's Algorithm 1)
            rng_hyb = np.random.default_rng(seed * 100_000 + j + 15)
            tr_h = simulate_hybrid_run(
                env, horizon, rng_hyb, gamma_exp3=gamma_exp3,
                offline_init=off, delta=delta
            )
            t_fw_hyb[si, j] = tr_h["subopt"].sum()
            conv_hyb[si, j] = convergence_round(tr_h["subopt"], conv_win, conv_thr, conv_k)
            if n_off == cum_target:
                cum_hyb_rows.append(np.cumsum(tr_h["subopt"].astype(float)))

            # Gated-FMUCB (certify-or-discard)
            rng_g = np.random.default_rng(seed * 100_000 + j + 13)
            tr_g = simulate_gated_run(
                env, horizon, rng_g, gamma_exp3=gamma_exp3, offline_init=off, delta=delta
            )
            if n_off > 0:
                thm1_transfer_gated[si, j] = float(tr_g["theorem1_transfer_ok"][0])
                offline_m0_gated[si, j] = float(tr_g["offline_m0_nonempty"][0])
            t_fw_gated[si, j] = tr_g["subopt"].sum()

            t_fw_base[si, j] = baseline_tfw
            conv_base[si, j] = baseline_conv
            if n_off == cum_target:
                cum_base_rows.append(baseline_cum)

            # Explore-then-commit baseline
            rng_etc = np.random.default_rng(seed * 100_000 + j + 17)
            tr_etc = simulate_explore_then_commit_run(
                env, horizon, rng_etc, gamma_exp3=gamma_exp3,
                offline_init=off, delta=delta
            )
            t_fw_etc[si, j] = tr_etc["subopt"].sum()

    x_labels = [str(n) for n in n_off_list]
    x_plot = np.array(n_off_list, dtype=float) + 1.0

    # Plot 1: N_off vs T_{f,w}
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    ax.set_xscale("log"); ax.grid(True, which="both", alpha=0.3)
    mb, eb = ci_mean(t_fw_base); mh, eh = ci_mean(t_fw_hyb)
    mg, eg = ci_mean(t_fw_gated); metc, eetc = ci_mean(t_fw_etc)
    ax.plot(x_plot, mb, "o-", color=COLORS["baseline"], lw=3, ms=8, label="FMUCB (online only)")
    ax.fill_between(x_plot, mb - eb, mb + eb, color=COLORS["baseline"], alpha=0.18)
    ax.plot(x_plot, mg, "^--", color=COLORS["gated"], lw=2.5, ms=7, label="Gated-FMUCB")
    ax.fill_between(x_plot, mg - eg, mg + eg, color=COLORS["gated"], alpha=0.15)
    ax.plot(x_plot, metc, "v:", color="#F4A261", lw=2.5, ms=7, label="Explore-then-commit")
    ax.fill_between(x_plot, metc - eetc, metc + eetc, color="#F4A261", alpha=0.15)
    ax.plot(x_plot, mh, "s-", color=COLORS["hybrid"], lw=3, ms=8, label="Hybrid-FMUCB")
    ax.fill_between(x_plot, mh - eh, mh + eh, color=COLORS["hybrid"], alpha=0.18)
    ax.set_xticks(x_plot); ax.set_xticklabels(x_labels)
    ax.set_xlabel(r"Offline dataset size $N_{\mathrm{off}}$")
    ax.set_ylabel(r"$T_{f,w}$ (mistakes vs.\ true best manipulation $F^{fm}$)")
    ax.set_title(r"Experiment 1: Offline data size vs $T_{f,w}$")
    ax.legend(frameon=True, fancybox=True, shadow=True)
    _style_axis(ax); fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "exp1_noff_vs_tfw.png"), bbox_inches="tight")
    plt.close(fig)

    # Plot 2: convergence rounds
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    ax.set_xscale("log"); ax.grid(True, which="both", alpha=0.3)
    cb, cbe = ci_mean(conv_base); ch, che = ci_mean(conv_hyb)
    ax.plot(x_plot, cb, "o-", color=COLORS["baseline"], lw=3, ms=8, label="FMUCB")
    ax.fill_between(x_plot, cb - cbe, cb + cbe, color=COLORS["baseline"], alpha=0.18)
    ax.plot(x_plot, ch, "s-", color=COLORS["hybrid"], lw=3, ms=8, label="Hybrid-FMUCB")
    ax.fill_between(x_plot, ch - che, ch + che, color=COLORS["hybrid"], alpha=0.18)
    ax.set_xticks(x_plot); ax.set_xticklabels(x_labels)
    ax.set_xlabel(r"$N_{\mathrm{off}}$")
    ax.set_ylabel("Rounds to reach sustained low error rate")
    ax.set_title(
        rf"Experiment 1: Convergence ($\leq{conv_thr}$, "
        rf"{conv_k} windows of {conv_win})"
    )
    ax.legend(frameon=True, fancybox=True, shadow=True)
    _style_axis(ax); fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "exp1_noff_vs_convergence.png"), bbox_inches="tight")
    plt.close(fig)

    # Plot 3: cumulative mistakes
    t_axis = np.arange(1, horizon + 1)
    mcb, ecb = ci_mean(np.stack(cum_base_rows))
    mch, ech = ci_mean(np.stack(cum_hyb_rows))
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    ax.plot(t_axis, mcb, "-", color=COLORS["baseline"], lw=3, label="FMUCB")
    ax.fill_between(t_axis, mcb - ecb, mcb + ecb, color=COLORS["baseline"], alpha=0.18)
    ax.plot(t_axis, mch, "-", color=COLORS["hybrid"], lw=3, label="Hybrid-FMUCB")
    ax.fill_between(t_axis, mch - ech, mch + ech, color=COLORS["hybrid"], alpha=0.18)
    ax.set_xlabel("Round $t$")
    ax.set_ylabel(r"Cumulative $T_{f,w}$ mistakes")
    ax.set_title(rf"Experiment 1: Cumulative mistakes ($N_{{\mathrm{{off}}}}={cum_target}$)")
    ax.legend(frameon=True, fancybox=True, shadow=True, loc="lower right")
    _style_axis(ax); fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"exp1_cumulative_mistakes_noff{cum_target}.png"), bbox_inches="tight")
    plt.close(fig)

    with open(os.path.join(out_dir, "exp1_summary.json"), "w") as f:
        summary = {
            "n_off": n_off_list,
            "reward_model": "Bernoulli",
            "confidence_sets": "regression_Hoeffding",
            "confidence_delta": delta,
            "gamma_exp3": gamma_exp3,
            "game_seed": game_seed,
            "data_quality": data_quality,
            "tfw_baseline_mean": t_fw_base.mean(0).tolist(),
            "tfw_baseline_std": t_fw_base.std(0, ddof=1).tolist(),
            "tfw_hybrid_mean": t_fw_hyb.mean(0).tolist(),
            "tfw_hybrid_std": t_fw_hyb.std(0, ddof=1).tolist(),
            "tfw_gated_mean": t_fw_gated.mean(0).tolist(),
            "tfw_gated_std": t_fw_gated.std(0, ddof=1).tolist(),
            "tfw_etc_mean": t_fw_etc.mean(0).tolist(),
            "tfw_etc_std": t_fw_etc.std(0, ddof=1).tolist(),
            "convergence_baseline_mean": conv_base.mean(0).tolist(),
            "convergence_hybrid_mean": conv_hyb.mean(0).tolist(),
            "theorem1_offline_transfer_rate_gated_mean": np.nanmean(
                thm1_transfer_gated, axis=0
            ).tolist(),
            "offline_m0_nonempty_rate_gated_mean": np.nanmean(
                offline_m0_gated, axis=0
            ).tolist(),
        }
        if fixed_env is not None:
            summary["c_man_per_noff"] = {str(k): v for k, v in c_man_per_noff.items()} if 'c_man_per_noff' in dir() else {}
            summary["delta3"] = float(delta3) if 'delta3' in dir() else None
        json.dump(summary, f, indent=2)


# ---------------------------------------------------------------------------
# Experiment 2: Coverage quality
# ---------------------------------------------------------------------------

def experiment2(
    out_dir: str, seeds: Sequence[int], n_a: int, n_b: int,
    horizon: int, gamma_exp3: float, n_off_fixed: int, progress: bool = True,
    delta: float = 0.05, game_seed: Optional[int] = None,
    game_seeds: Optional[Sequence[int]] = None,
) -> None:
    kinds = ("good", "neutral", "adversarial")
    methods = ("hybrid", "gated", "etc", "baseline")

    # Multi-game-seed: if game_seeds provided, run each game and average
    if game_seeds is not None and len(game_seeds) > 1:
        all_results = []
        for gs in game_seeds:
            sub_dir = os.path.join(out_dir, f"game_{gs}")
            os.makedirs(sub_dir, exist_ok=True)
            experiment2(
                sub_dir, seeds, n_a, n_b, horizon, gamma_exp3, n_off_fixed,
                progress=progress, delta=delta, game_seed=gs, game_seeds=None,
            )
            with open(os.path.join(sub_dir, "exp2_summary.json")) as f:
                all_results.append(json.load(f))
        # Average across games
        avg = {"n_off_fixed": n_off_fixed, "game_seeds": list(game_seeds), "num_games": len(game_seeds)}
        for key in all_results[0]:
            if key in ("n_off_fixed", "game_seeds", "num_games", "game_seed"):
                continue
            vals = [r[key] for r in all_results if key in r and isinstance(r[key], (int, float))]
            if vals:
                avg[key] = float(np.mean(vals))
        with open(os.path.join(out_dir, "exp2_summary.json"), "w") as f:
            json.dump(avg, f, indent=2)
        _plot_exp2_averaged(out_dir, all_results, n_off_fixed, kinds, methods)
        return

    # Per-method, per-quality T_fw tracking
    t_fw = {m: {k: np.zeros(len(seeds)) for k in kinds} for m in methods}

    if game_seed is not None:
        game_rng = np.random.default_rng(game_seed)
        fixed_env = StackelbergBandit.sample(n_a, n_b, game_rng)
    elif n_a == 4 and n_b == 4:
        fixed_env = StackelbergBandit.fixed_4x4()
    else:
        fixed_env = None

    builders = {
        "good": build_offline_good_coverage,
        "neutral": build_offline_uniform,
        "adversarial": build_offline_adversarial,
    }

    for kind in kinds:
        for si, seed in enumerate(
            tqdm(seeds, desc=f"Exp 2 ({kind})", unit="seed", disable=not progress)
        ):
            rng = np.random.default_rng(seed)
            env = fixed_env if fixed_env is not None else StackelbergBandit.sample(n_a, n_b, rng)
            rng_off = np.random.default_rng(seed + 91_000)
            off = builders[kind](env, n_off_fixed, rng_off)

            # Hybrid-FMUCB (paper's Algorithm 1)
            rng_h = np.random.default_rng(seed + 92_000)
            tr_h = simulate_hybrid_run(env, horizon, rng_h, gamma_exp3=gamma_exp3,
                                       offline_init=off, delta=delta)
            t_fw["hybrid"][kind][si] = tr_h["subopt"].sum()

            # Gated-FMUCB (certify-or-discard)
            rng_g = np.random.default_rng(seed + 93_000)
            tr_g = simulate_gated_run(env, horizon, rng_g, gamma_exp3=gamma_exp3,
                                      offline_init=off, delta=delta)
            t_fw["gated"][kind][si] = tr_g["subopt"].sum()

            # Explore-then-commit
            rng_etc = np.random.default_rng(seed + 94_000)
            tr_etc = simulate_explore_then_commit_run(env, horizon, rng_etc,
                                                      gamma_exp3=gamma_exp3,
                                                      offline_init=off, delta=delta)
            t_fw["etc"][kind][si] = tr_etc["subopt"].sum()

            # Baseline (no offline data)
            rng_base = np.random.default_rng(seed + 95_000)
            tr_base = simulate_hybrid_run(env, horizon, rng_base, gamma_exp3=gamma_exp3,
                                          offline_init=None, delta=delta)
            t_fw["baseline"][kind][si] = tr_base["subopt"].sum()

    # Plot: grouped bars — for each quality, show all 4 methods side by side
    method_labels = ["Hybrid-\nFMUCB", "Gated-\nFMUCB", "Explore-then-\ncommit", "FMUCB\n(no offline)"]
    method_colors = [COLORS["hybrid"], COLORS["gated"], "#F4A261", COLORS["baseline"]]
    kind_labels = ["Good (manip. relevant)", "Neutral (uniform)", "Adversarial (sparse manip.)"]

    fig, axes = plt.subplots(1, 3, figsize=(18.0, 5.5))
    n_methods = len(methods)
    bar_width = 0.2
    xpos = np.arange(n_methods)

    for ax, kind, kind_label in zip(axes, kinds, kind_labels):
        means = [float(t_fw[m][kind].mean()) for m in methods]
        errs = [1.96 * t_fw[m][kind].std(ddof=1) / np.sqrt(len(seeds)) for m in methods]
        bars = ax.bar(xpos, means, width=bar_width * 3.5, yerr=errs,
                      color=method_colors, edgecolor="white",
                      linewidth=1.2, capsize=6, error_kw={"linewidth": 1.5})
        for rect, m_val, e_val in zip(bars, means, errs):
            if np.isfinite(m_val):
                ax.text(rect.get_x() + rect.get_width() / 2,
                        rect.get_height() + e_val + max(errs) * 0.05,
                        f"{m_val:.0f}", ha="center", va="bottom", fontsize=13, fontweight="bold")
        ax.set_xticks(xpos); ax.set_xticklabels(method_labels, fontsize=16)
        ax.set_ylabel(r"$T_{f,w}$ (mistakes)", fontsize=16)
        ax.set_title(kind_label)
        _style_axis(ax)

    fig.suptitle(rf"Experiment 2: All methods $\times$ data quality ($N_{{\mathrm{{off}}}}={n_off_fixed}$)",
                 fontsize=22, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "exp2_coverage_bars.png"), bbox_inches="tight")
    plt.close(fig)

    with open(os.path.join(out_dir, "exp2_summary.json"), "w") as f:
        summary = {"n_off_fixed": n_off_fixed, "game_seed": game_seed}
        for m in methods:
            for k in kinds:
                summary[f"tfw_{m}_{k}_mean"] = float(t_fw[m][k].mean())
                summary[f"tfw_{m}_{k}_std"] = float(t_fw[m][k].std(ddof=1))
        json.dump(summary, f, indent=2)


def _plot_exp2_averaged(out_dir, all_results, n_off_fixed, kinds, methods):
    """Produce averaged Exp 2 bar chart from multiple per-game results."""
    method_labels = ["Hybrid-\nFMUCB", "Gated-\nFMUCB", "Explore-then-\ncommit", "FMUCB\n(no offline)"]
    method_colors = [COLORS["hybrid"], COLORS["gated"], "#F4A261", COLORS["baseline"]]
    kind_labels = ["Good (manip. relevant)", "Neutral (uniform)", "Adversarial (sparse manip.)"]
    num_games = len(all_results)

    fig, axes = plt.subplots(1, 3, figsize=(18.0, 5.5))
    n_methods = len(methods)
    bar_width = 0.2
    xpos = np.arange(n_methods)

    for ax, kind, kind_label in zip(axes, kinds, kind_labels):
        # Gather means across games
        game_means = []
        for r in all_results:
            game_means.append([r[f"tfw_{m}_{kind}_mean"] for m in methods])
        game_means = np.array(game_means)  # [num_games, n_methods]
        means = game_means.mean(0)
        errs = 1.96 * game_means.std(0, ddof=1) / np.sqrt(num_games) if num_games > 1 else np.zeros(n_methods)

        bars = ax.bar(xpos, means, width=bar_width * 3.5, yerr=errs,
                      color=method_colors, edgecolor="white",
                      linewidth=1.2, capsize=6, error_kw={"linewidth": 1.5})
        for rect, m_val, e_val in zip(bars, means, errs):
            if np.isfinite(m_val):
                ax.text(rect.get_x() + rect.get_width() / 2,
                        rect.get_height() + e_val + max(errs) * 0.05,
                        f"{m_val:.0f}", ha="center", va="bottom", fontsize=13, fontweight="bold")
        ax.set_xticks(xpos); ax.set_xticklabels(method_labels, fontsize=16)
        ax.set_ylabel(r"$T_{f,w}$ (mistakes)", fontsize=16)
        ax.set_title(kind_label)
        _style_axis(ax)

    fig.suptitle(rf"Experiment 2: All methods $\times$ data quality ($N_{{\mathrm{{off}}}}={n_off_fixed}$, {num_games} games avg)",
                 fontsize=22, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "exp2_coverage_bars.png"), bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Optional: learning curves
# ---------------------------------------------------------------------------

def learning_curve_figure(
    out_dir: str,
    seeds: Sequence[int],
    n_a: int,
    n_b: int,
    horizon: int,
    gamma_exp3: float,
    n_off: int,
    progress: bool = True,
    rolling_window: Optional[int] = None,
    delta: float = 0.05,
    caption_suffix: str = "",
) -> None:
    """
    Rolling **global** mistake rate vs time (subopt averaged over all rounds in the window).

    Paper context (*Hybrid Offline-Online Follower Manipulation...*): $T_{f,w}$ aggregates
    rounds where $b_t \\neq F^{fm}(a_t)$. The theory emphasizes **qualified manipulation**
    at the target and hybrid **sample complexity** (via $C_{\\mathrm{man}}$), not a monotone
    global error curve. EXP3 explores all leader rows, so the global rolling rate can stay high
    even when behavior at the Stackelberg row $a^*$ improves — see
    ``exp_optional_learning_curves_at_a_star.png``. For stronger global trends without changing
    Alg. 1, use smaller $\\gamma$, longer $T$, more offline data (CLI: ``--learning-curve-paper-profile``).
    """
    curves_b: List[np.ndarray] = []
    curves_h: List[np.ndarray] = []
    curves_b_as: List[np.ndarray] = []
    curves_h_as: List[np.ndarray] = []
    win = rolling_window if rolling_window is not None else max(50, horizon // 100)
    win = min(int(win), max(1, horizon))
    fb_hybrid_total = 0
    if progress:
        print(
            f"Learning curves: {len(seeds)} seeds, horizon={horizon}, N_off={n_off} "
            f"(per-round tqdm inside each simulate_gated_run; outer bar = seeds).",
            flush=True,
        )
    fixed_env = StackelbergBandit.fixed_4x4() if (n_a == 4 and n_b == 4) else None
    for seed in tqdm(seeds, desc="Learning curves", unit="seed", disable=not progress):
        rng = np.random.default_rng(seed)
        env = fixed_env if fixed_env is not None else StackelbergBandit.sample(n_a, n_b, rng)
        a_star_env = int(env.stackelberg_leader_action())
        rng_b = np.random.default_rng(seed + 3)
        rng_h = np.random.default_rng(seed + 5)
        off = build_offline_uniform(env, n_off, rng_h) if n_off > 0 else None
        tr_b = simulate_hybrid_run(
            env, horizon, rng_b, gamma_exp3=gamma_exp3, delta=delta
        )
        tr_h = simulate_hybrid_run(
            env, horizon, rng_h, gamma_exp3=gamma_exp3,
            offline_init=off, delta=delta
        )
        fb_hybrid_total += int(tr_h.get("fallback_count", 0))

        sub_b = tr_b["subopt"].astype(float)
        sub_h = tr_h["subopt"].astype(float)
        curves_b.append(np.convolve(sub_b, np.ones(win) / win, mode="valid"))
        curves_h.append(np.convolve(sub_h, np.ones(win) / win, mode="valid"))
        curves_b_as.append(
            rolling_subopt_rate_at_a_star(sub_b, tr_b["a_hist"], a_star_env, win)
        )
        curves_h_as.append(
            rolling_subopt_rate_at_a_star(sub_h, tr_h["a_hist"], a_star_env, win)
        )

    t_axis = np.arange(win, horizon + 1)
    mb, eb = ci_mean(np.stack(curves_b))
    mh, eh = ci_mean(np.stack(curves_h))
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    ax.plot(t_axis, mb, color=COLORS["baseline"], lw=2, label="FMUCB")
    ax.fill_between(t_axis, mb - eb, mb + eb, color=COLORS["baseline"], alpha=0.2)
    ax.plot(t_axis, mh, color=COLORS["hybrid"], lw=2, label="Hybrid-FMUCB")
    ax.fill_between(t_axis, mh - eh, mh + eh, color=COLORS["hybrid"], alpha=0.2)
    ax.set_xlabel("Round $t$")
    ax.set_ylabel(f"Rolling subopt. rate (window={win})")
    fb_rate = fb_hybrid_total / max(1, len(seeds) * horizon)
    ax.set_title(
        rf"Learning curves (global): $N_{{\mathrm{{off}}}}={n_off}$ — "
        f"hybrid row-fallback fraction ≈ {fb_rate:.3f} per round (mean over seeds)"
        + caption_suffix
        + "\n"
        + r"(EXP3 visits all leader rows; global rolling error often stays high — compare the $a_t=a^*$ plot.)"
    )
    ax.legend()
    ax.set_ylim(0, 1.0)
    _style_axis(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "exp_optional_learning_curves.png"), bbox_inches="tight")
    plt.close(fig)

    stack_b_as = np.stack(curves_b_as, axis=0)
    stack_h_as = np.stack(curves_h_as, axis=0)
    mb_a, eb_a = ci_mean_nan(stack_b_as, axis=0)
    mh_a, eh_a = ci_mean_nan(stack_h_as, axis=0)
    fig_a, ax_a = plt.subplots(figsize=(9.0, 5.0))
    ax_a.plot(t_axis, mb_a, color=COLORS["baseline"], lw=2, label="FMUCB")
    ax_a.fill_between(t_axis, mb_a - eb_a, mb_a + eb_a, color=COLORS["baseline"], alpha=0.2)
    ax_a.plot(t_axis, mh_a, color=COLORS["hybrid"], lw=2, label="Hybrid-FMUCB")
    ax_a.fill_between(t_axis, mh_a - eh_a, mh_a + eh_a, color=COLORS["hybrid"], alpha=0.2)
    ax_a.set_xlabel("Round $t$")
    ax_a.set_ylabel(f"Rolling subopt. at $a_t=a^*$ (window={win})")
    ax_a.set_title(
        rf"Learning curves (paper-relevant): conditional on Stackelberg leader row "
        rf"($N_{{\mathrm{{off}}}}={n_off}$)"
        + caption_suffix
    )
    ax_a.legend()
    ax_a.set_ylim(0, 1.0)
    _style_axis(ax_a)
    fig_a.tight_layout()
    fig_a.savefig(
        os.path.join(out_dir, "exp_optional_learning_curves_at_a_star.png"),
        bbox_inches="tight",
    )
    plt.close(fig_a)


# ===========================================================================
# Experiment 3: Full contextual Hybrid-FMUCB -- Algorithm 2
# ===========================================================================

# ---- Kronecker feature map: phi(x,a,b) = z_x ⊗ [e_a; e_b] ----
# This gives d = d_ctx * (n_a + n_b), with GENUINE cross-context sharing:
# mu(x,a,b) = z_x^T (w_a + w_b) where W = reshape(theta, d_ctx, n_a+n_b).
# Evidence at one context constrains theta (hence w_a, w_b), immediately
# informing all other contexts through their z_x embeddings.

def make_context_embeddings(n_x: int, d_ctx: int, seed: int = 42) -> np.ndarray:
    """Fixed random context embeddings z_x ∈ R^{d_ctx}, unit-norm rows.

    Generated deterministically from ``seed`` so that the same n_x always
    gives the same embeddings (different n_x values share no structure).
    """
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n_x, d_ctx))
    Z /= np.linalg.norm(Z, axis=1, keepdims=True)
    return Z  # shape (n_x, d_ctx)


def phi_vec(x: int, a: int, b: int, n_a: int, n_b: int,
            ctx_embed: np.ndarray) -> np.ndarray:
    """Affine Kronecker feature map phi(x,a,b) = [1; z_x ⊗ [e_a; e_b]] ∈ R^{1+d_ctx*(n_a+n_b)}.

    The leading 1 gives the intercept term, ensuring mu = <phi, theta> can
    represent an arbitrary constant offset (here 0.5) without violating
    linear realizability.
    """
    z = ctx_embed[x]           # (d_ctx,)
    ab = np.zeros(n_a + n_b)
    ab[a] = 1.0
    ab[n_a + b] = 1.0
    kron = np.kron(z, ab)      # (d_ctx * (n_a + n_b),)
    return np.concatenate(([1.0], kron))  # (1 + d_ctx * (n_a + n_b),)


def _mu_ctx(theta: np.ndarray, x: int, a: int, b: int,
            n_a: int, n_b: int, ctx_embed: np.ndarray) -> float:
    """Linear reward: mu(x,a,b) = <phi(x,a,b), theta>.

    With the affine feature map (leading intercept=1) and theta[0]=0.5,
    this equals 0.5 + <phi_kron, theta[1:]), centering rewards in [0,1]
    while satisfying exact linear realizability.
    """
    return float(np.clip(theta @ phi_vec(x, a, b, n_a, n_b, ctx_embed), 0.0, 1.0))


def make_fixed_contextual_game(
    d_ctx: int, n_a: int, n_b: int, game_seed: int = 123,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fixed theta_l, theta_f ∈ R^{1+d} for a contextual game with clean
    manipulation structure.  Generated deterministically from ``game_seed``.

    theta[0] = 0.5 is the intercept (shared by leader and follower), and
    theta[1:] ~ Uniform(-0.20, 0.20)^d provides the game-specific structure.
    Combined with the affine feature map phi = [1; z_x ⊗ [e_a; e_b]], this
    gives mu(x,a,b) = 0.5 + <phi_kron, theta[1:]> ∈ roughly [0.04, 0.94],
    satisfying linear realizability exactly.
    """
    d = d_ctx * (n_a + n_b)
    rng = np.random.default_rng(game_seed)
    theta_l_kron = rng.uniform(-0.20, 0.20, d)
    theta_f_kron = rng.uniform(-0.20, 0.20, d)
    # Prepend intercept
    theta_l = np.concatenate(([0.5], theta_l_kron))
    theta_f = np.concatenate(([0.5], theta_f_kron))
    return theta_l, theta_f


def true_best_contextual_manipulation(
    theta_l: np.ndarray,
    theta_f: np.ndarray,
    n_x: int, n_a: int, n_b: int,
    ctx_embed: np.ndarray,
    eps: float = 1e-9,
) -> np.ndarray:
    """
    Compute the true best contextual manipulation rule F^fm : X x A -> B.

    For each context x, enumerate all (a*, b*) candidate response rules
    F_{a*,b*}^x where F(x,a*)=b* and F(x,a)=argmin_b mu_l(x,a,b) for a!=a*.
    Among qualified ones (contextual contrast > 0), pick the one maximising
    mu_f(x, a*, b*).  Return an array F_fm of shape (n_x, n_a) giving
    F^fm(x, a) for each (x, a).
    """
    # Precompute mu_l and mu_f tables
    mu_l = np.array([[[_mu_ctx(theta_l, x, a, b, n_a, n_b, ctx_embed)
                        for b in range(n_b)]
                       for a in range(n_a)]
                      for x in range(n_x)])  # (n_x, n_a, n_b)
    mu_f = np.array([[[_mu_ctx(theta_f, x, a, b, n_a, n_b, ctx_embed)
                        for b in range(n_b)]
                       for a in range(n_a)]
                      for x in range(n_x)])  # (n_x, n_a, n_b)

    F_fm = np.zeros((n_x, n_a), dtype=np.int32)

    for x in range(n_x):
        # Worst-response for each a under mu_l[x]
        wr = np.array([int(np.argmin(mu_l[x, a])) for a in range(n_a)])
        # Default: myopic BR
        F_fm[x] = np.array([int(np.argmax(mu_f[x, a])) for a in range(n_a)])

        best_val = -np.inf
        for a_star in range(n_a):
            for b_star in range(n_b):
                # F_{a*,b*}^x
                F_x = wr.copy(); F_x[a_star] = b_star
                # Contextual contrast
                left = mu_l[x, a_star, b_star]
                others = [mu_l[x, a, F_x[a]] for a in range(n_a) if a != a_star]
                contrast = left - max(others) if others else left
                if contrast <= eps:
                    continue
                v = mu_f[x, a_star, b_star]
                if v > best_val:
                    best_val = v
                    F_fm[x] = F_x.copy()

    return F_fm  # shape (n_x, n_a)


def _linucb_confidence(
    a_mat_inv: np.ndarray,
    b_vec: np.ndarray,
    ph: np.ndarray,
    alpha: float,
) -> Tuple[float, float]:
    """
    Returns (predicted_value, confidence_width) for a LinUCB/LCB arm.
    pred = theta_hat @ ph = (A^{-1} b) @ ph
    conf = alpha * sqrt(ph^T A^{-1} ph)
    """
    theta_hat = a_mat_inv @ b_vec
    pred = float(theta_hat @ ph)
    conf = float(alpha * np.sqrt(max(0.0, ph @ a_mat_inv @ ph)))
    return pred, conf


def _build_offline_contextual(
    theta_l: np.ndarray,
    theta_f: np.ndarray,
    n_off: int,
    rng: np.random.Generator,
    n_x: int, n_a: int, n_b: int,
    ctx_embed: np.ndarray,
    ridge_reg: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build offline confidence sets for leader and follower from D_off.
    Returns (A_l, b_l, A_f, b_f) ridge-regression matrices.
    Rewards are Bernoulli (Deviation 1).
    """
    d = len(theta_l)
    A_l = ridge_reg * np.eye(d); b_l = np.zeros(d)
    A_f = ridge_reg * np.eye(d); b_f = np.zeros(d)
    for _ in range(n_off):
        x = int(rng.integers(0, n_x))
        a = int(rng.integers(0, n_a))
        b = int(rng.integers(0, n_b))
        ph = phi_vec(x, a, b, n_a, n_b, ctx_embed)
        mu_l_ab = _mu_ctx(theta_l, x, a, b, n_a, n_b, ctx_embed)
        mu_f_ab = _mu_ctx(theta_f, x, a, b, n_a, n_b, ctx_embed)
        r_l = bernoulli_sample(mu_l_ab, rng)
        r_f = bernoulli_sample(mu_f_ab, rng)
        A_l += np.outer(ph, ph); b_l += ph * r_l
        A_f += np.outer(ph, ph); b_f += ph * r_f
    return A_l, b_l, A_f, b_f


def simulate_contextual_hybrid_fmucb(
    theta_l: np.ndarray,
    theta_f: np.ndarray,
    n_x: int, n_a: int, n_b: int,
    horizon: int,
    rng: np.random.Generator,
    gamma_exp3: float,
    ctx_embed: np.ndarray,
    offline_init: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = None,
    alpha: float = 1.0,
    ridge_reg: float = 1.0,
    eps: float = 1e-9,
) -> np.ndarray:
    """
    Full contextual Hybrid-FMUCB -- Algorithm 2 (paper Sec. 3).

    Uses Kronecker feature map phi(x,a,b) = z_x ⊗ [e_a; e_b].

    T_{f,w} counts b_t != F^fm(x_t, a_t) where F^fm is the true contextual
    best manipulation (maximises mu_f among qualified response rules per context).

    Rewards are Bernoulli.
    """
    d = len(theta_l)
    F_fm = true_best_contextual_manipulation(theta_l, theta_f, n_x, n_a, n_b, ctx_embed)

    if offline_init is not None:
        A_l, b_l, A_f, b_f = [m.copy() for m in offline_init]
    else:
        A_l = ridge_reg * np.eye(d); b_l = np.zeros(d)
        A_f = ridge_reg * np.eye(d); b_f = np.zeros(d)

    weights = np.ones((n_x, n_a))
    subopt = np.zeros(horizon, dtype=np.bool_)

    # Pre-compute initial inverses (updated incrementally via Sherman-Morrison)
    A_l_inv = np.linalg.inv(A_l)
    A_f_inv = np.linalg.inv(A_f)

    for t in range(1, horizon + 1):
        x = int(rng.integers(0, n_x))
        a_t = exp3_sample(weights[x], gamma_exp3, rng)
        theta_l_hat = A_l_inv @ b_l

        # Precompute worst-response under theta_l_hat for this context (once)
        base_wr = np.array([
            int(np.argmin([float(np.clip(theta_l_hat @ phi_vec(x, a, b, n_a, n_b, ctx_embed), 0, 1))
                           for b in range(n_b)]))
            for a in range(n_a)
        ], dtype=np.int32)

        # Precompute phi and confidence for each (a, WR(a)) — used in comparison arms
        wr_ph = [phi_vec(x, a, int(base_wr[a]), n_a, n_b, ctx_embed) for a in range(n_a)]
        wr_pred_conf = [_linucb_confidence(A_l_inv, b_l, wr_ph[a], alpha) for a in range(n_a)]

        # Search M_t(x) and pick optimistic follower action
        best_ucb_f = -np.inf
        best_F_x: Optional[np.ndarray] = None  # shape (n_a,)

        for a_star in range(n_a):
            for b_star in range(n_b):
                # Pessimistic leader contrast (inf over F_{l,t})
                ph_target = phi_vec(x, a_star, b_star, n_a, n_b, ctx_embed)
                pred_lt, conf_lt = _linucb_confidence(A_l_inv, b_l, ph_target, alpha)
                lcb_target = pred_lt - conf_lt

                max_ucb_other = -np.inf
                for a in range(n_a):
                    if a == a_star:
                        continue
                    # For a != a_star, F(a) = WR(a) = base_wr[a]
                    pred_lo, conf_lo = wr_pred_conf[a]
                    ucb_other = pred_lo + conf_lo
                    if ucb_other > max_ucb_other:
                        max_ucb_other = ucb_other

                pess_contrast = lcb_target - max_ucb_other
                if pess_contrast <= eps:
                    continue  # not in M_t(x)

                # Optimistic follower value at (x, a*, b*) — reuse ph_target
                pred_f, conf_f = _linucb_confidence(A_f_inv, b_f, ph_target, alpha)
                ucb_f = pred_f + conf_f
                if ucb_f > best_ucb_f:
                    best_ucb_f = ucb_f
                    best_F_x = base_wr.copy()
                    best_F_x[a_star] = b_star

        # Play b_t = F_hat(x_t, a_t), or UCB-BR fallback if M_t = {}
        if best_F_x is not None:
            b_t = int(best_F_x[a_t])
        else:
            theta_f_hat = A_f_inv @ b_f
            ucb_vals = np.array([
                float(np.clip(theta_f_hat @ phi_vec(x, a_t, b, n_a, n_b, ctx_embed), 0, 1))
                + float(alpha * np.sqrt(max(0.0,
                    phi_vec(x, a_t, b, n_a, n_b, ctx_embed) @ A_f_inv
                    @ phi_vec(x, a_t, b, n_a, n_b, ctx_embed))))
                for b in range(n_b)
            ])
            b_t = int(np.argmax(ucb_vals))

        # Observe Bernoulli rewards
        r_l = bernoulli_sample(_mu_ctx(theta_l, x, a_t, b_t, n_a, n_b, ctx_embed), rng)
        r_f = bernoulli_sample(_mu_ctx(theta_f, x, a_t, b_t, n_a, n_b, ctx_embed), rng)

        # EXP3 update for leader
        exp3_update(weights[x], a_t, r_l, gamma_exp3)

        # Update confidence sets (Alg 2, line 12) with Sherman-Morrison
        ph = phi_vec(x, a_t, b_t, n_a, n_b, ctx_embed)
        A_l += np.outer(ph, ph); b_l += ph * r_l
        A_f += np.outer(ph, ph); b_f += ph * r_f
        # Sherman-Morrison: (A + vv^T)^{-1} = A^{-1} - A^{-1}vv^TA^{-1}/(1+v^TA^{-1}v)
        Ainv_ph_l = A_l_inv @ ph
        A_l_inv -= np.outer(Ainv_ph_l, Ainv_ph_l) / (1.0 + ph @ Ainv_ph_l)
        Ainv_ph_f = A_f_inv @ ph
        A_f_inv -= np.outer(Ainv_ph_f, Ainv_ph_f) / (1.0 + ph @ Ainv_ph_f)

        # T_{f,w}: compare to true best contextual manipulation
        subopt[t - 1] = (b_t != F_fm[x, a_t])

    return subopt


def experiment3_contextual(
    out_dir: str,
    seeds: Sequence[int],
    horizon: int,
    gamma_exp3: float,
    n_off: int,
    n_x_list: Sequence[int],
    n_a: int,
    n_b: int,
    d_ctx: int = 3,
    progress: bool = True,
    n_off_per_ctx: Optional[int] = None,
) -> None:
    """
    Experiment 3: Contextual Hybrid-FMUCB (Algorithm 2) vs per-context tabular
    Hybrid-FMUCB (Algorithm 1), sweeping over |X| to show the gap growing.

    Key design choices vs previous version:
      1. Kronecker feature map phi(x,a,b) = z_x ⊗ [e_a; e_b] — genuine
         cross-context sharing through universal theta (not additive).
      2. Fixed game instance (theta_l, theta_f from deterministic seed) —
         variation across seeds comes only from reward noise / EXP3 randomness.
      3. Sweep over n_x values on the x-axis — directly tests Corollary 1:
         the ratio T_fw^tab / T_fw^ctx should grow with |X|.
    """
    d = 1 + d_ctx * (n_a + n_b)  # affine feature map: intercept + Kronecker
    theta_l, theta_f = make_fixed_contextual_game(d_ctx, n_a, n_b)

    tfw_ctx_by_nx: Dict[int, np.ndarray] = {}
    tfw_tab_by_nx: Dict[int, np.ndarray] = {}
    tfw_gated_ctx_by_nx: Dict[int, np.ndarray] = {}
    tfw_etc_ctx_by_nx: Dict[int, np.ndarray] = {}
    tfw_online_ctx_by_nx: Dict[int, np.ndarray] = {}
    n_qualified_by_nx: Dict[int, float] = {}

    for n_x in n_x_list:
        ctx_embed = make_context_embeddings(n_x, d_ctx)
        # --- realizability check: verify all mu values are in [0, 1] without clipping ---
        for x in range(n_x):
            for a in range(n_a):
                for b in range(n_b):
                    raw = float(theta_l @ phi_vec(x, a, b, n_a, n_b, ctx_embed))
                    assert 0.0 <= raw <= 1.0, f"Realizability violated: mu_l({x},{a},{b})={raw:.4f}"
                    raw_f = float(theta_f @ phi_vec(x, a, b, n_a, n_b, ctx_embed))
                    assert 0.0 <= raw_f <= 1.0, f"Realizability violated: mu_f({x},{a},{b})={raw_f:.4f}"
        # Verify manipulation structure at this n_x
        F_fm = true_best_contextual_manipulation(theta_l, theta_f, n_x, n_a, n_b, ctx_embed)
        n_qual = 0
        for x in range(n_x):
            mu_l_x = np.array([[_mu_ctx(theta_l, x, a, b, n_a, n_b, ctx_embed)
                                for b in range(n_b)] for a in range(n_a)])
            # Check if the F_fm[x] actually constitutes a qualified manipulation
            for a_star in range(n_a):
                left = mu_l_x[a_star, F_fm[x, a_star]]
                others = [mu_l_x[a, F_fm[x, a]] for a in range(n_a) if a != a_star]
                if others and left > max(others) + 1e-9:
                    n_qual += 1
                    break
        n_qualified_by_nx[n_x] = n_qual / n_x
        if progress:
            print(f"  n_x={n_x}: qualified manipulation at {n_qual}/{n_x} contexts "
                  f"({100*n_qual/n_x:.0f}%), d={d}", flush=True)

        # Optionally scale N_off with |X| so tabular per-context coverage is constant
        n_off_eff = n_off_per_ctx * n_x if n_off_per_ctx is not None else n_off

        tfw_ctx_arr = np.zeros(len(seeds))
        tfw_tab_arr = np.zeros(len(seeds))
        tfw_gated_ctx_arr = np.zeros(len(seeds))
        tfw_etc_ctx_arr = np.zeros(len(seeds))
        tfw_online_ctx_arr = np.zeros(len(seeds))

        for si, seed in enumerate(
            tqdm(seeds, desc=f"Exp3 |X|={n_x}", unit="seed", disable=not progress)
        ):
            # ---- Contextual Algorithm 2 ----
            rng_ctx = np.random.default_rng(seed + 29)
            off_ctx = _build_offline_contextual(
                theta_l, theta_f, n_off_eff, rng_ctx, n_x, n_a, n_b,
                ctx_embed, ridge_reg=1.0,
            )
            sub_ctx = simulate_contextual_hybrid_fmucb(
                theta_l, theta_f, n_x, n_a, n_b, horizon,
                rng_ctx, gamma_exp3, ctx_embed, offline_init=off_ctx,
            )
            tfw_ctx_arr[si] = float(sub_ctx.sum())

            # ---- Non-contextual Algorithm 1 (per-context tabular) ----
            rng_tab = np.random.default_rng(seed + 17)
            # Build per-context tabular offline stats
            ctx_nv: Dict[int, np.ndarray] = {}
            ctx_sf: Dict[int, np.ndarray] = {}
            ctx_sl: Dict[int, np.ndarray] = {}
            for x in range(n_x):
                ctx_nv[x] = np.zeros((n_a, n_b), dtype=np.int64)
                ctx_sf[x] = np.zeros((n_a, n_b))
                ctx_sl[x] = np.zeros((n_a, n_b))
            # Distribute offline samples uniformly across contexts
            for _ in range(n_off_eff):
                x = int(rng_tab.integers(0, n_x))
                a = int(rng_tab.integers(0, n_a))
                b = int(rng_tab.integers(0, n_b))
                r_l = bernoulli_sample(
                    _mu_ctx(theta_l, x, a, b, n_a, n_b, ctx_embed), rng_tab)
                r_f = bernoulli_sample(
                    _mu_ctx(theta_f, x, a, b, n_a, n_b, ctx_embed), rng_tab)
                ctx_nv[x][a, b] += 1
                ctx_sl[x][a, b] += r_l
                ctx_sf[x][a, b] += r_f

            # Simulate per-context tabular Hybrid-FMUCB
            weights_tab = np.ones((n_x, n_a))
            subopt_tab = np.zeros(horizon, dtype=np.bool_)
            for t in range(1, horizon + 1):
                x = int(rng_tab.integers(0, n_x))
                a_t = exp3_sample(weights_tab[x], gamma_exp3, rng_tab)
                b_t, _ = hybrid_fmucb_pick(
                    a_t, ctx_nv[x], ctx_sf[x], ctx_sl[x], t, n_a, n_b, rng_tab
                )
                r_l = bernoulli_sample(
                    _mu_ctx(theta_l, x, a_t, b_t, n_a, n_b, ctx_embed), rng_tab)
                r_f = bernoulli_sample(
                    _mu_ctx(theta_f, x, a_t, b_t, n_a, n_b, ctx_embed), rng_tab)
                exp3_update(weights_tab[x], a_t, r_l, gamma_exp3)
                ctx_nv[x][a_t, b_t] += 1
                ctx_sf[x][a_t, b_t] += r_f
                ctx_sl[x][a_t, b_t] += r_l
                subopt_tab[t - 1] = (b_t != F_fm[x, a_t])
            tfw_tab_arr[si] = float(subopt_tab.sum())

            # ---- Contextual FMUCB (online only, no offline data) ----
            rng_online = np.random.default_rng(seed + 31)
            sub_online = simulate_contextual_hybrid_fmucb(
                theta_l, theta_f, n_x, n_a, n_b, horizon,
                rng_online, gamma_exp3, ctx_embed, offline_init=None,
            )
            tfw_online_ctx_arr[si] = float(sub_online.sum())

            # ---- Contextual Gated-FMUCB (certify-or-discard) ----
            rng_gated = np.random.default_rng(seed + 37)
            off_gated = _build_offline_contextual(
                theta_l, theta_f, n_off_eff, rng_gated, n_x, n_a, n_b,
                ctx_embed, ridge_reg=1.0,
            )
            # Check if offline estimate certifies the manipulation for all contexts
            A_l_g, b_l_g, A_f_g, b_f_g = off_gated
            A_l_g_inv = np.linalg.inv(A_l_g)
            theta_l_hat_g = A_l_g_inv @ b_l_g
            # Certification: for each context, check pessimistic contrast > 0
            all_certified = True
            for x_check in range(n_x):
                best_pess_contrast = -np.inf
                for a_star in range(n_a):
                    for b_star in range(n_b):
                        ph_t = phi_vec(x_check, a_star, b_star, n_a, n_b, ctx_embed)
                        pred_t, conf_t = _linucb_confidence(A_l_g_inv, b_l_g, ph_t, 1.0)
                        lcb_t = pred_t - conf_t
                        max_ucb_o = -np.inf
                        for a_o in range(n_a):
                            if a_o == a_star:
                                continue
                            wr_b = int(np.argmin([
                                float(theta_l_hat_g @ phi_vec(x_check, a_o, bb, n_a, n_b, ctx_embed))
                                for bb in range(n_b)
                            ]))
                            ph_o = phi_vec(x_check, a_o, wr_b, n_a, n_b, ctx_embed)
                            pred_o, conf_o = _linucb_confidence(A_l_g_inv, b_l_g, ph_o, 1.0)
                            ucb_o = pred_o + conf_o
                            if ucb_o > max_ucb_o:
                                max_ucb_o = ucb_o
                        pess = lcb_t - max_ucb_o
                        if pess > best_pess_contrast:
                            best_pess_contrast = pess
                if best_pess_contrast <= 1e-9:
                    all_certified = False
                    break

            if all_certified:
                # Commit to offline manipulation
                sub_gated = simulate_contextual_hybrid_fmucb(
                    theta_l, theta_f, n_x, n_a, n_b, horizon,
                    rng_gated, gamma_exp3, ctx_embed, offline_init=off_gated,
                )
            else:
                # Discard offline data and run online only
                sub_gated = simulate_contextual_hybrid_fmucb(
                    theta_l, theta_f, n_x, n_a, n_b, horizon,
                    rng_gated, gamma_exp3, ctx_embed, offline_init=None,
                )
            tfw_gated_ctx_arr[si] = float(sub_gated.sum())

            # ---- Contextual ETC (explore-then-commit) ----
            rng_etc = np.random.default_rng(seed + 41)
            off_etc = _build_offline_contextual(
                theta_l, theta_f, n_off_eff, rng_etc, n_x, n_a, n_b,
                ctx_embed, ridge_reg=1.0,
            )
            A_l_e, b_l_e, A_f_e, b_f_e = off_etc
            A_l_e_inv = np.linalg.inv(A_l_e)
            A_f_e_inv = np.linalg.inv(A_f_e)
            theta_l_hat_e = A_l_e_inv @ b_l_e
            theta_f_hat_e = A_f_e_inv @ b_f_e
            # Find best offline manipulation per context (greedy, no confidence)
            F_etc = np.zeros((n_x, n_a), dtype=np.int32)
            for x_e in range(n_x):
                wr_e = np.array([
                    int(np.argmin([float(theta_l_hat_e @ phi_vec(x_e, a, bb, n_a, n_b, ctx_embed))
                                   for bb in range(n_b)]))
                    for a in range(n_a)
                ], dtype=np.int32)
                best_val_e = -np.inf
                best_F_e = wr_e.copy()
                for a_star in range(n_a):
                    for b_star in range(n_b):
                        F_cand = wr_e.copy(); F_cand[a_star] = b_star
                        # Check if estimated contrast is positive
                        left_e = float(theta_l_hat_e @ phi_vec(x_e, a_star, b_star, n_a, n_b, ctx_embed))
                        others_e = [float(theta_l_hat_e @ phi_vec(x_e, a, F_cand[a], n_a, n_b, ctx_embed))
                                    for a in range(n_a) if a != a_star]
                        if others_e and left_e <= max(others_e) + 1e-9:
                            continue
                        v_e = float(theta_f_hat_e @ phi_vec(x_e, a_star, b_star, n_a, n_b, ctx_embed))
                        if v_e > best_val_e:
                            best_val_e = v_e
                            best_F_e = F_cand.copy()
                F_etc[x_e] = best_F_e

            # Commit: play F_etc for entire horizon
            weights_etc = np.ones((n_x, n_a))
            subopt_etc = np.zeros(horizon, dtype=np.bool_)
            for t in range(1, horizon + 1):
                x_e = int(rng_etc.integers(0, n_x))
                a_t = exp3_sample(weights_etc[x_e], gamma_exp3, rng_etc)
                b_t = int(F_etc[x_e, a_t])
                r_l = bernoulli_sample(
                    _mu_ctx(theta_l, x_e, a_t, b_t, n_a, n_b, ctx_embed), rng_etc)
                exp3_update(weights_etc[x_e], a_t, r_l, gamma_exp3)
                subopt_etc[t - 1] = (b_t != F_fm[x_e, a_t])
            tfw_etc_ctx_arr[si] = float(subopt_etc.sum())

        tfw_ctx_by_nx[n_x] = tfw_ctx_arr
        tfw_tab_by_nx[n_x] = tfw_tab_arr
        tfw_gated_ctx_by_nx[n_x] = tfw_gated_ctx_arr
        tfw_etc_ctx_by_nx[n_x] = tfw_etc_ctx_arr
        tfw_online_ctx_by_nx[n_x] = tfw_online_ctx_arr

    # ---- Plot: T_fw vs |X| for all methods ----
    nx_arr = np.array(n_x_list)
    m_tab = np.array([tfw_tab_by_nx[nx].mean() for nx in n_x_list])
    e_tab = np.array([1.96 * tfw_tab_by_nx[nx].std(ddof=1) / np.sqrt(len(seeds))
                       for nx in n_x_list])
    m_ctx = np.array([tfw_ctx_by_nx[nx].mean() for nx in n_x_list])
    e_ctx = np.array([1.96 * tfw_ctx_by_nx[nx].std(ddof=1) / np.sqrt(len(seeds))
                       for nx in n_x_list])
    m_gated = np.array([tfw_gated_ctx_by_nx[nx].mean() for nx in n_x_list])
    e_gated = np.array([1.96 * tfw_gated_ctx_by_nx[nx].std(ddof=1) / np.sqrt(len(seeds))
                         for nx in n_x_list])
    m_etc = np.array([tfw_etc_ctx_by_nx[nx].mean() for nx in n_x_list])
    e_etc = np.array([1.96 * tfw_etc_ctx_by_nx[nx].std(ddof=1) / np.sqrt(len(seeds))
                       for nx in n_x_list])
    m_online = np.array([tfw_online_ctx_by_nx[nx].mean() for nx in n_x_list])
    e_online = np.array([1.96 * tfw_online_ctx_by_nx[nx].std(ddof=1) / np.sqrt(len(seeds))
                          for nx in n_x_list])

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.2))

    # Panel 1: raw T_fw
    ax = axes[0]
    ax.plot(nx_arr, m_tab, "o-", color=COLORS["tabular"], lw=2.5, ms=8,
            label="Per-context tabular Hybrid-FMUCB")
    ax.fill_between(nx_arr, m_tab - e_tab, m_tab + e_tab,
                    color=COLORS["tabular"], alpha=0.18)
    ax.plot(nx_arr, m_online, "x--", color=COLORS["baseline"], lw=2.5, ms=8,
            label="Contextual FMUCB (online only)")
    ax.fill_between(nx_arr, m_online - e_online, m_online + e_online,
                    color=COLORS["baseline"], alpha=0.15)
    ax.plot(nx_arr, m_gated, "^--", color=COLORS["gated"], lw=2.5, ms=7,
            label="Contextual Gated-FMUCB")
    ax.fill_between(nx_arr, m_gated - e_gated, m_gated + e_gated,
                    color=COLORS["gated"], alpha=0.15)
    ax.plot(nx_arr, m_etc, "v:", color="#F4A261", lw=2.5, ms=7,
            label="Contextual ETC")
    ax.fill_between(nx_arr, m_etc - e_etc, m_etc + e_etc,
                    color="#F4A261", alpha=0.15)
    ax.plot(nx_arr, m_ctx, "s-", color=COLORS["contextual"], lw=2.5, ms=8,
            label="Contextual Hybrid-FMUCB (Alg 2)")
    ax.fill_between(nx_arr, m_ctx - e_ctx, m_ctx + e_ctx,
                    color=COLORS["contextual"], alpha=0.18)
    ax.set_xlabel(r"Number of contexts $|\mathcal{X}|$")
    ax.set_ylabel(r"$T_{f,w}$ (vs.\ true contextual $F^{fm}$)")
    ax.set_title(r"Follower error vs number of contexts")
    ax.legend(frameon=True, fancybox=True, shadow=True, fontsize=12)
    _style_axis(ax)

    # Panel 2: ratio T_fw^tab / T_fw^ctx
    ax2 = axes[1]
    ratio = m_tab / np.maximum(m_ctx, 1.0)
    ax2.plot(nx_arr, ratio, "D-", color="#E76F51", lw=2.5, ms=8,
             label="Empirical ratio")
    ax2.set_xlabel(r"Number of contexts $|\mathcal{X}|$")
    ax2.set_ylabel(r"$T_{f,w}^{\mathrm{tab}} / T_{f,w}^{\mathrm{ctx}}$")
    ax2.set_title(r"Sample complexity ratio")
    ax2.legend(frameon=True, fancybox=True, shadow=True)
    _style_axis(ax2)

    noff_label = (f"$N_{{\\mathrm{{off}}}}={n_off_per_ctx}\\cdot|\\mathcal{{X}}|$"
                  if n_off_per_ctx is not None else f"$N_{{\\mathrm{{off}}}}={n_off}$")
    fig.suptitle(
        rf"Experiment 3: Contextual generalization ({noff_label}, "
        rf"$d={d}$, $|A|={n_a}$, $|B|={n_b}$, $d_{{\mathrm{{ctx}}}}={d_ctx}$)",
        fontsize=22, y=1.01,
    )
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "exp3_contextual_vs_tabular.png"),
                bbox_inches="tight")
    plt.close(fig)

    with open(os.path.join(out_dir, "exp3_summary.json"), "w") as f:
        json.dump({
            "algorithm_1": "non_contextual_per_context_tabular_hybrid_fmucb",
            "algorithm_2": "full_contextual_hybrid_fmucb_linucb",
            "feature_map": "kronecker: z_x ⊗ [e_a; e_b]",
            "reward_model": "Bernoulli",
            "d_ctx": d_ctx, "d": d,
            "n_off": n_off, "n_off_per_ctx": n_off_per_ctx,
            "n_a": n_a, "n_b": n_b,
            "n_x_list": list(n_x_list),
            "tfw_tab_mean": {str(nx): float(tfw_tab_by_nx[nx].mean()) for nx in n_x_list},
            "tfw_ctx_mean": {str(nx): float(tfw_ctx_by_nx[nx].mean()) for nx in n_x_list},
            "tfw_gated_ctx_mean": {str(nx): float(tfw_gated_ctx_by_nx[nx].mean()) for nx in n_x_list},
            "tfw_etc_ctx_mean": {str(nx): float(tfw_etc_ctx_by_nx[nx].mean()) for nx in n_x_list},
            "tfw_online_ctx_mean": {str(nx): float(tfw_online_ctx_by_nx[nx].mean()) for nx in n_x_list},
            "ratio": {str(nx): float(tfw_tab_by_nx[nx].mean() / max(1, tfw_ctx_by_nx[nx].mean()))
                      for nx in n_x_list},
            "n_qualified_frac": {str(nx): n_qualified_by_nx[nx] for nx in n_x_list},
        }, f, indent=2)


# ---------------------------------------------------------------------------
# Plot-only: regenerate figures from existing JSON summaries
# ---------------------------------------------------------------------------

def _replot_from_json(out_dir: str, n_off_grid: List[int], game_seeds_list, args) -> None:
    """Re-read saved per-game JSONs and regenerate plots without running experiments."""
    n_off_list = list(n_off_grid)

    # --- Exp 1 ---
    exp1_path = os.path.join(out_dir, "exp1_summary.json")
    if os.path.exists(exp1_path) and game_seeds_list:
        all_results = []
        for gs in game_seeds_list:
            p = os.path.join(out_dir, f"game_{gs}", "exp1_summary.json")
            if os.path.exists(p):
                with open(p) as f:
                    all_results.append(json.load(f))
        if all_results:
            # Infer n_off_list from saved data
            if "n_off" in all_results[0]:
                n_off_list = all_results[0]["n_off"]
            cum_target = max(n_off_list)
            data_quality = all_results[0].get("data_quality", args.exp1_data_quality)
            _plot_exp1_averaged(out_dir, all_results, n_off_list, args.horizon, cum_target, data_quality)
            print(f"  [plot-only] exp1: regenerated from {len(all_results)} game JSONs", flush=True)

    # --- Exp 2 ---
    exp2_path = os.path.join(out_dir, "exp2_summary.json")
    if os.path.exists(exp2_path) and game_seeds_list:
        all_results = []
        for gs in game_seeds_list:
            p = os.path.join(out_dir, f"game_{gs}", "exp2_summary.json")
            if os.path.exists(p):
                with open(p) as f:
                    all_results.append(json.load(f))
        if all_results:
            kinds = ("good", "neutral", "adversarial")
            methods = ("hybrid", "gated", "etc", "baseline")
            n_off_fixed = all_results[0].get("n_off_fixed", args.exp2_n_off)
            _plot_exp2_averaged(out_dir, all_results, n_off_fixed, kinds, methods)
            print(f"  [plot-only] exp2: regenerated from {len(all_results)} game JSONs", flush=True)

    # --- Exp 3 ---
    exp3_path = os.path.join(out_dir, "exp3_summary.json")
    if os.path.exists(exp3_path):
        with open(exp3_path) as f:
            data = json.load(f)
        _replot_exp3(out_dir, data)
        print(f"  [plot-only] exp3: regenerated from exp3_summary.json", flush=True)


def _replot_exp3(out_dir: str, data: dict) -> None:
    """Regenerate exp3 plot from saved summary JSON."""
    n_x_list = data["n_x_list"]
    nx_arr = np.array(n_x_list)
    n_off = data.get("n_off")
    n_off_per_ctx = data.get("n_off_per_ctx")
    d = data.get("d", "?")
    n_a = data.get("n_a", 4)
    n_b = data.get("n_b", 4)
    d_ctx = data.get("d_ctx", 3)

    m_tab = np.array([data["tfw_tab_mean"][str(nx)] for nx in n_x_list])
    m_ctx = np.array([data["tfw_ctx_mean"][str(nx)] for nx in n_x_list])
    m_gated = np.array([data["tfw_gated_ctx_mean"][str(nx)] for nx in n_x_list])
    m_etc = np.array([data["tfw_etc_ctx_mean"][str(nx)] for nx in n_x_list])
    m_online = np.array([data["tfw_online_ctx_mean"][str(nx)] for nx in n_x_list])

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.2))

    ax = axes[0]
    ax.plot(nx_arr, m_tab, "o-", color=COLORS["tabular"], lw=2.5, ms=8,
            label="Per-context tabular Hybrid-FMUCB")
    ax.plot(nx_arr, m_online, "x--", color=COLORS["baseline"], lw=2.5, ms=8,
            label="Contextual FMUCB (online only)")
    ax.plot(nx_arr, m_gated, "^--", color=COLORS["gated"], lw=2.5, ms=7,
            label="Contextual Gated-FMUCB")
    ax.plot(nx_arr, m_etc, "v:", color="#F4A261", lw=2.5, ms=7,
            label="Contextual ETC")
    ax.plot(nx_arr, m_ctx, "s-", color=COLORS["contextual"], lw=2.5, ms=8,
            label="Contextual Hybrid-FMUCB (Alg 2)")
    ax.set_xlabel(r"Number of contexts $|\mathcal{X}|$")
    ax.set_ylabel(r"$T_{f,w}$ (vs.\ true contextual $F^{fm}$)")
    ax.set_title(r"Follower error vs number of contexts")
    ax.legend(frameon=True, fancybox=True, shadow=True, fontsize=12)
    _style_axis(ax)

    ax2 = axes[1]
    ratio = m_tab / np.maximum(m_ctx, 1.0)
    ax2.plot(nx_arr, ratio, "D-", color="#E76F51", lw=2.5, ms=8,
             label="Empirical ratio")
    ax2.set_xlabel(r"Number of contexts $|\mathcal{X}|$")
    ax2.set_ylabel(r"$T_{f,w}^{\mathrm{tab}} / T_{f,w}^{\mathrm{ctx}}$")
    ax2.set_title(r"Sample complexity ratio")
    ax2.legend(frameon=True, fancybox=True, shadow=True)
    _style_axis(ax2)

    noff_label = (f"$N_{{\\mathrm{{off}}}}={n_off_per_ctx}\\cdot|\\mathcal{{X}}|$"
                  if n_off_per_ctx else f"$N_{{\\mathrm{{off}}}}={n_off}$")
    fig.suptitle(
        rf"Experiment 3: Contextual generalization ({noff_label}, "
        rf"$d={d}$, $|A|={n_a}$, $|B|={n_b}$, $d_{{\mathrm{{ctx}}}}={d_ctx}$)",
        fontsize=22, y=1.01,
    )
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "exp3_contextual_vs_tabular.png"),
                bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    try:
        sys.stdout.reconfigure(line_buffering=True)  # py3.7+: show prints immediately
    except (AttributeError, OSError):
        pass
    parser = argparse.ArgumentParser(
        description="Stackelberg bandit offline-online experiments"
    )
    parser.add_argument("--out-dir", type=str, default="figures")
    parser.add_argument("--seeds", type=int, default=48)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--n-a", type=int, default=4)
    parser.add_argument("--n-b", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=8000)
    parser.add_argument(
        "--gamma-exp3",
        type=float,
        default=0.01,
        help=r"EXP3 exploration $\gamma$ (smaller ⇒ more weight on leader actions with high empirical payoff; typical 0.05–0.01)",
    )
    parser.add_argument(
        "--confidence-delta",
        type=float,
        default=0.05,
        help="High-probability parameter for regression confidence sets in Alg. 1 (finite-sample bounds; unchanged from implementation default)",
    )
    parser.add_argument("--exp1-cum-n-off", type=int, default=None)
    parser.add_argument("--exp2-n-off", type=int, default=1000)
    parser.add_argument("--skip-learning-curves", action="store_true")
    parser.add_argument(
        "--learning-curves-only",
        action="store_true",
        help="Run only exp_optional_learning_curves.png (skips Exp 1–2; overrides --exp1-only / --exp2-only)",
    )
    parser.add_argument(
        "--exp1-only",
        action="store_true",
        help="Run only Experiment 1 (no Exp 2, no learning curves unless combined with --exp2-only)",
    )
    parser.add_argument(
        "--exp2-only",
        action="store_true",
        help="Run only Experiment 2 (no Exp 1, no learning curves unless combined with --exp1-only)",
    )
    parser.add_argument("--exp3", action="store_true")
    parser.add_argument("--exp3-n-off", type=int, default=1500)
    parser.add_argument("--exp3-noff-per-ctx", type=int, default=None,
                        help="If set, N_off = noff_per_ctx * |X| (scales offline data with number of contexts)")
    parser.add_argument(
        "--exp3-n-x-list",
        type=str,
        default="5,10,20,50",
        help="Comma-separated list of |X| values to sweep in Experiment 3",
    )
    parser.add_argument("--d-ctx", type=int, default=3,
                        help="Context embedding dimension for Kronecker feature map")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument(
        "--learning-curve-horizon",
        type=int,
        default=None,
        help="Learning-curve online length (default: same as --horizon; use e.g. 1500 for quick tests)",
    )
    parser.add_argument(
        "--learning-curve-seeds",
        type=int,
        default=None,
        help="Number of seeds for learning curves only (default: same as --seeds)",
    )
    parser.add_argument(
        "--learning-curve-n-off",
        type=int,
        default=1000,
        help="Offline size for learning-curve runs",
    )
    parser.add_argument(
        "--learning-curve-window",
        type=int,
        default=None,
        help="Rolling window length (default: max(50, horizon_lc//100))",
    )
    parser.add_argument(
        "--learning-curve-gamma-exp3",
        type=float,
        default=None,
        help="EXP3 gamma for learning curves only (default: same as --gamma-exp3). Use e.g. 0.005 for more visits to a* in LC plots.",
    )
    parser.add_argument(
        "--learning-curve-paper-profile",
        action="store_true",
        help="Learning curves: use longer online horizon (max 20k vs --horizon), cap gamma at 0.01, floor N_off at 3000 — same Alg 1, stronger global signal",
    )
    parser.add_argument(
        "--game-seed",
        type=int,
        default=None,
        help="If set, generate a random game instance instead of using the fixed 4x4 game. "
             "Different game-seeds produce different games for robustness testing.",
    )
    parser.add_argument(
        "--exp1-n-off-grid",
        type=str,
        default="0,100,500,1000,2000,5000",
        help="Comma-separated N_off grid for Experiment 1",
    )
    parser.add_argument(
        "--exp1-data-quality",
        type=str,
        default="adversarial",
        choices=["good", "neutral", "poor", "adversarial", "mixed"],
        help="Quality of offline data in Experiment 1 (default: adversarial)",
    )
    parser.add_argument(
        "--game-seeds",
        type=str,
        default=None,
        help="Comma-separated list of game seeds to average over (overrides --game-seed for multi-game runs)",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Skip experiment computation; regenerate plots from existing JSON summaries in --out-dir",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    seeds = [args.base_seed + i for i in range(args.seeds)]
    n_off_grid = [int(x) for x in args.exp1_n_off_grid.split(",")]
    game_seeds_list = [int(x) for x in args.game_seeds.split(",")] if args.game_seeds else None
    show_p = not args.no_progress

    # --plot-only: regenerate figures from saved JSONs, no computation
    if args.plot_only:
        _replot_from_json(args.out_dir, n_off_grid, game_seeds_list, args)
        print(f"[plot-only] Regenerated figures in {os.path.abspath(args.out_dir)}", flush=True)
        return

    if args.learning_curves_only:
        run_exp1 = False
        run_exp2 = False
        run_lc = not args.skip_learning_curves
    elif args.exp3 and not args.exp1_only and not args.exp2_only and not args.learning_curves_only:
        # --exp3 alone: run only exp3
        run_exp1 = False
        run_exp2 = False
        run_lc = False
    elif args.exp1_only and args.exp2_only:
        run_exp1 = True
        run_exp2 = True
        run_lc = False
    elif args.exp1_only:
        run_exp1 = True
        run_exp2 = False
        run_lc = False
    elif args.exp2_only:
        run_exp1 = False
        run_exp2 = True
        run_lc = False
    else:
        run_exp1 = True
        run_exp2 = True
        run_lc = not args.skip_learning_curves

    if show_p:
        parts: List[str] = []
        if run_exp1:
            parts.append("exp1")
        if run_exp2:
            parts.append("exp2")
        if run_lc:
            parts.append("learning curves")
        if args.exp3:
            parts.append("exp3")
        msg = (
            f"Starting: {', '.join(parts)} | seeds={args.seeds} horizon={args.horizon} "
            f"n_a={args.n_a} n_b={args.n_b} — exp1: {len(n_off_grid)} N_off cells × "
            f"2 rollouts × {args.horizon} steps (per-round tqdm when progress is on)."
        )
        print(msg, file=sys.stderr, flush=True)

    if run_exp1:
        experiment1(
            args.out_dir,
            seeds,
            args.n_a,
            args.n_b,
            args.horizon,
            args.gamma_exp3,
            n_off_grid,
            cum_n_off=args.exp1_cum_n_off,
            progress=show_p,
            delta=args.confidence_delta,
            game_seed=args.game_seed,
            data_quality=args.exp1_data_quality,
            game_seeds=game_seeds_list,
        )
    if run_exp2:
        experiment2(
            args.out_dir,
            seeds,
            args.n_a,
            args.n_b,
            args.horizon,
            args.gamma_exp3,
            args.exp2_n_off,
            progress=show_p,
            delta=args.confidence_delta,
            game_seed=args.game_seed,
            game_seeds=game_seeds_list,
        )
    if run_lc:
        lc_horizon = (
            args.learning_curve_horizon
            if args.learning_curve_horizon is not None
            else args.horizon
        )
        lc_n = (
            args.learning_curve_seeds
            if args.learning_curve_seeds is not None
            else args.seeds
        )
        lc_seeds = [args.base_seed + i for i in range(lc_n)]
        lc_gamma = (
            args.learning_curve_gamma_exp3
            if args.learning_curve_gamma_exp3 is not None
            else args.gamma_exp3
        )
        lc_n_off = args.learning_curve_n_off
        lc_caption = ""
        if args.learning_curve_paper_profile:
            if args.learning_curve_horizon is None:
                lc_horizon = max(20000, args.horizon)
            lc_gamma = min(lc_gamma, 0.01)
            lc_n_off = max(lc_n_off, 3000)
            lc_caption = (
                rf" — paper-profile: $T={lc_horizon}$, "
                rf"$\gamma={lc_gamma}$, $N_{{\mathrm{{off}}}}={lc_n_off}$"
            )
        learning_curve_figure(
            args.out_dir,
            lc_seeds,
            args.n_a,
            args.n_b,
            lc_horizon,
            lc_gamma,
            n_off=lc_n_off,
            progress=show_p,
            rolling_window=args.learning_curve_window,
            delta=args.confidence_delta,
            caption_suffix=lc_caption,
        )
    if args.exp3:
        n_x_list = [int(x) for x in args.exp3_n_x_list.split(",")]
        experiment3_contextual(args.out_dir, seeds, args.horizon, args.gamma_exp3,
                               args.exp3_n_off, n_x_list, args.n_a, args.n_b,
                               d_ctx=args.d_ctx, progress=show_p,
                               n_off_per_ctx=args.exp3_noff_per_ctx)
    print(f"Wrote figures to {os.path.abspath(args.out_dir)}", flush=True)


if __name__ == "__main__":
    main()