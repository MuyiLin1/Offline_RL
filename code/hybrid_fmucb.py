"""
Tabular Hybrid-FMUCB helpers aligned with:
  "Hybrid Offline-Online Follower Manipulation in General-Sum Stackelberg Games"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

LeaderFeasibility = Literal["pessimistic"]

import numpy as np


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

def worst_response_leader_row(mu_l_row: np.ndarray) -> int:
    """argmin_b mu_l(a, b) for a fixed row (ties -> smallest index)."""
    return int(np.argmin(mu_l_row))


def build_rule_F(a_star: int, b_star: int, mu_l_hat: np.ndarray) -> np.ndarray:
    """
    Tabular rule F_{a*,b*} (paper Sec. 2 tabular special case):
      F(a*) = b*
      F(a)  = argmin_b mu_l_hat(a, b)   for a != a*
    """
    n_a = mu_l_hat.shape[0]
    F = np.zeros(n_a, dtype=np.int32)
    for a in range(n_a):
        if a == a_star:
            F[a] = b_star
        else:
            F[a] = worst_response_leader_row(mu_l_hat[a].astype(np.float64))
    return F


def manipulation_contrast(mu_l: np.ndarray, F: np.ndarray, a_star: int) -> float:
    """
    Delta_{F,a*}(g) = g(a*, F(a*)) - max_{a != a*} g(a, F(a))  [Eq. (2)].
    """
    n_a = mu_l.shape[0]
    left = float(mu_l[a_star, F[a_star]])
    others = [float(mu_l[a, F[a]]) for a in range(n_a) if a != a_star]
    if not others:
        return left
    return left - max(others)


# ---------------------------------------------------------------------------
# Deviation 2: Regression-based confidence sets  (Alg 1 lines 5-8)
# ---------------------------------------------------------------------------

def regression_confidence_radius(
    n_visits: int,
    t: int,
    n_a: int,
    n_b: int,
    delta: float = 0.05,
) -> float:
    """
    Per-cell confidence half-width for regression confidence sets.

    For tabular [0,1]-bounded Bernoulli rewards the ERM is the empirical mean.
    A union-bounded Hoeffding inequality over |A|*|B| cells and t^2 rounds gives:

        w_t(a,b) = sqrt( log(2 * |A| * |B| * t^2 / delta) / (2 * N_t(a,b)) )

    This is the tabular special case of the regression confidence set radius
    in Algorithm 1 (paper Sec. 2 and Appendix D).
    """
    log_term = np.log(2.0 * n_a * n_b * max(1, t) ** 2 / delta)
    n = max(1, int(n_visits))
    return float(np.sqrt(log_term / (2.0 * n)))


# ---------------------------------------------------------------------------
# Pessimistic feasibility using regression confidence sets
# ---------------------------------------------------------------------------

def manipulation_contrast_leader_pessimistic(
    sum_r_l: np.ndarray,
    n_visits: np.ndarray,
    F: np.ndarray,
    a_star: int,
    t: int,
    n_a: int,
    n_b: int,
    delta: float = 0.05,
) -> float:
    """
    Pessimistic lower bound on Delta_{F,a*}(mu_l) using regression confidence sets:

        LCB_l(a*, F(a*)) - max_{a'!=a*} UCB_l(a', F(a'))

    where
        LCB_l(a,b) = mu_hat(a,b) - w_t(a,b)
        UCB_l(a,b) = mu_hat(a,b) + w_t(a,b)

    This is the tabular implementation of
        inf_{g in G_{l,t}} Delta_{F,a*}(g) > 0   (Alg 1 line 6).
    """
    at, bt = a_star, int(F[a_star])
    nt = int(n_visits[at, bt])
    mean_t = float(sum_r_l[at, bt]) / max(1, nt)
    w_target = regression_confidence_radius(nt, t, n_a, n_b, delta)
    lcb_target = mean_t - w_target

    others: list[float] = []
    for a in range(n_a):
        if a == a_star:
            continue
        b = int(F[a])
        n_c = int(n_visits[a, b])
        mean_c = float(sum_r_l[a, b]) / max(1, n_c)
        w_c = regression_confidence_radius(n_c, t, n_a, n_b, delta)
        others.append(mean_c + w_c)

    if not others:
        return lcb_target
    return float(lcb_target - max(others))


# ---------------------------------------------------------------------------
# Deviation 3: Explicit C_man coefficient (Eq. 3) and offline certification
# ---------------------------------------------------------------------------

def compute_c_man(
    F: np.ndarray,
    a_star: int,
    n_a: int,
    n_b: int,
    nu_counts: np.ndarray,
) -> float:
    """
    Qualified-manipulation transfer coefficient C_man(F, a*) from Eq. (3):

        C_man(F, a*) = max(0,
            sup_{h in H}  Delta_{F,a*}(h)
            / sqrt(E_{(a,b)~nu}[h(a,b)^2])
        )

    For the tabular function class, the sup is achieved via Cauchy-Schwarz.
    The contrast Delta_{F,a*}(h) = h(a*, F(a*)) - max_{a!=a*} h(a, F(a))
    depends on at most |A| cells of the form (a, F(a)).  Write
    nu_a := nu(a, F(a)) / nu_total for these cells.  The optimal h
    concentrates mass on the target cell and one competitor, giving:

        C_man = sqrt(1/nu(a*, F(a*)) + 1/nu(a', F(a')))

    where a' is the competitor with least coverage.  We use the
    worst-case (minimum coverage) competitor for a tight bound.

    Parameters
    ----------
    F         : response rule, shape (n_a,)
    a_star    : target leader action
    n_a, n_b  : action space sizes
    nu_counts : offline visit counts proxy for nu, shape (n_a, n_b)
    """
    nu_total = float(nu_counts.sum())
    if nu_total <= 0.0:
        return float("inf")

    # Target cell coverage
    nu_target = float(nu_counts[a_star, int(F[a_star])])
    if nu_target <= 0.0:
        return float("inf")

    # Find the competitor with minimum coverage (worst case for C_man)
    min_nu_other = float("inf")
    for a in range(n_a):
        if a != a_star:
            nu_a = float(nu_counts[a, int(F[a])])
            if nu_a <= 0.0:
                return float("inf")
            if nu_a < min_nu_other:
                min_nu_other = nu_a

    if min_nu_other == float("inf"):
        # Only one leader action; no competitor
        return 0.0

    # C_man = sqrt(nu_total / nu_target + nu_total / min_nu_other)
    c_man = float(np.sqrt(nu_total / nu_target + nu_total / min_nu_other))
    return float(max(0.0, c_man))


def certify_offline(
    F: np.ndarray,
    a_star: int,
    mu_l_hat: np.ndarray,
    mu_l_true: np.ndarray,
    nu_counts: np.ndarray,
    delta3: float,
    n_a: int,
    n_b: int,
) -> Tuple[bool, float, float]:
    """
    Check Theorem 1 condition 4:
        C_man(F, a*) * sqrt(E_nu[(mu_l_hat - mu_l)^2]) <= delta3 / 4

    Returns (certified, lhs, threshold).
    'certified' is True when the offline data alone guarantees (F, a*) is a
    feasible manipulation before any online interaction (paper Sec. 2, Thm 1).

    Parameters
    ----------
    mu_l_hat  : offline leader reward estimate (ERM on D_off)
    mu_l_true : ground-truth mu_l
    nu_counts : offline visit counts, shape (n_a, n_b)
    delta3    : true manipulation contrast Delta_{F,a*}(mu_l)
    """
    c_man = compute_c_man(F, a_star, n_a, n_b, nu_counts)
    nu_total = float(nu_counts.sum())
    if nu_total <= 0:
        return False, float("inf"), delta3 / 4.0

    nu_prob = nu_counts.astype(np.float64) / nu_total
    mse_offline = float(np.sum(nu_prob * (mu_l_hat - mu_l_true) ** 2))
    lhs = c_man * float(np.sqrt(mse_offline))
    threshold = delta3 / 4.0
    return (lhs <= threshold), lhs, threshold


# ---------------------------------------------------------------------------
# Core algorithm: pooled mean and Hybrid-FMUCB pick
# ---------------------------------------------------------------------------

def pooled_mean(sum_r: np.ndarray, n_visits: np.ndarray) -> np.ndarray:
    """Entrywise sum_r / max(1, n_visits)  (paper Sec. 2 aggregated estimator)."""
    n = np.maximum(1, n_visits.astype(np.float64))
    return sum_r.astype(np.float64) / n


def select_best_feasible_manipulation_rule(
    sum_r_l: np.ndarray,
    sum_r_f: np.ndarray,
    n_visits: np.ndarray,
    t: int,
    n_a: int,
    n_b: int,
    delta: float,
    eps: float = 1e-9,
) -> Tuple[Optional[np.ndarray], int, int, float]:
    """
    Algorithm 1, lines 6--7 (tabular): among certifiably feasible (F, a*) in M_t,
    return argmax of optimistic follower value mu_hat_f(a*,b*) + w_t(a*,b*).

    Returns (F, a_star, b_star, best_ucb_f). If M_t is empty, (None, -1, -1, -inf).
    """
    mu_l_hat = pooled_mean(sum_r_l, n_visits)
    best_ucb_f = -np.inf
    best_F: Optional[np.ndarray] = None
    best_a, best_b = -1, -1

    for a_star in range(n_a):
        for b_star in range(n_b):
            F = build_rule_F(a_star, b_star, mu_l_hat)
            pess = manipulation_contrast_leader_pessimistic(
                sum_r_l, n_visits, F, a_star, t, n_a, n_b, delta
            )
            if pess <= eps:
                continue
            n_fb = max(1, int(n_visits[a_star, b_star]))
            mean_f = float(sum_r_f[a_star, b_star]) / n_fb
            w_f = regression_confidence_radius(n_fb, t, n_a, n_b, delta)
            ucb_f = mean_f + w_f
            if ucb_f > best_ucb_f:
                best_ucb_f = ucb_f
                best_F = F.copy()
                best_a, best_b = a_star, b_star

    if best_F is None:
        return None, -1, -1, float(-np.inf)
    return best_F, best_a, best_b, best_ucb_f


def offline_candidate_manipulation(
    sum_r_l: np.ndarray,
    sum_r_f: np.ndarray,
    n_visits_offline: np.ndarray,
    n_a: int,
    n_b: int,
    delta: float,
    eps: float = 1e-9,
    t_init: int = 1,
) -> Tuple[Optional[np.ndarray], int, int, float]:
    """
    Algorithm 1, lines 1--2: offline-only confidence sets G_{l,0}, G_{f,0} from D_off,
    then compute (F_0, a_0) maximizing follower optimistic value subject to feasibility.

    Uses pooled statistics from offline data only; time index ``t_init`` (default 1)
    matches the first online round before any new sample (same radii as first call to
    ``hybrid_fmucb_pick`` when online counts are still zero).
    """
    return select_best_feasible_manipulation_rule(
        sum_r_l,
        sum_r_f,
        n_visits_offline,
        t_init,
        n_a,
        n_b,
        delta,
        eps,
    )


def theorem1_offline_transfer_check(
    mu_l_true: np.ndarray,
    mu_f: np.ndarray,
    n_visits_offline: np.ndarray,
    sum_r_l_offline: np.ndarray,
    n_a: int,
    n_b: int,
) -> Tuple[bool, float, float, float]:
    """
    Theorem 1, condition 4: C_man * sqrt(E_nu[(hat_mu_l - mu_l)^2]) <= Delta_3 / 4
    for the true best manipulation (F^fm, a^fm).

    Returns (ok, lhs, threshold, delta3). If no qualified manipulation exists (delta3<=0),
    returns (False, nan, nan, delta3).
    """
    F_fm, a_fm = true_best_manipulation(mu_l_true, mu_f)
    delta3 = manipulation_contrast(mu_l_true, F_fm, a_fm)
    if delta3 <= 1e-12:
        return False, float("nan"), float("nan"), float(delta3)
    mu_l_hat = pooled_mean(sum_r_l_offline, n_visits_offline)
    ok, lhs, thr = certify_offline(
        F_fm,
        a_fm,
        mu_l_hat,
        mu_l_true,
        n_visits_offline.astype(np.float64),
        float(delta3),
        n_a,
        n_b,
    )
    return ok, lhs, thr, float(delta3)


def hybrid_fmucb_pick(
    a_t: int,
    n_visits: np.ndarray,
    sum_r_f: np.ndarray,
    sum_r_l: np.ndarray,
    t: int,
    n_a: int,
    n_b: int,
    rng: np.random.Generator,
    delta: float = 0.05,
    eps: float = 1e-9,
    *,
    follower_elim_eps: Optional[float] = None,
    leader_feasibility: LeaderFeasibility = "pessimistic",
    feasibility_margin: float = 0.0,
) -> Tuple[int, bool]:
    """
    Tabular Hybrid-FMUCB step (Algorithm 1, tabular reduction).

    Feasibility (pessimistic, Alg 1 line 6):
        inf_{g in G_{l,t}} Delta_{F,a*}(g) > 0
        implemented as: LCB_l(a*, F(a*)) - max_{a'!=a*} UCB_l(a', F(a')) > eps
        using regression_confidence_radius (calibrated to Bernoulli rewards).

    Follower objective (optimistic, Alg 1 line 7):
        sup_{gf in G_{f,t}} gf(a*, F(a*))
        = mu_hat_f(a*, b*) + w_t(a*, b*)

    Returns (b_t, fallback_used).
    fallback_used=True when M_t={} and UCB best-response is played instead.

    Optional kwargs match ``experiment_common`` / older scripts: only pessimistic
    leader feasibility is implemented; ``feasibility_margin`` tightens the
    feasibility threshold additively on ``eps``.
    """
    if leader_feasibility != "pessimistic":
        raise ValueError(
            "only leader_feasibility='pessimistic' is implemented "
            f"(got {leader_feasibility!r})"
        )
    eps_base = follower_elim_eps if follower_elim_eps is not None else eps
    eps_eff = float(eps_base) + float(feasibility_margin)
    best_F, _, _, _ = select_best_feasible_manipulation_rule(
        sum_r_l, sum_r_f, n_visits, t, n_a, n_b, delta, eps_eff
    )

    if best_F is None:
        # Deviation 4 fallback: UCB best-response on mu_f(a_t, .)
        b_fb = _ucb_best_response_row(a_t, n_visits, sum_r_f, t, n_a, n_b, delta, rng)
        return b_fb, True

    return int(best_F[a_t]), False


def _ucb_best_response_row(
    a: int,
    n_visits: np.ndarray,
    sum_r: np.ndarray,
    t: int,
    n_a: int,
    n_b: int,
    delta: float,
    rng: np.random.Generator,
) -> int:
    """UCB best-response on mu_f(a, .) -- fallback when M_t = {}."""
    n_b_local = sum_r.shape[1]
    ucb = np.array([
        float(sum_r[a, b]) / max(1, int(n_visits[a, b]))
        + regression_confidence_radius(int(n_visits[a, b]), t, n_a, n_b, delta)
        for b in range(n_b_local)
    ])
    maxv = ucb.max()
    cands = np.flatnonzero(np.isclose(ucb, maxv))
    return int(rng.choice(cands))


# ---------------------------------------------------------------------------
# Ground-truth best manipulation (for T_{f,w} evaluation)
# ---------------------------------------------------------------------------

def true_best_manipulation(
    mu_l: np.ndarray,
    mu_f: np.ndarray,
    eps: float = 1e-9,
) -> Tuple[np.ndarray, int]:
    """
    Among all qualified manipulations (Delta_{F,a*}(mu_l) > 0), return
    (F^fm, a^fm) that maximises mu_f(a^fm, F^fm(a^fm)).

    Falls back to myopic BR mapping if no manipulation is qualified.
    """
    n_a, n_b = mu_f.shape
    best_val = -np.inf
    best_F: Optional[np.ndarray] = None
    best_a_fm = 0

    for a_star in range(n_a):
        for b_star in range(n_b):
            F = build_rule_F(a_star, b_star, mu_l)
            if manipulation_contrast(mu_l, F, a_star) <= eps:
                continue
            v = float(mu_f[a_star, b_star])
            if v > best_val:
                best_val = v
                best_F = F.copy()
                best_a_fm = a_star

    if best_F is None:
        F_br = np.array(
            [int(np.argmax(mu_f[a])) for a in range(n_a)], dtype=np.int32
        )
        return F_br, int(np.argmax([mu_f[a, F_br[a]] for a in range(n_a)]))

    return best_F, best_a_fm


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class OfflineRewardStats:
    """Pooled offline counts and reward sums for both players (paper Sec. 2)."""
    n_visits: np.ndarray   # shape (n_a, n_b), dtype int64
    sum_r_f: np.ndarray    # shape (n_a, n_b)
    sum_r_l: np.ndarray    # shape (n_a, n_b)