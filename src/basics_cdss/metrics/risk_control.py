"""Distribution-free risk control for a clinically meaningful risk.

Conformal coverage controls the probability that the true label is inside the prediction
set. That is not the quantity a safety argument needs: a decision-support system is unsafe
when it misses a deteriorating patient, so the risk to control is a false-negative rate on
the high-urgency stratum, not set membership.

Learn-then-Test (Angelopoulos, Bates, Candes, Jordan and Lei, 2021) turns the choice of an
operating parameter lambda into multiple hypothesis testing. For each candidate lambda we
test

    H_lambda :  R(lambda) > alpha_risk        ("lambda is unsafe")

on held-out data and keep the lambdas whose null is rejected at level delta with a
family-wise correction. Any returned lambda then satisfies

    P( R(lambda) <= alpha_risk ) >= 1 - delta,

with no distributional assumption beyond i.i.d. calibration draws and a bounded risk. Two
p-values are provided: Hoeffding's, and Bentkus's (tighter for bounded losses, used in the
original paper). Fixed-sequence testing along a monotone lambda grid is offered because it
spends no alpha on correction when the risk is monotone in lambda, which it is here --
lowering the alert threshold can only decrease a false-negative rate.

The empirical check the manuscript reports is the one this guarantee invites: repeat the
whole select-then-deploy procedure over many simulated deployments and count how often the
selected lambda's true risk exceeds alpha_risk. That rate must not exceed delta.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Sequence

import numpy as np
from scipy.optimize import brentq
from scipy.stats import binom

__all__ = [
    "RiskControlResult",
    "hoeffding_p_value",
    "bentkus_p_value",
    "learn_then_test",
    "fnr_at_threshold",
]


@dataclass
class RiskControlResult:
    """Outcome of one Learn-then-Test selection."""

    lambda_hat: Optional[float]
    valid_lambdas: np.ndarray
    risks: np.ndarray
    p_values: np.ndarray
    alpha_risk: float
    delta: float
    method: str
    n_cal: int

    @property
    def selected(self) -> bool:
        return self.lambda_hat is not None


def hoeffding_p_value(risk_hat: float, n: int, alpha_risk: float) -> float:
    """Hoeffding p-value for H: R > alpha_risk, losses in [0, 1]."""
    if risk_hat >= alpha_risk:
        return 1.0
    return float(np.exp(-2 * n * (alpha_risk - risk_hat) ** 2))


def bentkus_p_value(risk_hat: float, n: int, alpha_risk: float) -> float:
    """Bentkus p-value (Angelopoulos et al., 2021, Eq. 7) for a [0, 1]-bounded loss."""
    if risk_hat >= alpha_risk:
        return 1.0
    k = int(np.ceil(n * risk_hat))
    return float(min(1.0, np.e * binom.cdf(k, n, alpha_risk)))


def _combine(p_hoeff: float, p_bent: float, method: str) -> float:
    if method == "hoeffding":
        return p_hoeff
    if method == "bentkus":
        return p_bent
    return min(p_hoeff, p_bent)  # "hb": the minimum of the two, as in the LTT paper


def learn_then_test(lambdas: Sequence[float], risk_fn: Callable[[float], float], n_cal: int,
                    alpha_risk: float = 0.05, delta: float = 0.10,
                    method: str = "hb", correction: str = "fixed_sequence",
                    pick: str = "most_permissive") -> RiskControlResult:
    """Select operating parameters whose risk is certified below ``alpha_risk``.

    Args:
        lambdas: candidate values, ordered from safest to most permissive. With
            ``correction="fixed_sequence"`` the order is part of the procedure: testing
            stops at the first non-rejection, which is why the ordering must be the one in
            which risk is non-decreasing.
        risk_fn: empirical risk on the calibration split for a given lambda, in [0, 1].
        n_cal: number of calibration points behind ``risk_fn``.
        alpha_risk: the risk level to certify (e.g. 0.05 for a 5% false-negative rate).
        delta: error probability of the certificate.
        correction: "fixed_sequence" or "bonferroni".
        pick: "most_permissive" returns the last surviving lambda (least conservative
            operating point that is still certified); "safest" returns the first.
    """
    lambdas = np.asarray(list(lambdas), dtype=float)
    risks = np.array([float(risk_fn(lam)) for lam in lambdas])
    p_vals = np.array([
        _combine(hoeffding_p_value(r, n_cal, alpha_risk),
                 bentkus_p_value(r, n_cal, alpha_risk), method)
        for r in risks
    ])

    if correction == "bonferroni":
        threshold = delta / len(lambdas)
        valid_mask = p_vals <= threshold
    elif correction == "fixed_sequence":
        valid_mask = np.zeros(len(lambdas), dtype=bool)
        for i, p in enumerate(p_vals):
            if p <= delta:
                valid_mask[i] = True
            else:
                break  # fixed-sequence testing stops at the first failure
    else:
        raise ValueError(f"unknown correction: {correction}")

    valid = lambdas[valid_mask]
    if len(valid) == 0:
        lambda_hat = None
    else:
        lambda_hat = float(valid[-1] if pick == "most_permissive" else valid[0])

    return RiskControlResult(lambda_hat=lambda_hat, valid_lambdas=valid, risks=risks,
                             p_values=p_vals, alpha_risk=alpha_risk, delta=delta,
                             method=method, n_cal=n_cal)


def fnr_at_threshold(y_true: np.ndarray, y_prob: np.ndarray, lam: float,
                     mask: Optional[np.ndarray] = None) -> float:
    """False-negative rate of the alert rule ``p >= lam``, optionally on a stratum.

    Returns 0.0 when the stratum contains no positive case, which is the convention the
    risk-control literature uses for an empty-risk sample; the caller is responsible for
    ensuring the calibration split has positives (the drivers here assert it).
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=float)
    if mask is not None:
        y_true, y_prob = y_true[mask], y_prob[mask]
    pos = y_true == 1
    if pos.sum() == 0:
        return 0.0
    return float((y_prob[pos] < lam).mean())
