"""Focused unit tests for pure/deterministic functions in basics_cdss.

These tests import the real vendored package under ``src/`` and exercise the
metric routines on tiny hand-made inputs where the correct answer is known by
hand. They avoid model training, network, large data, and shap so they run with
only numpy/pandas/scipy/scikit-learn installed.

Run:
    python -m pytest tests/test_units.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Make the vendored package importable without installation.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from basics_cdss.metrics.calibration import (  # noqa: E402
    brier_score,
    expected_calibration_error,
    reliability_curve,
)
from basics_cdss.metrics.coverage_risk import (  # noqa: E402
    abstention_rate,
    area_under_risk_coverage_curve,
    coverage_risk_curve,
)
from basics_cdss.metrics.harm import (  # noqa: E402
    asymmetric_cost_matrix,
    escalation_failure_analysis,
    weighted_harm_loss,
)
from basics_cdss.metrics.performance import (  # noqa: E402
    compute_performance_metrics,
    confusion_matrix,
)


# --------------------------------------------------------------------------- #
# performance.py
# --------------------------------------------------------------------------- #
def test_confusion_matrix_counts_and_prevalence():
    """Hand-counted 2x2 confusion matrix and prevalence."""
    y_true = np.array([0, 0, 1, 1, 1])
    y_pred = np.array([0, 1, 1, 1, 0])
    cm = confusion_matrix(y_true, y_pred)
    # TN=1 (idx0), FP=1 (idx1), TP=2 (idx2,3), FN=1 (idx4)
    assert (cm.tn, cm.fp, cm.fn, cm.tp) == (1, 1, 1, 2)
    assert cm.total == 5
    assert cm.prevalence == pytest.approx(3 / 5)


def test_performance_metrics_perfect_classifier():
    """A perfect classifier yields accuracy/precision/recall/F1 == 1 and 0 error rates."""
    y_true = np.array([0, 1, 0, 1, 1])
    y_pred = y_true.copy()
    m = compute_performance_metrics(y_true, y_pred)
    assert m.accuracy == pytest.approx(1.0)
    assert m.precision == pytest.approx(1.0)
    assert m.recall == pytest.approx(1.0)
    assert m.specificity == pytest.approx(1.0)
    assert m.f1_score == pytest.approx(1.0)
    assert m.fpr == pytest.approx(0.0)
    assert m.fnr == pytest.approx(0.0)


def test_performance_metrics_known_values():
    """Precision/recall/specificity match values derived by hand from the CM."""
    y_true = np.array([0, 0, 1, 1, 1])
    y_pred = np.array([0, 1, 1, 1, 0])
    m = compute_performance_metrics(y_true, y_pred)
    # TP=2 FP=1 FN=1 TN=1
    assert m.precision == pytest.approx(2 / 3)  # TP/(TP+FP)
    assert m.recall == pytest.approx(2 / 3)  # TP/(TP+FN)
    assert m.specificity == pytest.approx(1 / 2)  # TN/(TN+FP)
    assert m.accuracy == pytest.approx(3 / 5)


# --------------------------------------------------------------------------- #
# calibration.py
# --------------------------------------------------------------------------- #
def test_brier_score_known_value():
    """Brier = mean((p - y)^2) on a hand example."""
    y_true = np.array([1, 0, 1])
    y_prob = np.array([0.9, 0.1, 0.8])
    expected = np.mean((y_prob - y_true) ** 2)
    assert brier_score(y_true, y_prob) == pytest.approx(expected)


def test_brier_perfect_is_zero_and_bounds():
    """Perfect predictions give Brier 0; worst-case (confident & wrong) gives 1."""
    assert brier_score(np.array([1, 0]), np.array([1.0, 0.0])) == pytest.approx(0.0)
    assert brier_score(np.array([1, 0]), np.array([0.0, 1.0])) == pytest.approx(1.0)


def test_ece_perfectly_calibrated_is_zero():
    """When confidence equals empirical accuracy in every populated bin, ECE == 0."""
    # 10 samples at p=0.0 (all label 0) and 10 at p=1.0 (all label 1):
    # each bin's confidence matches its accuracy exactly.
    y_prob = np.array([0.0] * 10 + [1.0] * 10)
    y_true = np.array([0] * 10 + [1] * 10)
    assert expected_calibration_error(y_true, y_prob, n_bins=10) == pytest.approx(0.0)


def test_ece_bounds_and_empty():
    """ECE is in [0,1] and empty input returns 0."""
    rng = np.random.default_rng(42)
    y_prob = rng.random(50)
    y_true = (rng.random(50) > 0.5).astype(int)
    ece = expected_calibration_error(y_true, y_prob, n_bins=10)
    assert 0.0 <= ece <= 1.0
    assert expected_calibration_error(np.array([]), np.array([])) == 0.0


def test_reliability_curve_shapes_consistent():
    """All three reliability-curve arrays share the same (<= n_bins) length, counts sum to N."""
    rng = np.random.default_rng(42)
    y_prob = rng.random(40)
    y_true = (rng.random(40) > 0.5).astype(int)
    confs, accs, counts = reliability_curve(y_true, y_prob, n_bins=5)
    assert len(confs) == len(accs) == len(counts)
    assert len(confs) <= 5
    assert counts.sum() == 40  # every sample lands in exactly one bin


# --------------------------------------------------------------------------- #
# coverage_risk.py
# --------------------------------------------------------------------------- #
def test_abstention_rate_known_fraction():
    y_prob = np.array([0.9, 0.8, 0.3, 0.7, 0.2])
    # below 0.5: 0.3 and 0.2 -> 2/5
    assert abstention_rate(y_prob, threshold=0.5) == pytest.approx(2 / 5)
    assert abstention_rate(np.array([]), 0.5) == 0.0


def test_coverage_risk_curve_shapes_and_bounds():
    y_true = np.array([1, 1, 0, 1, 0])
    y_prob = np.array([0.9, 0.8, 0.3, 0.7, 0.2])
    coverages, risks, thresholds = coverage_risk_curve(y_true, y_prob, n_thresholds=20)
    assert len(coverages) == len(risks) == len(thresholds) == 20
    # coverage is a fraction; at threshold 0 everything is accepted.
    assert np.nanmax(coverages) == pytest.approx(1.0)
    assert np.all((coverages >= 0.0) & (coverages <= 1.0))


def test_aurc_trapezoid_known_value():
    """AURC is the trapezoidal area under risk vs coverage."""
    coverages = np.array([0.0, 0.5, 1.0])
    risks = np.array([0.0, 0.1, 0.2])
    # trapezoid: 0.5*(0+0.1)/2 + 0.5*(0.1+0.2)/2 = 0.025 + 0.075 = 0.1
    assert area_under_risk_coverage_curve(coverages, risks) == pytest.approx(0.1)


# --------------------------------------------------------------------------- #
# harm.py
# --------------------------------------------------------------------------- #
def test_weighted_harm_loss_tier_weighting():
    """High-tier error must count 10x a low-tier error per default weights."""
    y_true = np.array([1, 0])
    # One high-risk error only:
    loss_high = weighted_harm_loss(np.array([1, 0]), np.array([0, 0]),
                                   np.array(["high", "low"]))
    # One low-risk error only:
    loss_low = weighted_harm_loss(np.array([0, 1]), np.array([0, 0]),
                                  np.array(["high", "low"]))
    assert loss_high == pytest.approx(10.0 / 2)  # weight 10, N=2
    assert loss_low == pytest.approx(1.0 / 2)  # weight 1, N=2
    assert loss_high == pytest.approx(10 * loss_low)
    # No errors -> zero loss.
    assert weighted_harm_loss(y_true, y_true, np.array(["high", "low"])) == 0.0


def test_escalation_failure_analysis_counts():
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([0, 1, 1, 0])  # missed 1 high-risk, over-escalated 1 low-risk
    risk_tiers = np.array(["high", "high", "low", "low"])
    res = escalation_failure_analysis(y_true, y_pred, risk_tiers)
    assert res["escalation_failures"] == 1
    assert res["false_escalations"] == 1
    assert res["high_risk_samples"] == 2
    assert res["low_risk_samples"] == 2


def test_asymmetric_cost_matrix_fn_dominates_fp():
    """With cost_fn >> cost_fp, the average cost reflects only the FN here."""
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([1, 1, 0, 0])  # 1 FP (idx1), 1 FN (idx2)
    cost = asymmetric_cost_matrix(y_true, y_pred, cost_fn=10.0, cost_fp=1.0)
    # (10*1 FN + 1*1 FP) / 4 = 11/4
    assert cost == pytest.approx(11.0 / 4)
