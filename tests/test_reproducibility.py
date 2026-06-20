"""Reproducibility unit tests for the Beyond-Accuracy safety-evaluation framework.

These tests lock the verified key results of the seeded (seed = 42) pipeline so
that any regression -- in the synthetic-cohort generator, the metric routines,
or the governance verdict -- is caught immediately.

The expected values are the ACTUAL committed numbers in results/eval_results.json
and results/*.csv (not invented). Tests read those committed artifacts and assert
the locked properties with pytest.approx tolerances.

A separate, opt-in test (marked ``slow``) re-runs run_all.py end to end and
asserts the regenerated metrics match the committed JSON to tolerance. It is
skipped by default to keep the default suite fast and dependency-light; enable it
with ``pytest -m slow``.

Run:
    pytest tests/test_reproducibility.py
    pytest -m slow tests/test_reproducibility.py     # include the live re-run
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
EVAL_JSON = RESULTS_DIR / "eval_results.json"
METRICS_BY_TIER_CSV = RESULTS_DIR / "metrics_by_tier.csv"
GOVERNANCE_CSV = RESULTS_DIR / "governance_gates.csv"


# --------------------------------------------------------------------------- #
# Fixtures: load the committed reproducibility artifacts.
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def results():
    """The committed eval_results.json (source of truth for locked values)."""
    assert EVAL_JSON.is_file(), f"missing committed artifact: {EVAL_JSON}"
    return json.loads(EVAL_JSON.read_text())


@pytest.fixture(scope="module")
def tier_rows():
    """Parse results/metrics_by_tier.csv into {tier: {col: float}}."""
    assert METRICS_BY_TIER_CSV.is_file(), f"missing artifact: {METRICS_BY_TIER_CSV}"
    import csv

    rows = {}
    with METRICS_BY_TIER_CSV.open(newline="") as fh:
        for r in csv.DictReader(fh):
            rows[r["tier"]] = {
                "ece": float(r["ece"]),
                "aurc": float(r["aurc"]),
                "harm_framework": float(r["harm_framework"]),
                "harm_baseline": float(r["harm_baseline"]),
            }
    return rows


# --------------------------------------------------------------------------- #
# 1. Governance verdict is locked: "Conditional" at G = 0.60 (3/5 gates pass).
# --------------------------------------------------------------------------- #
def test_governance_verdict_conditional(results):
    gov = results["governance"]
    assert gov["verdict"] == "Conditional"
    assert gov["score"] == pytest.approx(0.60, abs=1e-9)
    assert gov["n_passed"] == 3
    assert len(gov["gates"]) == 5
    assert sum(bool(v) for v in gov["gates"].values()) == 3


def test_governance_gate_pattern(results):
    """The exact pass/fail pattern of the five gates is part of the verdict."""
    gates = results["governance"]["gates"]
    assert gates["calibration_high_tier"] is False
    assert gates["coverage_risk_high_tier"] is True
    assert gates["harm_weighted_loss"] is False
    assert gates["conformal_coverage"] is True
    assert gates["explanation_stability"] is True
    assert set(results["governance"]["failed_gates"]) == {
        "calibration_high_tier",
        "harm_weighted_loss",
    }


def test_governance_csv_matches_json(results):
    """The governance CSV must agree with the JSON verdict/score."""
    import csv

    score = verdict = None
    with GOVERNANCE_CSV.open(newline="") as fh:
        for row in csv.reader(fh):
            if row and row[0] == "SCORE":
                score = float(row[1])
            elif row and row[0] == "VERDICT":
                verdict = row[1]
    assert score == pytest.approx(results["governance"]["score"], abs=1e-9)
    assert verdict == results["governance"]["verdict"]


# --------------------------------------------------------------------------- #
# 2. Monotonic degradation: ECE and AURC increase Low -> Medium -> High.
# --------------------------------------------------------------------------- #
def test_ece_monotonic_increasing_by_tier(tier_rows):
    ece = [tier_rows[t]["ece"] for t in ("Low", "Medium", "High")]
    assert ece[0] < ece[1] < ece[2], f"ECE not monotonic across tiers: {ece}"


def test_aurc_monotonic_increasing_by_tier(tier_rows):
    aurc = [tier_rows[t]["aurc"] for t in ("Low", "Medium", "High")]
    assert aurc[0] < aurc[1] < aurc[2], f"AURC not monotonic across tiers: {aurc}"


def test_json_tier_dicts_monotonic(results):
    """Same monotonic-degradation property read from the JSON (full precision)."""
    ece = results["calibration"]["ece_by_tier"]
    aurc = results["coverage_risk"]["aurc_by_tier"]
    assert ece["Low"] < ece["Medium"] < ece["High"]
    assert aurc["Low"] < aurc["Medium"] < aurc["High"]


# --------------------------------------------------------------------------- #
# 3. Key metrics within tolerance (locked committed values).
# --------------------------------------------------------------------------- #
def test_overall_accuracy(results):
    assert results["performance"]["accuracy"] == pytest.approx(0.8095, abs=1e-3)


def test_ece_overall_and_high_tier(results):
    assert results["calibration"]["ece_overall"] == pytest.approx(0.0799, abs=2e-3)
    assert results["calibration"]["ece_by_tier"]["High"] == pytest.approx(
        0.1626, abs=2e-3
    )


def test_aurc_overall(results):
    assert results["coverage_risk"]["aurc_overall"] == pytest.approx(0.1461, abs=2e-3)


def test_conformal_coverage(results):
    cov = results["conformal"]["empirical_coverage"]
    assert cov == pytest.approx(0.9286, abs=2e-3)
    # Conformal guarantee: empirical coverage meets the 1 - alpha target.
    assert cov >= results["conformal"]["target_coverage"]


def test_explanation_stability(results):
    assert results["explanation_stability"][
        "eta_xai_spearman"
    ] == pytest.approx(0.9752, abs=2e-3)


def test_harm_reduction_fraction(results):
    # ~14.4% harm reduction vs the accuracy-only baseline.
    assert results["harm"]["reduction_fraction"] == pytest.approx(0.1438, abs=2e-3)


def test_cohort_shape(results):
    """Lock the deterministic cohort geometry (seed-42 fingerprint)."""
    meta = results["meta"]
    assert meta["seed"] == 42
    assert meta["cohort_n"] == 501
    assert meta["n_features"] == 48
    assert meta["split"] == {"train": 250, "calibration": 125, "test": 126}


# --------------------------------------------------------------------------- #
# 4. Determinism: re-running the pipeline reproduces the committed JSON.
#    Marked slow + opt-in (heavy deps: sklearn/shap, multi-second run).
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_live_rerun_matches_committed(results, tmp_path, monkeypatch):
    """Re-run run_all.main() and compare regenerated metrics to committed JSON.

    Skips gracefully if the scientific stack (numpy/sklearn/shap/basics_cdss)
    is not importable in the current environment.
    """
    import sys

    sys.path.insert(0, str(REPO_ROOT / "src"))
    sys.path.insert(0, str(REPO_ROOT))
    try:
        import run_all  # noqa: F401
    except Exception as exc:  # pragma: no cover - env-dependent
        pytest.skip(f"cannot import run_all (deps missing): {exc}")

    # Redirect output to a temp dir so the committed results/ is never touched.
    monkeypatch.setattr(run_all, "OUT", tmp_path, raising=True)
    run_all.main()

    regenerated = json.loads((tmp_path / "eval_results.json").read_text())

    assert regenerated["governance"]["verdict"] == results["governance"]["verdict"]
    assert regenerated["governance"]["score"] == pytest.approx(
        results["governance"]["score"], abs=1e-9
    )
    assert regenerated["performance"]["accuracy"] == pytest.approx(
        results["performance"]["accuracy"], abs=1e-6
    )
    assert regenerated["calibration"]["ece_overall"] == pytest.approx(
        results["calibration"]["ece_overall"], abs=1e-6
    )
    assert regenerated["coverage_risk"]["aurc_overall"] == pytest.approx(
        results["coverage_risk"]["aurc_overall"], abs=1e-6
    )
    assert regenerated["conformal"]["empirical_coverage"] == pytest.approx(
        results["conformal"]["empirical_coverage"], abs=1e-6
    )
