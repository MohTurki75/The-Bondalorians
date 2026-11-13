"""Paramètres appris sur les données : rétrécissement, comparables, difficulté."""

import numpy as np
import pytest

from bondalorians.quant.backtest import holdout_evaluation
from bondalorians.quant.estimator import ZSpreadEstimator
from bondalorians.quant.shrinkage import estimate_shrinkage_k
from bondalorians.quant.tuning import WEIGHT_KEYS
from bondalorians.synthetic import make_universe


def test_shrinkage_k_recovers_variance_ratio():
    rng = np.random.default_rng(0)
    sizes = rng.integers(1, 12, 400)
    issuers = np.repeat(np.arange(400), sizes)
    residuals = np.repeat(rng.normal(0, 0.20, 400), sizes) + rng.normal(0, 0.10, sizes.sum())
    result = estimate_shrinkage_k(residuals, issuers)
    assert result["k"] == pytest.approx(0.25, rel=0.25)  # 0,10² / 0,20²
    assert result["noise_sd"] == pytest.approx(0.10, rel=0.1)


def test_shrinkage_k_falls_back_without_repeated_issuers():
    result = estimate_shrinkage_k(np.array([0.1, -0.2, 0.3]), np.array(["a", "b", "c"]))
    assert result == {"k": 2.0, "noise_sd": None, "issuer_sd": None, "estimated": False}


def test_estimator_uses_estimated_or_forced_k(universe, settings):
    auto = ZSpreadEstimator(settings, use_ml=False).fit(universe, calibrate=False)
    assert auto.shrinkage_["estimated"] and 0.1 < auto.k_ < 1.0  # vrai rapport : 0,08² / 0,15² = 0,28
    settings["comparables"]["issuer_shrinkage_k"] = 3.5
    forced = ZSpreadEstimator(settings, use_ml=False).fit(universe, calibrate=False)
    assert forced.k_ == 3.5


def test_tuning_is_guarded(universe, settings):
    settings["comparables"].update(tune=True, tune_trials=40)
    estimator = ZSpreadEstimator(settings, use_ml=False).fit(universe)
    report = estimator.tuning_
    # Les valeurs par défaut sont le premier essai : le réglage ne peut pas faire pire dessus.
    assert report["tune_error_pct"]["tuned"] <= report["tune_error_pct"]["default"]
    # Adoption seulement si le gain est confirmé hors réglage.
    assert report["adopted"] == (report["valid_error_pct"]["tuned"] < report["valid_error_pct"]["default"])
    weights = estimator.params_["weights"]
    assert sum(weights.values()) == pytest.approx(1.0)
    if report["adopted"]:
        assert min(weights[k] for k in WEIGHT_KEYS) >= 0.05 - 1e-9
    else:
        assert weights == settings["comparables"]["weights"]
    # Les obligations de réglage ne servent pas à calibrer les intervalles.
    used = set(estimator.calibration_["oof"]["row_id"])
    assert not used & set(report["tune_row_ids"])
    assert estimator.calibration_report()["comparables_tuning"]["trials"] == 40


def test_adaptive_interval_balances_coverage_when_noise_depends_on_rating(settings):
    universe = make_universe(n_bonds=3000, n_issuers=400, seed=5, noise_by_rating=0.03)
    coverage = {}
    for adaptive in (False, True):
        settings["calibration"]["adaptive"] = adaptive
        e = holdout_evaluation(universe, settings, by_issuer=False)
        e = e[e["tier"] != "emetteur_inconnu"]
        coverage[adaptive] = (
            e["covered"].mean(), e[e["truth_bp"] < 150]["covered"].mean(),
            e[e["truth_bp"] >= 300]["covered"].mean(),
            e["score"].corr(e["abs_error_bp"] / e["truth_bp"], method="spearman"),
        )
    overall, tight, wide, rank = coverage[True]
    assert 0.85 <= overall <= 0.95
    # Intervalle fixe : trop large sur les spreads serrés, trop étroit sur les larges.
    fixed_gap = coverage[False][1] - coverage[False][2]
    assert fixed_gap > 0.05
    assert abs(tight - wide) < fixed_gap / 2
    # Un score élevé va avec une erreur relative faible.
    assert rank < -0.15


def test_refit_keeps_learned_parameters(universe, settings):
    estimator = ZSpreadEstimator(settings, use_ml=False).fit(universe)
    estimator.params_ = {**estimator.params_, "top_k": 25}
    calibration = estimator.calibration_
    estimator.refit(universe.head(1200))
    assert estimator.params_["top_k"] == 25 and estimator.calibration_ is calibration
    assert len(estimator.df_) == 1200
