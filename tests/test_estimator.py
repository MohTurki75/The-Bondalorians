import json

import pytest

from bondalorians.quant.backtest import holdout_evaluation
from bondalorians.quant.conformal import conformal_quantile
from bondalorians.quant.estimator import ZSpreadEstimator
from bondalorians.quant.gbm import catboost_available


def test_conformal_quantile_uses_finite_sample_rank():
    errors = list(range(1, 11))  # n = 10, alpha = 0.2 -> rang ceil(11 * 0.8) = 9
    assert conformal_quantile(errors, 0.2) == 9
    assert conformal_quantile(errors, 0.01) == 10  # rang plafonné à n


def test_estimate_is_complete_and_serializable(estimator, universe):
    result = estimator.estimate(universe["isin"].iloc[0])
    json.dumps(result)
    low, high = result["interval_bp"]
    assert low < result["z_spread_bp"] < high
    assert result["observed_z_spread_bp"] == universe["z_spread"].iloc[0]
    assert 0 <= result["confidence"]["score"] <= 100
    assert result["confidence"]["typical_error_bp"] > 0
    assert result["waterfall"][-1]["cumulative_bp"] == pytest.approx(result["z_spread_bp"], abs=0.2)


def test_bond_is_excluded_from_its_own_estimate(universe):
    isin = universe["isin"].iloc[0]
    distorted = universe.copy()
    distorted.loc[distorted["isin"] == isin, "z_spread"] *= 1.02
    base = ZSpreadEstimator(use_ml=False).fit(universe, calibrate=False)
    moved = ZSpreadEstimator(use_ml=False).fit(distorted, calibrate=False)
    result = base.estimate(isin)
    assert isin not in [c["isin"] for c in result["comparables"]]
    # Le propre spread de l'obligation n'agit que marginalement, via la courbe des pairs.
    assert moved.estimate(isin)["z_spread_bp"] == pytest.approx(result["z_spread_bp"], rel=1e-3)


def test_unseen_issuer_is_complex_with_wider_interval(estimator, universe, new_bond):
    unseen = estimator.estimate(new_bond)
    assert unseen["tier"] == "complexe" and unseen["issuer_seen"] is False
    assert unseen["issuer_offset"]["n_bonds"] == 0
    assert unseen["observed_z_spread_bp"] is None

    def relative_width(result):
        return (result["interval_bp"][1] - result["interval_bp"][0]) / result["z_spread_bp"]

    big_issuer = universe["issuer"].value_counts().index[0]
    known = estimator.estimate({**new_bond, "issuer": big_issuer})
    assert known["issuer_seen"] is True
    assert relative_width(unseen) > relative_width(known)
    assert unseen["confidence"]["score"] < known["confidence"]["score"]


def test_subordinated_bond_of_known_issuer_is_complex(estimator, universe, new_bond):
    known = universe["issuer"].value_counts().index[0]
    result = estimator.estimate({**new_bond, "issuer": known, "seniority": "HYBRID"})
    assert result["tier"] == "complexe" and result["issuer_seen"] is True


def test_worse_rating_gives_wider_spread(estimator, new_bond):
    spreads = [estimator.estimate({**new_bond, "rating": r})["z_spread_bp"] for r in ("A", "BBB", "BB", "B")]
    assert spreads == sorted(spreads)


def test_unknown_isin_and_bad_rating_are_rejected(estimator, new_bond):
    with pytest.raises(KeyError):
        estimator.estimate("XS_INCONNU")
    with pytest.raises(ValueError):
        estimator.estimate({**new_bond, "rating": "NR"})


def test_holdout_accuracy_and_coverage(universe, settings):
    evaluation = holdout_evaluation(universe, settings, by_issuer=False)
    seen = evaluation[evaluation["tier"] != "emetteur_inconnu"]
    relative = (seen["abs_error_bp"] / seen["truth_bp"]).median()
    assert relative < 0.10
    assert 0.84 <= seen["covered"].mean() <= 0.97


def test_uncalibrated_estimator_gives_point_estimate_only(universe, settings):
    result = ZSpreadEstimator(settings, use_ml=False).fit(universe, calibrate=False).estimate(universe["isin"].iloc[0])
    assert result["interval_bp"] is None and result["confidence"]["score"] is None
    assert result["narrative"].startswith("Z-Spread juste estimé")


@pytest.mark.skipif(not catboost_available(), reason="catboost non installé")
def test_catboost_models(universe, settings, new_bond):
    estimator = ZSpreadEstimator(settings, use_ml=True).fit(universe, calibrate=False)
    # Modèle B : monotone en notation.
    spreads = [estimator.estimate({**new_bond, "rating": r})["models"]["B_catboost_bp"]
               for r in ("AA", "A", "BBB", "BB", "B")]
    assert spreads == sorted(spreads)
    # Modèle C : une correction bornée du modèle A, expliquée par SHAP.
    result = estimator.estimate(new_bond)
    a, c = result["models"]["A_comparables_bp"], result["models"]["C_comparables_corriges_bp"]
    assert abs(c / a - 1) < 0.25
    assert "log_a" in result["shap_log_correction"] and "notch" in result["shap_log_catboost"]


@pytest.mark.skipif(not catboost_available(), reason="catboost non installé")
def test_ml_champion_needs_a_clear_margin(universe, settings):
    small = universe.head(700)
    settings["calibration"]["champion_margin"] = 0.99  # aucun modèle ML ne peut gagner
    estimator = ZSpreadEstimator(settings, use_ml=True).fit(small)
    assert set(estimator.calibration_["champion"].values()) == {"A"}
    assert estimator.calibration_["available_models"] == ["A", "B", "C", "P"]
    columns = estimator.backtest_summary()[0]
    assert "mae_bp_C_comparables_corriges" in columns and "mae_bp_B_catboost" in columns
