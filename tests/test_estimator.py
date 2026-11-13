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
    quantiles = estimator.calibration_["quantile"]
    assert quantiles["emetteur_inconnu"] > quantiles["simple"]


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


def test_holdout_accuracy_and_coverage(universe):
    evaluation = holdout_evaluation(universe, by_issuer=False)
    seen = evaluation[evaluation["tier"] != "emetteur_inconnu"]
    relative = (seen["abs_error_bp"] / seen["truth_bp"]).median()
    assert relative < 0.10
    assert 0.84 <= seen["covered"].mean() <= 0.97


@pytest.mark.skipif(not catboost_available(), reason="catboost non installé")
def test_catboost_challenger_is_monotone_in_rating(universe, new_bond):
    estimator = ZSpreadEstimator(use_ml=True).fit(universe, calibrate=False)
    spreads = [estimator.estimate({**new_bond, "rating": r})["models"]["B_catboost_bp"]
               for r in ("AA", "A", "BBB", "BB", "B")]
    assert spreads == sorted(spreads)
    assert "notch" in estimator.estimate(new_bond)["shap_log_catboost"]
