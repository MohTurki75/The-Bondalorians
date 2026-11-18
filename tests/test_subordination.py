"""Dette subordonnée ancrée sur la dette senior, contrôle de subordination, CatBoost."""

import math

import numpy as np
import pandas as pd
import pytest

from bondalorians.quant.estimator import ZSpreadEstimator
from bondalorians.quant.subordination import SubordinationModel, is_subordinated


def _subordinated_with_senior(estimator):
    df = estimator.df_
    for issuer, group in df.groupby("issuer"):
        juniors = group[group["sen_rank"] >= 3]
        if len(juniors) and (group["sen_rank"] <= 1).sum() >= 2:
            return juniors.iloc[0]
    pytest.skip("aucun émetteur avec dette senior et subordonnée")


def test_premium_measured_within_issuers(estimator):
    model = estimator.subordination_
    assert model is not None and model.categories_
    # Les primes mesurées sont positives et croissent avec la subordination.
    assert all(v > 0 for v in model.categories_.values())
    if {"TIER2", "AT1"} <= set(model.categories_):
        assert model.categories_["AT1"] > model.categories_["TIER2"]


def test_anchor_on_issuer_senior_debt(estimator):
    row = _subordinated_with_senior(estimator)
    result = estimator.estimate(row["isin"])
    senior = result["subordination"]
    assert senior["senior_bonds"] >= 1 and row["isin"] not in senior["senior_isins"]
    assert result["models"]["S_ancrage_senior_bp"] == senior["anchor_bp"]
    # Ancrage = senior de l'émetteur au profil de la cible + prime : au-dessus de la senior (prime positive).
    assert senior["anchor_bp"] > 0 and senior["premium_pct"] > 0
    control = next(c for c in result["controls"] if c["control"] == "Cohérence de subordination")
    assert control["status"] == ("ok" if result["z_spread_bp"] >= senior["senior_equivalent_bp"] else "alerte")
    assert "ramenée au profil de la cible" in control["detail"]


def test_no_anchor_for_senior_or_unknown_issuer(estimator, new_bond):
    assert not is_subordinated(1) and is_subordinated(3)
    assert "subordination" not in estimator.estimate(new_bond)
    unknown_sub = estimator.estimate({**new_bond, "seniority": "TIER2"})
    assert "subordination" not in unknown_sub and unknown_sub["models"]["S_ancrage_senior_bp"] is None
    steps = [s["step"] for s in unknown_sub["waterfall"]]
    assert not any(s.startswith("Écart émetteur") for s in steps)  # sans objet : omis


def test_s_can_be_retained_and_explained(universe, settings):
    settings["calibration"]["champion_margin"] = -10.0  # le meilleur gagne toujours
    estimator = ZSpreadEstimator(settings, use_ml=False).fit(universe)
    assert "S" in estimator.calibration_["available_models"]
    row = _subordinated_with_senior(estimator)
    result = estimator.estimate(row["isin"])
    if result["champion"] == "S":
        assert result["z_spread_bp"] == result["subordination"]["anchor_bp"]
        steps = [s["step"] for s in result["waterfall"]]
        assert any(s.startswith("Niveau de la dette senior") for s in steps)
        assert result["waterfall"][-1]["cumulative_bp"] == pytest.approx(result["z_spread_bp"], abs=0.2)


def test_controls_use_french_numbers(estimator, new_bond):
    details = " ".join(c["detail"] for c in estimator.estimate(new_bond)["controls"])
    assert not any(ch.isdigit() and nxt == "." and after.isdigit()
                   for ch, nxt, after in zip(details, details[1:], details[2:]))


def test_honest_coverage_is_not_tautological(estimator):
    from bondalorians.analysis.quality import honest_coverage

    oof = estimator.calibration_["oof"].copy()
    oof["issuer"] = estimator.df_["issuer"].to_numpy()[oof["row_id"].to_numpy()]
    coverage = honest_coverage(oof, estimator.calibration_)
    assert 0.8 <= coverage <= 0.97
    # Les quantiles sont recalculés sur l'autre moitié des émetteurs, jamais repris de la
    # calibration : multiplier toutes les erreurs ne change donc pas la couverture.
    scaled = oof.assign(err=oof["err"] * 2)
    assert honest_coverage(scaled, estimator.calibration_) == pytest.approx(coverage, abs=1e-9)
