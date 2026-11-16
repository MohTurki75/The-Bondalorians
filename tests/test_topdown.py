"""Lecture descendante et courbe des pairs comme candidate."""

import math

import numpy as np
import pandas as pd
import pytest

from bondalorians.quant.estimator import ZSpreadEstimator
from bondalorians.quant.topdown import LEVEL_NAMES, _interpolate, evaluate_top_down, top_down


def test_linear_interpolation_and_extrapolation():
    x = np.array([2.0, 4.0, 8.0])
    y = np.log(np.array([100.0, 120.0, 160.0]))
    flat = lambda m: 0.0  # noqa: E731 - courbe plate : extrapolation constante
    value, used, method = _interpolate(x, y, 3.0, flat, smooth=False)
    assert method == "interpolation linéaire" and sorted(used) == [0, 1]
    assert value == pytest.approx((np.log(100) + np.log(120)) / 2)
    value, used, method = _interpolate(x, y, 10.0, lambda m: 0.01 * m, smooth=False)
    assert method.startswith("extrapolation") and value == pytest.approx(np.log(160) + 0.02)
    value, used, _ = _interpolate(x, y, 3.0, flat, smooth=True)
    assert len(used) == 3 and value == pytest.approx(np.mean(y))


def test_levels_follow_available_information(estimator, universe, new_bond):
    issuer = universe["issuer"].value_counts().index[0]
    row = universe[universe["issuer"] == issuer].iloc[0]
    known = estimator.estimate(row["isin"])["top_down"]
    assert known["level"] in (1, 2) and known["points"]
    assert all(p["issuer"] == issuer for p in known["points"])
    assert all(p["isin"] != row["isin"] for p in known["points"])  # jamais elle-même
    unknown = estimator.estimate(new_bond)["top_down"]
    assert unknown["level"] >= 3 and unknown["level_name"] == LEVEL_NAMES[unknown["level"]]
    assert all(p["issuer"] != new_bond["issuer"] for p in unknown["points"])


def test_top_down_is_a_control_not_the_estimate(estimator, new_bond):
    result = estimator.estimate(new_bond)
    reading = result["top_down"]
    assert result["models"]["top_down_bp"] == reading["z_spread_bp"]
    assert reading["gap_pct"] == pytest.approx(100 * (reading["z_spread_bp"] / result["z_spread_bp"] - 1), abs=0.1)
    control = result["controls"][-1]
    assert control["control"] == "Accord avec la lecture descendante"
    assert control["status"] == ("ok" if abs(reading["gap_pct"]) <= 25 else "alerte")
    assert "Lecture descendante" in result["narrative"]


def test_peer_curve_can_be_champion(universe, settings, new_bond):
    settings["calibration"]["champion_margin"] = -10.0  # n'importe quel modèle meilleur passe
    estimator = ZSpreadEstimator(settings, use_ml=False).fit(universe)
    champions = estimator.calibration_["champion"]
    assert set(champions.values()) <= {"A", "P"}
    oof = estimator.calibration_["oof"]
    for key, champion in champions.items():
        group = oof[oof["key"] == key]
        errors = {m: (group[c] - group["y_log"]).abs().median() for m, c in (("A", "log_a"), ("P", "peer_log"))}
        assert champion == min(errors, key=errors.get)
    result = estimator.estimate(new_bond)
    if result["champion"] == "P":
        assert result["z_spread_bp"] == result["models"]["courbe_pairs_bp"]
        assert result["waterfall"][-1]["cumulative_bp"] == pytest.approx(result["z_spread_bp"], abs=0.2)
    settings["calibration"]["peer_curve_candidate"] = False
    without = ZSpreadEstimator(settings, use_ml=False).fit(universe)
    assert "P" not in without.calibration_["champion"].values()


def test_evaluation_tables(universe, settings):
    small = universe.head(500)
    estimator = ZSpreadEstimator(settings, use_ml=False).fit(small)
    tables = evaluate_top_down(estimator, modes=("strict",))
    assert {"Modèle A", "Top-down strict", "Courbe des pairs seule"} <= set(tables["par_palier"].columns)
    assert tables["par_niveau"]["Cas"].sum() == tables["par_palier"]["Cas"].sum()
    with pytest.raises(ValueError):
        top_down(estimator, estimator.df_.iloc[0].to_dict(), mode="autre")
