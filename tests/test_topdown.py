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
    target, exclude, _ = estimator.resolve(row["isin"])
    known = top_down(estimator, target, exclude_isin=exclude)
    assert known["level"] in (1, 2) and known["points"]
    assert all(p["issuer"] == issuer for p in known["points"])
    assert all(p["isin"] != row["isin"] for p in known["points"])  # jamais elle-même
    unknown = top_down(estimator, estimator.resolve(new_bond)[0])
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
    assert reading["reader"] == "Entonnoir avec maturité" and reading["level_name"]


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
    tables = evaluate_top_down(estimator, readers=("Descendante stricte", "Entonnoir"))
    assert {"Modèle A", "Descendante stricte", "Entonnoir", "Courbe des pairs seule"} <= set(tables["par_palier"].columns)
    assert tables["par_niveau"]["Cas"].sum() == tables["par_palier"]["Cas"].sum()
    with pytest.raises(ValueError):
        top_down(estimator, estimator.df_.iloc[0].to_dict(), mode="autre")


def test_funnel_ranking_basket_and_barycentre(estimator, new_bond):
    from bondalorians.quant.funnel import funnel, variable_ranking

    ranking = variable_ranking(estimator, include_maturity=False)
    assert ranking["part_expliquee"].is_monotonic_decreasing
    assert "maturite" not in set(ranking["variable"])
    assert "maturite" in set(variable_ranking(estimator, include_maturity=True)["variable"])
    target = estimator.resolve(new_bond)[0]
    reading = funnel(estimator, target, min_basket=40)
    assert reading["basket_size"] >= 40 and not reading["perfect_curve"]
    assert [s["variable"] for s in reading["steps"]] == list(ranking["libelle"])
    imposed = [s for s in reading["steps"] if s["imposee"]]
    assert all(s["restantes"] >= 40 for s in imposed)
    assert reading["steps"][-1]["restantes"] == reading["basket_size"]
    # Barycentre : moyenne des log-spreads corrigés de tout le panier.
    small = funnel(estimator, target, min_basket=10_000)  # aucune variable imposée : tout l'univers
    assert small["basket_size"] == len(estimator.df_)
    assert not any(s["imposee"] for s in small["steps"])


def test_funnel_perfect_curve(estimator):
    from bondalorians.quant.funnel import funnel

    df = estimator.df_
    groups = df.groupby(["issuer", "notch", "seniority", "currency"]).size()
    issuer, notch, seniority, currency = groups[groups >= 3].index[0]
    row = df[(df["issuer"] == issuer) & (df["notch"] == notch) & (df["seniority"] == seniority)
             & (df["currency"] == currency)].iloc[0].to_dict()
    reading = funnel(estimator, row, exclude_isin=row["isin"])
    assert reading["perfect_curve"] and reading["method"].startswith("courbe parfaite")
    assert all(p["isin"] != row["isin"] and p["issuer"] == issuer for p in reading["points"])


def test_model_quality(universe, settings):
    from bondalorians.analysis.quality import FAIL, NA, PASS, model_quality

    estimator = ZSpreadEstimator(settings, use_ml=False).fit(universe.head(800))
    quality = model_quality(estimator, include_forest=False, readers=("Entonnoir avec maturité",),
                            max_cases=200, in_sample_cases=200)
    indicators, tests = quality["tables"]["indicateurs"], quality["tables"]["tests"]
    assert {"Estimation retenue", "Courbe des pairs (P)", "Modèle A"} <= set(indicators["modele"])
    assert set(tests["statut"]) <= {PASS, FAIL, NA}
    retained = tests[(tests["modele"] == "Estimation retenue") & (tests["test"].str.startswith("Couverture"))]
    assert len(retained) == 2
    overfit = tests[(tests["modele"] == "Courbe des pairs (P)") & (tests["test"] == "Surapprentissage")]
    assert overfit.set_index("situation").loc["Émetteur connu", "statut"] in (PASS, FAIL)
    memory = tests[(tests["modele"] == "Modèle A") & (tests["test"] == "Surapprentissage")]
    assert (memory["statut"] == NA).all()
    assert indicators["moins_de_10_pct"].between(0, 1).all()
