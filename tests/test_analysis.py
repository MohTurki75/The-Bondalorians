"""Analyses de compréhension : échantillon, modèles, forêt aléatoire, clustering."""

import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")

from bondalorians.analysis import models as M  # noqa: E402
from bondalorians.analysis.clustering import ATYPICAL, METHODS, cluster_universe  # noqa: E402
from bondalorians.analysis.common import for_agent, rating_bucket  # noqa: E402
from bondalorians.analysis.forest import forest_out_of_fold  # noqa: E402
from bondalorians.analysis.sample import describe_sample  # noqa: E402


def _serializable(analysis):
    json.dumps(for_agent(analysis))
    assert all(isinstance(f, str) and f for f in analysis["findings"])


def test_rating_buckets():
    assert [rating_bucket(n) for n in (1, 2, 5, 9, 12, 15, 18)] == ["AAA", "AA", "A", "BBB", "BB", "B", "CCC et moins"]


def test_sample_description(estimator):
    analysis = describe_sample(estimator)
    _serializable(analysis)
    shares = analysis["tables"]["part_expliquee"]["part_expliquee"]
    assert shares.between(-1e-9, 1).all()
    assert analysis["tables"]["taille"]["obligations"] == len(analysis["points"])
    assert 0 <= analysis["tables"]["taille"]["part_du_reste_expliquee_par_emetteur"] <= 1


def test_peer_curve_rises_with_rating(estimator):
    view = M.peer_curve_view(estimator)
    _serializable(view)
    at_5y = view["points"]["curves"].assign(gap=lambda d: (d["maturity_years"] - 5).abs())
    by_letter = at_5y.loc[at_5y.groupby("lettre")["gap"].idxmin()].set_index("lettre")["z_spread"]
    assert by_letter["AA"] < by_letter["A"] < by_letter["BBB"] < by_letter["BB"] < by_letter["B"]


def test_shrinkage_view_applies_formula(estimator):
    view = M.shrinkage_view(estimator)
    points = view["points"]
    expected = points["n"] / (points["n"] + estimator.k_) * points["ecart_brut"]
    assert np.allclose(points["ecart_retenu"], expected)
    assert (points["ecart_retenu"].abs() <= points["ecart_brut"].abs() + 1e-12).all()


def test_views_without_catboost_say_so(estimator):
    for view in (M.catboost_view(estimator), M.corrector_view(estimator)):
        assert view["available"] is False and view["findings"]


def test_comparison_and_confidence(estimator):
    oof = forest_out_of_fold(estimator)
    calibration = estimator.calibration_["oof"]
    merged = calibration.merge(oof, on=["row_id", "scheme"], how="left")
    assert merged["log_rf"].notna().all()  # mêmes plis que le backtest
    view = M.model_comparison(estimator, oof)
    _serializable(view)
    assert {"Courbe des pairs", "A · Comparables", "Forêt aléatoire"} <= set(view["points"]["modele"])
    confidence = M.confidence_view(estimator)
    coverage = confidence["points"]["coverage"]
    assert coverage["couverture"].between(0.8, 1.0).all()
    assert len(confidence["points"]["deciles"]) == 10


def test_forest_and_effects(estimator):
    view = M.forest_view(estimator)
    assert view["points"].iloc[0]["variable"] == "Notation"
    effects = M.effect_curves(estimator, sample=80)
    notch = effects["points"].query("variable == 'notch' and modele == 'Courbe des pairs'").sort_values("valeur")
    assert notch["z_spread"].is_monotonic_increasing


def test_comparables_view(estimator, new_bond, universe):
    big = universe["issuer"].value_counts().index[0]
    result = estimator.estimate({**new_bond, "issuer": big})
    view = M.comparables_view(estimator, result)
    _serializable(view)
    assert len(view["points"]["comparables"]) == len(result["comparables"])


@pytest.mark.parametrize("method", list(METHODS))
def test_clustering_methods(estimator, new_bond, method):
    target = estimator.estimate(new_bond)
    analysis = cluster_universe(estimator, method, target=target)
    _serializable(analysis)
    families = analysis["tables"]["familles"]
    assert families["obligations"].sum() == len(analysis["points"])
    agreement = analysis["tables"]["accord"].set_index("methode")
    assert np.allclose(np.diag(agreement.to_numpy(dtype=float)), 1.0)
    shares = analysis["tables"]["part_expliquee"]
    assert shares["familles_et_lettre"] >= max(shares["familles"], shares["lettre"]) - 1e-9
    placement = analysis["tables"]["placement"]
    assert placement["famille"] in set(families["famille"]) | {ATYPICAL}
    if method != "hdbscan":
        assert ATYPICAL not in set(analysis["points"]["famille"])
        assert len(families) == analysis["k"]


def test_clustering_with_forced_k_and_spread(estimator):
    analysis = cluster_universe(estimator, "kmeans", k=4, include_spread=True)
    assert analysis["k"] == 4 and len(analysis["tables"]["familles"]) == 4
    assert any("flatteur" in f for f in analysis["findings"])


def test_target_without_size_uses_median_size(estimator, new_bond):
    target, _, _ = estimator.resolve(new_bond)
    assert target["log_amt"] == pytest.approx(float(np.median(estimator.a_["log_amt"])))
    sized, _, _ = estimator.resolve({**new_bond, "amount_outstanding": 1e9})
    assert sized["log_amt"] == pytest.approx(np.log(1e9))


def test_structure_verdict():
    from bondalorians.analysis.clustering import structure_verdict

    decreasing = pd.DataFrame({"k": range(2, 11), "silhouette": np.linspace(0.35, 0.12, 9)})
    verdict = structure_verdict(decreasing, reading_k=4)
    assert not verdict["clear"] and verdict["best_k"] == 2 and verdict["chosen_k"] == 4
    assert "continuum" in verdict["reason"]
    peaked = pd.DataFrame({"k": range(2, 11), "silhouette": [0.2, 0.3, 0.45, 0.32, 0.25, 0.2, 0.18, 0.15, 0.1]})
    verdict = structure_verdict(peaked, reading_k=4)
    assert verdict["clear"] and verdict["chosen_k"] == 4 and verdict["best_k"] == 4
    peaked["silhouette"] = [0.2, 0.3, 0.38, 0.32, 0.25, 0.2, 0.18, 0.15, 0.1]
    peaked.loc[peaked["k"] == 6, "silhouette"] = 0.39
    assert structure_verdict(peaked)["chosen_k"] == 6
    sharp = pd.DataFrame({"k": range(2, 11), "silhouette": np.linspace(0.7, 0.3, 9)})
    assert structure_verdict(sharp)["clear"] and structure_verdict(sharp)["chosen_k"] == 2


def test_clustering_uses_reading_k_without_structure(estimator):
    analysis = cluster_universe(estimator, "kmeans", reading_k=5)
    verdict = analysis["tables"]["structure"]
    assert list(analysis["tables"]["silhouette"]["k"]) == list(range(2, 11))
    expected = verdict["best_k"] if verdict["clear"] else 5
    assert analysis["k"] == expected
    assert verdict["reason"] in analysis["findings"]


def test_report_for_llm(estimator, universe):
    from bondalorians.analysis.report import build_report

    result = estimator.estimate(universe["isin"].iloc[0])
    report = build_report(estimator, result, "données synthétiques", include_forest=False, methods=("kmeans", "ward"),
                          include_quality=False)
    for heading in ("## Consigne", "## Glossaire", "## 1. Échantillon", "## 2. Les modèles un par un",
                    "## 3. Comparaison des modèles et confiance", "## 4. Clustering", "## 5. Obligation estimée"):
        assert heading in report
    assert "Spread observé dans la base" in report and "nan" not in report.lower().replace("nant", "")
    assert all(line.count("|") >= 2 for line in report.splitlines() if line.startswith("|"))
    without_bond = build_report(estimator, None, include_forest=False, methods=("kmeans",), include_quality=False)
    assert "## 5. Obligation estimée" not in without_bond
