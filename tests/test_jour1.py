"""Lecture d'une base inconnue : correspondance des colonnes, formats, colonnes supplémentaires."""

import numpy as np
import pandas as pd
import pytest

from bondalorians.analysis.extra_columns import evaluate_extra_columns
from bondalorians.data.automap import normalize, propose_schema_map, write_schema_map
from bondalorians.data.repository import apply_schema_map, load_schema_map
from bondalorians.data.schema import prepare, to_notch
from bondalorians.quant.peer_curve import PeerCurve
from bondalorians.synthetic import make_raw_database


@pytest.fixture(scope="module")
def raw():
    return make_raw_database(n_bonds=1500, seed=4)


@pytest.fixture(scope="module")
def proposal(raw):
    return propose_schema_map(raw, as_of="2026-11-24")


def test_ratings_in_many_formats():
    assert [to_notch(r) for r in ("BBB+", "Baa1", "BBB+ (S&P)", "Baa1 *-", "BBB-*-", "A- /NEG", "(P)A2", "BBB+u")] == \
        [8, 8, 8, 8, 10, 7, 6, 8]
    assert np.isnan(to_notch("NR")) and np.isnan(to_notch("WR"))


def test_columns_are_found(proposal):
    columns = proposal["schema_map"]["columns"]
    assert columns["isin"] == "Code ISIN" and columns["issuer"] == "Nom émetteur"
    assert columns["rating"] == "Notation composite" and columns["esg"] == "Note ESG MSCI"
    assert columns["seniority"] == "Rang de paiement" and columns["z_spread"] == "Z-Spread (pb)"
    assert columns["maturity_date"] == "Date échéance" and columns["issue_date"] == "Date émission"
    assert columns["duration"] == "Duration modifiée" and columns["price"] == "Prix"
    assert columns["ytm"] == "Rendement (%)" and columns["currency"] == "Devise"
    assert not proposal["problems"]
    report = proposal["report"].set_index("champ")
    assert report.loc["ytm", "role"].startswith("contrôle seulement")


def test_seniority_labels(proposal):
    mapping = proposal["schema_map"]["values"]["seniority"]
    assert mapping["Sr Unsecured"] == "SENIOR_UNSECURED"  # « unsecured » contient « secured »
    assert mapping["Secured"] == "SECURED" and mapping["Tier 2"] == "TIER2" and mapping["AT1"] == "AT1"
    assert mapping["Senior Non-Preferred"] == "SENIOR_NON_PREFERRED"
    assert mapping["Jr Subordinated"] == "SUBORDINATED" and mapping["Hybrid perpetual"] == "HYBRID"
    assert proposal["schema_map"]["values"]["coupon_type"] == {"Fixe": "FIXED", "Variable": "FLOATING"}


def test_apply_and_prepare(raw, proposal, tmp_path):
    path = write_schema_map(proposal, tmp_path / "schema_map.yaml")
    canonical = apply_schema_map(raw, load_schema_map(path))
    # Dates jour/mois/année : maturité et ancienneté cohérentes.
    assert canonical["maturity_years"].between(0, 31).all()
    assert canonical["age_years"].between(0, 8.1).all()
    assert canonical["callable"].dtype == bool
    prepared = prepare(canonical, require_target=True)
    assert len(prepared) == len(raw) and prepared["esg_score"].between(1, 7).all()
    assert set(prepared["seniority"]) <= {"SECURED", "SENIOR_PREFERRED", "SENIOR_UNSECURED", "SENIOR_NON_PREFERRED",
                                          "TIER2", "AT1", "SUBORDINATED", "HYBRID"}


def test_extra_columns_detects_esg_and_flags_leaks(raw, proposal):
    prepared = prepare(apply_schema_map(raw, proposal["schema_map"]), require_target=True)
    result = evaluate_extra_columns(prepared)
    table = result["tables"]["variables"].set_index("cle")
    assert table.loc["esg", "recommandation"] == "à activer"          # effet injecté
    assert table.loc["age", "recommandation"] == "inutile"            # aucun effet injecté
    assert table.loc["duration", "recommandation"] != "à activer"     # jamais d'office : fuite possible
    assert result["recommended"] == ["esg"]
    assert set(result["tables"]["fuite"]["colonne"]) == {"Rendement (YTM)", "Prix"}


def test_peer_curve_extra_feature(raw, proposal):
    prepared = prepare(apply_schema_map(raw, proposal["schema_map"]), require_target=True)
    curve = PeerCurve(enriched=True, extra=("esg",)).fit(prepared)
    good, bad = prepared.iloc[[0]].copy(), prepared.iloc[[0]].copy()
    good["esg_score"], bad["esg_score"] = 7.0, 1.0
    assert curve.predict_log(good)[0] < curve.predict_log(bad)[0]  # meilleure note ESG, spread plus serré
    missing = prepared.iloc[[0]].drop(columns=["esg_score"])
    assert np.isfinite(curve.predict_log(missing)[0])  # cible sans note ESG : effet nul


def test_normalize():
    assert normalize("Z-Spread (pb)") == "zspreadpb" and normalize("Émetteur") == "emetteur"
