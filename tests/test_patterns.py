"""Motifs pour les agents : poches de biais, aiguillage, fiche."""

import json

import numpy as np
import pandas as pd
import pytest

from bondalorians.analysis import patterns as P
from bondalorians.analysis.agent_brief import build_brief, patterns_for_bond
from bondalorians.quant.estimator import ZSpreadEstimator
from bondalorians.synthetic import make_universe


@pytest.fixture(scope="module")
def injected(fast_settings):
    universe = make_universe(n_bonds=2500, n_issuers=350, seed=9, profile="realiste")
    mask = (universe["sector"] == "Énergie") & (universe["currency"] == "USD")
    universe.loc[mask, "z_spread"] *= 1.3
    return ZSpreadEstimator(fast_settings, use_ml=False).fit(universe)


def test_injected_pocket_is_found(injected):
    pockets = P.find_pockets(injected, "courbe")
    hit = pockets[pockets["segment"] == "Secteur Énergie · Devise USD"]
    assert len(hit) >= 1 and (hit["ecart_pct"] > 15).all()
    assert (hit["emetteurs"] >= 8).all()


def test_no_pocket_on_structureless_errors(injected, monkeypatch):
    original = P._cases

    def shuffled(estimator, target):
        cases = original(estimator, target)
        cases["ecart"] = np.random.default_rng(3).permutation(cases["ecart"].to_numpy())
        return cases

    monkeypatch.setattr(P, "_cases", shuffled)
    assert P.find_pockets(injected, "retenu").empty


def test_routing_table(injected):
    table = P.routing_table(injected)
    assert set(table["emetteur"]) <= {"inconnu", "1 à 2 obligations", "3 obligations ou plus"}
    assert (table["cas"] >= 40).all()
    switched = table[table["changer"]]
    assert (switched["erreur_meilleure_pct"] < 0.95 * switched["erreur_actuelle_pct"]).all()
    assert (table["facteur_intervalle"] >= 1).all()


def test_brief_and_bond_lookup(injected):
    brief = build_brief(injected, include_extra=False)
    json.dumps(brief, default=str)
    assert brief["regles"] and brief["aiguillage"] and "mode_d_emploi" in brief["meta"]
    assert any(r["type"] == "garde-fou" for r in brief["regles"])
    row = injected.df_[(injected.df_["sector"] == "Énergie") & (injected.df_["currency"] == "USD")].iloc[0].to_dict()
    found = patterns_for_bond(brief, row, issuer_bonds=None)
    assert found["case"]["emetteur"] == "inconnu" and found["aiguillage"] is not None
    assert found["segments"]["secteur"] == "Énergie" and found["segments"]["devise"] == "USD"
    for pocket in found["poches"]:
        assert all(found["segments"][d] == v for d, v in pocket["dimensions"].items())
