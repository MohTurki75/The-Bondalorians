"""Format d'export Bloomberg du hackathon : dates JavaScript, deux dates de prix, ticker = émetteur."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bondalorians.analysis.dynamics import curve_stability, market_dynamics
from bondalorians.data import automap
from bondalorians.data.automap import detect_preset, propose_schema_map
from bondalorians.data.repository import apply_schema_map, load_universe_csv, select_pricing_date
from bondalorians.data.schema import prepare
from bondalorians.synthetic import make_bloomberg_export

SAMPLE = Path(__file__).parent / "data" / "bloomberg_echantillon.csv"


def _sample() -> pd.DataFrame:
    return pd.read_csv(SAMPLE, dtype={"pricing_date": str})


def test_sample_rows_are_read_correctly():
    raw = _sample()
    assert detect_preset(raw) == "bloomberg_fr"
    proposal = propose_schema_map(raw)
    assert proposal["schema_map"]["columns"]["issuer"] == "ticker"
    canonical = apply_schema_map(raw, proposal["schema_map"])
    df = prepare(canonical, require_target=True)
    assert len(df) == 6
    rzr = df[df["issuer"] == "RZR"].iloc[0]
    # 16/03/2026 -> 02/11/2036 : 10,6 ans, valorisé à la date de prix de la ligne, pas à aujourd'hui.
    assert rzr["maturity_years"] == pytest.approx(10.63, abs=0.02)
    assert rzr["age_years"] == pytest.approx(9.37, abs=0.02)
    # « CALLABLE » avec un call au pair trois mois avant l'échéance : pas remboursable économiquement.
    assert bool(rzr["callable_raw"]) and not bool(rzr["callable"])
    assert rzr["call_ratio"] == 1.0
    assert rzr["sector"] == "Health Care" and rzr["industry"] == "Pharmaceuticals"
    assert rzr["seniority"] == "SENIOR_UNSECURED" and rzr["notch"] == 5
    tb = df[df["issuer"] == "TB"].iloc[0]
    assert tb["holdco_flag"] == 1.0 and not bool(tb["callable"])
    assert np.isnan(df[df["issuer"] == "PJDL"].iloc[0]["holdco_flag"])  # « Insufficient Information »
    assert df["green_flag"].eq(0).all()


def test_issuer_guard_without_preset(monkeypatch):
    """Sans le format connu, le nom d'obligation n'est pas pris pour l'émetteur."""
    monkeypatch.setattr(automap, "PRESETS", {})
    proposal = propose_schema_map(_sample())
    assert proposal["schema_map"]["columns"]["issuer"] == "ticker"


def test_two_dates_select_latest_and_dynamics(tmp_path):
    raw = make_bloomberg_export(n_bonds=600, n_issuers=120)
    canonical = apply_schema_map(raw, propose_schema_map(raw)["schema_map"])
    one, day = select_pricing_date(canonical)
    assert str(day.date()) == "2026-03-16" and one["isin"].is_unique
    with pytest.raises(ValueError):
        select_pricing_date(canonical, "2024-01-01")
    path = tmp_path / "base.csv"
    canonical.to_csv(path, index=False)
    assert len(load_universe_csv(path)) == len(one)

    result = market_dynamics(canonical)
    by_rating = result["tables"]["notation"].set_index("Notation")["mouvement_median_pct"]
    assert by_rating.get("BB", 0) > by_rating.get("A", 0)  # le haut rendement a plus bougé (données factices)
    stability = curve_stability(canonical)
    # Le double de l'obligation à l'autre date est bien plus proche que la courbe : d'où l'interdiction du mélange.
    assert stability["erreur_double_autre_date_pct"] < stability["erreur_meme_date_pct"] / 2


def test_problem_reported_for_two_dates():
    raw = make_bloomberg_export(n_bonds=200, n_issuers=50)
    problems = " ".join(propose_schema_map(raw)["problems"])
    assert "Plusieurs dates de prix" in problems
