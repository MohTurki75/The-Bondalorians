"""Fiches obligataires Markdown : lecture, call, notation absente, estimation et explication."""

from pathlib import Path

import pytest

from bondalorians.data.automap import propose_schema_map
from bondalorians.data.fiche import fiche_to_bond, parse_fiche
from bondalorians.data.repository import apply_schema_map, select_pricing_date
from bondalorians.data.schema import prepare
from bondalorians.data.stores import JsonlStore
from bondalorians.service import Service
from bondalorians.synthetic import make_bloomberg_export

FICHES = Path(__file__).parent / "data" / "fiches"


def _read(name: str) -> str:
    return (FICHES / f"Fiche_{name}_Corp.md").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def universe():
    raw = make_bloomberg_export(n_bonds=800, n_issuers=150)
    canonical = apply_schema_map(raw, propose_schema_map(raw)["schema_map"])
    return select_pricing_date(canonical)[0]


def test_parse_fiche_fields():
    raw = parse_fiche(_read("PY366041"))
    assert raw["id"] == "PY366041 Corp" and raw["ticker"] == "VYQW"
    assert raw["paysemetteur"] == "FR" and raw["bondbailin"] == "Y" and raw["ratingmoyen"] == "A+"
    assert raw["nomsecteur"] == "Financials" and raw["structureaffaire"] == "Holdco"
    assert raw["niveauzspread"] is None


def test_call_and_seniority(universe):
    u = prepare(universe, require_target=True)
    sub, notes = fiche_to_bond(parse_fiche(_read("VY237195")), "2026-03-16", u)
    assert sub["seniority"] == "SUBORDINATED" and sub["coupon_type"] == "FLOATING"
    assert sub["callable"] and sub["maturity_years"] == pytest.approx(9.42, abs=0.02)  # cotée au call d'août 2035
    assert str(sub["rating_source"]).startswith("déduite")  # NR
    par_call, _ = fiche_to_bond(parse_fiche(_read("DT903899")), "2026-03-16", u)
    assert not par_call["callable"]  # « CALLABLE » mais échéance de calcul = échéance finale


def test_estimate_fiche_end_to_end(universe, tmp_path):
    service = Service(universe, use_ml=False, run_store=JsonlStore(tmp_path / "r.jsonl"),
                      feedback_store=JsonlStore(tmp_path / "f.jsonl"))
    assert str(service.pricing_date().date()) == "2026-03-16"
    for path in sorted(FICHES.glob("*.md")):
        out = service.estimate_fiche(path.read_text(encoding="utf-8"))
        assert out["result"]["z_spread_bp"] > 0 and out["markdown"].startswith("# ")
    nr = service.estimate_fiche(_read("WW655848"))
    assert nr["sensitivity"] and sum(s["retenue"] for s in nr["sensitivity"]) == 1
    spreads = [s["z_spread_bp"] for s in nr["sensitivity"]]
    assert spreads == sorted(spreads)  # moins bonne notation, spread plus large
