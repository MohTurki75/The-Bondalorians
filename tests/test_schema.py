import pandas as pd
import pytest

from bondalorians.data.repository import apply_schema_map
from bondalorians.data.schema import prepare, prepare_target, to_notch


def test_notch_handles_agencies_and_unknown():
    assert to_notch("AAA") == 1
    assert to_notch("bbb-") == 10
    assert to_notch("Baa3") == 10
    assert pd.isna(to_notch("NR"))
    assert pd.isna(to_notch(None))


def test_prepare_drops_unusable_rows(universe):
    bad = universe.head(4).copy()
    bad.loc[bad.index[0], "rating"] = "NR"
    bad.loc[bad.index[1], "z_spread"] = -5
    bad.loc[bad.index[2], "maturity_years"] = 0
    prepared = prepare(bad, require_target=True)
    assert len(prepared) == 1
    assert prepared.attrs["dropped"] == 3


def test_prepare_requires_canonical_columns(universe):
    with pytest.raises(ValueError, match="rating"):
        prepare(universe.drop(columns=["rating"]), require_target=True)


def test_target_needs_core_characteristics():
    with pytest.raises(ValueError, match="maturity_years"):
        prepare_target({"issuer": "X", "sector": "S", "currency": "EUR", "rating": "A", "seniority": "SECURED"})


def test_schema_map_renames_translates_and_derives_maturity():
    raw = pd.DataFrame({
        "code": ["A1"], "nom": ["Société X"], "sect": ["Industrie"], "dev": ["EUR"], "note": ["A"],
        "rang": ["Sr Unsecured"], "echeance": ["2031-11-20"], "zs": [88.0],
    })
    schema_map = {
        "source": {"as_of": "2026-11-20"},
        "columns": {"isin": "code", "issuer": "nom", "sector": "sect", "currency": "dev", "rating": "note",
                    "seniority": "rang", "maturity_years": None, "maturity_date": "echeance", "z_spread": "zs"},
        "values": {"seniority": {"Sr Unsecured": "SENIOR_UNSECURED"}},
    }
    out = apply_schema_map(raw, schema_map)
    assert out.loc[0, "seniority"] == "SENIOR_UNSECURED"
    assert out.loc[0, "maturity_years"] == pytest.approx(5.0, abs=0.01)
    assert len(prepare(out, require_target=True)) == 1
