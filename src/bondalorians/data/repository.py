"""Accès aux données : CSV (hors ligne) ou PostgreSQL, ramenés au schéma canonique."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import yaml


def apply_schema_map(raw: pd.DataFrame, schema_map: dict) -> pd.DataFrame:
    """Renomme et traduit un extrait brut selon `config/schema_map.yaml`."""
    columns = schema_map.get("columns", {})
    out = pd.DataFrame(index=raw.index)
    for canonical, real in columns.items():
        if real and real in raw.columns:
            out[canonical] = raw[real]

    if "maturity_years" not in out.columns:
        if "maturity_date" not in out.columns:
            raise ValueError("Ni maturity_years ni maturity_date dans la correspondance de schéma.")
        as_of = (schema_map.get("source") or {}).get("as_of")
        as_of = pd.Timestamp(as_of) if as_of else pd.Timestamp(date.today())
        out["maturity_years"] = (pd.to_datetime(out["maturity_date"]) - as_of).dt.days / 365.25
    out = out.drop(columns=["maturity_date"], errors="ignore")

    for column, mapping in (schema_map.get("values") or {}).items():
        if column in out.columns and mapping:
            out[column] = out[column].map(lambda v, m=mapping: m.get(v, v))
    return out


def load_schema_map(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_universe_csv(path: str | Path) -> pd.DataFrame:
    """Charge un univers déjà au schéma canonique (données synthétiques)."""
    return pd.read_csv(path)


def load_universe_postgres(dsn: str, schema_map: dict) -> pd.DataFrame:
    """Charge l'univers depuis PostgreSQL avec un rôle en lecture seule."""
    import psycopg  # dépendance optionnelle : pip install -e .[pg]

    query = schema_map["source"]["query"]
    with psycopg.connect(dsn) as connection, connection.cursor() as cursor:
        cursor.execute("SET TRANSACTION READ ONLY")
        cursor.execute(query)
        names = [d.name for d in cursor.description]
        raw = pd.DataFrame(cursor.fetchall(), columns=names)
    return apply_schema_map(raw, schema_map)
