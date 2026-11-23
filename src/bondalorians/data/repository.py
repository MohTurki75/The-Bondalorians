"""Accès aux données : CSV (hors ligne) ou PostgreSQL, ramenés au schéma canonique."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def apply_schema_map(raw: pd.DataFrame, schema_map: dict) -> pd.DataFrame:
    """Renomme et traduit un extrait brut selon `config/schema_map.yaml`."""
    columns = schema_map.get("columns", {})
    out = pd.DataFrame(index=raw.index)
    for canonical, real in columns.items():
        if real and real in raw.columns:
            out[canonical] = raw[real]

    as_of = (schema_map.get("source") or {}).get("as_of")
    as_of = pd.Timestamp(as_of) if as_of else pd.Timestamp(date.today())
    if "maturity_years" not in out.columns:
        if "maturity_date" not in out.columns:
            raise ValueError("Ni maturity_years ni maturity_date dans la correspondance de schéma.")
        out["maturity_years"] = (_to_dates(out["maturity_date"]) - as_of).dt.days / 365.25
    else:
        out["maturity_years"] = _to_number(out["maturity_years"])
    out = out.drop(columns=["maturity_date"], errors="ignore")
    if "issue_date" in out.columns:
        # Ancienneté de l'émission (années) : prime de nouvelle émission, liquidité.
        out["age_years"] = (as_of - _to_dates(out["issue_date"])).dt.days / 365.25
        out = out.drop(columns=["issue_date"])
    for column in ("z_spread", "amount_outstanding", "duration", "price", "ytm"):
        if column in out.columns:
            out[column] = _to_number(out[column])
    if "callable" in out.columns:
        out["callable"] = out["callable"].map(_to_bool)

    for column, mapping in (schema_map.get("values") or {}).items():
        if column in out.columns and mapping:
            out[column] = out[column].map(lambda v, m=mapping: m.get(v, v))
    return out


def _to_number(series: pd.Series) -> pd.Series:
    """Nombre, y compris « 1 234,5 », « 3,2 % » ou des chaînes."""
    if series.dtype == object:
        series = (series.astype(str).str.replace("%", "", regex=False).str.replace("\u00a0", "", regex=False)
                  .str.replace(" ", "", regex=False).str.replace(",", ".", regex=False))
    return pd.to_numeric(series, errors="coerce")


def _to_dates(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    import warnings

    text = series.astype(str).str.strip()
    # « 24/03/2039 » : jour en premier (usage français) ; sinon format ISO ou anglais.
    dayfirst = bool(text.str.fullmatch(r"\d{1,2}[/.-]\d{1,2}[/.-]\d{4}").mean() > 0.5) and bool(
        (text.str.extract(r"^(\d{1,2})")[0].astype(float) > 12).any())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parsed = pd.to_datetime(series, errors="coerce", dayfirst=dayfirst)
        if parsed.isna().mean() > 0.5:
            parsed = pd.to_datetime(series, errors="coerce", dayfirst=not dayfirst)
    return parsed


def _to_bool(value) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return False
    return str(value).strip().lower() in {"1", "true", "y", "yes", "oui", "o", "callable", "call", "vrai"}


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
