"""Accès aux données : CSV (hors ligne) ou PostgreSQL, ramenés au schéma canonique."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


# Champs de date convertis ligne à ligne par rapport à la date de prix de la ligne.
DATE_FIELDS = ("maturity_date", "final_maturity_date", "issue_date", "call_date")
FLAG_FIELDS = ("green", "social")
MATERIAL_CALL_YEARS = 1.0  # un call à moins d'un an de l'échéance finale (« par call ») ne compte pas


def apply_schema_map(raw: pd.DataFrame, schema_map: dict) -> pd.DataFrame:
    """Renomme et traduit un extrait brut selon `config/schema_map.yaml`.

    Si une colonne `pricing_date` est mappée, chaque ligne est valorisée à sa propre date de
    prix (maturité, ancienneté, années avant le call) ; sinon à `source.as_of` (défaut : aujourd'hui).
    """
    columns = schema_map.get("columns", {})
    out = pd.DataFrame(index=raw.index)
    for canonical, real in columns.items():
        if real and real in raw.columns:
            out[canonical] = raw[real]
    out = out.replace(list(MISSING_TOKENS), np.nan)

    as_of = (schema_map.get("source") or {}).get("as_of")
    as_of = pd.Timestamp(as_of) if as_of else pd.Timestamp(date.today())
    if "pricing_date" in out.columns:
        out["pricing_date"] = _to_dates(out["pricing_date"])
        reference = out["pricing_date"].fillna(as_of)
    else:
        reference = pd.Series(as_of, index=out.index)
    dates = {c: _to_dates(out[c]) for c in DATE_FIELDS if c in out.columns}

    def years(target: pd.Series) -> pd.Series:
        return (target - reference).dt.days / 365.25

    if "maturity_years" in out.columns and _to_number(out["maturity_years"]).notna().mean() > 0.5:
        out["maturity_years"] = _to_number(out["maturity_years"])
    elif "maturity_date" in dates:
        out["maturity_years"] = years(dates["maturity_date"])
    elif "final_maturity_date" in dates:
        out["maturity_years"] = years(dates["final_maturity_date"])
    else:
        raise ValueError("Ni maturity_years ni maturity_date dans la correspondance de schéma.")
    if "issue_date" in dates:
        # Ancienneté de l'émission (années) : prime de nouvelle émission, liquidité.
        out["age_years"] = -years(dates["issue_date"])
    final = dates.get("final_maturity_date", dates.get("maturity_date"))
    if final is not None:
        out["final_maturity_years"] = years(final)
    for column in ("z_spread", "amount_outstanding", "duration", "price", "ytm"):
        if column in out.columns:
            out[column] = _to_number(out[column])
    if "callable" in out.columns:
        out["callable"] = out["callable"].map(_to_bool)
    if "call_date" in dates:
        out["years_to_call"] = years(dates["call_date"])
        if "final_maturity_years" in out.columns:
            # Part de la vie restante avant le premier call : 1 = pas de call utile.
            ratio = (out["years_to_call"] / out["final_maturity_years"]).clip(0, 1)
            gap = out["final_maturity_years"] - out["years_to_call"]
            material = gap > MATERIAL_CALL_YEARS
            out["call_ratio"] = ratio.where(material, 1.0).fillna(1.0)
            if "callable" in out.columns:
                # « CALLABLE » avec un simple call au pair trois mois avant l'échéance : économiquement non remboursable.
                out["callable_raw"] = out["callable"]
                out["callable"] = out["callable"] & material.fillna(False)
    for column in FLAG_FIELDS:
        if column in out.columns:
            out[column] = out[column].map(_to_bool)
    out = out.drop(columns=[c for c in DATE_FIELDS if c in out.columns])

    for column, mapping in (schema_map.get("values") or {}).items():
        if column in out.columns and mapping:
            out[column] = out[column].map(lambda v, m=mapping: m.get(v, v))
    return out


MISSING_TOKENS = ("#N/A", "#N/A N/A", "#N/A Field Not Applicable", "#N/A Invalid Security", "N/A", "",
                  "nan", "NaN", "None", "null")


def pricing_dates(df: pd.DataFrame) -> pd.Series:
    """Nombre de lignes par date de prix (vide si la base n'a qu'une date)."""
    if "pricing_date" not in df.columns:
        return pd.Series(dtype=int)
    return pd.to_datetime(df["pricing_date"]).dt.normalize().value_counts().sort_index()


def select_pricing_date(df: pd.DataFrame, when: str | None = None) -> tuple[pd.DataFrame, pd.Timestamp | None]:
    """Garde une seule date de prix (par défaut la plus récente).

    Indispensable : avec deux dates, la même obligation apparaît deux fois et son double
    servirait de comparable presque parfait (fuite) ; le backtest serait faussement bon.
    """
    if "pricing_date" not in df.columns or df["pricing_date"].isna().all():
        return df, None
    days = pd.to_datetime(df["pricing_date"]).dt.normalize()
    chosen = pd.Timestamp(when).normalize() if when else days.max()
    if chosen not in set(days.dropna()):
        raise ValueError(f"Date de prix {chosen.date()} absente ; disponibles : "
                         + ", ".join(str(d.date()) for d in sorted(days.dropna().unique())))
    return df[days == chosen].reset_index(drop=True), chosen


def _to_number(series: pd.Series) -> pd.Series:
    """Nombre, y compris « 1 234,5 », « 3,2 % » ou des chaînes."""
    if series.dtype == object:
        series = (series.astype(str).str.replace("%", "", regex=False).str.replace("\u00a0", "", regex=False)
                  .str.replace(" ", "", regex=False).str.replace(",", ".", regex=False))
    return pd.to_numeric(series, errors="coerce")


_JS_DATE = r"^[A-Za-z]{3}\s+([A-Za-z]{3}\s+\d{1,2}\s+\d{4})"  # « Sun Nov 02 2036 00:00:00 GMT+0000 (…) »


def _to_dates(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    import warnings

    text = series.astype(str).str.strip()
    js = text.str.extract(_JS_DATE)[0]
    if js.notna().mean() > 0.3:  # dates JavaScript (export Bloomberg via un outil web)
        return pd.to_datetime(js, format="%b %d %Y", errors="coerce")
    if text.str.fullmatch(r"\d{8}(\.0)?").mean() > 0.5:  # 20260316
        return pd.to_datetime(text.str[:8], format="%Y%m%d", errors="coerce")
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
    text = str(value).strip().lower()
    if text in {"1", "true", "y", "yes", "oui", "o", "callable", "call", "vrai"}:
        return True
    # Types de maturité Bloomberg : « CALLABLE », « PERP/CALL », « CALL/SINK » ; pas « NON-CALLABLE ».
    return "call" in text and "non" not in text


def load_schema_map(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_universe_csv(path: str | Path, when: str | None = None) -> pd.DataFrame:
    """Charge un univers déjà au schéma canonique.

    S'il contient plusieurs dates de prix, seule la plus récente (ou `when`) est gardée.
    """
    df = pd.read_csv(path)
    df, chosen = select_pricing_date(df, when)
    if chosen is not None:
        df.attrs["pricing_date"] = str(chosen.date())
    return df


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
