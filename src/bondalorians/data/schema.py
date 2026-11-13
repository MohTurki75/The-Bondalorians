"""Schéma canonique et référentiels (notation, séniorité).

Tout le code en aval ne connaît que ces noms de colonnes.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

# Colonnes décrivant une obligation ; `z_spread` (en pb) n'est exigé que pour l'univers.
DESCRIPTIVE_COLUMNS = [
    "isin", "issuer", "sector", "country", "currency", "rating",
    "seniority", "coupon_type", "callable", "maturity_years", "amount_outstanding",
]
REQUIRED_TARGET = ["issuer", "sector", "currency", "rating", "seniority", "maturity_years"]
TARGET_COLUMN = "z_spread"

RATING_SCALE = [
    "AAA", "AA+", "AA", "AA-", "A+", "A", "A-", "BBB+", "BBB", "BBB-",
    "BB+", "BB", "BB-", "B+", "B", "B-", "CCC+", "CCC", "CCC-", "CC", "C", "D",
]
_MOODYS = [
    "AAA", "AA1", "AA2", "AA3", "A1", "A2", "A3", "BAA1", "BAA2", "BAA3",
    "BA1", "BA2", "BA3", "B1", "B2", "B3", "CAA1", "CAA2", "CAA3", "CA", "C", "D",
]
NOTCH = {rating: i + 1 for i, rating in enumerate(RATING_SCALE)}
NOTCH.update({rating: i + 1 for i, rating in enumerate(_MOODYS)})

SENIORITY_RANK = {
    "SECURED": 0, "SENIOR_PREFERRED": 1, "SENIOR_UNSECURED": 1,
    "SENIOR_NON_PREFERRED": 2, "TIER2": 3, "SUBORDINATED": 3, "AT1": 4, "HYBRID": 4,
}

_DEFAULTS = {
    "isin": None, "country": "NA", "coupon_type": "FIXED",
    "callable": False, "amount_outstanding": np.nan,
}


def to_notch(rating) -> float:
    """Convertit une notation S&P, Fitch ou Moody's en cran numérique (1 = AAA)."""
    if rating is None or (isinstance(rating, float) and math.isnan(rating)):
        return float("nan")
    return float(NOTCH.get(str(rating).strip().upper(), float("nan")))


def prepare(frame: pd.DataFrame, require_target: bool) -> pd.DataFrame:
    """Valide un jeu d'obligations et ajoute les variables dérivées.

    Les lignes inexploitables (notation inconnue, maturité ou spread non positif)
    sont écartées ; leur nombre est conservé dans `attrs["dropped"]`.
    """
    df = frame.copy()
    for column, default in _DEFAULTS.items():
        if column not in df.columns:
            df[column] = default
    missing = [c for c in REQUIRED_TARGET if c not in df.columns]
    if require_target and TARGET_COLUMN not in df.columns:
        missing.append(TARGET_COLUMN)
    if missing:
        raise ValueError(f"Colonnes canoniques manquantes : {missing}")

    for column in ["issuer", "sector", "country", "currency", "seniority", "coupon_type"]:
        df[column] = df[column].astype("object").where(df[column].notna(), "NA").astype(str)
    df["seniority"] = df["seniority"].str.upper()
    df["callable"] = df["callable"].fillna(False).astype(bool)
    df["notch"] = df["rating"].map(to_notch).astype(float)
    df["maturity_years"] = pd.to_numeric(df["maturity_years"], errors="coerce")
    df["sen_rank"] = df["seniority"].map(SENIORITY_RANK).fillna(1).astype(float)
    amount = pd.to_numeric(df["amount_outstanding"], errors="coerce")
    median = amount.median()
    df["log_amt"] = np.log(amount.fillna(median if pd.notna(median) else 1.0).clip(lower=1.0))

    valid = df["notch"].notna() & (df["maturity_years"] > 0)
    if require_target:
        df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")
        valid &= df[TARGET_COLUMN] > 0
    dropped = int((~valid).sum())
    df = df[valid].reset_index(drop=True)
    df["log_t"] = np.log(df["maturity_years"].astype(float))
    if "obs_weight" not in df.columns:
        df["obs_weight"] = 1.0
    df.attrs["dropped"] = dropped
    return df


def prepare_target(bond: dict) -> dict:
    """Prépare une obligation cible décrite par ses caractéristiques."""
    missing = [c for c in REQUIRED_TARGET if bond.get(c) in (None, "")]
    if missing:
        raise ValueError(f"Caractéristiques manquantes pour l'obligation : {missing}")
    prepared = prepare(pd.DataFrame([bond]), require_target=False)
    if prepared.empty:
        raise ValueError("Obligation inexploitable : notation inconnue ou maturité non positive.")
    return prepared.iloc[0].to_dict()
