"""Fiches obligataires au format Markdown (« # Fiche Obligataire - PY366041 Corp »).

Une fiche décrit l'obligation proposée par le broker, avec les libellés de l'export Bloomberg
(« Rang Seniorité », « Date Maturité Pricing »…). `parse_fiche` la ramène aux colonnes de la
base ; `fiche_to_bond` la passe par la même correspondance que la base (même traitement des
dates, du call, de la séniorité), puis complète une notation absente (NR).
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from bondalorians.data.automap import PRESETS
from bondalorians.data.repository import apply_schema_map
from bondalorians.data.schema import RATING_SCALE, SENIORITY_RANK, to_notch

# Libellé de la fiche (normalisé) -> colonne de l'export Bloomberg.
LABELS = {
    "nom": "nomobligation", "ticker": "ticker", "id": "id", "pays": "paysemetteur",
    "industrie": "industrieemetteur", "secteur": "nomsecteur", "groupe industrie": "nomgroupeindustrie",
    "classification secteur": "classificationsecteuremetteur",
    "classification niveau 1": "classificationsecteuremetteurniv1", "structure affaire": "structureaffaire",
    "date maturite pricing": "datematurit_pricingobligation", "date maturite": "datematurit",
    "rang seniorite": "rangseniorit_obligation", "type maturite": "typematurit", "devise": "devise",
    "type coupon": "typecoupon", "rating moyen": "ratingmoyen", "bail-in": "bondbailin",
    "green bond": "greenbond", "social bond": "socialbond", "niveau z spread": "niveauzspread",
    "rendement": "rendement", "prix": "prixbid", "duration": "durationbid", "date emission": "dateemission",
    "prochaine date call": "prochainedatecall", "date pricing": "pricing_date",
}
_LINE = re.compile(r"^\s*[-*]\s*\*\*(.+?)\s*:\*\*\s*(.*)$")


def _norm(label: str) -> str:
    import unicodedata

    text = unicodedata.normalize("NFKD", label).encode("ascii", "ignore").decode().lower()
    return re.sub(r"\s+", " ", text).strip()


def _value(text: str):
    """« `VYQW` » -> VYQW ; « FR (France) » -> FR ; « *(non renseigné)* » -> None."""
    text = text.strip().strip("`").strip()
    if not text or "non renseign" in text.lower() or text.upper() in {"#N/A", "N/A"}:
        return None
    # Code suivi de sa traduction : « Y (Oui) », « NR (Non Rated) ». Une date « (Central European…) » garde tout.
    match = re.fullmatch(r"([^()]+?)\s*\(([^()]*)\)", text)
    if match and not re.search(r"\d{4}", match.group(1)):
        return match.group(1).strip()
    return text


def parse_fiche(text: str) -> dict:
    """Champs de la fiche, sous les noms de colonnes de la base. Le premier « Nom » (obligation) l'emporte."""
    raw: dict = {}
    title = re.search(r"^#\s*Fiche Obligataire\s*-\s*(.+)$", text, flags=re.M)
    kind = re.search(r"\*\*Type\s*:\*\*\s*(.+)$", text, flags=re.M)
    for line in text.splitlines():
        match = _LINE.match(line)
        if not match:
            continue
        column = LABELS.get(_norm(match.group(1)))
        if column and column not in raw:
            raw[column] = _value(match.group(2))
    if "id" not in raw and title:
        raw["id"] = title.group(1).strip()
    raw["nomemetteur"] = raw.get("nomobligation")
    raw["_type"] = kind.group(1).strip() if kind else None
    return raw


def fiche_to_bond(raw: dict, as_of, universe: pd.DataFrame | None = None) -> tuple[dict, list[str]]:
    """Caractéristiques au schéma interne, prêtes pour `estimate`, et notes de lecture.

    `as_of` : date de prix de la base (les maturités sont comptées depuis cette date).
    `universe` : univers préparé, pour compléter une notation absente.
    """
    notes = []
    schema_map = {"source": {"as_of": str(pd.Timestamp(as_of).date())},
                  "columns": {k: v for k, v in PRESETS["bloomberg_fr"].items() if k != "pricing_date"}}
    frame = pd.DataFrame([{c: raw.get(c) for c in PRESETS["bloomberg_fr"].values()}])
    bond = apply_schema_map(frame, schema_map).iloc[0].to_dict()
    bond = {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in bond.items()}

    if raw.get("typematurit") and "CALL" in str(raw["typematurit"]).upper():
        final, workout = bond.get("final_maturity_years"), bond.get("maturity_years")
        if bond.get("callable"):
            notes.append(f"Cotée au call : maturité de calcul {workout:.1f} ans (échéance finale {final:.1f} ans) ; "
                         "traitée comme remboursable par anticipation.")
        else:
            notes.append("« CALLABLE » mais échéance de calcul = échéance finale : call de fin de vie (au pair ou "
                         "make-whole), sans effet sur le spread ; traitée comme non remboursable.")
    if str(raw.get("bondbailin") or "").upper() in {"Y", "YES", "OUI", "TRUE"}:
        notes.append("Dette « bail-in » (absorbe les pertes en résolution bancaire) : proche d'une senior "
                     "non préférée ; la base ne l'ajuste que si la variable bail-in est activée.")
    if raw.get("niveauzspread") is not None:
        notes.append(f"La fiche donne un Z-Spread ({raw['niveauzspread']}) : il n'est pas utilisé pour l'estimation.")

    notch = to_notch(bond.get("rating"))
    if notch is None or (isinstance(notch, float) and np.isnan(notch)):
        if universe is None:
            raise ValueError(f"Notation « {bond.get('rating')} » inutilisable et pas d'univers pour la déduire.")
        imputed, how = impute_rating(universe, bond)
        notes.append(f"Notation « {bond.get('rating') or 'absente'} » : {how} → {imputed} retenue. Estimation "
                     "fragile ; voir la sensibilité à la notation.")
        bond["rating_source"] = f"déduite : {how}"
        bond["rating_reported"] = bond.get("rating")
        bond["rating"] = imputed
    else:
        bond["rating_source"] = "lue sur la fiche"
    return bond, notes


def _sen_rank(seniority: str) -> float:
    return float(SENIORITY_RANK.get(str(seniority).upper(), 1))


def impute_rating(universe: pd.DataFrame, bond: dict) -> tuple[str, str]:
    """Notation probable d'une obligation non notée, d'après la base.

    1. Émetteur connu, même rang : notation médiane de ses obligations de ce rang.
    2. Émetteur connu, autre rang : sa notation senior + l'écart de crans médian observé, dans la base,
       entre ce rang et la dette senior d'un même émetteur.
    3. Émetteur inconnu : notation médiane des obligations du même secteur et du même rang.
    """
    df = universe[~universe["isin"].astype(str).str.startswith("FB_")]
    rank = _sen_rank(bond.get("seniority"))
    own = df[df["issuer"] == bond.get("issuer")]
    if len(own):
        same = own[own["sen_rank"] == rank]
        if len(same):
            return _label(same["notch"].median()), f"médiane des {len(same)} obligations de même rang de l'émetteur"
        senior = own[own["sen_rank"] <= 1]
        if len(senior):
            gap = _rank_gap(df, rank)
            return (_label(senior["notch"].median() + gap),
                    f"notation senior de l'émetteur {_label(senior['notch'].median())} + {gap:.0f} cran(s) "
                    "d'écart typique pour ce rang")
    pool = df[(df["sector"] == bond.get("sector")) & (df["sen_rank"] == rank)]
    if len(pool) < 20:
        pool = df[df["sen_rank"] == rank]
    if pool.empty:
        pool = df
    return _label(pool["notch"].median()), f"médiane de {len(pool)} obligations du même secteur et du même rang"


def _rank_gap(df: pd.DataFrame, rank: float) -> float:
    by_issuer = df.groupby(["issuer", "sen_rank"])["notch"].median().unstack()
    if rank not in by_issuer.columns or 1.0 not in by_issuer.columns:
        return 0.0
    gap = (by_issuer[rank] - by_issuer[1.0]).dropna()
    return float(np.round(gap.median())) if len(gap) >= 5 else 0.0


def _label(notch: float) -> str:
    index = int(np.clip(round(float(notch)), 1, len(RATING_SCALE))) - 1
    return RATING_SCALE[index]
