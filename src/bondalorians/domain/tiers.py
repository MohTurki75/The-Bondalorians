"""Classement d'un cas en palier : simple, intermédiaire ou complexe."""

from __future__ import annotations

import numpy as np

SIMPLE, INTERMEDIATE, COMPLEX = "simple", "intermediaire", "complexe"


def classify(
    target: dict,
    issuer_currency: np.ndarray,
    issuer_seniority: np.ndarray,
    issuer_maturity: np.ndarray,
    settings: dict,
) -> tuple[str, str, bool]:
    """Renvoie (palier, justification, émetteur_connu).

    Les trois tableaux décrivent les obligations du même émetteur présentes
    dans l'univers (cible exclue).
    """
    cfg = settings["tiers"]
    issuer_seen = len(issuer_currency) > 0
    if not issuer_seen:
        return COMPLEX, "Émetteur absent de l'univers : aucune courbe émetteur.", False
    if target["seniority"] in cfg["complex_seniorities"]:
        return COMPLEX, f"Séniorité {target['seniority']} : instrument subordonné ou hybride.", True

    same = (issuer_currency == target["currency"]) & (issuer_seniority == target["seniority"])
    count = int(same.sum())
    if count >= cfg["min_issuer_bonds_simple"]:
        low, high = issuer_maturity[same].min(), issuer_maturity[same].max()
        if low <= target["maturity_years"] <= high:
            return SIMPLE, (
                f"{count} obligations de l'émetteur, même devise et séniorité, "
                f"encadrant la maturité ({low:.1f} à {high:.1f} ans)."
            ), True
        return INTERMEDIATE, (
            f"{count} obligations de l'émetteur, mais maturité cible hors plage "
            f"({low:.1f} à {high:.1f} ans) : extrapolation."
        ), True
    return INTERMEDIATE, (
        f"Émetteur connu, mais seulement {count} obligation(s) de même devise et séniorité."
    ), True
