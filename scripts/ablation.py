"""Mesure l'effet de chaque amélioration sur des obligations mises de côté.

python scripts/ablation.py [chemin.csv ...] [--ml] [--historique]

Chaque ligne ajoute une amélioration à la précédente. Deux situations :
émetteur connu (obligations retirées au hasard) et émetteur inconnu
(émetteurs entiers retirés). Avec plusieurs fichiers, un tableau par fichier.

--historique : ajoute les premières étapes (k fixe, poids manuels).
--ml : ajoute CatBoost (modèles B et C) en dernière étape.
"""

import copy
import sys

import pandas as pd

from bondalorians.config import load_settings
from bondalorians.quant.backtest import holdout_evaluation

paths = [a for a in sys.argv[1:] if not a.startswith("--")] or ["data/synthetic_bonds.csv"]
use_ml = "--ml" in sys.argv
base = load_settings("config/settings.yaml")


def variant(k="auto", tune=True, adaptive=True, corrector=False, enriched=False, slope=False, dispersion=False):
    settings = copy.deepcopy(base)
    settings["comparables"].update(issuer_shrinkage_k=k, tune=tune, issuer_slope=slope)
    settings["calibration"]["adaptive"] = adaptive
    settings["ml"]["corrector"] = corrector
    settings["peer_curve"]["enriched"] = enriched
    settings["confidence"]["dispersion_features"] = dispersion
    return settings


variants = []
if "--historique" in sys.argv:
    variants += [
        ("Départ : k = 2, poids manuels", variant(k=2.0, tune=False, adaptive=False), False),
        ("+ k estimé", variant(tune=False, adaptive=False), False),
        ("+ poids de distance réglés", variant(adaptive=False), False),
    ]
variants += [
    ("Référence : version précédente", variant(), False),
    ("+ courbe des pairs enrichie", variant(enriched=True), False),
    ("+ pente de maturité par émetteur", variant(enriched=True, slope=True), False),
    ("+ dispersion dans la confiance", variant(enriched=True, slope=True, dispersion=True), False),
]
if use_ml:
    variants.append(("+ CatBoost (B et correcteur C)",
                     variant(enriched=True, slope=True, dispersion=True, corrector=True), True))

for path in paths:
    universe = pd.read_csv(path)
    rows = []
    for label, settings, ml in variants:
        for situation, by_issuer in [("connu", False), ("inconnu", True)]:
            e = holdout_evaluation(universe, settings, use_ml=ml, by_issuer=by_issuer)
            if not by_issuer:
                e = e[e["tier"] != "emetteur_inconnu"]
            relative = e["abs_error_bp"] / e["truth_bp"]
            complex_ = e[e["tier"] == "complexe"]["abs_error_bp"]
            rows.append({
                "variante": label, "émetteur": situation, "n": len(e),
                "erreur_méd_pb": round(e["abs_error_bp"].median(), 1),
                "erreur_méd_%": round(100 * relative.median(), 1),
                "complexe_méd_pb": round(complex_.median(), 1) if len(complex_) else None,
                "couverture": round(e["covered"].mean(), 3),
                "largeur": round(e["relative_width"].mean(), 3),
                # Le score classe-t-il bien les cas ? Négatif = bon (score haut, erreur basse).
                "rang_score_erreur": round(e["score"].corr(relative, method="spearman"), 2),
            })
    print(f"\n=== {path} ({len(universe)} obligations, ML={'oui' if use_ml else 'non'}) ===")
    print(pd.DataFrame(rows).to_string(index=False))
