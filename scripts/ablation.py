"""Mesure l'effet de chaque amélioration sur un échantillon mis de côté.

python scripts/ablation.py [chemin.csv] [--ml]
Chaque ligne ajoute une amélioration à la précédente.
"""

import copy
import sys

import pandas as pd

from bondalorians.config import load_settings
from bondalorians.quant.backtest import holdout_evaluation

args = [a for a in sys.argv[1:] if not a.startswith("--")]
use_ml = "--ml" in sys.argv
universe = pd.read_csv(args[0] if args else "data/synthetic_bonds.csv")
base = load_settings("config/settings.yaml")


def variant(k, tune, adaptive, corrector):
    settings = copy.deepcopy(base)
    settings["comparables"].update(issuer_shrinkage_k=k, tune=tune)
    settings["calibration"]["adaptive"] = adaptive
    settings["ml"]["corrector"] = corrector
    return settings


variants = [
    ("0. Départ (k = 2, poids manuels)", variant(2.0, False, False, False), False),
    ("1. + k estimé", variant("auto", False, False, False), False),
    ("2. + poids de distance réglés", variant("auto", True, False, False), False),
    ("3. + intervalle adaptatif", variant("auto", True, True, False), False),
]
if use_ml:
    variants.append(("4. + CatBoost (B et correcteur C)", variant("auto", True, True, True), True))

rows = []
for label, settings, ml in variants:
    for situation, by_issuer in [("émetteur connu", False), ("émetteur inconnu", True)]:
        e = holdout_evaluation(universe, settings, use_ml=ml, by_issuer=by_issuer)
        if not by_issuer:
            e = e[e["tier"] != "emetteur_inconnu"]
        relative = e["abs_error_bp"] / e["truth_bp"]
        # Le score classe-t-il bien les cas ? Corrélation de rang score / erreur relative.
        rank = e["score"].corr(relative, method="spearman")
        rows.append({
            "variante": label, "situation": situation, "n": len(e),
            "mediane_abs_bp": round(e["abs_error_bp"].median(), 1),
            "mae_bp": round(e["abs_error_bp"].mean(), 1),
            "couverture": round(e["covered"].mean(), 3),
            "largeur_relative": round(e["relative_width"].mean(), 3),
            "rang_score_erreur": round(rank, 2),
        })
print(pd.DataFrame(rows).to_string(index=False))
