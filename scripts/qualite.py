"""Qualité de chaque modèle et tests de validation.

python scripts/qualite.py [chemin.csv] [--complet] [--sans-ml]

Par défaut, les lectures de contrôle (plus lentes) sont évaluées sur 1 500 cas
tirés au hasard ; --complet les évalue sur tous les cas.
"""

import sys

import pandas as pd

from bondalorians.analysis.quality import model_quality
from bondalorians.config import load_settings
from bondalorians.quant.estimator import ZSpreadEstimator

args = [a for a in sys.argv[1:] if not a.startswith("--")]
path = args[0] if args else "data/synthetic_bonds.csv"
estimator = ZSpreadEstimator(load_settings("config/settings.yaml"), use_ml="--sans-ml" not in sys.argv)
estimator.fit(pd.read_csv(path))
quality = model_quality(estimator, max_cases=None if "--complet" in sys.argv else 1500)

pd.set_option("display.width", 220)
print(f"=== {path} ===\n")
for finding in quality["findings"]:
    print("-", finding)
indicators = quality["tables"]["indicateurs"]
print("\nIndicateurs hors échantillon :")
print(indicators[["modele", "situation", "cas", "erreur_mediane_pct", "erreur_mediane_pb", "biais_pct",
                  "moins_de_10_pct", "variation_entre_plis", "ratio_validation_apprentissage"]].round(3).to_string(index=False))
tests = quality["tables"]["tests"]
icons = {"réussi": "OK", "échoué": "ÉCHEC", "non applicable": "-"}
grid = tests.assign(cellule=tests["statut"].map(icons) + " " + tests["valeur"])
for situation, group in grid.groupby("situation"):
    print(f"\nTests · {situation} :")
    print(group.pivot(index="modele", columns="test", values="cellule").to_string())
