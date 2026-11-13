"""Backtest sur échantillon mis de côté.

python scripts/backtest.py [chemin.csv] [--ml]
Deux situations : émetteur connu (obligations retirées au hasard) et
émetteur inconnu (émetteurs entiers retirés).
"""

import sys

import pandas as pd

from bondalorians.config import load_settings
from bondalorians.quant.backtest import holdout_evaluation, summarize

args = [a for a in sys.argv[1:] if not a.startswith("--")]
use_ml = "--ml" in sys.argv
universe = pd.read_csv(args[0] if args else "data/synthetic_bonds.csv")
settings = load_settings("config/settings.yaml")

for label, by_issuer in [("Émetteur connu", False), ("Émetteur inconnu", True)]:
    evaluation = holdout_evaluation(universe, settings, use_ml=use_ml, by_issuer=by_issuer)
    print(f"\n== {label} ({len(evaluation)} obligations testées, ML={'oui' if use_ml else 'non'}) ==")
    print(summarize(evaluation).to_string(index=False))
    deciles = evaluation.groupby(pd.qcut(evaluation["score"], 5, duplicates="drop"), observed=True)
    print("\nErreur absolue moyenne (pb) par quintile de score de confiance :")
    print(deciles["abs_error_bp"].mean().round(1).to_string())
