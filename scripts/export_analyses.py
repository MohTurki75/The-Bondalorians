"""Écrit le rapport Markdown des analyses, à donner à un LLM pour interprétation.

Exemples :
    python scripts/export_analyses.py
    python scripts/export_analyses.py --isin XS0000000010
    python scripts/export_analyses.py --bond '{"issuer": "X", "sector": "Banques", "currency": "EUR",
        "rating": "BBB", "seniority": "SENIOR_UNSECURED", "maturity_years": 5}'
    python scripts/export_analyses.py --out rapports/mon_rapport.md --sans-foret

Source : BONDALORIANS_DSN (PostgreSQL), sinon BONDALORIANS_CSV, sinon les
données synthétiques. CatBoost est utilisé sauf si BONDALORIANS_USE_ML=0.
"""

import argparse
import json
import os
from datetime import date
from pathlib import Path

from bondalorians.analysis.report import build_report
from bondalorians.config import load_settings
from bondalorians.data import repository
from bondalorians.quant.estimator import ZSpreadEstimator

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--isin", help="obligation de l'univers à situer")
parser.add_argument("--bond", help="caractéristiques d'une obligation à situer (JSON)")
parser.add_argument("--out", help="fichier de sortie (défaut : rapports/analyses_<source>_<date>.md)")
parser.add_argument("--sans-foret", action="store_true", help="ne pas calculer la forêt aléatoire (plus rapide)")
args = parser.parse_args()

settings = load_settings("config/settings.yaml")
use_ml = os.environ.get("BONDALORIANS_USE_ML", "1") != "0"
if os.environ.get("BONDALORIANS_DSN"):
    universe = repository.load_universe_postgres(
        os.environ["BONDALORIANS_DSN"], repository.load_schema_map("config/schema_map.yaml"))
    source, tag = "PostgreSQL", "base"
else:
    path = os.environ.get("BONDALORIANS_CSV", "data/synthetic_bonds.csv")
    universe = repository.load_universe_csv(path)
    synthetic = Path(path).name.startswith("synthetic")
    source = "données synthétiques" if synthetic else f"fichier {Path(path).name}"
    tag = "donnees_synthetiques" if synthetic else Path(path).stem

print(f"Préparation des modèles sur {len(universe)} obligations ({source})…")
estimator = ZSpreadEstimator(settings, use_ml=use_ml).fit(universe)
result = None
if args.isin or args.bond:
    result = estimator.estimate(args.isin or json.loads(args.bond))
    result["run_id"] = "rapport"
print("Calcul des analyses…")
report = build_report(estimator, result, source, include_forest=not args.sans_foret)
out = Path(args.out or f"rapports/analyses_{tag}_{date.today().isoformat()}.md")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(report, encoding="utf-8")
print(f"Rapport écrit : {out} ({len(report.split())} mots)")
