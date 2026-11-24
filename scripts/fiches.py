"""Estime une ou plusieurs fiches obligataires (Markdown) et écrit l'explication de chacune.

    python scripts/fiches.py fiches/*.md
    python scripts/fiches.py dossier_des_fiches/ --csv data/base_jour1.csv --sortie rapports/fiches

Base : --csv, sinon BONDALORIANS_CSV, sinon data/base_jour1.csv (une seule date de prix).
Sorties : une explication par fiche (<ID>.md) et un récapitulatif (recapitulatif.md et .csv).
"""

import argparse
import os
import re
import sys
from pathlib import Path

import pandas as pd

from bondalorians.config import load_settings
from bondalorians.data.repository import load_universe_csv
from bondalorians.data.stores import JsonlStore
from bondalorians.service import Service

parser = argparse.ArgumentParser(description="Estimation des fiches obligataires.")
parser.add_argument("fiches", nargs="+", help="fichiers .md ou dossiers")
parser.add_argument("--csv", default=os.environ.get("BONDALORIANS_CSV", "data/base_jour1.csv"))
parser.add_argument("--sortie", default="rapports/fiches")
parser.add_argument("--sans-ml", action="store_true", help="sans CatBoost (plus rapide)")
args = parser.parse_args()

paths = []
for item in args.fiches:
    path = Path(item)
    paths += sorted(path.glob("*.md")) if path.is_dir() else [path]
if not paths:
    sys.exit("Aucune fiche .md trouvée.")
if not Path(args.csv).exists():
    sys.exit(f"Base introuvable : {args.csv}. Lancez d'abord scripts/jour1.py --csv <export>.")

print(f"Préparation des modèles sur {args.csv}…", flush=True)
service = Service(load_universe_csv(args.csv), load_settings(), use_ml=not args.sans_ml,
                  run_store=JsonlStore("runs/runs.jsonl"), feedback_store=JsonlStore("runs/feedback.jsonl"))
print(f"Date de prix de la base : {service.pricing_date().date()}\n")

out = Path(args.sortie)
out.mkdir(parents=True, exist_ok=True)
rows = []
for path in paths:
    try:
        estimate = service.estimate_fiche(path.read_text(encoding="utf-8"))
    except (KeyError, ValueError) as error:
        print(f"{path.name} : estimation impossible ({error})")
        rows.append({"fichier": path.name, "erreur": str(error)})
        continue
    fiche, bond, result = estimate["fiche"], estimate["bond"], estimate["result"]
    name = re.sub(r"[^A-Za-z0-9_-]+", "_", str(fiche.get("id") or path.stem)).strip("_")
    (out / f"{name}.md").write_text(estimate["markdown"], encoding="utf-8")
    low, high = result["interval_bp"] or (None, None)
    rows.append({
        "fichier": path.name, "id": fiche.get("id"), "obligation": fiche.get("nomobligation"),
        "emetteur": bond["issuer"], "emetteur_connu": result["issuer_seen"], "notation": bond["rating"],
        "notation_source": bond.get("rating_source"), "seniorite": bond["seniority"],
        "maturite_ans": round(bond["maturity_years"], 2), "z_spread_pb": result["z_spread_bp"],
        "intervalle_bas_pb": low, "intervalle_haut_pb": high, "confiance": result["confidence"].get("score"),
        "modele": result["champion"], "palier": result["tier"],
        "alertes": sum(c["status"] != "ok" for c in result["controls"]), "points_attention": len(estimate["notes"]),
    })
    print(f"{fiche.get('id')} · {fiche.get('nomobligation')} : {result['z_spread_bp']:.0f} pb "
          f"[{low:.0f} ; {high:.0f}], confiance {result['confidence'].get('score') or 0:.0f}/100"
          + (f", notation déduite {bond['rating']}" if str(bond.get("rating_source")).startswith("déduite") else ""))
    for note in estimate["notes"]:
        print("    -", note)

table = pd.DataFrame(rows)
table.to_csv(out / "recapitulatif.csv", index=False)
header = "| " + " | ".join(table.columns) + " |\n|" + "---|" * len(table.columns) + "\n"
body = "".join("| " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |\n" for row in table.itertuples(index=False))
(out / "recapitulatif.md").write_text("# Récapitulatif des fiches\n\n" + header + body, encoding="utf-8")
print(f"\nÉcrit : {out}/ (une explication par fiche, recapitulatif.md, recapitulatif.csv)")
