"""Jour 1 : lire la base, proposer la correspondance des colonnes, profiler, tester les colonnes en plus.

Depuis un fichier :
    python scripts/jour1.py --csv chemin/base.csv
Depuis PostgreSQL :
    python scripts/jour1.py --dsn "host=... dbname=... user=..." --query "SELECT * FROM schema.table"

Options : --as-of 2026-11-24 (date de valorisation, pour les maturités et l'ancienneté),
          --sep ";" (séparateur CSV), --ecrire (écrit config/schema_map.yaml, l'ancien gardé en .bak).

Étapes :
 1. lecture brute et aperçu des colonnes ;
 2. correspondance proposée (nom + contenu), avec confiance et autres candidates ;
 3. traduction des valeurs (séniorité, type de coupon) et notations non reconnues ;
 4. application, nettoyage, profil de la base (lignes écartées, obligations par émetteur…) ;
 5. impact des colonnes ESG, date d'émission, duration ; contrôle prix et rendement ;
 6. écriture de data/base_jour1.csv au schéma interne, utilisable par l'interface :
        BONDALORIANS_CSV=data/base_jour1.csv scripts/start.sh
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

from bondalorians.analysis.extra_columns import evaluate_extra_columns
from bondalorians.data.automap import propose_schema_map, write_schema_map
from bondalorians.data.repository import apply_schema_map
from bondalorians.data.schema import prepare

parser = argparse.ArgumentParser(description="Lecture de la base du jour 1.")
parser.add_argument("--csv")
parser.add_argument("--sep", default=None)
parser.add_argument("--dsn")
parser.add_argument("--query")
parser.add_argument("--as-of", dest="as_of")
parser.add_argument("--ecrire", action="store_true", help="écrire config/schema_map.yaml")
parser.add_argument("--sortie", default="data/base_jour1.csv")
args = parser.parse_args()

pd.set_option("display.width", 220)
pd.set_option("display.max_colwidth", 60)


def title(text: str) -> None:
    print(f"\n{'=' * 8} {text} {'=' * max(0, 70 - len(text))}")


# 1. Lecture brute.
if args.csv:
    raw = pd.read_csv(args.csv, sep=args.sep, engine="python")
elif args.dsn and args.query:
    import psycopg

    with psycopg.connect(args.dsn) as connection, connection.cursor() as cursor:
        cursor.execute("SET TRANSACTION READ ONLY")
        cursor.execute(args.query)
        raw = pd.DataFrame(cursor.fetchall(), columns=[d.name for d in cursor.description])
else:
    sys.exit("Indiquez --csv, ou --dsn et --query.")
title(f"1. Base brute : {len(raw)} lignes, {raw.shape[1]} colonnes")
overview = pd.DataFrame({
    "type": raw.dtypes.astype(str), "renseignée": raw.notna().mean().round(3),
    "valeurs distinctes": raw.nunique(),
    "exemples": [", ".join(map(str, raw[c].dropna().unique()[:4])) for c in raw.columns],
})
print(overview.to_string())

# 2. Correspondance proposée.
proposal = propose_schema_map(raw, as_of=args.as_of)
title("2. Correspondance proposée (à relire)")
print(proposal["report"].to_string(index=False))
if proposal["unmapped"]:
    print("\nColonnes non utilisées :", ", ".join(map(str, proposal["unmapped"])))

# 3. Valeurs et problèmes.
title("3. Traduction des valeurs")
for field, mapping in proposal["schema_map"]["values"].items():
    print(f"{field} :")
    for source, target in mapping.items():
        print(f"    {source!r:40} -> {target}")
for problem in proposal["problems"]:
    print("ATTENTION :", problem)
if any("obligatoire" in p for p in proposal["problems"]):
    print("\nCorrigez config/schema_map.yaml à la main, puis relancez avec --ecrire désactivé.")
if args.ecrire:
    path = write_schema_map(proposal, "config/schema_map.yaml", query=args.query)
    print(f"\nÉcrit : {path} (ancienne version dans {path}.bak)")

# 4. Application et profil.
canonical = apply_schema_map(raw, proposal["schema_map"])
prepared = prepare(canonical, require_target=True)
title("4. Profil de la base au schéma interne")
print(f"Lignes gardées : {len(prepared)} ; écartées : {prepared.attrs['dropped']} "
      "(notation inconnue, maturité ou spread non positif).")
per_issuer = prepared.groupby("issuer").size()
print(f"Émetteurs : {len(per_issuer)} ; obligations par émetteur : médiane {per_issuer.median():.0f}, "
      f"{(per_issuer == 1).mean():.0%} n'en ont qu'une, {(per_issuer >= 3).mean():.0%} en ont au moins 3.")
print("Z-Spread (pb) :", prepared["z_spread"].describe(percentiles=[.01, .25, .5, .75, .99]).round(1).to_dict())
print("Maturité (ans) :", prepared["maturity_years"].describe(percentiles=[.5]).round(2).to_dict())
for column in ("rating", "seniority", "currency", "sector", "coupon_type", "callable"):
    counts = prepared[column].value_counts()
    print(f"{column} ({len(counts)}) :", ", ".join(f"{k} {v}" for k, v in counts.head(12).items()))

# 5. Colonnes supplémentaires.
title("5. Colonnes supplémentaires : aident-elles à estimer le spread ?")
extra = evaluate_extra_columns(prepared)
for finding in extra["findings"]:
    print("-", finding)
if len(extra["tables"]["variables"]):
    print(extra["tables"]["variables"].to_string(index=False))

# 6. Sortie au schéma interne.
output = Path(args.sortie)
output.parent.mkdir(parents=True, exist_ok=True)
canonical.to_csv(output, index=False)
title("6. Fichier prêt pour l'interface")
print(f"{output} ({len(canonical)} lignes). Pour lancer l'interface sur ces données :")
print(f"    BONDALORIANS_CSV={output} scripts/start.sh")
