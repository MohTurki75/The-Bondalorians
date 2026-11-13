"""Génère un univers synthétique : python scripts/make_synthetic.py [chemin.csv] [--hetero]"""

import sys
from pathlib import Path

from bondalorians.synthetic import make_universe

args = [a for a in sys.argv[1:] if not a.startswith("--")]
path = Path(args[0] if args else "data/synthetic_bonds.csv")
path.parent.mkdir(parents=True, exist_ok=True)
# --hetero : bruit croissant avec la dégradation de la notation
universe = make_universe(noise_by_rating=0.03 if "--hetero" in sys.argv else 0.0)
universe.to_csv(path, index=False)
print(f"{len(universe)} obligations, {universe['issuer'].nunique()} émetteurs -> {path}")
