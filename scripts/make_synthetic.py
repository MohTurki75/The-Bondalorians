"""Génère un univers synthétique : python scripts/make_synthetic.py [chemin.csv]"""

import sys
from pathlib import Path

from bondalorians.synthetic import make_universe

path = Path(sys.argv[1] if len(sys.argv) > 1 else "data/synthetic_bonds.csv")
path.parent.mkdir(parents=True, exist_ok=True)
universe = make_universe()
universe.to_csv(path, index=False)
print(f"{len(universe)} obligations, {universe['issuer'].nunique()} émetteurs -> {path}")
