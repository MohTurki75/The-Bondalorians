"""Génère un univers synthétique.

python scripts/make_synthetic.py [chemin.csv] [--simple] [--hetero]

Par défaut, profil « réaliste » : il contient des effets que le modèle ne
capture pas d'origine, pour que les améliorations soient mesurables.
--simple : ancien profil log-linéaire, très favorable au modèle A.
--hetero : profil simple avec un bruit croissant selon la notation.
"""

import sys
from pathlib import Path

from bondalorians.synthetic import make_universe

args = [a for a in sys.argv[1:] if not a.startswith("--")]
path = Path(args[0] if args else "data/synthetic_bonds.csv")
path.parent.mkdir(parents=True, exist_ok=True)
if "--hetero" in sys.argv:
    universe, label = make_universe(noise_by_rating=0.03), "simple, bruit croissant"
elif "--simple" in sys.argv:
    universe, label = make_universe(), "simple"
else:
    universe, label = make_universe(profile="realiste"), "réaliste"
universe.to_csv(path, index=False)
print(f"{len(universe)} obligations, {universe['issuer'].nunique()} émetteurs, profil {label} -> {path}")
