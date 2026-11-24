"""Motifs pour orienter les agents : poches de biais, table d'aiguillage, fiche JSON.

python scripts/motifs.py [chemin.csv] [--sans-ml] [--avec-lectures]

Écrit rapports/fiche_agents.json (à donner aux agents, ou servie par l'outil MCP
`agent_brief`) et rapports/fiche_agents.md (lecture humaine).
--avec-lectures : fait aussi concourir l'entonnoir dans l'aiguillage (plus lent).
"""

import sys
from pathlib import Path

import pandas as pd

from bondalorians.analysis.agent_brief import build_brief, save_brief
from bondalorians.analysis.report import md_table
from bondalorians.config import load_settings
from bondalorians.quant.estimator import ZSpreadEstimator

args = [a for a in sys.argv[1:] if not a.startswith("--")]
path = args[0] if args else "data/synthetic_bonds.csv"
estimator = ZSpreadEstimator(load_settings("config/settings.yaml"), use_ml="--sans-ml" not in sys.argv)
print(f"Préparation des modèles sur {path}…")
estimator.fit(pd.read_csv(path))
readings = None
if "--avec-lectures" in sys.argv:
    from bondalorians.quant.topdown import readings_out_of_fold

    readings = readings_out_of_fold(estimator, ("Entonnoir avec maturité",), max_cases=3000)
print("Recherche des motifs…")
brief = build_brief(estimator, readings=readings, source=Path(path).name)
Path("rapports").mkdir(exist_ok=True)
save_brief(brief, "rapports/fiche_agents.json")

lines = ["# Fiche pour les agents", "", f"- Généré le : {brief['meta']['genere_le']} ; source : {brief['meta']['source']}",
         f"- Mode d'emploi : {brief['meta']['mode_d_emploi']}", "", "## Règles", ""]
lines += [f"- **{r['id']} ({r['type']})** {r['consigne']} _{r['preuve']}_" for r in brief["regles"]]
lines += ["", "## Aiguillage", "", md_table(pd.DataFrame(brief["aiguillage"]), {
    "notation": "Notation", "dette": "Dette", "emetteur": "Émetteur", "cas": "Cas",
    "erreur_actuelle_pct": "Erreur actuelle (%)", "meilleure_methode": "Meilleure méthode",
    "erreur_meilleure_pct": "Son erreur (%)", "methode_recommandee": "Recommandation",
    "couverture_intervalle": "Couverture", "facteur_intervalle": "Facteur intervalle"}, digits=2), "",
          "## Poches de biais de l'estimation retenue", "",
          md_table(pd.DataFrame(brief["poches"]), {"situation": "Situation", "segment": "Segment", "obligations": "Obligations",
                                                   "emetteurs": "Émetteurs", "ecart_pct": "Écart (%)", "action": "Action"}),
          "", "## Effets non captés par la courbe des pairs", "",
          md_table(pd.DataFrame(brief["poches_courbe_des_pairs"]), {"situation": "Situation", "segment": "Segment",
                                                                   "obligations": "Obligations", "ecart_pct": "Écart (%)"})]
Path("rapports/fiche_agents.md").write_text("\n".join(lines), encoding="utf-8")
print(f"{len(brief['regles'])} règles, {len(brief['poches'])} poches, {len(brief['aiguillage'])} cases d'aiguillage.")
print("Écrit : rapports/fiche_agents.json et rapports/fiche_agents.md")
for rule in brief["regles"]:
    print(f"- {rule['id']} [{rule['type']}] {rule['consigne']}")
