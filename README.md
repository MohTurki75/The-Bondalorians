# The Bondalorians

Préparation du hackathon CDC. Agent IA qui estime le **Z-Spread juste** d'une
obligation à partir d'un univers d'environ 4 000 obligations stockées dans
PostgreSQL. Le gérant compare ensuite lui-même ce chiffre au spread proposé
par son broker.

## Principe

Le LLM orchestre et explique ; il ne calcule aucun chiffre. Tous les nombres
viennent d'outils déterministes, journalisés et rejouables.

## Ce que renvoie une estimation

- le Z-Spread juste, en points de base ;
- un intervalle à 90 % (prédiction conforme) ;
- un score de confiance sur 100 ;
- le palier du cas : simple, intermédiaire ou complexe ;
- les comparables retenus et une cascade en points de base ;
- les contrôles de cohérence et les réserves.

## Méthode

1. **Courbe des pairs** : régression du log du spread sur la notation, la
   maturité, la séniorité, le secteur, la devise.
2. **Modèle A (transparent)** : courbe des pairs, plus l'écart propre à
   l'émetteur (rétréci s'il a peu d'obligations), plus l'écart des comparables
   les plus proches.
3. **Modèles CatBoost** : B, indépendant, sert de contrôle ; C corrige
   l'erreur du modèle A.
4. **Backtest** : il désigne le champion par palier et calibre l'intervalle,
   en séparant émetteur connu et émetteur jamais vu.
5. **Confiance apprise** : un modèle prédit la difficulté de chaque cas ; il
   fixe le score et la largeur de l'intervalle.
6. **Retours utilisateurs** : un spread traité ou un avis devient une
   observation pondérée, prise en compte immédiatement.

Les coefficients sont appris sur les données quand c'est possible : poids de
la courbe émetteur, poids de distance des comparables, champion, confiance.

- Déroulé pas à pas : [docs/FONCTIONNEMENT.md](docs/FONCTIONNEMENT.md)
- Lire l'onglet « Analyse » (graphiques, clustering, glossaire) : [docs/ANALYSE.md](docs/ANALYSE.md)
- Décisions et limites : [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- Synthèse du problème et des modèles, à donner à un assistant : [docs/PROBLEME_ESTIMATION.md](docs/PROBLEME_ESTIMATION.md)

## Démarrage

    pip install -e ".[ml,pg,mcp,ui,dev]"
    python scripts/make_synthetic.py        # univers synthétique dans data/
    python scripts/backtest.py --ml         # erreur et couverture hors échantillon
    python scripts/ablation.py --ml         # effet de chaque amélioration
    python scripts/export_analyses.py       # rapport Markdown des analyses, pour un LLM
    python scripts/qualite.py               # indicateurs et tests de chaque modèle
    python scripts/jour1.py --csv base.csv  # jour 1 : correspondance des colonnes, profil, colonnes en plus
    python scripts/motifs.py data/base_jour1.csv   # fiche pour les agents : aiguillage, poches, règles
    python scripts/fiches.py dossier_fiches/       # estime des fiches obligataires .md, une explication par fiche
    pytest

Interface de démonstration :

    scripts/start.sh                        # démarre, affiche l'adresse
    scripts/status.sh
    scripts/stop.sh

Installation, configuration, arrêt et dépannage :
[docs/LANCEMENT.md](docs/LANCEMENT.md).

Serveur MCP sur données synthétiques :

    BONDALORIANS_CSV=data/synthetic_bonds.csv python -m bondalorians.mcp_server.server

Serveur MCP sur PostgreSQL (jour 1) :

    psql "$DSN" -f sql/01_ref.sql -f sql/02_app.sql
    BONDALORIANS_DSN="$DSN" python -m bondalorians.mcp_server.server

## Le jour 1

Le seul fichier à adapter au schéma réel est `config/schema_map.yaml`.
`scripts/jour1.py` le propose automatiquement (voir docs/LANCEMENT.md) ; l'export Bloomberg à deux
dates de prix est reconnu tel quel (`config/schema_map_bloomberg.yaml`). Les
paramètres métier (paliers, poids, seuils) sont dans `config/settings.yaml`.

## Structure

    config/                 correspondance de schéma et paramètres
    sql/                    référentiels (ref) et tables applicatives (app)
    src/bondalorians/
      data/                 schéma canonique, accès CSV et PostgreSQL, journaux
      domain/               paliers et contrôles
      quant/                courbe des pairs, estimateur, réglage, CatBoost, confiance, backtest
      explain/              cascade en points de base, récit
      analysis/             compréhension : échantillon, modèles, forêt aléatoire, clustering
      learning/             retours utilisateurs
      mcp_server/           outils MCP
      service.py            façade unique (utilisable sans MCP ni LLM)
    app/                    interface Streamlit
    scripts/                démarrage et arrêt, données synthétiques, backtest, mesures
    tests/

## Outils MCP

`describe_universe`, `get_bond`, `search_bonds`, `classify_complexity`,
`find_comparables`, `estimate_zspread`, `validate_estimate`,
`explain_estimate`, `submit_feedback`, `get_run`, `backtest_summary`,
`calibration_report`, `analyze_sample`, `analyze_models`, `analyze_comparables`,
`analyze_clusters`, `analysis_report`, `model_quality`, `agent_brief`, `patterns_for_bond`.

## État

La solution tourne de bout en bout sur données synthétiques, interface comprise. Les erreurs
mesurées sur ces données ne disent rien de la précision réelle : elles
valident la chaîne, pas la méthode.
