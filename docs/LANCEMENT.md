# Lancer, vérifier et arrêter la solution

La solution a trois composants. Seule l'interface est un service à démarrer
et à arrêter.

| Composant | Rôle | Démarrage | Arrêt |
|---|---|---|---|
| Interface Streamlit | Écran de démonstration | `scripts/start.sh` | `scripts/stop.sh` |
| Serveur MCP | Outils pour l'agent LLM | Lancé par le client MCP | Fermer le client MCP |
| PostgreSQL | Données et journaux | Fourni par les organisateurs | Hors de notre périmètre |

L'interface et le serveur MCP sont indépendants : chacun charge les données
et prépare ses propres modèles.

## 1. Installation (une seule fois)

Prérequis : Python 3.10 ou plus récent.

    git clone https://github.com/MohTurki75/the-bondalorians.git
    cd the-bondalorians
    python -m venv .venv
    source .venv/bin/activate          # Windows : .venv\Scripts\activate
    pip install -e ".[ml,pg,mcp,ui,dev]"      # ui inclut scikit-learn pour l'onglet Analyse

Vérification :

    pytest

## 2. Configuration

    cp .env.example .env

Le fichier `.env` n'est jamais versionné. Trois cas :

| Cas | Réglage dans `.env` |
|---|---|
| Essai sans base | Rien : les données synthétiques sont générées et utilisées |
| Fichier CSV au schéma canonique | `BONDALORIANS_CSV=chemin/vers/fichier.csv` |
| PostgreSQL (jour du hackathon) | `BONDALORIANS_DSN=host=... port=5432 dbname=... user=... password=...` |

Autres réglages : `BONDALORIANS_USE_ML=0` coupe CatBoost (préparation en
5 secondes au lieu d'une minute) ; `BONDALORIANS_PORT` change le port.

### Lire la base du jour 1 en une commande

Le script propose la correspondance des colonnes (par leur nom et leur
contenu), traduit les valeurs (séniorité, type de coupon), lit les dates au
format français, reconnaît les notations S&P, Moody's et Fitch sous leurs
formes courantes (« Baa1 *- », « BBB+ (S&P) »), dresse le profil de la base et
teste les colonnes supplémentaires.

Depuis un fichier exporté :

    python scripts/jour1.py --csv base.csv --sep ";" --as-of 2026-11-24

Depuis PostgreSQL :

    python scripts/jour1.py --dsn "host=... dbname=... user=..." --query "SELECT * FROM schema.table" --as-of 2026-11-24

À relire dans la sortie :

1. **Correspondance proposée** : colonne retenue pour chaque champ, confiance,
   autres candidates. Une confiance faible ou un champ obligatoire manquant se
   corrige à la main dans `config/schema_map.yaml` (option `--ecrire` pour
   l'écrire, l'ancien fichier étant gardé en `.bak`).
2. **Traduction des valeurs** : chaque libellé de séniorité et sa catégorie.
3. **Profil** : lignes écartées, obligations par émetteur, répartitions.
4. **Colonnes supplémentaires** :
   - ESG, ancienneté de l'émission (calculée depuis la date d'émission) :
     recommandées seulement si elles réduisent l'erreur hors échantillon d'au
     moins 1 %, émetteur connu comme inconnu ;
   - duration : jamais recommandée d'office. Une duration modifiée se calcule
     au rendement de marché, qui contient le spread : son gain peut être une
     fuite ;
   - prix et rendement (YTM) : jamais utilisés comme variables. Ils sont liés
     mécaniquement au spread ; les utiliser reviendrait à recalculer le spread
     de la cotation au lieu de l'estimer.
5. **Réglage à reporter** : la ligne `extra: [...]` à copier dans
   `config/settings.yaml`, section `peer_curve`.

Le script écrit `data/base_jour1.csv` au schéma interne ; l'interface le lit
directement :

    BONDALORIANS_CSV=data/base_jour1.csv scripts/start.sh

Ensuite, la fiche pour les agents :

    python scripts/motifs.py data/base_jour1.csv

Elle écrit `rapports/fiche_agents.json` (à donner aux agents) et
`rapports/fiche_agents.md` (à relire).

`--as-of` doit être la date des spreads de la base : c'est elle qui sert à
calculer les maturités et l'ancienneté.

### Format d'export Bloomberg (base du hackathon)

Le fichier à colonnes `pricing_date, id, ticker, nomemetteur, …, niveauzspread, ratingmoyen,
prochainedatecall` est reconnu seul : la correspondance est fixe (`config/schema_map_bloomberg.yaml`).

    python scripts/jour1.py --csv base.csv                     # estime sur la date la plus récente
    python scripts/jour1.py --csv base.csv --date 2026-03-16   # ou sur une date choisie

Ce que fait la lecture, et pourquoi :

| Colonne | Traitement | Raison |
|---|---|---|
| `ticker` | émetteur | `nomemetteur` contient le nom de l'obligation (« RZR 1 ⅜ 11/02/36 ») |
| `pricing_date` | une seule date pour estimer ; l'autre pour la dynamique | la même obligation aux deux dates serait un comparable presque parfait (fuite) |
| `datematurit_pricingobligation` | maturité, calculée depuis la date de prix de la ligne | échéance de calcul du Z-Spread |
| `datematurit`, `prochainedatecall`, `typematurit` | remboursable seulement si le call tombe plus d'un an avant l'échéance finale ; part de vie avant le call | « CALLABLE » désigne souvent un simple call au pair trois mois avant l'échéance |
| `nomsecteur` | secteur (courbe des pairs) | une dizaine de valeurs : assez d'obligations par case |
| `industrieemetteur` | industrie (motifs, dynamique) | trop fin pour la régression |
| `structureaffaire` | holdco / opco, variable candidate | subordination structurelle |
| `greenbond` | variable candidate | prime verte éventuelle |
| `dateemission` | ancienneté, variable candidate | |
| `durationbid` | candidate, signalée fuite probable | calculée au rendement de marché |
| `prixbid`, `rendement`, `discountmargenextcall` | contrôles seulement | liés mécaniquement au spread |
| `niveaumid`, `nomgroupeindustrie`, `classificationsecteuremetteur`, `premieredatesettlement` | ignorées | sens incertain ou redondant |

Les dates au format JavaScript (« Sun Nov 02 2036 00:00:00 GMT+0000 (…) »), les dates `20260316` et les
`#N/A` sont lus directement.

Sorties : `data/base_jour1.csv` (une date, pour l'interface et le backtest) et
`data/base_jour1_historique.csv` (toutes les dates). La section 7 du script mesure le mouvement du
marché entre les deux dates (par notation, séniorité, secteur, maturité), la part expliquée par le
segment, les obligations atypiques, et la stabilité de la courbe des pairs d'une date à l'autre.

### Estimer des fiches obligataires (.md)

Une fiche « # Fiche Obligataire - PY366041 Corp » décrit l'obligation proposée par le broker. Elle
passe par la même lecture que la base (dates, call, séniorité), avec les maturités comptées depuis la
date de prix de la base.

    python scripts/fiches.py dossier_fiches/ --csv data/base_jour1.csv     # ou des fichiers .md un à un

Sorties dans `rapports/fiches/` : une explication par fiche (`<ID>.md` : ce qui a été lu, points
d'attention, modèles, cascade, comparables, contrôles) et un récapitulatif (`recapitulatif.md`, `.csv`).
Dans l'interface : onglet Estimation, mode « Depuis des fiches (.md) », glisser une ou plusieurs fiches.

Cas particuliers :
- **Notation NR** : déduite de la base (obligations de même rang de l'émetteur ; sinon sa dette senior
  plus l'écart de crans habituel ; sinon médiane du secteur et du rang). L'explication donne la
  sensibilité du spread à deux crans au-dessus et en dessous : à lire avant de conclure.
- **Cotée au call** : « Date Maturité Pricing » bien avant « Date Maturité » → maturité jusqu'au call,
  remboursable. Si les deux dates sont égales, le « CALLABLE » est un call de fin de vie, ignoré.
- **Bail-in** : signalé, ce n'est pas un rang (la senior préférée d'une banque est aussi « bail-in »).
  Le rang vient de « Rang Seniorité » : « Senior Bail-In » = senior non préférée. La variable `bailin`
  n'agit que si elle est activée (`peer_curve.extra`) et que `jour1.py` montre qu'elle réduit l'erreur.
- **Obligation déjà dans la base** (même ID) : exclue de ses propres comparables ; son spread observé
  est affiché à côté de l'estimation, pour contrôle.

### Préparer PostgreSQL (jour 1)

1. Créer les référentiels et les tables de journal, avec un compte autorisé à écrire :

       psql "$BONDALORIANS_DSN" -f sql/01_ref.sql -f sql/02_app.sql

2. Adapter `config/schema_map.yaml` au schéma réel : requête, noms de
   colonnes, traduction des valeurs.
3. Démarrer, puis lire dans l'interface le nombre d'obligations chargées.

## 3. Démarrer

    scripts/start.sh

Le script lit `.env`, génère les données synthétiques si aucune source n'est
configurée, lance l'interface en arrière-plan et affiche son adresse, par
défaut http://localhost:8551. Le premier affichage prend jusqu'à une minute :
c'est la préparation des modèles et du backtest.

## 4. Vérifier

    scripts/status.sh

- Journal de l'interface : `runs/ui.log`.
- Dans l'interface : cliquer sur « Estimer » ; un Z-Spread, un intervalle et
  une cascade doivent s'afficher.
- L'onglet « Backtest et calibrage » montre l'erreur par palier.

## 5. Arrêter

    scripts/stop.sh

Les estimations et les retours restent enregistrés : dans `runs/*.jsonl` en
mode fichier, dans le schéma `app` en mode PostgreSQL.

## 6. Redémarrer après un changement

| Changement | Action |
|---|---|
| `.env`, `config/settings.yaml`, `config/schema_map.yaml`, code | `scripts/stop.sh` puis `scripts/start.sh` |
| Nouvelles données dans la base | Bouton « Tout recalculer » de l'interface |
| Retours utilisateurs à intégrer aux modèles CatBoost et aux intervalles | Bouton « Tout recalculer » |

Repartir de zéro en mode fichier (efface estimations et retours) :

    scripts/stop.sh
    rm -rf runs
    scripts/start.sh

## 7. Serveur MCP

Le serveur communique par l'entrée et la sortie standard : il n'écoute sur
aucun port. C'est le client MCP qui le lance à son démarrage et l'arrête à
sa fermeture. Exemple de déclaration pour Claude Desktop
(`claude_desktop_config.json`), à adapter à vos chemins :

    {
      "mcpServers": {
        "bondalorians": {
          "command": "/chemin/vers/the-bondalorians/.venv/bin/python",
          "args": ["-m", "bondalorians.mcp_server.server"],
          "cwd": "/chemin/vers/the-bondalorians",
          "env": {
            "BONDALORIANS_CSV": "data/synthetic_bonds.csv",
            "BONDALORIANS_CONFIG_DIR": "config"
          }
        }
      }
    }

Pour PostgreSQL, remplacer `BONDALORIANS_CSV` par `BONDALORIANS_DSN`.
Test manuel hors client (Ctrl+C pour arrêter) :

    BONDALORIANS_CSV=data/synthetic_bonds.csv python -m bondalorians.mcp_server.server

Le serveur prépare ses modèles au premier appel d'outil, qui peut donc
prendre jusqu'à une minute.

## 8. Sans les scripts (Windows)

Les scripts `start.sh`, `stop.sh` et `status.sh` demandent un shell Unix
(macOS, Linux, WSL, Git Bash). Sous Windows, lancer directement, après avoir
défini les variables d'environnement :

    python scripts/make_synthetic.py
    python -m streamlit run app/main.py

L'interface s'ouvre sur http://localhost:8551 (port fixé dans
`.streamlit/config.toml`). Arrêt : Ctrl+C dans le terminal.

## 9. Problèmes courants

| Symptôme | Cause probable | Remède |
|---|---|---|
| « Déjà démarré » | Une interface tourne déjà | `scripts/stop.sh` puis `scripts/start.sh` |
| « Le port 8551 est déjà utilisé » | Une autre application Streamlit tourne sur ce port | Mettre `BONDALORIANS_PORT=8552` dans `.env`, ou arrêter l'autre application |
| La page affiche une autre application | Même cause : le navigateur ouvre l'application qui occupe le port | Vérifier l'adresse affichée par `scripts/start.sh` |
| « Dépendances manquantes » au démarrage, ou `No module named 'sklearn'` | Paquets installés dans un autre Python que celui qui lance l'interface | Activer l'environnement virtuel (`source .venv/bin/activate`), puis `python -m pip install -e ".[ml,pg,mcp,ui,dev]"` |
| « Échec du démarrage » | Port occupé ou dépendance manquante | Lire `runs/ui.log` ; changer `BONDALORIANS_PORT` |
| « Aucune source de données » | Ni base, ni CSV, ni données synthétiques | `python scripts/make_synthetic.py` ou renseigner `.env` |
| « Colonnes canoniques manquantes » | `schema_map.yaml` ne correspond pas au schéma | Corriger la section `columns` |
| Beaucoup de lignes écartées | Notations ou séniorités non reconnues | Compléter la section `values` de `schema_map.yaml` |
| Page blanche pendant une minute | Préparation des modèles | Attendre, ou `BONDALORIANS_USE_ML=0` |
