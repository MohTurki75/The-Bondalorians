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
