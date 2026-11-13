#!/usr/bin/env bash
# Démarre l'interface en arrière-plan. Usage : scripts/start.sh
# Réglages lus dans le fichier .env à la racine (voir .env.example).
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -f .env ]; then set -a; . ./.env; set +a; fi
PORT="${BONDALORIANS_PORT:-8501}"
mkdir -p runs

if [ -f runs/ui.pid ] && kill -0 "$(cat runs/ui.pid)" 2>/dev/null; then
  echo "Déjà démarré (PID $(cat runs/ui.pid)) : http://localhost:$PORT"
  exit 0
fi

# Sans base ni fichier configuré, on travaille sur des données d'essai.
if [ -z "${BONDALORIANS_DSN:-}" ] && [ -z "${BONDALORIANS_CSV:-}" ] && [ ! -f data/synthetic_bonds.csv ]; then
  echo "Aucune source configurée : génération de données synthétiques."
  python scripts/make_synthetic.py
fi

nohup python -m streamlit run app/main.py \
  --server.port "$PORT" --server.headless true --browser.gatherUsageStats false \
  > runs/ui.log 2>&1 &
echo $! > runs/ui.pid

for _ in $(seq 1 30); do
  if curl -fs "http://localhost:$PORT/_stcore/health" >/dev/null 2>&1; then
    echo "Interface démarrée (PID $(cat runs/ui.pid)) : http://localhost:$PORT"
    echo "Premier affichage : jusqu'à une minute, le temps de préparer les modèles."
    exit 0
  fi
  if ! kill -0 "$(cat runs/ui.pid)" 2>/dev/null; then break; fi
  sleep 1
done
echo "Échec du démarrage. Dernières lignes de runs/ui.log :" >&2
tail -n 20 runs/ui.log >&2
rm -f runs/ui.pid
exit 1
