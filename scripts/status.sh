#!/usr/bin/env bash
# Indique si l'interface tourne.
cd "$(dirname "$0")/.."
if [ -f .env ]; then set -a; . ./.env; set +a; fi
PORT="${BONDALORIANS_PORT:-8501}"
if [ -f runs/ui.pid ] && kill -0 "$(cat runs/ui.pid)" 2>/dev/null; then
  echo "En marche (PID $(cat runs/ui.pid)) : http://localhost:$PORT"
else
  echo "À l'arrêt."
fi
