#!/usr/bin/env bash
# Arrête l'interface démarrée par scripts/start.sh.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -f runs/ui.pid ]; then
  echo "Rien à arrêter : aucune interface démarrée par scripts/start.sh."
  exit 0
fi
PID="$(cat runs/ui.pid)"
if kill -0 "$PID" 2>/dev/null; then
  kill "$PID"
  for _ in $(seq 1 10); do kill -0 "$PID" 2>/dev/null || break; sleep 0.5; done
  kill -0 "$PID" 2>/dev/null && kill -9 "$PID"
  echo "Interface arrêtée (PID $PID)."
else
  echo "Le processus $PID ne tournait plus."
fi
rm -f runs/ui.pid
