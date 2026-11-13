"""Journal d'audit et retours utilisateurs : JSONL en local, PostgreSQL en cible."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path


class JsonlStore:
    """Stockage en ajout seul dans un fichier JSONL (une ligne par enregistrement)."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def save(self, payload: dict, prefix: str = "run") -> str:
        record_id = f"{prefix}_{uuid.uuid4().hex[:12]}"
        record = {
            "id": record_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "payload": payload,
        }
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        return record_id

    def all(self) -> list[dict]:
        if not self.path.exists():
            return []
        with open(self.path, encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    def get(self, record_id: str) -> dict | None:
        return next((r for r in self.all() if r["id"] == record_id), None)


class PostgresStore:
    """Même interface, sur une table `(id text, created_at, payload jsonb)` du schéma app."""

    def __init__(self, dsn: str, table: str, id_column: str):
        self.dsn, self.table, self.id_column = dsn, table, id_column

    def _connect(self):
        import psycopg  # dépendance optionnelle : pip install -e .[pg]

        return psycopg.connect(self.dsn)

    def save(self, payload: dict, prefix: str = "run") -> str:
        record_id = f"{prefix}_{uuid.uuid4().hex[:12]}"
        with self._connect() as connection:
            connection.execute(
                f"INSERT INTO {self.table} ({self.id_column}, payload) VALUES (%s, %s::jsonb)",
                (record_id, json.dumps(payload, ensure_ascii=False, default=str)),
            )
        return record_id

    def all(self) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT {self.id_column}, created_at, payload FROM {self.table} ORDER BY created_at"
            ).fetchall()
        return [{"id": r[0], "created_at": r[1].isoformat(), "payload": r[2]} for r in rows]

    def get(self, record_id: str) -> dict | None:
        with self._connect() as connection:
            row = connection.execute(
                f"SELECT {self.id_column}, created_at, payload FROM {self.table} "
                f"WHERE {self.id_column} = %s",
                (record_id,),
            ).fetchone()
        return {"id": row[0], "created_at": row[1].isoformat(), "payload": row[2]} if row else None
