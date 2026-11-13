-- Tables applicatives : seul schéma où l'application écrit.
CREATE SCHEMA IF NOT EXISTS app;

-- Journal d'audit : une ligne par estimation, rejouable.
CREATE TABLE IF NOT EXISTS app.runs (
    run_id      text PRIMARY KEY,
    created_at  timestamptz NOT NULL DEFAULT now(),
    payload     jsonb NOT NULL      -- entrées, comparables, version, sorties
);

-- Retours utilisateurs.
CREATE TABLE IF NOT EXISTS app.feedback (
    feedback_id text PRIMARY KEY,
    created_at  timestamptz NOT NULL DEFAULT now(),
    payload     jsonb NOT NULL      -- run_id, type (trade/opinion), spread, motif
);

-- Registre des modèles promus.
CREATE TABLE IF NOT EXISTS app.model_registry (
    model_version text PRIMARY KEY,
    created_at    timestamptz NOT NULL DEFAULT now(),
    metrics       jsonb NOT NULL,
    is_champion   boolean NOT NULL DEFAULT false
);
