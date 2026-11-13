"""Serveur MCP : expose des outils métier, jamais de SQL libre.

Lancement : python -m bondalorians.mcp_server.server
Le LLM orchestre ces outils et rédige ; tous les chiffres viennent d'ici.
"""

from __future__ import annotations

try:  # mcp >= 2
    from mcp.server.mcpserver import MCPServer as _Server
except ImportError:  # mcp 1.x
    from mcp.server.fastmcp import FastMCP as _Server

from bondalorians.service import Service

INSTRUCTIONS = (
    "Outils d'estimation du Z-Spread juste d'une obligation. Règles : ne jamais "
    "calculer ni inventer un chiffre ; citer uniquement les valeurs renvoyées par "
    "les outils ; toujours donner l'intervalle, le score de confiance et les réserves."
)

mcp = _Server("bondalorians", instructions=INSTRUCTIONS)
_service: Service | None = None


def service() -> Service:
    global _service
    if _service is None:
        _service = Service.from_env()
    return _service


def _bond(isin: str | None, characteristics: dict | None) -> str | dict:
    if characteristics:
        return characteristics
    if isin:
        return isin
    raise ValueError("Fournir un ISIN de l'univers ou les caractéristiques de l'obligation.")


@mcp.tool()
def describe_universe() -> dict:
    """Statistiques de l'univers : effectifs, répartition, distribution des Z-Spreads."""
    return service().describe_universe()


@mcp.tool()
def get_bond(isin: str) -> dict:
    """Caractéristiques et Z-Spread observé d'une obligation de l'univers."""
    return service().get_bond(isin)


@mcp.tool()
def search_bonds(filters: dict | None = None, limit: int = 25) -> list[dict]:
    """Recherche par égalité (issuer, sector, country, currency, rating, seniority,
    coupon_type) et par plage de maturité (min_maturity, max_maturity, en années)."""
    return service().search_bonds(filters, limit)


@mcp.tool()
def classify_complexity(isin: str | None = None, characteristics: dict | None = None) -> dict:
    """Classe le cas : simple, intermediaire ou complexe, avec la justification."""
    return service().classify_complexity(_bond(isin, characteristics))


@mcp.tool()
def find_comparables(isin: str | None = None, characteristics: dict | None = None) -> dict:
    """Comparables retenus, distances, poids et spreads ajustés à la cible."""
    return service().find_comparables(_bond(isin, characteristics))


@mcp.tool()
def estimate_zspread(isin: str | None = None, characteristics: dict | None = None) -> dict:
    """Estime le Z-Spread juste (pb), son intervalle et son score de confiance.

    `characteristics` attend : issuer, sector, currency, rating, seniority,
    maturity_years ; optionnels : country, coupon_type, callable, amount_outstanding.
    Renvoie aussi un `run_id` pour l'audit, l'explication et le retour utilisateur.
    """
    return service().estimate(_bond(isin, characteristics))


@mcp.tool()
def validate_estimate(run_id: str) -> dict:
    """Contrôles de cohérence d'une estimation enregistrée."""
    return service().validate_estimate(run_id)


@mcp.tool()
def explain_estimate(run_id: str) -> dict:
    """Cascade en points de base, comparables, modèles et récit d'une estimation."""
    return service().explain_estimate(run_id)


@mcp.tool()
def submit_feedback(run_id: str, kind: str, z_spread_bp: float, reason: str = "", author: str = "") -> dict:
    """Enregistre un retour : kind = 'trade' (spread réellement traité) ou 'opinion'."""
    return service().submit_feedback(run_id, kind, z_spread_bp, reason, author)


@mcp.tool()
def get_run(run_id: str) -> dict:
    """Rejeu d'audit : entrées et sorties complètes d'une estimation."""
    return service().get_run(run_id)


@mcp.tool()
def backtest_summary() -> list[dict]:
    """Erreur hors échantillon par palier (pb) et modèle champion retenu."""
    return service().backtest_summary()


if __name__ == "__main__":
    mcp.run()
