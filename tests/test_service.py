import asyncio

import pytest

from bondalorians.data.stores import JsonlStore
from bondalorians.service import Service


@pytest.fixture()
def service(universe, fast_settings, tmp_path):
    return Service(
        universe, fast_settings, use_ml=False,
        run_store=JsonlStore(tmp_path / "runs.jsonl"),
        feedback_store=JsonlStore(tmp_path / "feedback.jsonl"),
    )


def test_estimate_is_logged_and_replayable(service, new_bond):
    result = service.estimate(new_bond)
    record = service.get_run(result["run_id"])
    assert record["payload"]["input"] == new_bond
    assert record["payload"]["result"]["z_spread_bp"] == result["z_spread_bp"]
    explanation = service.explain_estimate(result["run_id"])
    assert explanation["narrative"].startswith("Z-Spread juste estimé")
    assert service.validate_estimate(result["run_id"])["controls"]


def test_trade_feedback_moves_estimate_more_than_opinion(universe, fast_settings, new_bond, tmp_path):
    moves = {}
    for kind in ("trade", "opinion"):
        svc = Service(universe, fast_settings, use_ml=False,
                      run_store=JsonlStore(tmp_path / f"r_{kind}.jsonl"),
                      feedback_store=JsonlStore(tmp_path / f"f_{kind}.jsonl"))
        before = svc.estimate(new_bond)
        svc.submit_feedback(before["run_id"], kind, before["z_spread_bp"] * 1.5, reason="test")
        after = svc.estimate(new_bond)
        moves[kind] = after["z_spread_bp"] - before["z_spread_bp"]
        assert after["issuer_seen"] is True  # le retour crée une observation de l'émetteur
        assert svc.describe_universe()["n_feedback_observations"] == 1
    assert moves["trade"] > moves["opinion"] > 0


def test_feedback_is_validated(service, new_bond):
    run_id = service.estimate(new_bond)["run_id"]
    with pytest.raises(ValueError):
        service.submit_feedback(run_id, "rumeur", 100)
    with pytest.raises(ValueError):
        service.submit_feedback(run_id, "trade", -1)
    with pytest.raises(KeyError):
        service.submit_feedback("run_absent", "trade", 100)


def test_search_and_describe(service):
    found = service.search_bonds({"currency": "eur", "min_maturity": 5, "max_maturity": 7}, limit=10)
    assert 0 < len(found) <= 10
    assert all(b["currency"] == "EUR" and 5 <= b["maturity_years"] <= 7 for b in found)
    with pytest.raises(ValueError):
        service.search_bonds({"z_spread": 100})
    assert service.describe_universe()["n_bonds"] == 1500
    assert {row["palier"] for row in service.backtest_summary()} >= {"simple", "emetteur_inconnu"}
    assert service.calibration_report()["issuer_shrinkage"]["estimated"] is True


def test_mcp_server_exposes_business_tools(service, new_bond, monkeypatch):
    server = pytest.importorskip("bondalorians.mcp_server.server")
    monkeypatch.setattr(server, "_service", service)
    names = {tool.name for tool in asyncio.run(server.mcp.list_tools())}
    assert names == {
        "describe_universe", "get_bond", "search_bonds", "classify_complexity", "find_comparables",
        "estimate_zspread", "validate_estimate", "explain_estimate", "submit_feedback", "get_run",
        "backtest_summary", "calibration_report", "analyze_sample", "analyze_models",
        "analyze_comparables", "analyze_clusters",
    }
    result = server.estimate_zspread(characteristics=new_bond)
    assert server.validate_estimate(result["run_id"])["run_id"] == result["run_id"]
    assert server.classify_complexity(characteristics=new_bond)["tier"] == "complexe"


def test_analyses_for_the_agent(service, new_bond):
    import json

    pytest.importorskip("sklearn")
    run_id = service.estimate(new_bond)["run_id"]
    results = [service.analyze_sample(), service.analyze_models(include_forest=False),
               service.analyze_comparables(run_id), service.analyze_clusters("ward", run_id)]
    for result in results:
        json.dumps(result)
    assert "points" not in results[0] and results[3]["tables"]["placement"]
    with pytest.raises(ValueError):
        service.analyze_clusters("dbscan")
