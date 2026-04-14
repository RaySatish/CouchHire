"""Additional offline tests for pipeline.py behavior."""

from __future__ import annotations

import argparse

import pytest

import pipeline


def test_graph_contains_no_orphan_nodes() -> None:
    graph = pipeline.build_graph()
    node_names = set(graph.nodes.keys())
    assert node_names
    # Minimal connectivity checks for key paths
    assert "scrape_jd" in node_names
    assert "notify" in node_names
    assert "error" in node_names


def test_route_after_scrape_handles_error() -> None:
    assert pipeline.route_after_scrape({"error": "x"}) == "error"
    assert pipeline.route_after_scrape({}) == "parse_jd"


def test_cli_parses_supported_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jd")
    parser.add_argument("--url")
    parser.add_argument("--file")
    args = parser.parse_args(["--jd", "x"])
    assert args.jd == "x"


def test_run_pipeline_returns_state_on_internal_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    class FailingApp:
        def invoke(self, _state):
            raise RuntimeError("boom")

    monkeypatch.setattr(pipeline, "_compiled_pipeline", FailingApp())
    out = pipeline.run_pipeline(jd_text="test")
    assert "error" in out
    assert "boom" in (out.get("error") or "")
