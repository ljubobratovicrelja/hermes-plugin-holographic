"""0.26.7: a fact_store add/change/correct that arrives without 'content'.

A small model omits the long string argument, gets a bare "Missing required argument: 'content'", and replays
the identical call until a loop guard stops the turn. Two contracts: the schema requires content for exactly the
actions that cannot work without it (search/list/stats/probe stay free of it), and the runtime error names the
action, the field and the shape the next call should take.
"""
from __future__ import annotations

import json

from hermes_plugin_holographic import FACT_STORE_SCHEMA, HolographicMemoryProvider


def _provider(tmp_path):
    provider = HolographicMemoryProvider(config={"db_path": str(tmp_path / "facts.db")})
    provider.initialize(session_id="test")
    return provider


def test_content_is_required_conditionally_for_the_actions_that_need_it():
    params = FACT_STORE_SCHEMA["parameters"]
    assert params["required"] == ["action"]  # no action is forced to carry a fact text
    (rule,) = params["allOf"]
    assert rule["if"]["properties"]["action"]["enum"] == ["add", "change", "correct"]
    assert rule["then"]["required"] == ["content"]


def test_a_missing_content_names_the_action_and_the_expected_shape(tmp_path):
    provider = _provider(tmp_path)
    for action in ("add", "change"):
        body = json.loads(provider.handle_tool_call("fact_store", {"action": action, "category": "project"}))
        assert body["missing"] == "content"
        assert body["action"] == action
        assert body["expected"] == {"action": action, "content": "<one standalone sentence>"}
        assert "Nothing was changed" in body["error"]


def test_a_rejected_call_leaves_the_store_empty(tmp_path):
    provider = _provider(tmp_path)
    provider.handle_tool_call("fact_store", {"action": "add", "category": "project"})
    assert json.loads(provider.handle_tool_call("fact_store", {"action": "stats"}))["facts"] == 0
