# -*- coding: utf-8 -*-

"""
Tests for the Claude model-ID dash-format extension.

The extension adds a response middleware that rewrites /v1/models model IDs from
the gateway's dotted form (claude-sonnet-4.6) to the dashed form
(claude-sonnet-4-6) recognized by Claude Code / Claude Desktop — but ONLY when a
request header (other than Authorization) contains "claude". Non-Claude clients
keep seeing the dotted form. Dashing is safe because kiro's normalize_model_name
maps dash→dot before resolution, so both forms resolve to the same model.
"""

import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from extensions.model_id_format import (
    _dashify_claude_id,
    _is_claude_client,
    _rewrite_models_body,
    install_model_id_format,
    uninstall_model_id_format,
)

# Mirrors the real /v1/models payload: dotted Claude IDs plus a non-Claude "auto".
_MODELS_PAYLOAD = {
    "object": "list",
    "data": [
        {"id": "claude-sonnet-4.6", "object": "model", "owned_by": "anthropic"},
        {"id": "claude-3.7-sonnet", "object": "model", "owned_by": "anthropic"},
        {"id": "claude-opus-4", "object": "model", "owned_by": "anthropic"},
        {"id": "auto", "object": "model", "owned_by": "anthropic"},
    ],
}


@pytest.fixture
def client():
    """A minimal app with the same /v1/models shape and the middleware installed."""
    app = FastAPI()

    @app.get("/v1/models")
    async def models():
        return _MODELS_PAYLOAD

    @app.get("/health")
    async def health():
        return {"status": "ok", "note": "claude"}  # 'claude' in body, not relevant

    install_model_id_format(app)
    try:
        yield TestClient(app)
    finally:
        uninstall_model_id_format(app)


def _ids(response):
    return [m["id"] for m in response.json()["data"]]


# =============================================================================
# _dashify_claude_id
# =============================================================================

class TestDashifyClaudeId:
    def test_dotted_version_is_dashed(self):
        assert _dashify_claude_id("claude-sonnet-4.6") == "claude-sonnet-4-6"

    def test_legacy_dotted_version_is_dashed(self):
        assert _dashify_claude_id("claude-3.7-sonnet") == "claude-3-7-sonnet"

    def test_no_dot_unchanged(self):
        assert _dashify_claude_id("claude-opus-4") == "claude-opus-4"

    def test_non_claude_id_unchanged(self):
        # No "claude" substring → never touched, even with a dot.
        assert _dashify_claude_id("gpt-4.1") == "gpt-4.1"
        assert _dashify_claude_id("auto") == "auto"

    def test_empty_unchanged(self):
        assert _dashify_claude_id("") == ""
        assert _dashify_claude_id(None) is None


# =============================================================================
# _is_claude_client
# =============================================================================

class _FakeHeaders:
    def __init__(self, pairs):
        self._pairs = pairs

    def items(self):
        return list(self._pairs)


class _FakeRequest:
    def __init__(self, pairs):
        self.headers = _FakeHeaders(pairs)


class TestIsClaudeClient:
    def test_matches_user_agent(self):
        req = _FakeRequest([("user-agent", "claude-cli/1.2.0 (external)")])
        assert _is_claude_client(req) is True

    def test_case_insensitive_value(self):
        req = _FakeRequest([("x-app", "Claude-Desktop")])
        assert _is_claude_client(req) is True

    def test_no_match(self):
        req = _FakeRequest([("user-agent", "curl/8.0"), ("accept", "*/*")])
        assert _is_claude_client(req) is False

    def test_authorization_is_ignored(self):
        # A key that happens to contain "claude" must NOT trigger the rewrite.
        req = _FakeRequest([("authorization", "Bearer claude-secret-key")])
        assert _is_claude_client(req) is False


# =============================================================================
# _rewrite_models_body
# =============================================================================

class TestRewriteModelsBody:
    def test_rewrites_claude_ids_only(self):
        out = json.loads(_rewrite_models_body(json.dumps(_MODELS_PAYLOAD).encode()))
        ids = [m["id"] for m in out["data"]]
        assert ids == ["claude-sonnet-4-6", "claude-3-7-sonnet", "claude-opus-4", "auto"]

    def test_no_change_returns_same_object_identity(self):
        # All-dashed already → nothing to do → original bytes returned unchanged.
        body = json.dumps(
            {"object": "list", "data": [{"id": "claude-opus-4"}, {"id": "auto"}]}
        ).encode()
        assert _rewrite_models_body(body) is body

    def test_malformed_json_passes_through(self):
        body = b"not json"
        assert _rewrite_models_body(body) is body

    def test_missing_data_list_passes_through(self):
        body = json.dumps({"object": "list"}).encode()
        assert _rewrite_models_body(body) is body


# =============================================================================
# Middleware end-to-end
# =============================================================================

class TestMiddleware:
    def test_claude_client_gets_dashed_ids(self, client):
        resp = client.get("/v1/models", headers={"User-Agent": "claude-cli/1.0"})
        assert resp.status_code == 200
        assert _ids(resp) == [
            "claude-sonnet-4-6",
            "claude-3-7-sonnet",
            "claude-opus-4",
            "auto",
        ]

    def test_non_claude_client_gets_dotted_ids(self, client):
        resp = client.get("/v1/models", headers={"User-Agent": "curl/8.0"})
        assert resp.status_code == 200
        assert _ids(resp) == [
            "claude-sonnet-4.6",
            "claude-3.7-sonnet",
            "claude-opus-4",
            "auto",
        ]

    def test_other_endpoints_untouched(self, client):
        # /health body contains "claude" but is not /v1/models → not rewritten.
        resp = client.get("/health", headers={"User-Agent": "claude-cli/1.0"})
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok", "note": "claude"}

    def test_response_is_valid_json_with_correct_length(self, client):
        # Rebuilt response must carry a content-length matching the new body.
        resp = client.get("/v1/models", headers={"User-Agent": "claude-cli/1.0"})
        assert int(resp.headers["content-length"]) == len(resp.content)
        assert resp.headers["content-type"].startswith("application/json")


def test_uninstall_removes_rewrite():
    app = FastAPI()

    @app.get("/v1/models")
    async def models():
        return _MODELS_PAYLOAD

    install_model_id_format(app)
    client = TestClient(app)
    dashed = [m["id"] for m in client.get(
        "/v1/models", headers={"User-Agent": "claude-cli/1.0"}
    ).json()["data"]]
    assert "claude-sonnet-4-6" in dashed

    uninstall_model_id_format(app)
    client = TestClient(app)
    dotted = [m["id"] for m in client.get(
        "/v1/models", headers={"User-Agent": "claude-cli/1.0"}
    ).json()["data"]]
    assert "claude-sonnet-4.6" in dotted
