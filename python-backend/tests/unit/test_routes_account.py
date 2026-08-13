# -*- coding: utf-8 -*-

"""
Unit tests for the /usage and /account extension routes.

Two things are covered:

1. Registration — the routes must be mounted by main.py. The Docker image runs
   `python main.py`, which previously never executed app_entry.py's
   include_router, so both endpoints served 404 in every container deployment.

2. Host resolution — GetUsageLimits only answers on q.{region}.amazonaws.com and
   returns 400 UnknownOperationException on runtime.kiro.dev. The handlers must
   resolve the host themselves rather than depending on
   install_control_plane_host_redirect() having patched the q_host property,
   since main.py does not install that extension.
"""

import ast
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from extensions.routes_account import router as account_router


_BACKEND_ROOT = Path(__file__).resolve().parents[2]


def _include_router_arg_names(module_filename):
    """Return the argument source of every `*.include_router(...)` call, in order.

    Uses the AST so the result reflects the actual call graph. Matching source
    text instead would miss real double registrations written as
    `include_router(account_router, prefix="")`, `include_router( account_router )`,
    a call split across lines, or `alias = account_router; include_router(alias)`.

    A router passed via an alias shows up under the alias name, so it will not
    silently satisfy an expected list of real router names.
    """
    tree = ast.parse((_BACKEND_ROOT / module_filename).read_text())
    names = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "include_router"
        ):
            args = [ast.unparse(arg) for arg in node.args]
            args += [f"{kw.arg}={ast.unparse(kw.value)}" for kw in node.keywords]
            names.append(", ".join(args))
    return names


# Mirrors a real GetUsageLimits payload: an expired trial, a live credit
# allowance, and one ACTIVE bonus grant.
_USAGE_RESPONSE = {
    "userInfo": {"email": "user@example.com", "provider": "google"},
    "subscriptionInfo": {
        "subscriptionTitle": "Kiro Power",
        "type": "POWER",
        "status": "Active",
        "expiryDate": "2026-12-31",
    },
    "usageBreakdownList": [
        {
            "displayName": "Credits",
            "usageLimit": 10000,
            "currentUsage": 2500,
            "resetDate": "2026-09-01",
            "freeTrialInfo": {
                "freeTrialStatus": "EXPIRED",
                "usageLimit": 0,
                "currentUsage": 0,
            },
            "bonuses": [
                {"status": "ACTIVE", "usageLimit": 500, "currentUsage": 100},
                {"status": "EXPIRED", "usageLimit": 999, "currentUsage": 0},
            ],
        }
    ],
    "nextDateReset": 1757000000,
}


def _upstream_response(status_code=200, payload=None, text=""):
    """A stand-in for the httpx.Response returned by the shared client."""
    response = Mock()
    response.status_code = status_code
    response.json = Mock(return_value=payload if payload is not None else {})
    response.text = text
    return response


def _auth_manager(q_host="https://runtime.us-east-1.kiro.dev"):
    """An auth manager whose q_host is un-redirected, as under main.py."""
    manager = Mock()
    manager.q_host = q_host
    manager.fingerprint = "test-fingerprint"
    manager.get_access_token = AsyncMock(return_value="test-access-token")
    return manager


def _build_app(auth_manager=None, upstream=None, post_side_effect=None):
    """A minimal app carrying just the router and the two app.state attributes
    the handlers read (account_manager, http_client)."""
    app = FastAPI()
    app.include_router(account_router)

    account = Mock()
    account.auth_manager = auth_manager

    account_manager = Mock()
    account_manager.get_first_account = Mock(return_value=account if auth_manager else None)

    http_client = Mock()
    if post_side_effect is not None:
        http_client.post = AsyncMock(side_effect=post_side_effect)
    else:
        http_client.post = AsyncMock(return_value=upstream or _upstream_response(payload=_USAGE_RESPONSE))

    app.state.account_manager = account_manager
    app.state.http_client = http_client
    return app


@pytest.fixture
def client():
    return TestClient(_build_app(auth_manager=_auth_manager()))


@pytest.fixture
def auth_header(valid_proxy_api_key):
    return {"Authorization": f"Bearer {valid_proxy_api_key}"}


# =============================================================================
# Route registration — the regression this change exists to prevent
# =============================================================================

class TestRouteRegistration:
    def test_main_registers_usage_and_account(self):
        """
        What it does: Verifies main.py mounts both extension routes.
        Purpose: `python main.py` is the Docker CMD; before this change it served
        404 for /usage and /account because only app_entry.py mounted them.

        Asserted against the OpenAPI schema rather than app.routes: this FastAPI
        version stores include_router() results as lazy _IncludedRouter entries
        that carry no .path until the app starts, so a path scan of app.routes
        finds nothing — not even /v1/models.
        """
        import main

        paths = main.app.openapi()["paths"]
        assert "/usage" in paths
        assert "/account" in paths
        # Sanity: the upstream routes resolve the same way, so an empty result
        # would indicate a broken assertion rather than a missing registration.
        assert "/v1/models" in paths

    def test_main_registers_exactly_three_routers(self):
        """
        What it does: Verifies main.py makes exactly three include_router calls —
        openai_router, anthropic_router, account_router.
        Purpose: Catch a duplicate registration of any router. The OpenAPI schema
        cannot show this (paths are dict keys, so duplicates collapse), so the
        check is on the call graph.

        Counted via AST rather than by matching source text: a text search for
        "include_router(account_router)" is defeated by an added keyword argument,
        inner whitespace, a line break, or passing the router through a local
        alias — all of which are real double registrations.

        A legitimate fourth router should update this expectation. That is
        deliberate: main.py is meant to stay close to upstream, so a new
        registration here is worth a second look.
        """
        assert _include_router_arg_names("main.py") == [
            "openai_router",
            "anthropic_router",
            "account_router",
        ]

    def test_app_entry_registers_no_routers(self):
        """
        What it does: Verifies app_entry.py makes no include_router calls at all.
        Purpose: It imports main's already-registered app object, so any
        registration here is a duplicate of main.py's.

        Asserted against the AST rather than by importing app_entry: that import
        installs process-wide monkey-patches (q_host redirect, role widening,
        prompt rewriting) which would leak into unrelated tests in the same
        session. An empty call list is a stronger claim than "does not mention
        account_router" and cannot be sidestepped by how the call is written.
        """
        assert _include_router_arg_names("app_entry.py") == []

    def test_no_router_is_included_twice_at_runtime(self):
        """
        What it does: Verifies the live app object holds exactly three included
        routers.
        Purpose: A runtime backstop that does not depend on source shape at all.
        This FastAPI version keeps one _IncludedRouter entry per include_router
        call, so a duplicate registration raises the count even when the extra
        call is written in a form no source scan would match.
        """
        import main

        included = [
            route
            for route in main.app.routes
            if type(route).__name__ == "_IncludedRouter"
        ]
        assert len(included) == 3


# =============================================================================
# Authentication
# =============================================================================

class TestAuthentication:
    @pytest.mark.parametrize("path", ["/usage", "/account"])
    def test_missing_key_returns_401(self, client, path):
        assert client.get(path).status_code == 401

    @pytest.mark.parametrize("path", ["/usage", "/account"])
    def test_wrong_key_returns_401(self, client, path, invalid_proxy_api_key):
        response = client.get(path, headers={"Authorization": f"Bearer {invalid_proxy_api_key}"})
        assert response.status_code == 401

    @pytest.mark.parametrize("path", ["/usage", "/account"])
    def test_valid_key_returns_200(self, client, path, auth_header):
        assert client.get(path, headers=auth_header).status_code == 200


# =============================================================================
# Control-plane host resolution — the functional half of the fix
# =============================================================================

class TestHostResolution:
    @pytest.mark.parametrize("path", ["/usage", "/account"])
    def test_request_goes_to_q_amazonaws(self, path, auth_header):
        """
        What it does: Verifies the upstream call targets q.{region}.amazonaws.com
        even though the auth manager's q_host still points at runtime.kiro.dev.
        Purpose: GetUsageLimits 400s on runtime.kiro.dev. Registering the route
        without resolving the host would yield a reachable endpoint that always
        fails.
        """
        app = _build_app(auth_manager=_auth_manager())
        response = TestClient(app).get(path, headers=auth_header)

        assert response.status_code == 200
        called_url = app.state.http_client.post.await_args.args[0]
        assert called_url == "https://q.us-east-1.amazonaws.com"

    @pytest.mark.parametrize("path", ["/usage", "/account"])
    def test_region_is_preserved(self, path, auth_header):
        app = _build_app(auth_manager=_auth_manager("https://runtime.eu-west-1.kiro.dev"))
        TestClient(app).get(path, headers=auth_header)

        assert app.state.http_client.post.await_args.args[0] == "https://q.eu-west-1.amazonaws.com"

    @pytest.mark.parametrize("path", ["/usage", "/account"])
    def test_already_redirected_host_is_left_alone(self, path, auth_header):
        """Covers the app_entry path, where the extension already rewrote q_host."""
        app = _build_app(auth_manager=_auth_manager("https://q.us-east-1.amazonaws.com"))
        TestClient(app).get(path, headers=auth_header)

        assert app.state.http_client.post.await_args.args[0] == "https://q.us-east-1.amazonaws.com"

    @pytest.mark.parametrize("path", ["/usage", "/account"])
    def test_get_usage_limits_target_header_is_sent(self, path, auth_header):
        app = _build_app(auth_manager=_auth_manager())
        TestClient(app).get(path, headers=auth_header)

        headers = app.state.http_client.post.await_args.kwargs["headers"]
        assert headers["x-amz-target"].endswith("GetUsageLimits")
        assert headers["Content-Type"] == "application/x-amz-json-1.0"


# =============================================================================
# /usage — raw passthrough
# =============================================================================

class TestUsageEndpoint:
    def test_returns_upstream_payload_verbatim(self, client, auth_header):
        response = client.get("/usage", headers=auth_header)
        assert response.json() == _USAGE_RESPONSE


# =============================================================================
# /account — field mapping
# =============================================================================

class TestAccountFieldMapping:
    @pytest.fixture
    def body(self, client, auth_header):
        return client.get("/account", headers=auth_header).json()

    def test_identity_fields(self, body):
        assert body["email"] == "user@example.com"
        assert body["accountName"] == "user@example.com"
        assert body["provider"] == "google"

    def test_subscription_fields(self, body):
        assert body["subscriptionTitle"] == "Kiro Power"
        assert body["planType"] == "POWER"
        assert body["accountStatus"] == "Active"
        assert body["subscriptionExpiryDate"] == "2026-12-31"

    def test_quota_totals_sum_trial_and_free(self, body):
        # Trial is expired (0/0), so the totals come from the credit allowance.
        assert body["totalQuota"] == 10000
        assert body["currentUsage"] == 2500
        assert body["remainingQuota"] == 7500
        assert body["usagePercentage"] == 25.0

    def test_only_active_bonuses_are_counted(self, body):
        # The EXPIRED 999-credit grant must be excluded.
        assert body["bonusQuota"] == 500
        assert body["bonusUsage"] == 100
        assert body["bonusRemaining"] == 400

    def test_reset_date_is_exposed(self, body):
        assert body["resetDate"] == "2026-09-01"

    def test_trial_flag_false_when_expired(self, body):
        assert body["isTrial"] is False

    def test_trial_flag_true_when_active(self, auth_header):
        payload = {
            "userInfo": {"email": "trial@example.com"},
            "subscriptionInfo": {},
            "usageBreakdownList": [
                {
                    "usageLimit": 0,
                    "currentUsage": 0,
                    "freeTrialInfo": {
                        "freeTrialStatus": "ACTIVE",
                        "usageLimit": 200,
                        "currentUsage": 50,
                    },
                }
            ],
        }
        app = _build_app(
            auth_manager=_auth_manager(),
            upstream=_upstream_response(payload=payload),
        )
        body = TestClient(app).get("/account", headers=auth_header).json()

        assert body["isTrial"] is True
        assert body["accountStatus"] == "Trial"
        assert body["trialQuota"] == 200
        assert body["trialUsage"] == 50
        assert body["totalQuota"] == 200


# =============================================================================
# Error paths
# =============================================================================

class TestErrorPaths:
    @pytest.mark.parametrize("path", ["/usage", "/account"])
    def test_no_initialized_account_returns_503(self, path, auth_header):
        app = _build_app(auth_manager=None)
        response = TestClient(app).get(path, headers=auth_header)

        assert response.status_code == 503
        assert "No initialized account" in response.json()["detail"]

    @pytest.mark.parametrize("path", ["/usage", "/account"])
    def test_upstream_status_is_propagated(self, path, auth_header):
        """A 400 from GetUsageLimits must surface as 400, not be masked as 502."""
        app = _build_app(
            auth_manager=_auth_manager(),
            upstream=_upstream_response(status_code=400, text="UnknownOperationException"),
        )
        response = TestClient(app).get(path, headers=auth_header)

        assert response.status_code == 400
        assert "UnknownOperationException" in response.json()["detail"]

    @pytest.mark.parametrize("path", ["/usage", "/account"])
    def test_transport_failure_returns_502(self, path, auth_header):
        app = _build_app(
            auth_manager=_auth_manager(),
            post_side_effect=RuntimeError("connection reset"),
        )
        response = TestClient(app).get(path, headers=auth_header)

        assert response.status_code == 502
        assert "connection reset" in response.json()["detail"]

    def test_empty_usage_list_does_not_crash(self, auth_header):
        payload = {"userInfo": {}, "subscriptionInfo": {}, "usageBreakdownList": []}
        app = _build_app(
            auth_manager=_auth_manager(),
            upstream=_upstream_response(payload=payload),
        )
        response = TestClient(app).get("/account", headers=auth_header)

        assert response.status_code == 200
        body = response.json()
        assert body["totalQuota"] == 0
        assert body["accountName"] == "User"
        assert body["bonusQuota"] is None
