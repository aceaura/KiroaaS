# -*- coding: utf-8 -*-

"""
Tests for the control-plane host redirect extension.

runtime.kiro.dev only serves the chat operation. The extension redirects
KiroAuthManager.q_host (used by ListAvailableModels and /mcp) to
q.{region}.amazonaws.com, leaves api_host (chat) alone, and forces
account_manager._is_runtime_endpoint() to False so dynamic model fetching runs.
"""

import pytest

import kiro.account_manager as account_manager
from kiro.auth import KiroAuthManager
from extensions.control_plane_host import (
    install_control_plane_host_redirect,
    to_q_amazonaws_host,
    uninstall_control_plane_host_redirect,
)


@pytest.fixture(autouse=True)
def clean_redirect():
    uninstall_control_plane_host_redirect()
    install_control_plane_host_redirect()
    yield
    uninstall_control_plane_host_redirect()


def _manager(region="us-east-1"):
    return KiroAuthManager(refresh_token="rt", region=region)


# =============================================================================
# to_q_amazonaws_host — the pure resolver, usable without installing the redirect
# =============================================================================

def test_helper_rewrites_runtime_host():
    assert (
        to_q_amazonaws_host("https://runtime.us-east-1.kiro.dev")
        == "https://q.us-east-1.amazonaws.com"
    )


def test_helper_preserves_region():
    assert (
        to_q_amazonaws_host("https://runtime.eu-west-1.kiro.dev")
        == "https://q.eu-west-1.amazonaws.com"
    )


def test_helper_is_idempotent():
    # Applying it to an already-redirected host must not rewrite it again.
    once = to_q_amazonaws_host("https://runtime.us-east-1.kiro.dev")
    assert to_q_amazonaws_host(once) == once


def test_helper_passes_through_unknown_host():
    assert to_q_amazonaws_host("https://example.invalid") == "https://example.invalid"


def test_helper_accepts_trailing_slash():
    assert (
        to_q_amazonaws_host("https://runtime.us-east-1.kiro.dev/")
        == "https://q.us-east-1.amazonaws.com"
    )


def test_helper_handles_empty_host():
    assert to_q_amazonaws_host("") == ""


def test_q_host_redirected_for_runtime():
    manager = _manager()
    # Sanity: upstream config resolves q_host to runtime.kiro.dev.
    assert manager._q_host == "https://runtime.us-east-1.kiro.dev"
    assert manager.q_host == "https://q.us-east-1.amazonaws.com"


def test_region_preserved_in_redirect():
    manager = _manager(region="eu-west-1")
    assert manager.q_host == "https://q.eu-west-1.amazonaws.com"


def test_api_host_left_untouched():
    manager = _manager()
    # Chat must keep using runtime.kiro.dev.
    assert manager.api_host == "https://runtime.us-east-1.kiro.dev"


def test_non_runtime_q_host_passes_through():
    manager = _manager()
    manager._q_host = "https://q.us-east-1.amazonaws.com"
    assert manager.q_host == "https://q.us-east-1.amazonaws.com"


def test_is_runtime_endpoint_forced_false():
    manager = _manager()
    # api_host is still runtime, but fetching is now valid → must report False.
    assert manager.api_host.startswith("https://runtime.")
    assert account_manager._is_runtime_endpoint(manager) is False


def test_uninstall_restores_q_host_and_runtime_check():
    manager = _manager()
    assert manager.q_host == "https://q.us-east-1.amazonaws.com"
    assert account_manager._is_runtime_endpoint(manager) is False

    uninstall_control_plane_host_redirect()

    assert manager.q_host == "https://runtime.us-east-1.kiro.dev"
    assert account_manager._is_runtime_endpoint(manager) is True
