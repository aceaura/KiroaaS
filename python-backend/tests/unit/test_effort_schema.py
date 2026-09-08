# -*- coding: utf-8 -*-

"""
Unit tests for effort_schema module.

Tests for native effort tier resolution:
- clamp_effort tier clamping rules
- resolve_effort_decision outcome matrix
"""

from kiro.config import EFFORT_FALLBACK, MODEL_EFFORT_SCHEMA
from kiro.effort_schema import (
    clamp_effort,
    lookup_effort_schema,
    resolve_effort_decision,
    resolve_first_token_timeout,
    resolve_native_effort,
)


class TestLookupEffortSchema:
    """Tests for lookup_effort_schema function."""

    def test_supported_claude_model(self):
        """
        What it does: Verifies claude-opus-5 has an output_config schema
        Purpose: Ensure supported Claude models expose the native channel
        """
        print("Looking up schema for claude-opus-5...")
        schema = lookup_effort_schema("claude-opus-5")

        print(f"Schema: {schema}")
        assert schema is not None
        assert schema[0] == "output_config"

    def test_supported_gpt_model(self):
        """
        What it does: Verifies gpt-5.5 has a reasoning schema including 'none'
        Purpose: Ensure GPT models use the reasoning schema path
        """
        print("Looking up schema for gpt-5.5...")
        schema = lookup_effort_schema("gpt-5.5")

        print(f"Schema: {schema}")
        assert schema is not None
        assert schema[0] == "reasoning"
        assert "none" in schema[1]

    def test_unsupported_model_returns_none(self):
        """
        What it does: Verifies unknown/legacy models have no schema
        Purpose: Ensure models outside the table never receive native fields
        """
        print("Looking up schema for claude-sonnet-4.5...")
        schema = lookup_effort_schema("claude-sonnet-4.5")

        print(f"Schema: {schema}")
        assert schema is None


class TestClampEffort:
    """Tests for clamp_effort function."""

    def test_exact_tier_passes_through(self):
        """
        What it does: Verifies an allowed tier is returned unchanged
        Purpose: Ensure exact requests are not modified
        """
        print("Clamping 'high' against claude-opus-5 enum...")
        result = clamp_effort("high", MODEL_EFFORT_SCHEMA["claude-opus-5"][1])

        print(f"Comparing: expected='high', got='{result}'")
        assert result == "high"

    def test_clamps_down_to_nearest_lower_tier(self):
        """
        What it does: Verifies xhigh clamps down to high on claude-sonnet-4.6
        Purpose: Ensure clamping never escalates effort above the request
        """
        print("Clamping 'xhigh' against claude-sonnet-4.6 enum...")
        result = clamp_effort("xhigh", MODEL_EFFORT_SCHEMA["claude-sonnet-4.6"][1])

        print(f"Comparing: expected='high', got='{result}'")
        assert result == "high"

    def test_clamps_up_only_to_minimum_when_nothing_lower(self):
        """
        What it does: Verifies a below-minimum tier uses the model's lowest tier
        Purpose: Ensure the model still receives a valid enum value
        """
        print("Clamping 'low' against an enum without low...")
        result = clamp_effort("low", ("high", "max"))

        print(f"Comparing: expected='high', got='{result}'")
        assert result == "high"

    def test_none_accepted_when_model_supports_it(self):
        """
        What it does: Verifies 'none' passes through for GPT models
        Purpose: Ensure explicit reasoning disable reaches GPT models natively
        """
        print("Clamping 'none' against gpt-5.5 enum...")
        result = clamp_effort("none", MODEL_EFFORT_SCHEMA["gpt-5.5"][1])

        print(f"Comparing: expected='none', got='{result}'")
        assert result == "none"

    def test_none_rejected_when_model_lacks_it(self):
        """
        What it does: Verifies 'none' returns None for Claude models
        Purpose: Ensure Claude never receives a reasoning tier it lacks
        """
        print("Clamping 'none' against claude-opus-5 enum...")
        result = clamp_effort("none", MODEL_EFFORT_SCHEMA["claude-opus-5"][1])

        print(f"Comparing: expected=None, got={result}")
        assert result is None

    def test_unknown_value_uses_fallback(self):
        """
        What it does: Verifies an unknown tier falls back to EFFORT_FALLBACK
        Purpose: Ensure defensive default for out-of-schema values
        """
        print("Clamping 'ultra' against claude-opus-5 enum...")
        result = clamp_effort("ultra", MODEL_EFFORT_SCHEMA["claude-opus-5"][1])

        print(f"Comparing: expected='{EFFORT_FALLBACK}', got='{result}'")
        assert result == EFFORT_FALLBACK


class TestResolveEffortDecision:
    """Tests for resolve_effort_decision outcome matrix."""

    def test_native_exact(self):
        """
        What it does: Verifies an exact tier produces a native fragment
        Purpose: Ensure the payload fragment matches the model schema path
        """
        print("Resolving 'high' for claude-opus-5...")
        decision = resolve_effort_decision("claude-opus-5", "high")

        print(f"Decision: {decision}")
        assert decision.outcome == "native"
        assert decision.reason == "exact"
        assert decision.clamped is False
        assert decision.fragment == {"output_config": {"effort": "high"}}
        assert decision.field == "output_config.effort"

    def test_native_clamped(self):
        """
        What it does: Verifies clamping is reported on the decision
        Purpose: Ensure audits can see when the tier was lowered
        """
        print("Resolving 'xhigh' for claude-sonnet-4.6...")
        decision = resolve_effort_decision("claude-sonnet-4.6", "xhigh")

        print(f"Decision: {decision}")
        assert decision.outcome == "native"
        assert decision.reason == "clamped"
        assert decision.clamped is True
        assert decision.adopted == "high"
        assert decision.fragment == {"output_config": {"effort": "high"}}

    def test_no_request(self):
        """
        What it does: Verifies missing effort omits the native field
        Purpose: Ensure no field is fabricated when the client did not ask
        """
        print("Resolving without effort for claude-opus-5...")
        decision = resolve_effort_decision("claude-opus-5", None)

        print(f"Decision: {decision}")
        assert decision.outcome == "omitted"
        assert decision.reason == "no_request"
        assert decision.fragment is None

    def test_default_tier_applied_when_client_silent(self):
        """
        What it does: Verifies the default tier fills in for silent clients
        Purpose: Silent clients get a deterministic native tier on schema models
        """
        print("Resolving silent request with default 'medium' for claude-opus-5...")
        decision = resolve_effort_decision("claude-opus-5", None, default_tier="medium")

        print(f"Decision: {decision}")
        assert decision.outcome == "native"
        assert decision.reason == "default"
        assert decision.requested is None
        assert decision.adopted == "medium"
        assert decision.clamped is False
        assert decision.fragment == {"output_config": {"effort": "medium"}}

    def test_default_tier_skipped_for_unsupported_model(self):
        """
        What it does: Verifies the default tier is not sent to unknown models
        Purpose: Models outside the schema table must never receive the field
        """
        print("Resolving silent request with default for claude-sonnet-4.5...")
        decision = resolve_effort_decision("claude-sonnet-4.5", None, default_tier="medium")

        print(f"Decision: {decision}")
        assert decision.outcome == "omitted"
        assert decision.reason == "no_request"
        assert decision.fragment is None

    def test_default_tier_none_disables_on_gpt(self):
        """
        What it does: Verifies a 'none' default sends reasoning.effort=none on GPT
        Purpose: Allow operators to default silent clients to no reasoning
        """
        print("Resolving silent request with default 'none' for gpt-5.5...")
        decision = resolve_effort_decision("gpt-5.5", None, default_tier="none")

        print(f"Decision: {decision}")
        assert decision.outcome == "native"
        assert decision.reason == "default"
        assert decision.fragment == {"reasoning": {"effort": "none"}}

    def test_default_tier_none_omitted_on_claude(self):
        """
        What it does: Verifies a 'none' default is omitted on Claude models
        Purpose: Claude has no native 'none' tier, so nothing may be fabricated
        """
        print("Resolving silent request with default 'none' for claude-opus-5...")
        decision = resolve_effort_decision("claude-opus-5", None, default_tier="none")

        print(f"Decision: {decision}")
        assert decision.outcome == "omitted"
        assert decision.reason == "no_request"

    def test_invalid_default_tier_falls_back(self):
        """
        What it does: Verifies a misconfigured default tier lands on EFFORT_FALLBACK
        Purpose: A bad env value must degrade to the safe tier, never pass through
        """
        print("Resolving silent request with invalid default 'turbo'...")
        decision = resolve_effort_decision("claude-opus-5", None, default_tier="turbo")

        print(f"Decision: {decision}")
        assert decision.outcome == "native"
        assert decision.reason == "default"
        assert decision.adopted == EFFORT_FALLBACK
        assert decision.fragment == {"output_config": {"effort": EFFORT_FALLBACK}}

    def test_explicit_effort_overrides_default(self):
        """
        What it does: Verifies an explicit client tier wins over the default
        Purpose: The default must never override what the client asked for
        """
        print("Resolving explicit 'high' with default 'medium' for claude-opus-5...")
        decision = resolve_effort_decision("claude-opus-5", "high", default_tier="medium")

        print(f"Decision: {decision}")
        assert decision.outcome == "native"
        assert decision.reason == "exact"
        assert decision.adopted == "high"

    def test_unsupported_model(self):
        """
        What it does: Verifies an effort request for an unknown model is omitted
        Purpose: Ensure no field is sent to models outside the schema table
        """
        print("Resolving 'high' for claude-sonnet-4.5...")
        decision = resolve_effort_decision("claude-sonnet-4.5", "high")

        print(f"Decision: {decision}")
        assert decision.outcome == "omitted"
        assert decision.reason == "unsupported_model"
        assert decision.fragment is None

    def test_unsupported_none(self):
        """
        What it does: Verifies 'none' on a Claude model is omitted
        Purpose: Ensure Claude disable requests do not fabricate a field
        """
        print("Resolving 'none' for claude-opus-5...")
        decision = resolve_effort_decision("claude-opus-5", "none")

        print(f"Decision: {decision}")
        assert decision.outcome == "omitted"
        assert decision.reason == "unsupported_none"
        assert decision.fragment is None

    def test_gpt_none_produces_fragment(self):
        """
        What it does: Verifies 'none' on a GPT model produces reasoning.effort=none
        Purpose: Ensure GPT models receive the native disable signal
        """
        print("Resolving 'none' for gpt-5.5...")
        decision = resolve_effort_decision("gpt-5.5", "none")

        print(f"Decision: {decision}")
        assert decision.outcome == "native"
        assert decision.fragment == {"reasoning": {"effort": "none"}}
        assert decision.field == "reasoning.effort"

    def test_native_disabled(self, monkeypatch):
        """
        What it does: Verifies NATIVE_EFFORT_ENABLED=False omits all fields
        Purpose: Ensure the kill switch restores legacy tag behavior
        """
        print("Disabling native effort...")
        monkeypatch.setattr("kiro.effort_schema.NATIVE_EFFORT_ENABLED", False)

        print("Resolving 'high' for claude-opus-5...")
        decision = resolve_effort_decision("claude-opus-5", "high")

        print(f"Decision: {decision}")
        assert decision.outcome == "omitted"
        assert decision.reason == "native_disabled"
        assert decision.fragment is None


class TestResolveNativeEffort:
    """Tests for resolve_native_effort compatibility wrapper."""

    def test_returns_decision_when_fragment_present(self):
        """
        What it does: Verifies the wrapper returns the decision for native outcomes
        Purpose: Ensure the convenience API exposes sendable decisions
        """
        print("Resolving 'medium' for claude-opus-5...")
        decision = resolve_native_effort("claude-opus-5", "medium")

        print(f"Decision: {decision}")
        assert decision is not None
        assert decision.fragment == {"output_config": {"effort": "medium"}}

    def test_returns_none_when_omitted(self):
        """
        What it does: Verifies the wrapper returns None for omitted outcomes
        Purpose: Ensure callers can treat omission as absence
        """
        print("Resolving 'high' for claude-sonnet-4.5...")
        decision = resolve_native_effort("claude-sonnet-4.5", "high")

        print(f"Decision: {decision}")
        assert decision is None


class TestResolveFirstTokenTimeout:
    """Tests for effort-aware first-byte timeout resolution."""

    def test_no_effort_uses_base_timeout(self):
        """
        What it does: Verifies requests without effort keep the global timeout
        Purpose: Preserve existing behavior for non-reasoning requests
        """
        print("Resolving timeout without effort...")
        timeout = resolve_first_token_timeout("gpt-5.6-sol", None)

        print(f"Comparing: expected=120.0, got={timeout}")
        assert timeout == 120.0

    def test_none_uses_base_timeout(self):
        """
        What it does: Verifies explicit no-reasoning requests keep the base timeout
        Purpose: Avoid extending waits when reasoning was disabled
        """
        print("Resolving timeout for effort='none'...")
        timeout = resolve_first_token_timeout("gpt-5.6-sol", "none")

        print(f"Comparing: expected=120.0, got={timeout}")
        assert timeout == 120.0

    def test_high_effort_extends_timeout(self):
        """
        What it does: Verifies high effort scales the first-byte wait
        Purpose: Prevent premature retries for expensive reasoning
        """
        print("Resolving timeout for effort='high'...")
        timeout = resolve_first_token_timeout("gpt-5.6-sol", "high")

        # 120 * 4.0 exceeds the cap, so the wait lands on the cap itself.
        print(f"Comparing: expected=280.0, got={timeout}")
        assert timeout == 280.0

    def test_clamped_xhigh_uses_adopted_tier_timeout(self, monkeypatch):
        """
        What it does: Verifies Sonnet 4.6 xhigh uses the clamped high timeout
        Purpose: Match the actual effort sent to Kiro instead of the request

        The base timeout and cap are pinned here so the adopted tier stays
        distinguishable from the requested one. Under the production defaults
        both multipliers exceed the cap and collapse to the same value, which
        would satisfy this assertion even if clamping were ignored.
        """
        monkeypatch.setattr("kiro.effort_schema.FIRST_TOKEN_TIMEOUT", 15.0)
        monkeypatch.setattr("kiro.effort_schema.EFFORT_FIRST_TOKEN_TIMEOUT_CAP", 0.0)

        print("Resolving timeout for clamped xhigh on claude-sonnet-4.6...")
        timeout = resolve_first_token_timeout("claude-sonnet-4.6", "xhigh")

        print(f"Comparing: expected=60.0 (adopted 'high' 4.0x), got={timeout}")
        assert timeout == 60.0
        assert timeout != 90.0, "timeout followed requested 'xhigh' instead of adopted 'high'"

    def test_unsupported_model_still_uses_requested_tier(self):
        """
        What it does: Verifies prompt-tag reasoning models also get a longer wait
        Purpose: Keep legacy fake-reasoning requests from premature retries
        """
        print("Resolving timeout for xhigh on unsupported claude-sonnet-4.5...")
        timeout = resolve_first_token_timeout("claude-sonnet-4.5", "xhigh")

        print(f"Comparing: expected=280.0, got={timeout}")
        assert timeout == 280.0

    def test_unknown_tier_uses_medium_timeout(self):
        """
        What it does: Verifies unknown effort values use the safe fallback tier
        Purpose: Prevent invalid values from bypassing dynamic timeout handling
        """
        print("Resolving timeout for unknown effort='ultra'...")
        timeout = resolve_first_token_timeout("gpt-5.6-sol", "ultra")

        print(f"Comparing: expected=240.0, got={timeout}")
        assert timeout == 240.0

    def test_default_tier_scales_silent_client(self):
        """
        What it does: Verifies silent clients scale on the configured default tier
        Purpose: Match the wait to the tier actually sent upstream (medium 2.0x)
        """
        print("Resolving timeout for silent client with default 'medium'...")
        timeout = resolve_first_token_timeout("gpt-5.6-sol", None, default_tier="medium")

        print(f"Comparing: expected=240.0, got={timeout}")
        assert timeout == 240.0

    def test_default_tier_none_uses_base_timeout(self):
        """
        What it does: Verifies a 'none' default keeps the base timeout
        Purpose: Defaulting to no reasoning must not extend the wait
        """
        print("Resolving timeout for silent client with default 'none'...")
        timeout = resolve_first_token_timeout("gpt-5.6-sol", None, default_tier="none")

        print(f"Comparing: expected=120.0, got={timeout}")
        assert timeout == 120.0

    def test_explicit_effort_overrides_default_timeout(self):
        """
        What it does: Verifies an explicit tier wins over the default for timeouts
        Purpose: The default must never widen or shrink a requested tier's wait
        """
        print("Resolving timeout for explicit 'low' with default 'high'...")
        timeout = resolve_first_token_timeout("gpt-5.6-sol", "low", default_tier="high")

        print(f"Comparing: expected=180.0 (low 1.5x), got={timeout}")
        assert timeout == 180.0

    def test_non_string_effort_falls_through_to_default(self):
        """
        What it does: Verifies a non-string effort is replaced by the default tier
        Purpose: Malformed client values must not silently skip timeout scaling
        """
        print("Resolving timeout for effort=123 with default 'medium'...")
        timeout = resolve_first_token_timeout("gpt-5.6-sol", 123, default_tier="medium")

        print(f"Comparing: expected=240.0, got={timeout}")
        assert timeout == 240.0

    def test_cap_limits_scaled_timeout(self, monkeypatch):
        """
        What it does: Verifies the configured cap limits high-tier waits
        Purpose: Keep dynamic timeouts bounded in production
        """
        print("Setting effort timeout cap to 45 seconds...")
        monkeypatch.setattr("kiro.effort_schema.EFFORT_FIRST_TOKEN_TIMEOUT_CAP", 45.0)

        print("Resolving timeout for effort='max'...")
        timeout = resolve_first_token_timeout("gpt-5.6-sol", "max")

        print(f"Comparing: expected=45.0, got={timeout}")
        assert timeout == 45.0
