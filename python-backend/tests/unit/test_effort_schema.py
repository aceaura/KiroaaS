# -*- coding: utf-8 -*-

"""
Unit tests for kiro.effort_schema.

Covers the cc 5-tier effort vocabulary, the medium fallback for unknown tiers,
the "none" disabled-thinking path, and downward clamping for unavailable tiers
(e.g. "xhigh" on claude-sonnet-4.6).
"""

from kiro.config import (
    EFFORT_FALLBACK,
    EFFORT_ORDER,
    MODEL_EFFORT_SCHEMA,
)
from kiro.effort_schema import (
    clamp_effort,
    resolve_native_effort,
)


OPUS_ALLOWED = MODEL_EFFORT_SCHEMA["claude-opus-5"][1]
SONNET46_ALLOWED = MODEL_EFFORT_SCHEMA["claude-sonnet-4.6"][1]
GPT_ALLOWED = MODEL_EFFORT_SCHEMA["gpt-5.6-sol"][1]


class TestEffortVocabulary:
    """The single cc five-tier vocabulary and its fallback."""

    def test_five_tiers_only(self):
        """EFFORT_ORDER is exactly cc's five tiers, with no none/minimal."""
        assert EFFORT_ORDER == ("low", "medium", "high", "xhigh", "max")

    def test_fallback_is_medium(self):
        """Unknown tiers land on "medium"."""
        assert EFFORT_FALLBACK == "medium"

    def test_medium_in_every_model(self):
        """The fallback tier must be accepted by every model in the schema."""
        for model_id, (_path, allowed) in MODEL_EFFORT_SCHEMA.items():
            assert "medium" in allowed, model_id


class TestClampEffort:
    """clamp_effort: hit verbatim / none special / unknown->medium / clamp down."""

    def test_hit_returns_verbatim(self):
        for tier in EFFORT_ORDER:
            assert clamp_effort(tier, OPUS_ALLOWED) == tier

    def test_xhigh_clamps_down_on_sonnet46(self):
        """sonnet-4.6 has no xhigh, so it clamps to the highest tier below it."""
        assert clamp_effort("xhigh", SONNET46_ALLOWED) == "high"

    def test_max_hits_on_sonnet46(self):
        assert clamp_effort("max", SONNET46_ALLOWED) == "max"

    def test_none_kept_for_gpt(self):
        """GPT models accept none, so the disabled request is kept."""
        assert clamp_effort("none", GPT_ALLOWED) == "none"

    def test_none_omitted_for_claude(self):
        """Claude has no none tier; substituting one would invent reasoning."""
        assert clamp_effort("none", OPUS_ALLOWED) is None

    def test_minimal_falls_back_to_medium(self):
        """OpenAI's minimal is outside the cc vocabulary -> medium."""
        assert clamp_effort("minimal", OPUS_ALLOWED) == "medium"
        assert clamp_effort("minimal", GPT_ALLOWED) == "medium"

    def test_unknown_word_falls_back_to_medium(self):
        assert clamp_effort("bogus", OPUS_ALLOWED) == "medium"
        assert clamp_effort("xhigh1", GPT_ALLOWED) == "medium"

    def test_unrankable_tier_in_enum_does_not_crash(self):
        """A model enum may carry a tier outside EFFORT_ORDER.

        Ranking it with EFFORT_ORDER.index() raises ValueError ("tuple.index(x): x not
        in tuple"), which surfaced as a 400 on every clamped request. Such tiers must be
        skipped during comparison, not ranked.
        """
        assert clamp_effort("xhigh", ("low", "high", "server-only-tier")) == "high"

    def test_enum_with_no_rankable_tier_falls_back(self):
        """Nothing comparable left -> medium, which every model accepts."""
        assert clamp_effort("xhigh", ("none", "server-only-tier")) == "medium"


class TestResolveNativeEffort:
    """resolve_native_effort end-to-end decisions."""

    def test_master_switch_off_returns_none(self, monkeypatch):
        monkeypatch.setattr("kiro.effort_schema.NATIVE_EFFORT_ENABLED", False)
        assert resolve_native_effort("claude-opus-5", "high") is None

    def test_no_effort_returns_none(self):
        assert resolve_native_effort("claude-opus-5", None) is None

    def test_model_without_channel_returns_none(self):
        assert resolve_native_effort("claude-haiku-4.5", "high") is None

    def test_minimal_resolves_to_medium_clamped(self):
        d = resolve_native_effort("claude-opus-5", "minimal")
        assert d is not None
        assert d.adopted == "medium"
        assert d.clamped is True
        assert d.fragment == {"output_config": {"effort": "medium"}}

    def test_unknown_resolves_to_medium(self):
        d = resolve_native_effort("claude-opus-5", "bogus")
        assert d.adopted == "medium"

    def test_none_claude_omits(self):
        assert resolve_native_effort("claude-opus-5", "none") is None

    def test_none_gpt_kept(self):
        d = resolve_native_effort("gpt-5.6-sol", "none")
        assert d is not None
        assert d.adopted == "none"
        assert d.fragment == {"reasoning": {"effort": "none"}}

    def test_full_matrix_no_exception(self):
        """Every model x five-tier combination resolves without error and inside enum."""
        for model_id, (_path, allowed) in MODEL_EFFORT_SCHEMA.items():
            for tier in EFFORT_ORDER:
                d = resolve_native_effort(model_id, tier)
                if d is not None:
                    assert d.adopted in allowed, (model_id, tier, d.adopted)
