# -*- coding: utf-8 -*-

# Kiro Gateway
# https://github.com/jwadow/kiro-gateway
# Copyright (C) 2025 Jwadow
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Resolve client effort tiers into Kiro's native additionalModelRequestFields."""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from loguru import logger

from kiro.config import (
    EFFORT_FALLBACK,
    EFFORT_FIRST_TOKEN_TIMEOUT_CAP,
    EFFORT_FIRST_TOKEN_TIMEOUT_MULTIPLIERS,
    EFFORT_ORDER,
    FIRST_TOKEN_TIMEOUT,
    MODEL_EFFORT_SCHEMA,
    NATIVE_EFFORT_ENABLED,
)

NATIVE_EFFORT_FIELD = "additionalModelRequestFields"


@dataclass
class EffortDecision:
    """Auditable outcome of resolving one requested tier against one model."""

    requested: Optional[str]
    adopted: Optional[str]
    schema_path: Optional[str]
    fragment: Optional[Dict[str, Dict[str, str]]]
    clamped: bool
    outcome: str
    reason: str

    @property
    def field(self) -> Optional[str]:
        """Return the dotted native field path, or None when omitted."""
        return f"{self.schema_path}.effort" if self.schema_path else None


def lookup_effort_schema(model_id: str) -> Optional[Tuple[str, Tuple[str, ...]]]:
    """Return the native effort schema for a resolved Kiro model ID."""
    return MODEL_EFFORT_SCHEMA.get(model_id)


def clamp_effort(requested: str, allowed: Tuple[str, ...]) -> Optional[str]:
    """Clamp a requested tier into the model's accepted enum.

    Returns None only when the request means "no reasoning" and the model has no
    native "none" tier. Unknown values use EFFORT_FALLBACK as a defensive default.
    """
    if requested in allowed:
        return requested

    if requested == "none":
        return "none" if "none" in allowed else None

    if requested not in EFFORT_ORDER:
        return EFFORT_FALLBACK

    candidates = tuple(tier for tier in allowed if tier in EFFORT_ORDER)
    if not candidates:
        return EFFORT_FALLBACK

    requested_rank = EFFORT_ORDER.index(requested)
    lower = [tier for tier in candidates if EFFORT_ORDER.index(tier) < requested_rank]
    if lower:
        return max(lower, key=EFFORT_ORDER.index)

    return min(candidates, key=EFFORT_ORDER.index)


def resolve_effort_decision(
    model_id: str,
    effort: Optional[str],
    default_tier: Optional[str] = None,
) -> EffortDecision:
    """Resolve a requested tier into an auditable native-field decision.

    Args:
        model_id: Resolved Kiro model identifier.
        effort: Client-requested tier, or None when the client was silent.
        default_tier: Tier to apply for silent clients. Callers must pass None
            when the client explicitly disabled thinking so that an explicit
            disable is never overridden by the default.
    """
    if not NATIVE_EFFORT_ENABLED:
        return EffortDecision(
            requested=effort,
            adopted=None,
            schema_path=None,
            fragment=None,
            clamped=False,
            outcome="omitted",
            reason="native_disabled",
        )

    if not effort:
        if default_tier:
            schema = lookup_effort_schema(model_id)
            if schema is not None:
                # clamp_effort also guards a misconfigured default tier.
                adopted = clamp_effort(default_tier, schema[1])
                if adopted is not None:
                    return EffortDecision(
                        requested=None,
                        adopted=adopted,
                        schema_path=schema[0],
                        fragment={schema[0]: {"effort": adopted}},
                        clamped=False,
                        outcome="native",
                        reason="default",
                    )
        return EffortDecision(
            requested=effort,
            adopted=None,
            schema_path=None,
            fragment=None,
            clamped=False,
            outcome="omitted",
            reason="no_request",
        )

    schema = lookup_effort_schema(model_id)
    if schema is None:
        return EffortDecision(
            requested=effort,
            adopted=None,
            schema_path=None,
            fragment=None,
            clamped=False,
            outcome="omitted",
            reason="unsupported_model",
        )

    path, allowed = schema
    adopted = clamp_effort(effort, allowed)
    if adopted is None:
        return EffortDecision(
            requested=effort,
            adopted=None,
            schema_path=None,
            fragment=None,
            clamped=False,
            outcome="omitted",
            reason="unsupported_none",
        )

    clamped = adopted != effort
    if clamped:
        logger.warning(
            f"Clamped effort for '{model_id}': client requested '{effort}', "
            f"sending '{adopted}' (allowed: {', '.join(allowed)})"
        )

    return EffortDecision(
        requested=effort,
        adopted=adopted,
        schema_path=path,
        fragment={path: {"effort": adopted}},
        clamped=clamped,
        outcome="native",
        reason="clamped" if clamped else "exact",
    )


def resolve_native_effort(model_id: str, effort: Optional[str]) -> Optional[EffortDecision]:
    """Return the native decision only when a field should be sent."""
    decision = resolve_effort_decision(model_id, effort)
    return decision if decision.fragment is not None else None


def resolve_first_token_timeout(
    model_id: str,
    effort: Optional[str],
    default_tier: Optional[str] = None,
) -> float:
    """Resolve the first-byte wait appropriate for a requested effort tier.

    The adopted tier takes precedence so a model-specific clamp also lowers the
    wait. Unsupported models still scale on the normalized requested tier because
    their prompt-tag reasoning path can also delay first output. Silent clients
    scale on default_tier, matching the tier that will actually be sent upstream.
    """
    if not effort or not isinstance(effort, str):
        effort = default_tier
        if not effort:
            return FIRST_TOKEN_TIMEOUT

    requested = effort.strip().lower()
    if not requested or requested == "none":
        return FIRST_TOKEN_TIMEOUT

    effective = None
    if NATIVE_EFFORT_ENABLED:
        schema = lookup_effort_schema(model_id)
        if schema is not None:
            effective = clamp_effort(requested, schema[1])
    if effective is None:
        effective = requested if requested in EFFORT_ORDER else EFFORT_FALLBACK
    if effective == "none":
        return FIRST_TOKEN_TIMEOUT

    multiplier = EFFORT_FIRST_TOKEN_TIMEOUT_MULTIPLIERS.get(
        effective,
        EFFORT_FIRST_TOKEN_TIMEOUT_MULTIPLIERS[EFFORT_FALLBACK],
    )
    timeout = FIRST_TOKEN_TIMEOUT * multiplier
    if EFFORT_FIRST_TOKEN_TIMEOUT_CAP > 0:
        timeout = min(timeout, EFFORT_FIRST_TOKEN_TIMEOUT_CAP)

    if timeout != FIRST_TOKEN_TIMEOUT:
        logger.debug(
            f"Adjusted first-token timeout for model='{model_id}': "
            f"requested='{requested}', effective='{effective}', timeout={timeout}s"
        )
    return timeout
