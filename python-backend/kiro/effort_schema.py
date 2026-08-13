# -*- coding: utf-8 -*-

# Kiro Gateway
# https://github.com/jwadow/kiro-gateway
# Copyright (C) 2025 Jwadow
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

"""
Resolve a requested effort tier into Kiro's native ``additionalModelRequestFields``.

Kiro validates this field before invoking the model and rejects anything outside the
target model's enum, so every decision has to be made against MODEL_EFFORT_SCHEMA
(kiro/config.py) rather than guessed from the model name. Three properties follow from
that and are the reason this module exists:

- The schema path is per-family and mutually exclusive. Claude accepts only
  ``output_config.effort``, GPT-5.6 only ``reasoning.effort``. Crossing them is an
  immediate 400, not a silent downgrade.
- A model absent from the table has no native channel and must not receive the field at
  all -- five models answer 400 "not supported for this model".
- Enums differ within a family, so a tier the client asked for may be unavailable and
  has to be clamped.

Kept separate from converters_core (already ~1600 lines) because this is a self-contained
capability with no I/O: pure functions over the config table, straightforward to test in
isolation.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from loguru import logger

from kiro.config import (
    EFFORT_FALLBACK,
    EFFORT_ORDER,
    MODEL_EFFORT_SCHEMA,
    NATIVE_EFFORT_ENABLED,
)

# Top-level payload key, camelCase. Sibling of conversationState / profileArn.
# The inner keys are snake_case. Misspelling either side is a 400, so the literal lives
# in exactly one place.
NATIVE_EFFORT_FIELD = "additionalModelRequestFields"


@dataclass
class EffortDecision:
    """The outcome of resolving one requested tier against one model.

    Attributes:
        fragment: Value for the ``additionalModelRequestFields`` key, e.g.
            ``{"output_config": {"effort": "high"}}``.
        adopted: The tier actually sent, always inside the model's enum.
        clamped: True when ``adopted`` differs from what the client requested.
        reason: Human-readable explanation, for logs. Empty when nothing was clamped.
    """
    fragment: Dict[str, Dict[str, str]]
    adopted: str
    clamped: bool
    reason: str = ""


def lookup_effort_schema(model_id: str) -> Optional[Tuple[str, Tuple[str, ...]]]:
    """Return ``(schema_path, allowed_enum)`` for a model, or None if it has no channel.

    A miss is a definite answer, not a gap to paper over: the field must not be sent.

    Deliberately a plain dict lookup on the resolved model id. Do NOT reach for
    extract_model_family() -- its regex only matches haiku|sonnet|opus, so every GPT
    model and claude-fable-5 come back None, and the enums are not family-uniform
    anyway (claude-sonnet-4.6 has no "xhigh" while the Claude 5 models do).
    """
    return MODEL_EFFORT_SCHEMA.get(model_id)


def clamp_effort(requested: str, allowed: Tuple[str, ...]) -> Optional[str]:
    """Clamp a requested tier into ``allowed``.

    Returns the tier to send, or None when the request cannot be honoured at all and the
    field should be omitted.

    Rules:
        - Already allowed: returned untouched.
        - ``none``: a request for no reasoning at all. Kept for models whose enum includes
          it (the GPT models), and None for the rest -- substituting a real tier would
          manufacture reasoning the client explicitly declined. Reached only via
          ``thinking.type=disabled``, since ``none`` is outside the cc vocabulary.
        - Outside EFFORT_ORDER (e.g. OpenAI's "minimal", typos): EFFORT_FALLBACK
          ("medium"). The converters already map unknown words here, so this is a
          defensive backstop -- and "medium" is accepted by every model.
        - In EFFORT_ORDER but unavailable (e.g. "xhigh" on claude-sonnet-4.6): the highest
          allowed tier strictly below it. Never a higher one -- clamping upward would
          spend credits the client never approved.
    """
    if requested in allowed:
        return requested

    if requested == "none":
        return "none" if "none" in allowed else None

    if requested not in EFFORT_ORDER:
        return EFFORT_FALLBACK

    # Only rankable tiers can take part in the comparison. A model enum may legitimately
    # carry a value outside EFFORT_ORDER ("none" today, a future server-side tier
    # tomorrow); EFFORT_ORDER.index() would raise ValueError on those, so they are
    # filtered out rather than ranked.
    candidates = tuple(t for t in allowed if t in EFFORT_ORDER)
    if not candidates:
        return EFFORT_FALLBACK

    requested_rank = EFFORT_ORDER.index(requested)
    lower = [t for t in candidates if EFFORT_ORDER.index(t) < requested_rank]
    if lower:
        return max(lower, key=EFFORT_ORDER.index)

    # Nothing lower exists; the floor is the least real reasoning the model can do.
    return min(candidates, key=EFFORT_ORDER.index)


def resolve_native_effort(model_id: str, effort: Optional[str]) -> Optional[EffortDecision]:
    """Resolve a tier into a payload fragment, or None to send no native field.

    None is returned -- and the field omitted -- when the master switch is off, no tier
    was requested, the model has no native channel, or the request is "none" against a
    model with no "none" tier. Anything else resolves to a concrete tier: unknown words
    fall back to EFFORT_FALLBACK inside clamp_effort.
    """
    if not NATIVE_EFFORT_ENABLED:
        return None

    if not effort:
        return None

    schema = lookup_effort_schema(model_id)
    if schema is None:
        logger.debug(
            f"Model '{model_id}' has no native effort channel, skipping "
            f"{NATIVE_EFFORT_FIELD} injection (requested effort='{effort}')"
        )
        return None

    path, allowed = schema
    adopted = clamp_effort(effort, allowed)
    if adopted is None:
        # Only "none" against a model without a "none" tier reaches here (every Claude
        # model): substituting a real tier would manufacture reasoning the client
        # declined. Routine, so DEBUG.
        logger.debug(
            f"Model '{model_id}' has no 'none' effort tier, skipping "
            f"{NATIVE_EFFORT_FIELD} injection"
        )
        return None

    clamped = adopted != effort
    reason = ""
    if clamped:
        reason = (
            f"'{effort}' is not accepted by '{model_id}'; clamped to '{adopted}' "
            f"(allowed: {', '.join(allowed)})"
        )
        logger.warning(
            f"Clamped effort for '{model_id}': client requested '{effort}', "
            f"sending '{adopted}' (allowed: {', '.join(allowed)})"
        )
    else:
        logger.debug(
            f"Native effort for '{model_id}': {path}.effort='{adopted}'"
        )

    return EffortDecision(
        fragment={path: {"effort": adopted}},
        adopted=adopted,
        clamped=clamped,
        reason=reason,
    )
