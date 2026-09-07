# -*- coding: utf-8 -*-

"""Privacy-safe request audit state for effort and Kiro credit usage."""

import math
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional, TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from kiro.effort_schema import EffortDecision


_CREDIT_KEYS = ("credits", "credits_used", "usage")


def extract_credit_value(usage: Any) -> Optional[float]:
    """Extract a finite, non-negative credit value from a metering payload.

    Kiro currently sends ``meteringEvent.usage`` as a number. Dictionary support
    is retained for compatibility with older gateway tests and alternate upstream
    wrappers, but arbitrary objects and strings are never serialized to logs.

    Args:
        usage: Raw usage value from a parsed Kiro metering event.

    Returns:
        A finite non-negative float, or ``None`` when no valid credit value exists.
    """
    value: Any = usage
    if isinstance(value, dict):
        value = next((value[key] for key in _CREDIT_KEYS if key in value), None)

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None

    credits = float(value)
    if not math.isfinite(credits) or credits < 0:
        return None
    return credits


def _format_credits(value: float) -> str:
    """Format credits without adding insignificant trailing zeroes."""
    return format(value, ".15g")


@dataclass
class RequestAudit:
    """Mutable audit state owned by exactly one client request.

    The object is passed explicitly across converter and streaming layers. This
    avoids cross-request leakage under asyncio concurrency and preserves the same
    audit ID across account failover and first-token retries.
    """

    protocol: str
    client_model: str
    audit_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    model_id: Optional[str] = None
    adopted_effort: Optional[str] = None
    effort_field: Optional[str] = None
    credits: Optional[float] = None
    metering_events: int = 0
    _effort_logged: bool = field(default=False, init=False, repr=False)
    _credit_logged: bool = field(default=False, init=False, repr=False)

    def record_effort(self, model_id: str, decision: "EffortDecision") -> None:
        """Record and log the final native effort decision once.

        Args:
            model_id: Resolved Kiro model identifier.
            decision: Final native effort decision used to build the payload.
        """
        self.model_id = model_id
        self.adopted_effort = decision.adopted
        self.effort_field = decision.field
        if self._effort_logged:
            return

        logger.info(
            "effort_decision "
            f"audit_id={self.audit_id} "
            f"protocol={self.protocol} "
            f"model={model_id} "
            f"requested={decision.requested or 'none'} "
            f"adopted={decision.adopted or 'none'} "
            f"field={decision.field or 'none'} "
            f"outcome={decision.outcome} "
            f"clamped={str(decision.clamped).lower()} "
            f"reason={decision.reason}"
        )
        self._effort_logged = True

    def record_metering(self, usage: Any) -> None:
        """Accumulate one valid upstream metering event.

        Args:
            usage: Raw ``meteringEvent.usage`` value.
        """
        credits = extract_credit_value(usage)
        if credits is None:
            return
        self.credits = (self.credits or 0.0) + credits
        self.metering_events += 1

    def log_credit_once(self, status: str = "completed") -> None:
        """Emit one privacy-safe credit line for this request.

        Args:
            status: Completion state used to distinguish a normal response from
                an interrupted or failed stream.
        """
        if self._credit_logged:
            return

        credits = (
            _format_credits(self.credits)
            if self.credits is not None
            else "unavailable"
        )
        logger.info(
            "credit_usage "
            f"audit_id={self.audit_id} "
            f"protocol={self.protocol} "
            f"model={self.model_id or self.client_model} "
            f"effort={self.adopted_effort or 'none'} "
            f"field={self.effort_field or 'none'} "
            f"credits={credits} "
            f"metering_events={self.metering_events} "
            f"status={status}"
        )
        self._credit_logged = True
