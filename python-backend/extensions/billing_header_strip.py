"""
Strip Claude Code's per-request billing attribution from the system prompt.

Claude Code (2.1.x) prepends a system text block of the form

    x-anthropic-billing-header: cc_version=...; cc_entrypoint=...; cch=<5hex>;

as the FIRST system block. The ``cch`` segment (when present, e.g. in
``claude -p`` / SDK mode) is a fresh random hex per request; the
``cc_version`` / ``cc_entrypoint`` segments differ per client + version.
``extract_system_prompt()`` concatenates all system text blocks into one
string before forwarding to Kiro, so this line lands at the very front of
the forwarded prompt — poisoning any upstream cache keyed on the prompt
prefix (per-request churn from ``cch``, per-client fragmentation from the
version/entrypoint tags).

The line is billing metadata meant to ship as an HTTP header; it carries no
semantic content for the model and is safe to drop before forwarding.

Rather than patch upstream kiro/converters_anthropic.py (keeping the backend
zero-diff against kiro-gateway-origin), we wrap extract_system_prompt() at
runtime. The billing block is always the first block and pure-attribution, so
after the join it is the leading line of the returned string — stripping a
single leading ``x-anthropic-billing-header:`` line yields the same result as
upstream's per-block handling.

Gated by STRIP_BILLING_HEADER (default true) so it can be disabled for A/B
comparison or to debug attribution behavior. Order-insensitive: it wraps a
module function at runtime, so unlike role widening it does not care about
import/registration order.
"""

import os
import re
from typing import Any, Callable, Dict

from loguru import logger


# Anchored to the start of the (joined) prompt; matches only the single
# attribution line and its trailing newline.
_BILLING_HEADER_LINE_PATTERN = re.compile(
    r"^x-anthropic-billing-header:[^\n]*\n?", re.IGNORECASE
)

_installed = False
_originals: Dict[str, Any] = {}


def _strip_enabled() -> bool:
    return os.getenv("STRIP_BILLING_HEADER", "true").lower() in ("true", "1", "yes")


def strip_billing_attribution(text: str) -> str:
    """Remove a leading Claude Code billing-attribution line from ``text``."""
    if not text or not _strip_enabled():
        return text
    stripped = _BILLING_HEADER_LINE_PATTERN.sub("", text, count=1)
    if stripped != text:
        # Drop any blank lines the removal left at the very top.
        return stripped.lstrip("\n")
    return text


def install_billing_header_strip() -> None:
    """Wrap extract_system_prompt() to strip billing attribution at runtime."""
    global _installed
    if _installed:
        return

    import kiro.converters_anthropic as converters_anthropic

    _originals["extract_system_prompt"] = converters_anthropic.extract_system_prompt

    def extract_system_prompt_stripped(system: Any) -> str:
        result = _originals["extract_system_prompt"](system)
        return strip_billing_attribution(result)

    converters_anthropic.extract_system_prompt = extract_system_prompt_stripped

    _installed = True
    logger.info("Installed billing-header strip extension")


def uninstall_billing_header_strip() -> None:
    """Restore the patched function. Intended for unit tests."""
    global _installed
    if _originals:
        import kiro.converters_anthropic as converters_anthropic

        converters_anthropic.extract_system_prompt = _originals["extract_system_prompt"]

    _originals.clear()
    _installed = False
