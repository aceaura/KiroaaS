# -*- coding: utf-8 -*-

"""Tests for privacy-safe request effort and credit auditing."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kiro.converters_core import ThinkingConfig, UnifiedMessage, build_kiro_payload
from kiro.effort_schema import EffortDecision
from kiro.request_audit import RequestAudit, extract_credit_value
from kiro.streaming_core import KiroEvent, collect_stream_to_result
from kiro.streaming_openai import stream_kiro_to_openai_internal


@pytest.mark.parametrize(
    ("usage", "expected"),
    [
        (0, 0.0),
        (0.06745126666668, 0.06745126666668),
        ({"credits": 1.25}, 1.25),
        ({"credits_used": 2}, 2.0),
        ({"usage": 3.5}, 3.5),
    ],
)
def test_extract_credit_value_accepts_safe_numeric_values(usage, expected):
    assert extract_credit_value(usage) == expected


@pytest.mark.parametrize(
    "usage",
    [
        True,
        False,
        "1.25",
        -0.01,
        float("nan"),
        float("inf"),
        float("-inf"),
        {},
        {"credits": "1.25"},
        {"nested": {"credits": 1.25}},
        object(),
        None,
    ],
)
def test_extract_credit_value_rejects_unsafe_values(usage):
    assert extract_credit_value(usage) is None


def _native_decision() -> EffortDecision:
    return EffortDecision(
        requested="xhigh",
        adopted="xhigh",
        schema_path="reasoning",
        fragment={"reasoning": {"effort": "xhigh"}},
        clamped=False,
        outcome="native",
        reason="exact",
    )


def test_audit_logs_linked_effort_and_accumulated_credit_once():
    audit = RequestAudit(
        protocol="openai",
        client_model="CLIENT_MODEL_SECRET",
        audit_id="audit123",
    )

    with patch("kiro.request_audit.logger.info") as mock_info:
        audit.record_effort("gpt-5.6-sol", _native_decision())
        audit.record_effort("gpt-5.6-sol", _native_decision())
        audit.record_metering(0.0)
        audit.record_metering(0.06745126666668)
        audit.record_metering({"credits": 0.03254873333332})
        audit.record_metering("INVALID_SECRET")
        audit.log_credit_once(status="completed")
        audit.log_credit_once(status="failed")

    logs = [call.args[0] for call in mock_info.call_args_list]
    assert len([line for line in logs if line.startswith("effort_decision ")]) == 1
    assert len([line for line in logs if line.startswith("credit_usage ")]) == 1
    assert all("audit_id=audit123" in line for line in logs)
    assert "credits=0.1" in logs[-1]
    assert "metering_events=3" in logs[-1]
    assert "status=completed" in logs[-1]
    assert "CLIENT_MODEL_SECRET" not in "\n".join(logs)
    assert "INVALID_SECRET" not in "\n".join(logs)


def test_converter_links_effort_to_audit_without_sensitive_payload_data():
    audit = RequestAudit(
        protocol="openai",
        client_model="gpt-5.6-sol",
        audit_id="linked789",
    )

    with patch("kiro.request_audit.logger.info") as mock_info:
        result = build_kiro_payload(
            messages=[UnifiedMessage(role="user", content="MESSAGE_SECRET")],
            system_prompt="SYSTEM_SECRET",
            model_id="gpt-5.6-sol",
            tools=None,
            conversation_id="CONVERSATION_SECRET",
            profile_arn="PROFILE_SECRET",
            thinking_config=ThinkingConfig(effort="xhigh"),
            request_audit=audit,
        )

    effort_log = mock_info.call_args_list[0].args[0]
    assert effort_log.startswith(
        "effort_decision audit_id=linked789 protocol=openai model=gpt-5.6-sol "
    )
    assert "adopted=xhigh" in effort_log
    assert "field=reasoning.effort" in effort_log
    for secret in (
        "MESSAGE_SECRET",
        "SYSTEM_SECRET",
        "CONVERSATION_SECRET",
        "PROFILE_SECRET",
    ):
        assert secret not in effort_log
    assert result.payload["additionalModelRequestFields"] == {
        "reasoning": {"effort": "xhigh"}
    }


def test_audit_logs_unavailable_without_metering():
    audit = RequestAudit(
        protocol="anthropic",
        client_model="claude-opus-5",
        audit_id="audit456",
    )

    with patch("kiro.request_audit.logger.info") as mock_info:
        audit.record_effort("claude-opus-5", _native_decision())
        audit.log_credit_once(status="client_disconnected")

    credit_log = mock_info.call_args_list[-1].args[0]
    assert "credits=unavailable" in credit_log
    assert "metering_events=0" in credit_log
    assert "status=client_disconnected" in credit_log


@pytest.mark.asyncio
async def test_request_audits_remain_isolated_under_concurrency():
    async def record(audit: RequestAudit, values: list[float]) -> None:
        for value in values:
            await asyncio.sleep(0)
            audit.record_metering(value)

    first = RequestAudit(protocol="openai", client_model="model-a", audit_id="first")
    second = RequestAudit(protocol="anthropic", client_model="model-b", audit_id="second")

    await asyncio.gather(record(first, [0.1, 0.2]), record(second, [1.0, 2.0]))

    assert first.audit_id == "first"
    assert first.credits == pytest.approx(0.3)
    assert first.metering_events == 2
    assert second.audit_id == "second"
    assert second.credits == pytest.approx(3.0)
    assert second.metering_events == 2


@pytest.mark.asyncio
async def test_core_collector_accumulates_every_scalar_metering_event():
    response = AsyncMock()
    audit = RequestAudit(protocol="anthropic", client_model="claude-opus-5")

    async def events(*args, **kwargs):
        yield KiroEvent(type="content", content="ok")
        yield KiroEvent(type="usage", usage=0.04)
        yield KiroEvent(type="usage", usage=0.06)

    with patch("kiro.streaming_core.parse_kiro_stream", events):
        with patch("kiro.streaming_core.parse_bracket_tool_calls", return_value=[]):
            result = await collect_stream_to_result(response, request_audit=audit)

    assert result.content == "ok"
    assert result.usage == 0.06
    assert audit.credits == pytest.approx(0.1)
    assert audit.metering_events == 2


@pytest.mark.asyncio
async def test_openai_stream_records_scalar_metering_and_preserves_response_usage():
    response = AsyncMock()
    response.aclose = AsyncMock()
    model_cache = MagicMock()
    model_cache.get_max_input_tokens.return_value = 200000
    audit = RequestAudit(protocol="openai", client_model="gpt-5.6")

    async def events(*args, **kwargs):
        yield KiroEvent(type="content", content="ok")
        yield KiroEvent(type="usage", usage=0.06745126666668)
        yield KiroEvent(type="context_usage", context_usage_percentage=1.0)

    chunks = []
    with patch("kiro.streaming_openai.parse_kiro_stream", events):
        with patch("kiro.streaming_openai.parse_bracket_tool_calls", return_value=[]):
            async for chunk in stream_kiro_to_openai_internal(
                AsyncMock(),
                response,
                "gpt-5.6",
                model_cache,
                MagicMock(),
                request_audit=audit,
            ):
                chunks.append(chunk)

    payloads = [
        json.loads(chunk.removeprefix("data:").strip())
        for chunk in chunks
        if chunk.startswith("data:") and "[DONE]" not in chunk
    ]
    final_usage = payloads[-1]["usage"]
    assert final_usage["credits_used"] == 0.06745126666668
    assert audit.credits == pytest.approx(0.06745126666668)
    assert audit.metering_events == 1
