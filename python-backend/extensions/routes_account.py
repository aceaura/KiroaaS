"""
Extension routes: /usage and /account.

Lives outside the upstream kiro/ tree so upstream can be sync'd without conflicts.
Mounted by main.py, so the routes exist for every launch path (the Docker image
runs `python main.py`; app_entry.py imports that same app object).
"""

import json
from datetime import datetime
from typing import Any, Dict

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from loguru import logger

from extensions.control_plane_host import to_q_amazonaws_host
from kiro.auth import KiroAuthManager
from kiro.network_errors import classify_network_error, get_short_error_message
from kiro.routes_openai import verify_api_key
from kiro.utils import get_kiro_headers


router = APIRouter()

# Extra attempts for transient timeout blips (e.g. proxy upstream flapping).
# Chat paths get retries from KiroHttpClient; these routes call the shared
# client directly, so a single timeout would otherwise surface as a bare 502.
USAGE_TIMEOUT_RETRIES = 1


def _resolve_auth_manager(request: Request) -> KiroAuthManager:
    account = request.app.state.account_manager.get_first_account()
    if not account or not account.auth_manager:
        raise HTTPException(status_code=503, detail="No initialized account available")
    return account.auth_manager


def _readable_cause(error: Exception) -> str:
    """Return a one-line, user-readable cause for an upstream failure.

    httpx timeout exceptions stringify to an empty message, so logging raw
    str(e) yields "Error fetching usage: " with nothing after it; classify
    those through network_errors instead. Non-httpx exceptions keep their
    message, falling back to the type name when that is empty too.
    """
    if isinstance(error, httpx.HTTPError):
        return get_short_error_message(classify_network_error(error))
    return str(error) or type(error).__name__


async def _post_usage_limits(
    request: Request,
    url: str,
    body: Dict[str, Any],
    headers: Dict[str, Any],
) -> httpx.Response:
    """POST to GetUsageLimits, retrying once on timeout.

    Args:
        request: Incoming request carrying the shared http_client in app.state
        url: Resolved q.amazonaws.com endpoint
        body: GetUsageLimits request body
        headers: Kiro auth headers including the x-amz-target

    Returns:
        The upstream response (any status code; caller maps non-200).

    Raises:
        httpx.TimeoutException: When every attempt timed out.
    """
    attempts = USAGE_TIMEOUT_RETRIES + 1
    for attempt in range(1, attempts + 1):
        try:
            return await request.app.state.http_client.post(url, json=body, headers=headers)
        except httpx.TimeoutException as e:
            if attempt == attempts:
                raise
            logger.warning(
                f"GetUsageLimits attempt {attempt}/{attempts} failed: {_readable_cause(e)}; retrying"
            )
    raise RuntimeError("unreachable: the final attempt always returns or raises")


async def _get_usage_limits(request: Request, auth_manager: KiroAuthManager) -> dict:
    profile_arn = auth_manager.profile_arn
    if not profile_arn:
        raise HTTPException(
            status_code=503,
            detail="Initialized account has no profileArn for usage queries",
        )

    token = await auth_manager.get_access_token()
    headers = get_kiro_headers(auth_manager, token)
    headers["x-amz-target"] = "com.amazon.aws.codewhisperer.runtime.AmazonCodeWhispererService.GetUsageLimits"
    headers["Content-Type"] = "application/x-amz-json-1.0"

    # GetUsageLimits lives on q.amazonaws.com; runtime.kiro.dev 400s with
    # UnknownOperationException. Resolve the host here instead of relying on the
    # control_plane_host redirect being installed.
    url = to_q_amazonaws_host(auth_manager.q_host)
    body = {
        "profileArn": profile_arn,
        "origin": "AI_EDITOR",
        "resourceType": "AGENTIC_REQUEST",
    }
    response = await _post_usage_limits(request, url, body, headers)
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail=response.text)
    return response.json()


@router.get("/usage", dependencies=[Depends(verify_api_key)])
async def get_usage(request: Request):
    """
    Query Kiro credit usage via GetUsageLimits API.

    Returns subscription info, credit usage breakdown, overage config,
    and reset date — the same data shown by kiro-cli's /usage command.
    """
    logger.info("Request to /usage")

    auth_manager = _resolve_auth_manager(request)

    try:
        usage_data = await _get_usage_limits(request, auth_manager)
        return JSONResponse(content=usage_data)
    except HTTPException:
        raise
    except Exception as e:
        reason = _readable_cause(e)
        logger.error(f"Error fetching usage: {reason}", exc_info=True)
        raise HTTPException(status_code=502, detail=f"Failed to fetch usage: {reason}")


@router.get("/account", dependencies=[Depends(verify_api_key)])
async def get_account(request: Request):
    """
    Query Kiro account profile by calling the usage endpoint internally.

    Returns account name, email, subscription plan, quota information,
    trial plan details, and expiration dates.
    """
    logger.info("Request to /account")

    auth_manager = _resolve_auth_manager(request)

    try:
        usage_data = await _get_usage_limits(request, auth_manager)

        user_info = usage_data.get("userInfo", {})
        subscription_info = usage_data.get("subscriptionInfo", {})
        usage_list = usage_data.get("usageBreakdownList", [])

        logger.info(f"Full API response: {json.dumps(usage_data, indent=2)}")

        usage_info = usage_list[0] if usage_list else {}

        free_trial_info = usage_info.get("freeTrialInfo", {})
        is_trial = free_trial_info.get("freeTrialStatus") == "ACTIVE"

        trial_expiry_ts = free_trial_info.get("freeTrialExpiry")
        trial_expiry = None
        if trial_expiry_ts:
            trial_expiry = datetime.fromtimestamp(trial_expiry_ts).strftime("%Y-%m-%d")

        subscription_expiry = subscription_info.get("expiryDate") or subscription_info.get("subscriptionExpiryDate")

        account_status = "Trial" if is_trial else "Active"
        if subscription_info.get("status"):
            account_status = subscription_info.get("status")

        bonuses = usage_info.get("bonuses", [])
        active_bonuses = [b for b in bonuses if b.get("status") == "ACTIVE"]
        bonus_quota = sum(b.get("usageLimitWithPrecision", b.get("usageLimit", 0)) for b in active_bonuses) if active_bonuses else None
        bonus_usage = sum(b.get("currentUsageWithPrecision", b.get("currentUsage", 0)) for b in active_bonuses) if active_bonuses else None
        bonus_remaining = (bonus_quota - bonus_usage) if bonus_quota and bonus_usage else None

        trial_quota = free_trial_info.get("usageLimitWithPrecision") or free_trial_info.get("usageLimit") or 0
        trial_usage = free_trial_info.get("currentUsageWithPrecision") or free_trial_info.get("currentUsage") or 0

        free_quota = usage_info.get("usageLimitWithPrecision") or usage_info.get("usageLimit") or 0
        free_usage = usage_info.get("currentUsageWithPrecision") or usage_info.get("currentUsage") or 0

        total_quota = trial_quota + free_quota
        total_usage = trial_usage + free_usage

        remaining_quota = (total_quota - total_usage) if total_quota and total_usage else None
        usage_percentage = round((total_usage / total_quota) * 100, 2) if total_quota and total_usage else 0

        account_data = {
            "accountName": user_info.get("email") or "User",
            "email": user_info.get("email"),
            "provider": user_info.get("provider"),
            "planType": subscription_info.get("type", "Free"),
            "subscriptionTitle": subscription_info.get("subscriptionTitle"),
            "isTrial": is_trial,
            "trialExpiryDate": trial_expiry,
            "subscriptionExpiryDate": subscription_expiry,
            "totalQuota": total_quota,
            "currentUsage": total_usage,
            "remainingQuota": remaining_quota,
            "usagePercentage": usage_percentage,
            "trialQuota": trial_quota,
            "trialUsage": trial_usage,
            "freeQuota": free_quota,
            "freeUsage": free_usage,
            "bonusQuota": bonus_quota,
            "bonusUsage": bonus_usage,
            "bonusRemaining": bonus_remaining,
            "accountStatus": account_status,
            "resetDate": usage_info.get("resetDate"),
            "overageEnabled": usage_info.get("overageEnabled", False),
        }

        logger.info(f"Account retrieved: {account_data.get('email')} (trial={is_trial}, quota={total_quota}, bonus={bonus_quota})")
        return JSONResponse(content=account_data)

    except HTTPException:
        raise
    except Exception as e:
        reason = _readable_cause(e)
        logger.error(f"Error fetching account: {reason}", exc_info=True)
        raise HTTPException(status_code=502, detail=f"Failed to fetch account: {reason}")
