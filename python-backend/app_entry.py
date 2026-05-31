"""
Entry point for launching the Kiro Gateway with local extensions.

Responsibilities:
- Import the upstream FastAPI app (zero-diff upstream)
- Mount local extension routers (/usage, /account)
- Handle Tauri-specific launch behavior (localhost binding, ready signal)
  without modifying upstream main.py

Use this instead of main.py when embedding the gateway into Tauri.
"""

import os
import sys

# Ensure python-backend root is on sys.path so `extensions` and `kiro` import.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import uvicorn
from loguru import logger

# Importing main runs module-level setup (logging, FastAPI instance, lifespan).
from main import (
    app,
    validate_configuration,
    parse_cli_args,
    resolve_server_config,
    print_startup_banner,
    UVICORN_LOG_CONFIG,
    SERVER_HOST,
    SERVER_PORT,
    DEFAULT_SERVER_HOST,
    DEFAULT_SERVER_PORT,
)

from extensions.routes_account import router as account_router
from extensions.tool_name_alias import install_tool_name_aliasing
from extensions.role_widening import install_role_widening
from extensions.profile_arn_autofetch import install_profile_arn_autofetch


# Mount extension routers on the upstream app.
install_tool_name_aliasing()
install_role_widening()
install_profile_arn_autofetch()
app.include_router(account_router)


def _is_tauri_managed() -> bool:
    return os.getenv("TAURI_MANAGED") == "true"


def _resolve_host_for_tauri(cli_host, env_host: str, default_host: str) -> str:
    """Resolve bind host. CLI > Rust-provided SERVER_HOST > env > default.

    When Tauri-managed, Rust always sets SERVER_HOST explicitly (either
    127.0.0.1 or 0.0.0.0 depending on the user's "Allow LAN Access" toggle),
    so we honor it. Fall back to 127.0.0.1 only if the env var is missing.
    """
    if cli_host is not None:
        return cli_host
    if _is_tauri_managed():
        if os.getenv("SERVER_HOST") is not None:
            return env_host
        return "127.0.0.1"
    if env_host != default_host:
        return env_host
    return default_host


def _run() -> None:
    args = parse_cli_args()
    validate_configuration()

    # Host/port resolution: mostly delegate to upstream, but override host for Tauri.
    cli_host = args.host
    cli_port = args.port
    final_host = _resolve_host_for_tauri(cli_host, SERVER_HOST, DEFAULT_SERVER_HOST)
    if cli_port is not None:
        final_port = cli_port
    elif SERVER_PORT != DEFAULT_SERVER_PORT:
        final_port = SERVER_PORT
    else:
        final_port = DEFAULT_SERVER_PORT

    if _is_tauri_managed():
        # Ready signal for any consumer that watches stdout; Tauri itself polls /health.
        print(f"KIRO_GATEWAY_READY:{final_port}", flush=True)
        logger.info(f"Server ready on port {final_port} (Tauri-managed)")
    else:
        print_startup_banner(final_host, final_port)

    logger.info(f"Starting Uvicorn server on {final_host}:{final_port}...")

    # Patch UVICORN_LOG_CONFIG to reference main.InterceptHandler
    # (upstream already uses "main.InterceptHandler", which is fine since we import main).
    uvicorn.run(
        app,
        host=final_host,
        port=final_port,
        log_config=UVICORN_LOG_CONFIG,
    )


if __name__ == "__main__":
    _run()
