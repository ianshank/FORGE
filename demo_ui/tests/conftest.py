"""conftest.py — Shared pytest fixtures for demo_ui tests."""

import pytest


# Use anyio as the async backend (supports asyncio and trio)
@pytest.fixture(scope="session")
def anyio_backend() -> str:
    return "asyncio"
