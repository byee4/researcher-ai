"""Pytest configuration for researcher-ai test suite.

Live tests (marked @pytest.mark.live) require real network access and API keys.
They are skipped by default; opt in with:

    pytest --run-live

Snapshot tests (marked @pytest.mark.snapshot) use frozen fixtures and run
normally without any special flag.
"""

from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-live",
        action="store_true",
        default=False,
        help="Run live end-to-end tests that require network access and API keys.",
    )


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    if config.getoption("--run-live"):
        return  # opt-in: run everything
    skip_live = pytest.mark.skip(reason="Live tests skipped by default; use --run-live to enable.")
    for item in items:
        if item.get_closest_marker("live"):
            item.add_marker(skip_live)
