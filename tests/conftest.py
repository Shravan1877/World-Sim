"""Shared pytest setup: live LLM tests are skipped unless --live is passed (CLAUDE.md §1.6)."""

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--live", action="store_true", default=False, help="run @pytest.mark.live tests")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--live"):
        return
    skip_live = pytest.mark.skip(reason="live LLM test; pass --live to run")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip_live)
