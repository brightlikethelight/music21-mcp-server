"""Tests for BaseTool timeout, progress, and error_handling paths."""

import asyncio
import logging
import time
from typing import Any

import pytest

from music21_mcp.tools.base_tool import BaseTool


class _StubTool(BaseTool):
    """Minimal concrete subclass for testing BaseTool behavior."""

    async def execute(self, **kwargs: Any) -> dict[str, Any]:
        delay = kwargs.get("delay", 0)
        if delay:
            await asyncio.sleep(delay)
        return self.create_success_response("ok")

    def validate_inputs(self, **kwargs: Any) -> str | None:
        return None


class TestExecuteWithTimeout:
    @pytest.mark.asyncio
    async def test_execute_with_timeout_exceeds(self):
        """Slow execute triggers timeout and returns error response."""
        tool = _StubTool(score_manager={}, timeout=0.1)
        result = await tool.execute_with_timeout(timeout=0.1, delay=5)
        assert result["status"] == "error"
        assert "timed out" in result["message"]
        assert result["tool"] == "_StubTool"

    @pytest.mark.asyncio
    async def test_execute_with_timeout_success(self):
        """Normal execution completes within timeout."""
        tool = _StubTool(score_manager={}, timeout=5)
        result = await tool.execute_with_timeout(timeout=5, delay=0)
        assert result["status"] == "success"


class TestRunWithProgress:
    @pytest.mark.asyncio
    async def test_run_with_progress(self):
        """Progress callback is invoked during run_with_progress."""
        tool = _StubTool(score_manager={}, timeout=10)
        progress_updates: list[tuple[float, str]] = []

        def _cb(percent: float, message: str) -> None:
            progress_updates.append((percent, message))

        tool.set_progress_callback(_cb)

        result = await tool.run_with_progress(
            lambda: 42,
            progress_start=0.0,
            progress_end=1.0,
            message="testing",
            timeout=10,
        )
        assert result == 42
        # Callback should have been invoked at least at start and end
        assert len(progress_updates) >= 2
        assert progress_updates[0][0] == 0.0
        assert progress_updates[-1][0] == 1.0


class TestErrorHandling:
    def test_error_handling_slow_operation(self, caplog):
        """Operations >1s inside error_handling log an info message."""
        tool = _StubTool(score_manager={})
        with caplog.at_level(logging.INFO), tool.error_handling("slow op"):
            time.sleep(1.1)
        assert any(
            "slow op" in r.message and "completed" in r.message for r in caplog.records
        )

    def test_error_handling_exception(self):
        """Exceptions inside error_handling are logged and re-raised."""
        tool = _StubTool(score_manager={})
        with pytest.raises(ValueError, match="boom"), tool.error_handling("failing op"):
            raise ValueError("boom")
