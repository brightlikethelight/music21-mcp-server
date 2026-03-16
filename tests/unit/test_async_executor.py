"""Tests for async_executor module"""

import asyncio

import pytest

from music21_mcp.async_executor import (
    Music21AsyncExecutor,
    async_music21,
    get_executor_stats,
    run_in_thread,
    shutdown_executor,
)


class TestMusic21AsyncExecutor:
    """Tests for Music21AsyncExecutor"""

    def setup_method(self):
        # Reset singleton between tests
        Music21AsyncExecutor._instance = None
        Music21AsyncExecutor._lock = None

    def teardown_method(self):
        if Music21AsyncExecutor._instance:
            Music21AsyncExecutor._instance.shutdown(wait=False)
            Music21AsyncExecutor._instance = None
            Music21AsyncExecutor._lock = None

    @pytest.mark.asyncio
    async def test_singleton(self):
        inst1 = await Music21AsyncExecutor.get_instance()
        inst2 = await Music21AsyncExecutor.get_instance()
        assert inst1 is inst2

    @pytest.mark.asyncio
    async def test_run_simple_function(self):
        executor = Music21AsyncExecutor(max_workers=2)
        result = await executor.run(lambda: 42)
        assert result == 42
        executor.shutdown(wait=False)

    @pytest.mark.asyncio
    async def test_run_with_args(self):
        executor = Music21AsyncExecutor(max_workers=2)

        def add(a, b):
            return a + b

        result = await executor.run(add, 3, 4)
        assert result == 7
        executor.shutdown(wait=False)

    @pytest.mark.asyncio
    async def test_run_with_kwargs(self):
        executor = Music21AsyncExecutor(max_workers=2)

        def greet(name="world"):
            return f"hello {name}"

        result = await executor.run(greet, name="test")
        assert result == "hello test"
        executor.shutdown(wait=False)

    @pytest.mark.asyncio
    async def test_run_timeout(self):
        import time

        executor = Music21AsyncExecutor(max_workers=2)

        def slow():
            time.sleep(5)

        with pytest.raises(asyncio.TimeoutError):
            await executor.run(slow, timeout=0.1)
        executor.shutdown(wait=False)

    @pytest.mark.asyncio
    async def test_run_exception_propagation(self):
        executor = Music21AsyncExecutor(max_workers=2)

        def fail():
            raise ValueError("boom")

        with pytest.raises(ValueError, match="boom"):
            await executor.run(fail)
        executor.shutdown(wait=False)

    @pytest.mark.asyncio
    async def test_get_stats(self):
        executor = Music21AsyncExecutor(max_workers=2)
        await executor.run(lambda: 1)
        stats = executor.get_stats()
        assert stats["total_operations"] == 1
        assert stats["max_workers"] == 2
        assert stats["total_time_seconds"] > 0
        executor.shutdown(wait=False)

    @pytest.mark.asyncio
    async def test_shutdown(self):
        executor = Music21AsyncExecutor(max_workers=2)
        executor.shutdown(wait=True)
        # Should not raise


class TestConvenienceFunctions:
    """Tests for module-level convenience functions"""

    def setup_method(self):
        Music21AsyncExecutor._instance = None
        Music21AsyncExecutor._lock = None

    def teardown_method(self):
        if Music21AsyncExecutor._instance:
            Music21AsyncExecutor._instance.shutdown(wait=False)
            Music21AsyncExecutor._instance = None
            Music21AsyncExecutor._lock = None

    @pytest.mark.asyncio
    async def test_run_in_thread(self):
        result = await run_in_thread(lambda: 99)
        assert result == 99

    @pytest.mark.asyncio
    async def test_async_music21_decorator(self):
        @async_music21
        def compute(x):
            return x * 3

        result = await compute(5)
        assert result == 15

    @pytest.mark.asyncio
    async def test_get_executor_stats_no_instance(self):
        stats = await get_executor_stats()
        assert stats["total_operations"] == 0

    @pytest.mark.asyncio
    async def test_get_executor_stats_with_instance(self):
        await run_in_thread(lambda: 1)
        stats = await get_executor_stats()
        assert stats["total_operations"] >= 1

    @pytest.mark.asyncio
    async def test_shutdown_executor(self):
        await run_in_thread(lambda: 1)  # ensure instance exists
        await shutdown_executor()
        assert Music21AsyncExecutor._instance is None
