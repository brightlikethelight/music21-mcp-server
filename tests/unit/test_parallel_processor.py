"""Tests for parallel_processor module"""

import asyncio

import pytest

from music21_mcp.parallel_processor import ParallelProcessor, get_parallel_processor


class TestParallelProcessor:
    """Tests for ParallelProcessor"""

    def setup_method(self):
        self.processor = ParallelProcessor(max_workers=2)

    def teardown_method(self):
        if hasattr(self, "processor") and hasattr(self.processor, "_executor"):
            self.processor._executor.shutdown(wait=False)

    def test_init_default_workers(self):
        p = ParallelProcessor()
        assert p.max_workers == 4
        p._executor.shutdown(wait=False)

    def test_init_custom_workers(self):
        assert self.processor.max_workers == 2

    @pytest.mark.asyncio
    async def test_process_batch_empty(self):
        result = await self.processor.process_batch([], lambda x: x * 2)
        assert result == []

    @pytest.mark.asyncio
    async def test_process_batch_normal(self):
        results = await self.processor.process_batch(
            [1, 2, 3, 4], lambda x: x * 2, batch_size=2
        )
        assert results == [2, 4, 6, 8]

    @pytest.mark.asyncio
    async def test_process_batch_with_progress(self):
        progress_calls = []

        def on_progress(done, total):
            progress_calls.append((done, total))

        await self.processor.process_batch(
            [1, 2, 3], lambda x: x, batch_size=2, progress_callback=on_progress
        )
        assert len(progress_calls) >= 1
        # Last call should report all items done
        assert progress_calls[-1] == (3, 3)

    @pytest.mark.asyncio
    async def test_process_batch_error_propagation(self):
        def fail(x):
            raise ValueError(f"bad {x}")

        results = await self.processor.process_batch([1], fail)
        # Errors are returned as exceptions (gather return_exceptions=True)
        assert isinstance(results[0], ValueError)

    @pytest.mark.asyncio
    async def test_map_reduce(self):
        result = await self.processor.map_reduce(
            [1, 2, 3], lambda x: x * 2, sum, batch_size=2
        )
        assert result == 12

    @pytest.mark.asyncio
    async def test_process_chord_batch(self):
        results = await self.processor.process_chord_batch(
            ["a", "b"], lambda x: {"note": x}
        )
        assert len(results) == 2
        assert results[0] == {"note": "a"}

    @pytest.mark.asyncio
    async def test_process_chord_batch_error_handling(self):
        def bad_analysis(item):
            raise RuntimeError("analysis failed")

        results = await self.processor.process_chord_batch(["x"], bad_analysis)
        assert results[0]["failed"] is True
        assert "error" in results[0]


class TestGetParallelProcessor:
    """Tests for the singleton accessor"""

    def test_singleton_returns_same_instance(self):
        import music21_mcp.parallel_processor as mod

        mod._global_processor = None  # reset
        p1 = get_parallel_processor()
        p2 = get_parallel_processor()
        assert p1 is p2
        mod._global_processor = None  # cleanup
