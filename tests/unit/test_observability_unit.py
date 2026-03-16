"""Tests for observability module"""

import pytest

from music21_mcp.observability import (
    MetricsCollector,
    StructuredLogger,
    get_logger,
    get_metrics,
    monitor_performance,
    record_error,
    record_metric,
    with_context,
)


class TestStructuredLogger:
    """Tests for StructuredLogger"""

    def test_create_logger(self):
        logger = StructuredLogger("test_obs")
        assert logger.logger.name == "test_obs"

    def test_all_log_levels(self):
        logger = get_logger("test_levels")
        # Should not raise
        logger.debug("debug msg", key="val")
        logger.info("info msg")
        logger.warning("warn msg")
        logger.error("error msg")
        logger.critical("critical msg")

    def test_error_with_exception(self):
        logger = get_logger("test_err")
        logger.error("failed", error=ValueError("test"))

    def test_critical_with_exception(self):
        logger = get_logger("test_crit")
        logger.critical("critical fail", error=RuntimeError("bad"))


class TestMetricsCollector:
    """Tests for MetricsCollector"""

    def setup_method(self):
        self.collector = MetricsCollector()

    def test_increment_counter(self):
        self.collector.increment_counter("requests")
        self.collector.increment_counter("requests")
        metrics = self.collector.get_metrics()
        assert metrics["counters"]["requests"] == 2

    def test_increment_counter_with_labels(self):
        self.collector.increment_counter("requests", status="ok")
        metrics = self.collector.get_metrics()
        assert any("status=ok" in k for k in metrics["counters"])

    def test_record_histogram(self):
        self.collector.record_histogram("latency", 0.5)
        self.collector.record_histogram("latency", 1.5)
        metrics = self.collector.get_metrics()
        assert "latency" in metrics["histograms"]
        assert metrics["histograms"]["latency"]["count"] == 2

    def test_set_gauge(self):
        self.collector.set_gauge("memory", 100.0)
        metrics = self.collector.get_metrics()
        assert metrics["gauges"]["memory"] == 100.0

    def test_record_timer(self):
        self.collector.record_timer("op_duration", 0.25)
        metrics = self.collector.get_metrics()
        assert "op_duration" in metrics["timers"]
        assert metrics["timers"]["op_duration"]["count"] == 1

    def test_record_metric(self):
        self.collector.record_metric("cpu", 45.0)
        metrics = self.collector.get_metrics()
        assert metrics["gauges"]["cpu"] == 45.0

    def test_record_error(self):
        self.collector.record_error("parse", ValueError("bad input"))
        metrics = self.collector.get_metrics()
        assert any("errors" in k for k in metrics["counters"])

    def test_reset_metrics(self):
        self.collector.increment_counter("x")
        self.collector.reset_metrics()
        metrics = self.collector.get_metrics()
        assert len(metrics["counters"]) == 0

    def test_get_metrics_has_metadata(self):
        metrics = self.collector.get_metrics()
        assert "metadata" in metrics
        assert "uptime_seconds" in metrics["metadata"]


class TestWithContext:
    """Tests for context manager"""

    def test_basic_context(self):
        with with_context(request_id="r1", user_id="u1", operation="op1") as ctx:
            assert ctx.request_id == "r1"

    def test_auto_request_id(self):
        with with_context() as ctx:
            assert ctx.request_id  # auto-generated UUID


class TestMonitorPerformance:
    """Tests for monitor_performance decorator"""

    @pytest.mark.asyncio
    async def test_async_decorator(self):
        @monitor_performance(operation_name="test_async_op")
        async def async_fn():
            return "ok"

        result = await async_fn()
        assert result == "ok"

    def test_sync_decorator(self):
        @monitor_performance(operation_name="test_sync_op")
        def sync_fn():
            return 42

        assert sync_fn() == 42

    @pytest.mark.asyncio
    async def test_async_decorator_error(self):
        @monitor_performance(operation_name="test_err_op")
        async def bad_fn():
            raise RuntimeError("fail")

        with pytest.raises(RuntimeError, match="fail"):
            await bad_fn()

    def test_sync_decorator_error(self):
        @monitor_performance(operation_name="test_sync_err")
        def bad_fn():
            raise ValueError("bad")

        with pytest.raises(ValueError, match="bad"):
            bad_fn()

    def test_default_operation_name(self):
        @monitor_performance()
        def my_func():
            return 1

        assert my_func() == 1


class TestGlobalFunctions:
    """Tests for module-level convenience functions"""

    def test_record_metric(self):
        record_metric("test_global", 77.0)
        metrics = get_metrics()
        assert "test_global" in metrics["gauges"]

    def test_record_error(self):
        record_error("test_op", ValueError("test"))
        metrics = get_metrics()
        assert any("errors" in k for k in metrics["counters"])

    def test_get_logger(self):
        logger = get_logger("custom_name")
        assert isinstance(logger, StructuredLogger)
