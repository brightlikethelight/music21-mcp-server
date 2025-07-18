"""
Music21 MCP Server Monitoring Module

Comprehensive observability and monitoring infrastructure:
- Prometheus metrics collection
- Circuit breakers and resilience patterns
- Performance monitoring and optimization
- OpenTelemetry integration
- Grafana dashboards
- Enterprise-grade monitoring
"""

from .metrics import (
    MetricsRegistry,
    MetricsCollector,
    MetricsConfig,
    MCPRequestTracker,
    MusicAnalysisTracker,
    instrument_mcp_tool,
    get_metrics_registry,
    get_metrics_collector,
    initialize_metrics,
    HAS_PROMETHEUS
)

__all__ = [
    'MetricsRegistry',
    'MetricsCollector', 
    'MetricsConfig',
    'MCPRequestTracker',
    'MusicAnalysisTracker',
    'instrument_mcp_tool',
    'get_metrics_registry',
    'get_metrics_collector',
    'initialize_metrics',
    'HAS_PROMETHEUS'
]