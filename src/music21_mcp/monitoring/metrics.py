"""
Comprehensive Prometheus Metrics for Music21 MCP Server

Enterprise-grade monitoring with:
- MCP protocol metrics
- Music analysis performance metrics
- Security and authentication metrics
- Resource utilization monitoring
- Business intelligence metrics
- OpenTelemetry integration
- Custom Grafana dashboards

Complies with 2024 observability standards for production deployment.
"""

import asyncio
import functools
import logging
import os
import psutil
import time
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Set
from dataclasses import dataclass

# Prometheus client imports
try:
    from prometheus_client import (
        Counter, Gauge, Histogram, Summary, Info, Enum,
        CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST,
        multiprocess, values
    )
    HAS_PROMETHEUS = True
except ImportError:
    HAS_PROMETHEUS = False
    # Mock classes for when prometheus_client is not available
    class Counter:
        def __init__(self, *args, **kwargs): pass
        def labels(self, *args, **kwargs): return self
        def inc(self, *args, **kwargs): pass
    
    class Gauge:
        def __init__(self, *args, **kwargs): pass
        def labels(self, *args, **kwargs): return self
        def set(self, *args, **kwargs): pass
        def inc(self, *args, **kwargs): pass
        def dec(self, *args, **kwargs): pass
    
    class Histogram:
        def __init__(self, *args, **kwargs): pass
        def labels(self, *args, **kwargs): return self
        def observe(self, *args, **kwargs): pass
    
    class Summary:
        def __init__(self, *args, **kwargs): pass
        def labels(self, *args, **kwargs): return self
        def observe(self, *args, **kwargs): pass
    
    class Info:
        def __init__(self, *args, **kwargs): pass
        def info(self, *args, **kwargs): pass
    
    class Enum:
        def __init__(self, *args, **kwargs): pass
        def labels(self, *args, **kwargs): return self
        def state(self, *args, **kwargs): pass
    
    def generate_latest(*args, **kwargs): return b""
    CONTENT_TYPE_LATEST = "text/plain"

# Setup metrics registry
if HAS_PROMETHEUS and os.environ.get('PROMETHEUS_MULTIPROC_DIR'):
    # Multi-process mode for production
    REGISTRY = CollectorRegistry()
    multiprocess.MultiProcessCollector(REGISTRY)
else:
    # Single process mode for development
    REGISTRY = CollectorRegistry() if HAS_PROMETHEUS else None

logger = logging.getLogger(__name__)


@dataclass
class MetricsConfig:
    """Configuration for metrics collection"""
    enabled: bool = True
    system_metrics_interval: int = 30  # seconds
    max_memory_gb: float = 4.0
    max_cpu_percent: float = 80.0
    track_client_usage: bool = True
    anonymize_client_ids: bool = True
    
    # Histogram buckets optimized for music analysis
    analysis_duration_buckets: tuple = (
        0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60, 120, 300, 600
    )
    
    # Request duration buckets for MCP
    request_duration_buckets: tuple = (
        0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10
    )


class MetricsRegistry:
    """Centralized registry for all Prometheus metrics"""
    
    def __init__(self, config: Optional[MetricsConfig] = None):
        self.config = config or MetricsConfig()
        self.registry = REGISTRY
        
        if not HAS_PROMETHEUS:
            logger.warning("Prometheus client not available. Metrics will be disabled.")
            return
        
        # Initialize all metrics
        self._init_mcp_metrics()
        self._init_music_metrics()
        self._init_security_metrics()
        self._init_system_metrics()
        self._init_business_metrics()
        self._init_server_info()
        
        logger.info("Prometheus metrics registry initialized")
    
    def _init_mcp_metrics(self):
        """Initialize MCP protocol metrics"""
        self.mcp_requests_total = Counter(
            'mcp_requests_total',
            'Total number of MCP requests',
            ['method', 'tool', 'status'],
            registry=self.registry
        )
        
        self.mcp_request_duration = Histogram(
            'mcp_request_duration_seconds',
            'MCP request duration in seconds',
            ['method', 'tool'],
            buckets=self.config.request_duration_buckets,
            registry=self.registry
        )
        
        self.mcp_active_sessions = Gauge(
            'mcp_active_sessions',
            'Number of active MCP sessions',
            registry=self.registry
        )
        
        self.mcp_protocol_errors = Counter(
            'mcp_protocol_errors_total',
            'Total number of MCP protocol errors',
            ['error_type', 'tool'],
            registry=self.registry
        )
        
        self.mcp_server_state = Enum(
            'mcp_server_state',
            'Current state of the MCP server',
            states=['starting', 'running', 'stopping', 'error'],
            registry=self.registry
        )
        
        self.mcp_concurrent_requests = Gauge(
            'mcp_concurrent_requests',
            'Number of concurrent MCP requests being processed',
            registry=self.registry
        )
    
    def _init_music_metrics(self):
        """Initialize music analysis metrics"""
        self.music_files_processed = Counter(
            'music21_files_processed_total',
            'Total number of music files processed',
            ['format', 'analysis_type', 'status'],
            registry=self.registry
        )
        
        self.music_analysis_duration = Histogram(
            'music21_analysis_duration_seconds',
            'Time spent analyzing music files',
            ['analysis_type', 'complexity'],
            buckets=self.config.analysis_duration_buckets,
            registry=self.registry
        )
        
        self.music_features_extracted = Counter(
            'music21_features_extracted_total',
            'Total number of music features extracted',
            ['feature_type', 'format'],
            registry=self.registry
        )
        
        self.music_analysis_queue_size = Gauge(
            'music21_analysis_queue_size',
            'Current size of the music analysis queue',
            registry=self.registry
        )
        
        self.music_cache_hits = Counter(
            'music21_cache_hits_total',
            'Number of cache hits for music analysis',
            ['cache_type'],
            registry=self.registry
        )
        
        self.music_cache_misses = Counter(
            'music21_cache_misses_total',
            'Number of cache misses for music analysis',
            ['cache_type'],
            registry=self.registry
        )
        
        self.music_corpus_size = Gauge(
            'music21_corpus_size',
            'Number of scores in the active corpus',
            registry=self.registry
        )
    
    def _init_security_metrics(self):
        """Initialize security monitoring metrics"""
        self.auth_attempts = Counter(
            'music21_auth_attempts_total',
            'Total authentication attempts',
            ['method', 'status'],
            registry=self.registry
        )
        
        self.security_violations = Counter(
            'music21_security_violations_total',
            'Security violations detected',
            ['violation_type', 'severity'],
            registry=self.registry
        )
        
        self.rate_limit_hits = Counter(
            'music21_rate_limit_hits_total',
            'Number of rate limit violations',
            ['limit_type', 'client_type'],
            registry=self.registry
        )
        
        self.blocked_requests = Counter(
            'music21_blocked_requests_total',
            'Number of blocked requests',
            ['block_reason', 'source_type'],
            registry=self.registry
        )
        
        self.active_threats = Gauge(
            'music21_active_threats',
            'Number of currently active security threats',
            ['threat_level'],
            registry=self.registry
        )
        
        self.oauth_tokens_issued = Counter(
            'music21_oauth_tokens_issued_total',
            'Number of OAuth tokens issued',
            ['grant_type'],
            registry=self.registry
        )
        
        self.oauth_tokens_revoked = Counter(
            'music21_oauth_tokens_revoked_total',
            'Number of OAuth tokens revoked',
            ['reason'],
            registry=self.registry
        )
    
    def _init_system_metrics(self):
        """Initialize system resource metrics"""
        self.memory_usage_bytes = Gauge(
            'music21_memory_usage_bytes',
            'Memory usage in bytes',
            ['type'],  # 'rss', 'vms', 'shared'
            registry=self.registry
        )
        
        self.cpu_usage_percent = Gauge(
            'music21_cpu_usage_percent',
            'CPU usage percentage',
            registry=self.registry
        )
        
        self.disk_usage_bytes = Gauge(
            'music21_disk_usage_bytes',
            'Disk usage in bytes',
            ['path', 'type'],  # 'used', 'free', 'total'
            registry=self.registry
        )
        
        self.network_bytes = Counter(
            'music21_network_bytes_total',
            'Network traffic in bytes',
            ['direction'],  # 'sent', 'received'
            registry=self.registry
        )
        
        self.gc_collections = Counter(
            'music21_gc_collections_total',
            'Number of garbage collections',
            ['generation'],
            registry=self.registry
        )
        
        self.file_descriptors = Gauge(
            'music21_open_file_descriptors',
            'Number of open file descriptors',
            registry=self.registry
        )
        
        self.thread_count = Gauge(
            'music21_thread_count',
            'Number of active threads',
            registry=self.registry
        )
    
    def _init_business_metrics(self):
        """Initialize business intelligence metrics"""
        self.api_usage_by_client = Counter(
            'music21_api_usage_by_client',
            'API usage by client',
            ['client_id', 'endpoint'],
            registry=self.registry
        )
        
        self.music_scores_created = Counter(
            'music21_scores_created_total',
            'Total music scores created',
            ['output_format'],
            registry=self.registry
        )
        
        self.user_sessions = Summary(
            'music21_user_session_duration_seconds',
            'Duration of user sessions',
            registry=self.registry
        )
        
        self.feature_usage = Counter(
            'music21_feature_usage_total',
            'Usage count for different features',
            ['feature_name', 'user_type'],
            registry=self.registry
        )
        
        self.error_budget_remaining = Gauge(
            'music21_error_budget_remaining_percent',
            'Remaining error budget percentage',
            ['service'],
            registry=self.registry
        )
        
        self.sla_compliance = Gauge(
            'music21_sla_compliance_percent',
            'SLA compliance percentage',
            ['service', 'period'],
            registry=self.registry
        )
    
    def _init_server_info(self):
        """Initialize server information metrics"""
        self.server_info = Info(
            'music21_server',
            'Music21 MCP server information',
            registry=self.registry
        )
        
        # Set server information
        import music21
        import sys
        import platform
        
        self.server_info.info({
            'version': '1.0.0',
            'music21_version': music21.VERSION_STR,
            'python_version': platform.python_version(),
            'platform': platform.platform(),
            'deployment': os.environ.get('DEPLOYMENT_ENV', 'development'),
            'mcp_transport': 'stdio',
            'security_enabled': 'true',
            'prometheus_enabled': str(HAS_PROMETHEUS)
        })


# Global metrics registry
_metrics_registry: Optional[MetricsRegistry] = None


def get_metrics_registry() -> MetricsRegistry:
    """Get the global metrics registry"""
    global _metrics_registry
    if _metrics_registry is None:
        _metrics_registry = MetricsRegistry()
    return _metrics_registry


def initialize_metrics(config: Optional[MetricsConfig] = None) -> MetricsRegistry:
    """Initialize the global metrics registry"""
    global _metrics_registry
    _metrics_registry = MetricsRegistry(config)
    return _metrics_registry


class MetricsCollector:
    """Handles metrics collection and reporting"""
    
    def __init__(self, registry: Optional[MetricsRegistry] = None):
        self.registry = registry or get_metrics_registry()
        self.process = psutil.Process()
        self.start_time = time.time()
        self.active_requests = 0
        
        # Background monitoring
        self._monitoring_task: Optional[asyncio.Task] = None
        
    async def start_monitoring(self):
        """Start background system monitoring"""
        if self._monitoring_task is None:
            self._monitoring_task = asyncio.create_task(self._monitor_system_resources())
            logger.info("Started background system monitoring")
    
    async def stop_monitoring(self):
        """Stop background system monitoring"""
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
            self._monitoring_task = None
            logger.info("Stopped background system monitoring")
    
    async def _monitor_system_resources(self):
        """Background task to monitor system resources"""
        while True:
            try:
                await self._collect_system_metrics()
                await asyncio.sleep(self.registry.config.system_metrics_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"System metrics collection error: {e}")
                await asyncio.sleep(self.registry.config.system_metrics_interval)
    
    async def _collect_system_metrics(self):
        """Collect system resource metrics"""
        if not HAS_PROMETHEUS:
            return
        
        try:
            # Memory metrics
            memory_info = self.process.memory_info()
            self.registry.memory_usage_bytes.labels(type='rss').set(memory_info.rss)
            self.registry.memory_usage_bytes.labels(type='vms').set(memory_info.vms)
            
            # CPU metrics
            cpu_percent = self.process.cpu_percent()
            self.registry.cpu_usage_percent.set(cpu_percent)
            
            # File descriptors
            try:
                fd_count = self.process.num_fds()
                self.registry.file_descriptors.set(fd_count)
            except AttributeError:
                # Windows doesn't have num_fds
                pass
            
            # Thread count
            thread_count = self.process.num_threads()
            self.registry.thread_count.set(thread_count)
            
            # Disk usage for common paths
            for path in ['/', '/var/log', '/tmp']:
                try:
                    disk_usage = psutil.disk_usage(path)
                    self.registry.disk_usage_bytes.labels(path=path, type='used').set(disk_usage.used)
                    self.registry.disk_usage_bytes.labels(path=path, type='free').set(disk_usage.free)
                    self.registry.disk_usage_bytes.labels(path=path, type='total').set(disk_usage.total)
                except (OSError, FileNotFoundError):
                    # Path might not exist on all systems
                    pass
            
            # Network metrics
            try:
                net_io = self.process.net_io_counters()
                self.registry.network_bytes.labels(direction='sent').inc(net_io.bytes_sent)
                self.registry.network_bytes.labels(direction='received').inc(net_io.bytes_recv)
            except AttributeError:
                # Might not be available on all platforms
                pass
            
            # Garbage collection metrics
            import gc
            for i, count in enumerate(gc.get_counts()):
                self.registry.gc_collections.labels(generation=str(i)).inc(count)
            
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
    
    def track_mcp_request(self, method: str, tool: str):
        """Context manager for tracking MCP requests"""
        return MCPRequestTracker(self.registry, method, tool)
    
    def track_music_analysis(self, analysis_type: str, file_format: str, complexity: str = 'medium'):
        """Context manager for tracking music analysis"""
        return MusicAnalysisTracker(self.registry, analysis_type, file_format, complexity)
    
    def record_auth_attempt(self, method: str, success: bool):
        """Record authentication attempt"""
        status = 'success' if success else 'failure'
        self.registry.auth_attempts.labels(method=method, status=status).inc()
    
    def record_security_violation(self, violation_type: str, severity: str):
        """Record security violation"""
        self.registry.security_violations.labels(
            violation_type=violation_type,
            severity=severity
        ).inc()
    
    def record_rate_limit_hit(self, limit_type: str, client_type: str = 'unknown'):
        """Record rate limit violation"""
        self.registry.rate_limit_hits.labels(
            limit_type=limit_type,
            client_type=client_type
        ).inc()
    
    def record_blocked_request(self, block_reason: str, source_type: str = 'unknown'):
        """Record blocked request"""
        self.registry.blocked_requests.labels(
            block_reason=block_reason,
            source_type=source_type
        ).inc()
    
    def update_active_threats(self, threat_level: str, count: int):
        """Update active threat count"""
        self.registry.active_threats.labels(threat_level=threat_level).set(count)
    
    def record_feature_usage(self, feature_name: str, user_type: str = 'anonymous'):
        """Record feature usage"""
        if self.registry.config.anonymize_client_ids and user_type != 'anonymous':
            user_type = 'authenticated_user'
        
        self.registry.feature_usage.labels(
            feature_name=feature_name,
            user_type=user_type
        ).inc()
    
    def update_error_budget(self, service: str, remaining_percent: float):
        """Update error budget remaining"""
        self.registry.error_budget_remaining.labels(service=service).set(remaining_percent)
    
    def update_sla_compliance(self, service: str, period: str, compliance_percent: float):
        """Update SLA compliance"""
        self.registry.sla_compliance.labels(
            service=service,
            period=period
        ).set(compliance_percent)
    
    def get_metrics_data(self) -> bytes:
        """Get metrics data in Prometheus format"""
        if not HAS_PROMETHEUS:
            return b"# Prometheus metrics not available\n"
        
        return generate_latest(self.registry.registry)


class MCPRequestTracker:
    """Context manager for tracking MCP requests"""
    
    def __init__(self, registry: MetricsRegistry, method: str, tool: str):
        self.registry = registry
        self.method = method
        self.tool = tool
        self.start_time = None
        self.status = 'success'
    
    async def __aenter__(self):
        self.start_time = time.time()
        self.registry.mcp_concurrent_requests.inc()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.status = 'error'
            error_type = exc_type.__name__
            self.registry.mcp_protocol_errors.labels(
                error_type=error_type,
                tool=self.tool
            ).inc()
        
        # Record duration
        if self.start_time:
            duration = time.time() - self.start_time
            self.registry.mcp_request_duration.labels(
                method=self.method,
                tool=self.tool
            ).observe(duration)
        
        # Record request completion
        self.registry.mcp_requests_total.labels(
            method=self.method,
            tool=self.tool,
            status=self.status
        ).inc()
        
        self.registry.mcp_concurrent_requests.dec()
    
    def set_error(self, error_type: str):
        """Manually set error status"""
        self.status = 'error'


class MusicAnalysisTracker:
    """Context manager for tracking music analysis operations"""
    
    def __init__(self, registry: MetricsRegistry, analysis_type: str, file_format: str, complexity: str):
        self.registry = registry
        self.analysis_type = analysis_type
        self.file_format = file_format
        self.complexity = complexity
        self.start_time = None
        self.status = 'success'
    
    async def __aenter__(self):
        self.start_time = time.time()
        self.registry.music_analysis_queue_size.inc()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.status = 'error'
        
        # Record processing
        self.registry.music_files_processed.labels(
            format=self.file_format,
            analysis_type=self.analysis_type,
            status=self.status
        ).inc()
        
        # Record duration
        if self.start_time:
            duration = time.time() - self.start_time
            self.registry.music_analysis_duration.labels(
                analysis_type=self.analysis_type,
                complexity=self.complexity
            ).observe(duration)
        
        self.registry.music_analysis_queue_size.dec()
    
    def record_feature_extracted(self, feature_type: str):
        """Record extracted feature"""
        self.registry.music_features_extracted.labels(
            feature_type=feature_type,
            format=self.file_format
        ).inc()


# Instrumentation decorator
def instrument_mcp_tool(tool_name: str):
    """Decorator to instrument MCP tools with Prometheus metrics"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            collector = MetricsCollector()
            
            async with collector.track_mcp_request('tool_call', tool_name) as tracker:
                try:
                    result = await func(*args, **kwargs)
                    
                    # Record feature usage
                    collector.record_feature_usage(tool_name)
                    
                    return result
                    
                except Exception as e:
                    tracker.set_error(type(e).__name__)
                    raise
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            # Similar implementation for sync functions
            collector = MetricsCollector()
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                
                # Record metrics
                collector.record_feature_usage(tool_name)
                
                return result
                
            except Exception as e:
                collector.registry.mcp_protocol_errors.labels(
                    error_type=type(e).__name__,
                    tool=tool_name
                ).inc()
                raise
            
            finally:
                duration = time.time() - start_time
                collector.registry.mcp_request_duration.labels(
                    method='tool_call',
                    tool=tool_name
                ).observe(duration)
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator


# Global metrics collector
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector"""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector


# Export main classes
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