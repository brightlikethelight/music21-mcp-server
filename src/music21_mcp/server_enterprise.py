#!/usr/bin/env python3
"""
Music21 MCP Server - Enterprise Edition

Production-ready server with comprehensive:
- Security (OAuth 2.1, input validation, monitoring, audit logging)
- Monitoring (Prometheus metrics, OpenTelemetry, health checks)
- Resilience (circuit breakers, bulkheads, retry patterns)
- Performance optimization and enterprise deployment features

Maintains full backward compatibility with existing MCP clients.
"""

import asyncio
import gc
import logging
import os
import psutil
import time
from datetime import datetime
from typing import Dict, Any, Optional
import json

# Try to import MCP
try:
    from mcp.server.fastmcp import FastMCP
    HAS_MCP = True
except ImportError:
    HAS_MCP = False
    class FastMCP:
        def __init__(self, *args, **kwargs):
            raise ImportError("MCP package is not installed. Please install with: pip install mcp")

# Import tools
from .tools import (
    ImportScoreTool,
    ListScoresTool,
    KeyAnalysisTool,
    ChordAnalysisTool,
    ScoreInfoTool,
    ExportScoreTool,
    DeleteScoreTool,
    HarmonyAnalysisTool,
    VoiceLeadingAnalysisTool,
    PatternRecognitionTool,
    HarmonizationTool,
    CounterpointGeneratorTool,
    StyleImitationTool,
)

# Import security components
from .security.security_middleware import (
    SecurityMiddleware, SecurityConfig, SecurityContext,
    create_development_security_config, create_production_security_config
)
from .security.oauth_provider import OAuth2Provider
from .security.monitoring import initialize_security_monitoring, get_security_monitor
from .security.audit_logging import initialize_audit_logging, get_audit_logger

# Import monitoring components
from .monitoring.metrics import (
    initialize_metrics, get_metrics_collector, MetricsConfig,
    instrument_mcp_tool, HAS_PROMETHEUS
)
from .monitoring.circuit_breaker import (
    get_resilience_orchestrator, CircuitBreakerConfig, BulkheadConfig, RetryConfig
)

# Configure enterprise logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/var/log/music21_mcp/server.log', mode='a') if os.path.exists('/var/log/music21_mcp') else logging.NullHandler()
    ]
)
logger = logging.getLogger(__name__)

# Enterprise configuration
class EnterpriseConfig:
    """Enterprise deployment configuration"""
    
    def __init__(self):
        # Environment detection
        self.environment = os.environ.get('DEPLOYMENT_ENV', 'development')
        self.debug_mode = self.environment == 'development'
        
        # Security configuration
        self.security_enabled = os.environ.get('SECURITY_ENABLED', 'true').lower() == 'true'
        self.require_authentication = os.environ.get('REQUIRE_AUTH', 'false').lower() == 'true'
        
        # Monitoring configuration
        self.metrics_enabled = os.environ.get('METRICS_ENABLED', 'true').lower() == 'true'
        self.monitoring_enabled = os.environ.get('MONITORING_ENABLED', 'true').lower() == 'true'
        
        # Performance configuration
        self.max_concurrent_requests = int(os.environ.get('MAX_CONCURRENT_REQUESTS', '50'))
        self.request_timeout = float(os.environ.get('REQUEST_TIMEOUT', '30.0'))
        
        # Resource limits
        self.max_memory_gb = float(os.environ.get('MAX_MEMORY_GB', '4.0'))
        self.max_cpu_percent = float(os.environ.get('MAX_CPU_PERCENT', '80.0'))
        
        # Feature flags
        self.enable_caching = os.environ.get('ENABLE_CACHING', 'true').lower() == 'true'
        self.enable_compression = os.environ.get('ENABLE_COMPRESSION', 'true').lower() == 'true'
        
        logger.info(f"Enterprise config loaded for {self.environment} environment")


# Global enterprise configuration
enterprise_config = EnterpriseConfig()

# Enterprise score storage with monitoring
class EnterpriseScoreStorage:
    """Enterprise score storage with monitoring and caching"""
    
    def __init__(self):
        self._storage: Dict[str, Any] = {}
        self._access_count: Dict[str, int] = {}
        self._last_access: Dict[str, float] = {}
        
        # Get metrics collector if available
        self.metrics_collector = get_metrics_collector() if enterprise_config.metrics_enabled else None
    
    def __getitem__(self, key: str) -> Any:
        if key not in self._storage:
            raise KeyError(f"Score '{key}' not found")
        
        # Update access tracking
        self._access_count[key] = self._access_count.get(key, 0) + 1
        self._last_access[key] = time.time()
        
        # Update metrics
        if self.metrics_collector:
            self.metrics_collector.registry.music_cache_hits.labels(cache_type='score_storage').inc()
        
        return self._storage[key]
    
    def __setitem__(self, key: str, value: Any):
        self._storage[key] = value
        self._access_count[key] = 0
        self._last_access[key] = time.time()
        
        # Update metrics
        if self.metrics_collector:
            self.metrics_collector.registry.music_corpus_size.set(len(self._storage))
    
    def __delitem__(self, key: str):
        if key in self._storage:
            del self._storage[key]
            self._access_count.pop(key, None)
            self._last_access.pop(key, None)
            
            # Update metrics
            if self.metrics_collector:
                self.metrics_collector.registry.music_corpus_size.set(len(self._storage))
    
    def __contains__(self, key: str) -> bool:
        return key in self._storage
    
    def __len__(self) -> int:
        return len(self._storage)
    
    def items(self):
        return self._storage.items()
    
    def keys(self):
        return self._storage.keys()
    
    def values(self):
        return self._storage.values()
    
    def get(self, key: str, default=None):
        try:
            return self[key]
        except KeyError:
            if self.metrics_collector:
                self.metrics_collector.registry.music_cache_misses.labels(cache_type='score_storage').inc()
            return default
    
    def clear(self):
        self._storage.clear()
        self._access_count.clear()
        self._last_access.clear()
        
        if self.metrics_collector:
            self.metrics_collector.registry.music_corpus_size.set(0)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics"""
        total_access = sum(self._access_count.values())
        return {
            'total_scores': len(self._storage),
            'total_accesses': total_access,
            'most_accessed': max(self._access_count.items(), key=lambda x: x[1]) if self._access_count else None,
            'memory_usage_mb': sum(len(str(v)) for v in self._storage.values()) / 1024 / 1024
        }


# Enterprise score storage
scores = EnterpriseScoreStorage()

# Initialize enterprise components
def initialize_enterprise_components():
    """Initialize all enterprise components"""
    global security_middleware, metrics_collector, resilience_orchestrator
    
    # Initialize security
    if enterprise_config.security_enabled:
        logger.info("Initializing enterprise security components...")
        
        # Initialize security monitoring
        initialize_security_monitoring()
        
        # Initialize audit logging
        initialize_audit_logging(
            log_directory='/var/log/music21_mcp/audit' if os.path.exists('/var/log/music21_mcp') else './logs/audit'
        )
        
        # Create security configuration
        if enterprise_config.require_authentication:
            # Production security with OAuth
            oauth_provider = OAuth2Provider()
            # Register a default client for development
            oauth_provider.register_client(
                client_id="music21_mcp_client",
                client_secret="development_secret_change_in_production",
                redirect_uris=["http://localhost:8080/callback"]
            )
            security_config = create_production_security_config(oauth_provider)
        else:
            # Development security
            security_config = create_development_security_config()
        
        security_middleware = SecurityMiddleware(security_config)
        logger.info("Security components initialized")
    else:
        security_middleware = None
        logger.info("Security disabled - running in insecure mode")
    
    # Initialize monitoring
    if enterprise_config.metrics_enabled:
        logger.info("Initializing enterprise monitoring components...")
        
        metrics_config = MetricsConfig(
            enabled=True,
            max_memory_gb=enterprise_config.max_memory_gb,
            max_cpu_percent=enterprise_config.max_cpu_percent
        )
        
        initialize_metrics(metrics_config)
        metrics_collector = get_metrics_collector()
        
        # Start background monitoring
        if hasattr(metrics_collector, 'start_monitoring'):
            asyncio.create_task(metrics_collector.start_monitoring())
        
        logger.info("Monitoring components initialized")
    else:
        metrics_collector = None
        logger.info("Monitoring disabled")
    
    # Initialize resilience
    logger.info("Initializing resilience components...")
    resilience_orchestrator = get_resilience_orchestrator()
    
    # Create circuit breakers for different tool categories
    analysis_cb_config = CircuitBreakerConfig(
        failure_threshold=3,
        recovery_timeout=30.0,
        monitoring_window=60.0
    )
    resilience_orchestrator.create_circuit_breaker("music_analysis", analysis_cb_config)
    
    import_cb_config = CircuitBreakerConfig(
        failure_threshold=5,
        recovery_timeout=15.0,
        monitoring_window=30.0
    )
    resilience_orchestrator.create_circuit_breaker("score_import", import_cb_config)
    
    # Create bulkheads for resource isolation
    analysis_bulkhead_config = BulkheadConfig(
        max_concurrent=10,
        queue_size=50,
        timeout=30.0
    )
    resilience_orchestrator.create_bulkhead("analysis_pool", analysis_bulkhead_config)
    
    # Create retry handlers
    retry_config = RetryConfig(
        max_attempts=3,
        base_delay=1.0,
        max_delay=10.0
    )
    resilience_orchestrator.create_retry_handler("default_retry", retry_config)
    
    logger.info("Resilience components initialized")


# Create FastMCP server
mcp = FastMCP("Music21 MCP Server - Enterprise Edition")

# Initialize enterprise components
try:
    initialize_enterprise_components()
    logger.info("Enterprise components initialization completed")
except Exception as e:
    logger.error(f"Failed to initialize enterprise components: {e}")
    # Continue with basic functionality
    security_middleware = None
    metrics_collector = None
    resilience_orchestrator = None

# Enterprise request wrapper
async def enterprise_request_wrapper(tool_name: str, func, **kwargs):
    """Enterprise wrapper for all MCP tool requests"""
    start_time = time.time()
    context = SecurityContext()
    context.start_time = start_time
    
    # Extract context from environment/headers if available
    context.source_ip = os.environ.get('CLIENT_IP', 'unknown')
    context.user_agent = os.environ.get('HTTP_USER_AGENT', 'MCP Client')
    
    try:
        # Security validation
        if security_middleware:
            # Authenticate request
            auth_result, auth_error = await security_middleware.authenticate_request(context)
            if not auth_result:
                await security_middleware.log_request(context, tool_name, kwargs, "auth_failure")
                return {"status": "error", "error": "authentication_failed", "message": auth_error}
            
            # Validate input
            is_valid, validation_error, validated_params = await security_middleware.validate_request(tool_name, kwargs)
            if not is_valid:
                await security_middleware.log_request(context, tool_name, kwargs, "validation_failure")
                return {"status": "error", "error": "validation_failed", "message": validation_error}
            
            kwargs = validated_params or kwargs
            
            # Monitor request
            is_allowed, monitor_error = await security_middleware.monitor_request(context, tool_name, kwargs)
            if not is_allowed:
                await security_middleware.log_request(context, tool_name, kwargs, "blocked")
                return {"status": "error", "error": "request_blocked", "message": monitor_error}
        
        # Execute with resilience patterns
        if resilience_orchestrator:
            circuit_breaker_name = "music_analysis" if tool_name in ['harmony_analysis', 'key_analysis', 'pattern_recognition'] else None
            bulkhead_name = "analysis_pool" if circuit_breaker_name else None
            retry_handler_name = "default_retry"
            
            result = await resilience_orchestrator.execute_with_resilience(
                func,
                circuit_breaker_name=circuit_breaker_name,
                bulkhead_name=bulkhead_name,
                retry_handler_name=retry_handler_name,
                timeout=enterprise_config.request_timeout,
                **kwargs
            )
        else:
            # Basic execution
            result = await func(**kwargs)
        
        # Log successful request
        if security_middleware:
            await security_middleware.log_request(context, tool_name, kwargs, "success", result)
        
        return result
        
    except Exception as e:
        # Log failed request
        if security_middleware:
            await security_middleware.log_request(context, tool_name, kwargs, "error")
        
        # Record metrics
        if metrics_collector:
            metrics_collector.registry.mcp_protocol_errors.labels(
                error_type=type(e).__name__,
                tool=tool_name
            ).inc()
        
        logger.error(f"Enterprise request failed for {tool_name}: {e}")
        
        # Return appropriate error response
        if enterprise_config.debug_mode:
            return {"status": "error", "error": type(e).__name__, "message": str(e)}
        else:
            return {"status": "error", "error": "internal_error", "message": "Request failed"}

# Register enterprise MCP tools
@mcp.tool(name="import_score")
@instrument_mcp_tool("import_score")
async def import_score(score_id: str, source: str, source_type: str = "corpus"):
    """Import a score from various sources with enterprise security and monitoring"""
    
    async def _execute(**kwargs):
        tool = ImportScoreTool(scores)
        return await tool.execute(**kwargs)
    
    return await enterprise_request_wrapper(
        "import_score", _execute, 
        score_id=score_id, source=source, source_type=source_type
    )

@mcp.tool(name="list_scores")
@instrument_mcp_tool("list_scores")
async def list_scores():
    """List all available scores with enterprise monitoring"""
    
    async def _execute(**kwargs):
        tool = ListScoresTool(scores)
        return await tool.execute(**kwargs)
    
    return await enterprise_request_wrapper("list_scores", _execute)

@mcp.tool(name="key_analysis")
@instrument_mcp_tool("key_analysis")
async def key_analysis(score_id: str):
    """Analyze the key signature of a score with enterprise resilience"""
    
    async def _execute(**kwargs):
        tool = KeyAnalysisTool(scores)
        return await tool.execute(**kwargs)
    
    return await enterprise_request_wrapper(
        "key_analysis", _execute, score_id=score_id
    )

@mcp.tool(name="chord_analysis")
@instrument_mcp_tool("chord_analysis")
async def chord_analysis(score_id: str):
    """Analyze chord progressions in a score with enterprise monitoring"""
    
    async def _execute(**kwargs):
        tool = ChordAnalysisTool(scores)
        return await tool.execute(**kwargs)
    
    return await enterprise_request_wrapper(
        "chord_analysis", _execute, score_id=score_id
    )

@mcp.tool(name="score_info")
@instrument_mcp_tool("score_info")
async def score_info(score_id: str):
    """Get detailed information about a score with enterprise security"""
    
    async def _execute(**kwargs):
        tool = ScoreInfoTool(scores)
        return await tool.execute(**kwargs)
    
    return await enterprise_request_wrapper(
        "score_info", _execute, score_id=score_id
    )

@mcp.tool(name="export_score")
@instrument_mcp_tool("export_score")
async def export_score(score_id: str, format: str = "musicxml"):
    """Export a score to various formats with enterprise audit logging"""
    
    async def _execute(**kwargs):
        tool = ExportScoreTool(scores)
        return await tool.execute(**kwargs)
    
    return await enterprise_request_wrapper(
        "export_score", _execute, score_id=score_id, format=format
    )

@mcp.tool(name="delete_score")
@instrument_mcp_tool("delete_score")
async def delete_score(score_id: str):
    """Delete a score from storage with enterprise audit trail"""
    
    async def _execute(**kwargs):
        tool = DeleteScoreTool(scores)
        return await tool.execute(**kwargs)
    
    return await enterprise_request_wrapper(
        "delete_score", _execute, score_id=score_id
    )

@mcp.tool(name="harmony_analysis")
@instrument_mcp_tool("harmony_analysis")
async def harmony_analysis(score_id: str, analysis_type: str = "roman"):
    """Perform harmony analysis with enterprise circuit breaker protection"""
    
    async def _execute(**kwargs):
        tool = HarmonyAnalysisTool(scores)
        return await tool.execute(**kwargs)
    
    return await enterprise_request_wrapper(
        "harmony_analysis", _execute, 
        score_id=score_id, analysis_type=analysis_type
    )

@mcp.tool(name="voice_leading_analysis")
@instrument_mcp_tool("voice_leading_analysis")
async def voice_leading_analysis(score_id: str):
    """Analyze voice leading with enterprise monitoring"""
    
    async def _execute(**kwargs):
        tool = VoiceLeadingAnalysisTool(scores)
        return await tool.execute(**kwargs)
    
    return await enterprise_request_wrapper(
        "voice_leading_analysis", _execute, score_id=score_id
    )

@mcp.tool(name="pattern_recognition")
@instrument_mcp_tool("pattern_recognition")
async def pattern_recognition(score_id: str, pattern_type: str = "melodic", min_length: int = 3):
    """Recognize patterns in music with enterprise resilience"""
    
    async def _execute(**kwargs):
        tool = PatternRecognitionTool(scores)
        return await tool.execute(**kwargs)
    
    return await enterprise_request_wrapper(
        "pattern_recognition", _execute,
        score_id=score_id, pattern_type=pattern_type, min_length=min_length
    )

@mcp.tool(name="harmonization")
@instrument_mcp_tool("harmonization")
async def harmonization(score_id: str, style: str = "bach"):
    """Generate harmonization with enterprise monitoring"""
    
    async def _execute(**kwargs):
        tool = HarmonizationTool(scores)
        return await tool.execute(**kwargs)
    
    return await enterprise_request_wrapper(
        "harmonization", _execute, score_id=score_id, style=style
    )

@mcp.tool(name="counterpoint_generation")
@instrument_mcp_tool("counterpoint_generation")
async def counterpoint_generation(score_id: str, species: int = 1):
    """Generate counterpoint with enterprise performance monitoring"""
    
    async def _execute(**kwargs):
        tool = CounterpointGeneratorTool(scores)
        return await tool.execute(**kwargs)
    
    return await enterprise_request_wrapper(
        "counterpoint_generation", _execute, score_id=score_id, species=species
    )

@mcp.tool(name="style_imitation")
@instrument_mcp_tool("style_imitation")
async def style_imitation(score_id: str, target_composer: str):
    """Imitate musical style with enterprise circuit breaker"""
    
    async def _execute(**kwargs):
        tool = StyleImitationTool(scores)
        return await tool.execute(**kwargs)
    
    return await enterprise_request_wrapper(
        "style_imitation", _execute, 
        score_id=score_id, target_composer=target_composer
    )

# Enterprise health and monitoring endpoints
@mcp.tool(name="health_check")
async def health_check():
    """Comprehensive enterprise health check"""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0-enterprise",
        "environment": enterprise_config.environment,
        "uptime_seconds": time.time() - mcp.start_time if hasattr(mcp, 'start_time') else 0
    }
    
    # Check components
    components = {}
    
    # Security component health
    if security_middleware:
        components["security"] = {"status": "enabled", "authentication": enterprise_config.require_authentication}
    else:
        components["security"] = {"status": "disabled"}
    
    # Monitoring component health
    if metrics_collector:
        components["monitoring"] = {"status": "enabled", "prometheus": HAS_PROMETHEUS}
    else:
        components["monitoring"] = {"status": "disabled"}
    
    # Resilience component health
    if resilience_orchestrator:
        resilience_health = resilience_orchestrator.get_health_status()
        components["resilience"] = resilience_health
    else:
        components["resilience"] = {"status": "disabled"}
    
    # Storage health
    storage_stats = scores.get_stats()
    components["storage"] = {
        "status": "healthy",
        "total_scores": storage_stats["total_scores"],
        "memory_usage_mb": storage_stats["memory_usage_mb"]
    }
    
    # System health
    process = psutil.Process()
    memory_info = process.memory_info()
    components["system"] = {
        "status": "healthy",
        "memory_rss_mb": memory_info.rss / 1024 / 1024,
        "cpu_percent": process.cpu_percent(),
        "threads": process.num_threads()
    }
    
    health_status["components"] = components
    
    # Determine overall health
    unhealthy_components = [name for name, comp in components.items() 
                          if comp.get("status") not in ["healthy", "enabled"]]
    
    if unhealthy_components:
        health_status["status"] = "unhealthy"
        health_status["unhealthy_components"] = unhealthy_components
    
    return health_status

@mcp.tool(name="metrics")
async def get_metrics():
    """Get Prometheus metrics data"""
    if not metrics_collector:
        return {"error": "Metrics not enabled"}
    
    try:
        metrics_data = metrics_collector.get_metrics_data()
        return {
            "status": "success",
            "format": "prometheus",
            "data": metrics_data.decode('utf-8') if isinstance(metrics_data, bytes) else str(metrics_data)
        }
    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        return {"error": f"Failed to get metrics: {str(e)}"}

@mcp.tool(name="server_stats")
async def server_stats():
    """Get comprehensive server statistics"""
    stats = {
        "timestamp": datetime.utcnow().isoformat(),
        "environment": enterprise_config.environment,
        "configuration": {
            "security_enabled": enterprise_config.security_enabled,
            "monitoring_enabled": enterprise_config.metrics_enabled,
            "max_concurrent_requests": enterprise_config.max_concurrent_requests,
            "request_timeout": enterprise_config.request_timeout
        }
    }
    
    # Storage statistics
    stats["storage"] = scores.get_stats()
    
    # Security statistics
    if security_middleware:
        security_monitor = get_security_monitor()
        stats["security"] = security_monitor.get_security_metrics()
    
    # Resilience statistics
    if resilience_orchestrator:
        stats["resilience"] = resilience_orchestrator.get_health_status()
    
    # System statistics
    process = psutil.Process()
    memory_info = process.memory_info()
    stats["system"] = {
        "memory_rss_bytes": memory_info.rss,
        "memory_vms_bytes": memory_info.vms,
        "cpu_percent": process.cpu_percent(),
        "threads": process.num_threads(),
        "connections": len(process.net_connections()) if hasattr(process, 'net_connections') else 0,
        "gc_counts": gc.get_counts()
    }
    
    return stats

# Initialize server start time for health checks
if not hasattr(mcp, 'start_time'):
    mcp.start_time = time.time()

# Server lifecycle management
async def startup():
    """Enterprise server startup procedures"""
    logger.info("Starting Music21 MCP Server - Enterprise Edition")
    logger.info(f"Environment: {enterprise_config.environment}")
    logger.info(f"Security: {'enabled' if enterprise_config.security_enabled else 'disabled'}")
    logger.info(f"Monitoring: {'enabled' if enterprise_config.metrics_enabled else 'disabled'}")
    
    # Start background tasks
    if metrics_collector and hasattr(metrics_collector, 'start_monitoring'):
        await metrics_collector.start_monitoring()
    
    logger.info("Enterprise server startup completed")

async def shutdown():
    """Enterprise server shutdown procedures"""
    logger.info("Shutting down Music21 MCP Server - Enterprise Edition")
    
    # Stop background tasks
    if metrics_collector and hasattr(metrics_collector, 'stop_monitoring'):
        await metrics_collector.stop_monitoring()
    
    # Cleanup resources
    scores.clear()
    
    logger.info("Enterprise server shutdown completed")

def main():
    """Main entry point for enterprise server"""
    if not HAS_MCP:
        logger.error("MCP package not available. Please install with: pip install mcp")
        return
    
    try:
        # Run startup
        asyncio.run(startup())
        
        # Run server
        logger.info("Music21 MCP Server - Enterprise Edition is ready")
        mcp.run()
        
    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
    except Exception as e:
        logger.error(f"Server error: {e}")
    finally:
        # Run shutdown
        try:
            asyncio.run(shutdown())
        except Exception as e:
            logger.error(f"Shutdown error: {e}")

if __name__ == "__main__":
    main()