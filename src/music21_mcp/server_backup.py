#!/usr/bin/env python3
"""
Music21 MCP Server - Modern Architecture Implementation
Implements 2024 MCP best practices with microservices patterns and clean architecture
"""
import asyncio
import gc
import json
import logging
import os
import signal
import sys
import tempfile
import time
import traceback
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import psutil

# Try to import MCP, but make it optional
try:
    from mcp.server.fastmcp import FastMCP

    HAS_MCP = True
except ImportError:
    HAS_MCP = False

    # Create a dummy FastMCP class for compatibility
    class FastMCP:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "MCP package is not installed. Please install with: pip install mcp"
            )

# Import new architecture components
from .core.mcp_registry import registry, MCPTool, MCPResource, MCPPrompt
from .core.services import (
    container,
    ScoreManagementService,
    AnalysisService,
    ServiceConfig,
    ServiceError,
)
from .core.mcp_adapters import (
    MCPAdapterService,
    create_tool_adapter,
    create_resource_adapter,
    create_prompt_adapter,
)


from music21 import (
    analysis,
    chord,
    converter,
    corpus,
    expressions,
    harmony,
    interval,
    key,
    meter,
    note,
    pitch,
    roman,
    scale,
    stream,
    tempo,
    voiceLeading,
)

# Import all tools
from .tools import (
    ChordAnalysisTool,
    CounterpointGeneratorTool,
    DeleteScoreTool,
    ExportScoreTool,
    HarmonizationTool,
    HarmonyAnalysisTool,
    ImportScoreTool,
    KeyAnalysisTool,
    ListScoresTool,
    PatternRecognitionTool,
    ScoreInfoTool,
    StyleImitationTool,
    VoiceLeadingAnalysisTool,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Server Configuration
class ServerConfig:
    def __init__(self):
        self.max_scores = 100
        self.max_score_size_mb = 50
        self.memory_limit_mb = 2048
        self.gc_threshold_mb = 1500
        self.cache_ttl_seconds = 3600
        self.rate_limit_per_minute = 100
        self.circuit_breaker_threshold = 5
        self.circuit_breaker_timeout = 60


config = ServerConfig()


# Simple resilience implementations
class SimpleCircuitBreaker:
    """Lightweight circuit breaker without external dependencies"""

    def __init__(self, name: str, threshold: int = 5, timeout: int = 60):
        self.name = name
        self.threshold = threshold
        self.timeout = timeout
        self.failures = 0
        self.last_failure = None
        self.state = "closed"

    def record_success(self):
        if self.state == "half_open":
            self.state = "closed"
            self.failures = 0
            logger.info(f"Circuit {self.name} closed")

    def record_failure(self):
        self.failures += 1
        self.last_failure = time.time()
        if self.failures >= self.threshold:
            self.state = "open"
            logger.warning(f"Circuit {self.name} opened after {self.failures} failures")

    def can_execute(self) -> bool:
        if self.state == "closed":
            return True
        if self.state == "open":
            if time.time() - self.last_failure > self.timeout:
                self.state = "half_open"
                logger.info(f"Circuit {self.name} half-open")
                return True
            return False
        return self.state == "half_open"


class SimpleRateLimiter:
    """Token bucket rate limiter"""

    def __init__(self, rate: float, burst: int):
        self.rate = rate
        self.burst = burst
        self.tokens = burst
        self.last_update = time.time()

    def acquire(self) -> bool:
        now = time.time()
        elapsed = now - self.last_update
        self.last_update = now

        # Add tokens based on elapsed time
        self.tokens = min(self.burst, self.tokens + elapsed * self.rate)

        if self.tokens >= 1:
            self.tokens -= 1
            return True
        return False


class ScoreManager:
    """Manages music scores with memory-aware caching"""

    def __init__(self, max_scores: int = 100):
        self.scores: Dict[str, Any] = {}
        self.access_times: Dict[str, float] = {}
        self.max_scores = max_scores
        self.lock = asyncio.Lock()

    async def add_score(self, score_id: str, score: Any):
        async with self.lock:
            # Check memory before adding
            if self._get_memory_usage_mb() > config.gc_threshold_mb:
                await self._cleanup_old_scores()

            # Remove oldest if at capacity
            if len(self.scores) >= self.max_scores:
                oldest = min(self.access_times.items(), key=lambda x: x[1])
                del self.scores[oldest[0]]
                del self.access_times[oldest[0]]

            self.scores[score_id] = score
            self.access_times[score_id] = time.time()

    async def get_score(self, score_id: str) -> Optional[Any]:
        async with self.lock:
            if score_id in self.scores:
                self.access_times[score_id] = time.time()
                return self.scores[score_id]
            return None

    async def remove_score(self, score_id: str) -> bool:
        async with self.lock:
            if score_id in self.scores:
                del self.scores[score_id]
                del self.access_times[score_id]
                return True
            return False

    async def list_scores(self) -> List[Dict[str, Any]]:
        async with self.lock:
            return [
                {
                    "id": score_id,
                    "last_accessed": self.access_times[score_id],
                    "title": getattr(
                        self.scores[score_id].metadata, "title", "Untitled"
                    ),
                }
                for score_id in self.scores
            ]

    def _get_memory_usage_mb(self) -> float:
        """Get current process memory usage in MB"""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024

    async def _cleanup_old_scores(self):
        """Remove old scores to free memory"""
        logger.info("Running memory cleanup...")
        cutoff_time = time.time() - config.cache_ttl_seconds

        to_remove = [
            score_id
            for score_id, access_time in self.access_times.items()
            if access_time < cutoff_time
        ]

        for score_id in to_remove:
            del self.scores[score_id]
            del self.access_times[score_id]

        # Force garbage collection
        gc.collect()
        logger.info(f"Cleaned up {len(to_remove)} old scores")


# Create FastMCP server
mcp = FastMCP("Music21 MCP Server - Production")

# Initialize modern architecture components
rate_limiter = SimpleRateLimiter(rate=config.rate_limit_per_minute / 60, burst=10)

# Circuit breakers for different operations
circuit_breakers = {
    "import": SimpleCircuitBreaker(
        "import", config.circuit_breaker_threshold, config.circuit_breaker_timeout
    ),
    "analysis": SimpleCircuitBreaker(
        "analysis", config.circuit_breaker_threshold, config.circuit_breaker_timeout
    ),
    "export": SimpleCircuitBreaker(
        "export", config.circuit_breaker_threshold, config.circuit_breaker_timeout
    ),
    "generation": SimpleCircuitBreaker(
        "generation", config.circuit_breaker_threshold, config.circuit_breaker_timeout
    ),
}

# Global services - will be initialized in main()
score_service = None
analysis_service = None
adapter_service = None


# Modern health check tool using new architecture
@mcp.tool()
async def health_check() -> Dict[str, Any]:
    """Check server health and resource usage with new architecture"""
    process = psutil.Process()
    memory = process.memory_info()

    # Get health from all services
    services_health = await container.health_check_all()
    registry_health = await registry.health_check()

    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "uptime_seconds": (
            datetime.now() - datetime.fromtimestamp(process.create_time())
        ).total_seconds(),
        "memory": {
            "used_mb": memory.rss / 1024 / 1024,
            "limit_mb": config.memory_limit_mb,
            "percent": (memory.rss / 1024 / 1024) / config.memory_limit_mb * 100,
        },
        "scores": {
            "count": len(score_service.scores) if score_service else 0,
            "max": config.max_scores,
        },
        "circuit_breakers": {name: cb.state for name, cb in circuit_breakers.items()},
        "services": services_health,
        "registry": registry_health,
        "architecture": {
            "version": "2024.1",
            "features": [
                "microservices_patterns",
                "dependency_injection",
                "mcp_2024_compliance",
                "service_adapters",
                "health_monitoring",
            ],
        },
    }


@mcp.tool()
async def cleanup_memory() -> Dict[str, Any]:
    """Force memory cleanup and garbage collection with new architecture"""
    try:
        before = psutil.Process().memory_info().rss / 1024 / 1024

        # Clear old scores using service
        if score_service:
            await score_service._cleanup_old_scores()

        # Force multiple GC passes
        for _ in range(3):
            gc.collect()

        after = psutil.Process().memory_info().rss / 1024 / 1024

        return {
            "status": "success",
            "memory_before_mb": round(before, 2),
            "memory_after_mb": round(after, 2),
            "freed_mb": round(before - after, 2),
            "scores_remaining": len(score_service.scores) if score_service else 0,
        }
    except Exception as e:
        logger.error(f"Cleanup failed: {str(e)}")
        return {"error": str(e)}


# Graceful shutdown handler
def shutdown_handler(signum, frame):
    logger.info("Received shutdown signal, cleaning up...")
    # Cleanup any resources
    sys.exit(0)


signal.signal(signal.SIGINT, shutdown_handler)
signal.signal(signal.SIGTERM, shutdown_handler)


# Initialize modern architecture with services and adapters
async def initialize_architecture():
    """Initialize the modern service architecture"""
    global score_service, analysis_service, adapter_service
    
    logger.info("Initializing modern MCP architecture...")
    
    # Create service configurations
    score_config = ServiceConfig(
        name="score_management",
        description="Music score storage and management service",
        enabled=True,
        timeout=30.0,
        retry_count=3,
        health_check_interval=60.0,
    )
    
    analysis_config = ServiceConfig(
        name="analysis",
        description="Music analysis service",
        enabled=True,
        timeout=30.0,
        retry_count=3,
        health_check_interval=60.0,
    )
    
    adapter_config = ServiceConfig(
        name="mcp_adapters",
        description="MCP adapter management service",
        enabled=True,
        timeout=30.0,
        retry_count=3,
        health_check_interval=60.0,
    )
    
    # Create services
    score_service = ScoreManagementService(score_config, max_scores=config.max_scores)
    analysis_service = AnalysisService(analysis_config)
    adapter_service = MCPAdapterService(adapter_config)
    
    # Register services with container
    container.register_service(score_service)
    container.register_service(analysis_service, dependencies=["score_management"])
    container.register_service(adapter_service)
    
    # Initialize all services
    await container.initialize_all()
    
    logger.info("Service architecture initialized successfully")


# Initialize and register all tools with modern architecture
async def initialize_tools():
    """Initialize all tool instances with modern MCP architecture"""
    # Create legacy tool instances for backward compatibility
    # Note: These expect Dict[str, Any] but we'll adapt them
    legacy_score_manager = score_service.scores  # Direct access for compatibility
    
    tool_instances = {
        "import": ImportScoreTool(legacy_score_manager),
        "list": ListScoresTool(legacy_score_manager),
        "key_analysis": KeyAnalysisTool(legacy_score_manager),
        "chord_analysis": ChordAnalysisTool(legacy_score_manager),
        "score_info": ScoreInfoTool(legacy_score_manager),
        "export": ExportScoreTool(legacy_score_manager),
        "delete": DeleteScoreTool(legacy_score_manager),
        "harmony_analysis": HarmonyAnalysisTool(legacy_score_manager),
        "voice_leading": VoiceLeadingAnalysisTool(legacy_score_manager),
        "pattern_recognition": PatternRecognitionTool(legacy_score_manager),
        "harmonization": HarmonizationTool(legacy_score_manager),
        "counterpoint": CounterpointGeneratorTool(legacy_score_manager),
        "style_imitation": StyleImitationTool(legacy_score_manager),
    }
    
    # Create MCP tool adapters for each tool
    for name, tool in tool_instances.items():
        description = getattr(tool, 'description', f'Music21 {name} tool')
        adapter = create_tool_adapter(name, description, tool)
        adapter_service.register_tool_adapter(adapter)
    
    # Create MCP Resources for data access
    await create_mcp_resources()
    
    # Create MCP Prompts for common usage patterns
    await create_mcp_prompts()
    
    # Initialize the registry
    await registry.initialize()
    
    # Register legacy tools with FastMCP for backward compatibility
    for name, tool in tool_instances.items():
        # Determine circuit breaker category
        if name in ["import", "export", "delete"]:
            cb_name = name
        elif name in ["harmonization", "counterpoint", "style_imitation"]:
            cb_name = "generation"
        else:
            cb_name = "analysis"

        # Get the circuit breaker for this tool
        cb = circuit_breakers.get(cb_name, circuit_breakers["analysis"])

        # Register the tool's execute method with MCP
        tool_func = create_resilient_tool(tool, cb, rate_limiter)
        mcp.tool(name=tool.name)(tool_func)
    
    logger.info(f"Initialized {len(tool_instances)} tools with modern architecture")


async def create_mcp_resources():
    """Create MCP Resources for data access endpoints"""
    logger.info("Creating MCP Resources...")
    
    # Resource for accessing score list
    async def get_score_list(**kwargs):
        """Get list of all scores"""
        return await score_service.list_scores()
    
    score_list_resource = create_resource_adapter(
        "score_list",
        "List of all available music scores",
        "scores/list",
        get_score_list
    )
    adapter_service.register_resource_adapter(score_list_resource)
    
    # Resource for accessing specific score metadata
    async def get_score_metadata(score_id: str, **kwargs):
        """Get metadata for a specific score"""
        scores = await score_service.list_scores()
        score_info = next((s for s in scores if s["id"] == score_id), None)
        return score_info or {"error": "Score not found"}
    
    score_metadata_resource = create_resource_adapter(
        "score_metadata",
        "Metadata for a specific music score",
        "scores/{score_id}/metadata",
        get_score_metadata
    )
    adapter_service.register_resource_adapter(score_metadata_resource)
    
    logger.info("MCP Resources created successfully")


async def create_mcp_prompts():
    """Create MCP Prompts for common usage patterns"""
    logger.info("Creating MCP Prompts...")
    
    # Prompt for basic score analysis workflow
    analysis_prompt = create_prompt_adapter(
        "analyze_score",
        "Complete analysis workflow for a music score",
        """Please analyze the score '{score_id}' using the following workflow:
1. First, get basic score information
2. Perform key analysis to understand the tonal structure
3. Analyze chord progressions and harmonic content
4. Examine voice leading patterns
5. Provide a comprehensive summary of findings

Score ID: {score_id}
Analysis Focus: {focus}""",
        [
            {"name": "score_id", "type": "string", "description": "ID of the score to analyze"},
            {"name": "focus", "type": "string", "description": "Specific aspect to focus on", "default": "comprehensive"}
        ]
    )
    adapter_service.register_prompt_adapter(analysis_prompt)
    
    # Prompt for composition workflow
    composition_prompt = create_prompt_adapter(
        "compose_music",
        "Guided composition workflow using various tools",
        """Let's create a new musical composition with the following parameters:
1. Start with harmonization of the melody in '{key}' 
2. Apply counterpoint techniques for voice leading
3. Use style imitation based on '{style}' period
4. Export the final result in '{format}' format

Key: {key}
Style: {style}
Format: {format}
Melody: {melody}""",
        [
            {"name": "key", "type": "string", "description": "Musical key for the composition"},
            {"name": "style", "type": "string", "description": "Musical style to imitate"},
            {"name": "format", "type": "string", "description": "Output format (midi, xml, etc.)"},
            {"name": "melody", "type": "string", "description": "Base melody or theme"}
        ]
    )
    adapter_service.register_prompt_adapter(composition_prompt)
    
    logger.info("MCP Prompts created successfully")


def create_resilient_tool(tool, circuit_breaker, rate_limiter):
    """Create a resilient wrapper for a tool"""

    async def resilient_execute(**kwargs):
        # Rate limiting
        if not rate_limiter.acquire():
            return {"error": "Rate limit exceeded, please try again later"}

        # Circuit breaker
        if not circuit_breaker.can_execute():
            return {
                "error": f"Service temporarily unavailable, retry in {circuit_breaker.timeout}s"
            }

        try:
            # Execute the tool
            result = await tool.execute(**kwargs)
            circuit_breaker.record_success()
            return result
        except Exception as e:
            circuit_breaker.record_failure()
            logger.error(f"Tool {tool.name} failed: {str(e)}")
            return {"error": str(e)}

    # Copy metadata
    resilient_execute.__name__ = tool.name
    resilient_execute.__doc__ = tool.description

    return resilient_execute


# Main entry point with modern architecture
async def main():
    logger.info(f"Starting Music21 MCP Server (Modern Architecture)")
    logger.info(f"Configuration: {json.dumps(config.__dict__, indent=2)}")
    
    try:
        # Initialize the modern architecture
        await initialize_architecture()
        
        # Initialize all tools with adapters
        await initialize_tools()
        
        logger.info("🚀 Modern MCP Server ready with 2024 architecture!")
        logger.info("✅ Services: Score Management, Analysis, Adapters")
        logger.info("✅ MCP Primitives: Tools, Resources, Prompts")
        logger.info("✅ Features: Dependency Injection, Health Monitoring, Circuit Breakers")
        
        # Run the server
        mcp.run()
        
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)
    finally:
        # Clean up services
        try:
            await container.cleanup_all()
            await registry.cleanup()
            logger.info("Services cleaned up successfully")
        except Exception as e:
            logger.error(f"Cleanup error: {str(e)}")


def run_server():
    """Synchronous wrapper for the async main function"""
    asyncio.run(main())


if __name__ == "__main__":
    run_server()
