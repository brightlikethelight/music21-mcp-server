#!/usr/bin/env python3
"""
Music21 MCP Server - Production-Ready Consolidated Version
Combines all features from various server implementations into a single, clean server
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
            raise ImportError("MCP package is not installed. Please install with: pip install mcp")

from music21 import (
    analysis, chord, converter, corpus, expressions, harmony, interval,
    key, meter, note, pitch, roman, scale, stream, tempo, voiceLeading
)

# Import all tools
from .tools import (
    ImportScoreTool, ListScoresTool, KeyAnalysisTool,
    ChordAnalysisTool, ScoreInfoTool, ExportScoreTool,
    DeleteScoreTool, HarmonyAnalysisTool, VoiceLeadingAnalysisTool,
    PatternRecognitionTool, HarmonizationTool,
    CounterpointGeneratorTool, StyleImitationTool
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
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
                    "title": getattr(self.scores[score_id].metadata, 'title', 'Untitled')
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
            score_id for score_id, access_time in self.access_times.items()
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

# Initialize components
score_manager = ScoreManager(max_scores=config.max_scores)
rate_limiter = SimpleRateLimiter(rate=config.rate_limit_per_minute/60, burst=10)

# Circuit breakers for different operations
circuit_breakers = {
    "import": SimpleCircuitBreaker("import", config.circuit_breaker_threshold, config.circuit_breaker_timeout),
    "analysis": SimpleCircuitBreaker("analysis", config.circuit_breaker_threshold, config.circuit_breaker_timeout),
    "export": SimpleCircuitBreaker("export", config.circuit_breaker_threshold, config.circuit_breaker_timeout),
    "generation": SimpleCircuitBreaker("generation", config.circuit_breaker_threshold, config.circuit_breaker_timeout)
}

# Health check tool (server-specific)
@mcp.tool()
async def health_check() -> Dict[str, Any]:
    """Check server health and resource usage"""
    process = psutil.Process()
    memory = process.memory_info()
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "uptime_seconds": (datetime.now() - datetime.fromtimestamp(process.create_time())).total_seconds(),
        "memory": {
            "used_mb": memory.rss / 1024 / 1024,
            "limit_mb": config.memory_limit_mb,
            "percent": (memory.rss / 1024 / 1024) / config.memory_limit_mb * 100
        },
        "scores": {
            "count": len(score_manager.scores),
            "max": config.max_scores
        },
        "circuit_breakers": {
            name: cb.state for name, cb in circuit_breakers.items()
        }
    }

@mcp.tool()
async def cleanup_memory() -> Dict[str, Any]:
    """Force memory cleanup and garbage collection"""
    try:
        before = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Clear old scores
        await score_manager._cleanup_old_scores()
        
        # Force multiple GC passes
        for _ in range(3):
            gc.collect()
        
        after = psutil.Process().memory_info().rss / 1024 / 1024
        
        return {
            "status": "success",
            "memory_before_mb": round(before, 2),
            "memory_after_mb": round(after, 2),
            "freed_mb": round(before - after, 2),
            "scores_remaining": len(score_manager.scores)
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

# Initialize and register all tools
def initialize_tools():
    """Initialize all tool instances with circuit breakers"""
    tool_instances = {
        'import': ImportScoreTool(score_manager),
        'list': ListScoresTool(score_manager),
        'key_analysis': KeyAnalysisTool(score_manager),
        'chord_analysis': ChordAnalysisTool(score_manager),
        'score_info': ScoreInfoTool(score_manager),
        'export': ExportScoreTool(score_manager),
        'delete': DeleteScoreTool(score_manager),
        'harmony_analysis': HarmonyAnalysisTool(score_manager),
        'voice_leading': VoiceLeadingAnalysisTool(score_manager),
        'pattern_recognition': PatternRecognitionTool(score_manager),
        'harmonization': HarmonizationTool(score_manager),
        'counterpoint': CounterpointGeneratorTool(score_manager),
        'style_imitation': StyleImitationTool(score_manager)
    }
    
    # Wrap each tool with resilience features
    for name, tool in tool_instances.items():
        # Determine circuit breaker category
        if name in ['import', 'export', 'delete']:
            cb_name = name
        elif name in ['harmonization', 'counterpoint', 'style_imitation']:
            cb_name = 'generation'
        else:
            cb_name = 'analysis'
        
        # Get the circuit breaker for this tool
        cb = circuit_breakers.get(cb_name, circuit_breakers['analysis'])
        
        # Register the tool's execute method with MCP
        tool_func = create_resilient_tool(tool, cb, rate_limiter)
        mcp.tool(name=tool.name)(tool_func)
    
    logger.info(f"Initialized {len(tool_instances)} tools with resilience features")

def create_resilient_tool(tool, circuit_breaker, rate_limiter):
    """Create a resilient wrapper for a tool"""
    async def resilient_execute(**kwargs):
        # Rate limiting
        if not rate_limiter.acquire():
            return {"error": "Rate limit exceeded, please try again later"}
        
        # Circuit breaker
        if not circuit_breaker.can_execute():
            return {"error": f"Service temporarily unavailable, retry in {circuit_breaker.timeout}s"}
        
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

# Main entry point
def main():
    logger.info(f"Starting Music21 MCP Server (Production)")
    logger.info(f"Configuration: {json.dumps(config.__dict__, indent=2)}")
    
    # Initialize all tools
    initialize_tools()
    
    # Run the server
    try:
        mcp.run()
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()