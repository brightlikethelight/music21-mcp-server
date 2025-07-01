#\!/usr/bin/env python3
"""
Basic resilient MCP server for production testing
Uses FastMCP for simplicity and stability
"""
import asyncio
import logging
import time
import psutil
import gc
from typing import Dict, Any, Optional
from datetime import datetime

from mcp.server.fastmcp import FastMCP
from music21 import converter, corpus, stream, key

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastMCP server
mcp = FastMCP("Music21 Basic Resilient Server")

# Simple circuit breaker
class SimpleCircuitBreaker:
    def __init__(self, name: str, threshold=5, timeout=60):
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
            logger.warning(f"Circuit {self.name} opened")
    
    def can_execute(self):
        if self.state == "closed":
            return True
        if self.state == "open":
            if time.time() - self.last_failure > self.timeout:
                self.state = "half_open"
                return True
        return self.state == "half_open"

# Circuit breakers for each operation
circuit_breakers = {
    "import": SimpleCircuitBreaker("import"),
    "analyze": SimpleCircuitBreaker("analyze"),
    "export": SimpleCircuitBreaker("export")
}

# Simple rate limiter
class SimpleRateLimiter:
    def __init__(self, rate: float):
        self.rate = rate
        self.last_call = 0
    
    def can_proceed(self):
        now = time.time()
        if now - self.last_call < 1.0 / self.rate:
            return False
        self.last_call = now
        return True

rate_limiter = SimpleRateLimiter(10.0)  # 10 requests per second

# Score storage
scores = {}

@mcp.tool()
async def health_check() -> Dict[str, Any]:
    """Check server health and resource usage"""
    process = psutil.Process()
    memory = process.memory_info()
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "memory_mb": memory.rss / 1024 / 1024,
        "cpu_percent": process.cpu_percent(),
        "circuit_breakers": {
            name: cb.state for name, cb in circuit_breakers.items()
        },
        "active_scores": len(scores)
    }

@mcp.tool()
async def import_score(score_id: str, source: str) -> Dict[str, Any]:
    """Import a score with circuit breaker protection"""
    
    if not rate_limiter.can_proceed():
        return {"error": "Rate limit exceeded"}
    
    cb = circuit_breakers["import"]
    if not cb.can_execute():
        return {"error": f"Circuit breaker open, retry in {cb.timeout}s"}
    
    try:
        # Force garbage collection before import
        gc.collect()
        
        # Import the score
        if '/' in source and not source.endswith(('.xml', '.mid', '.midi')):
            score = corpus.parse(source)
        else:
            score = converter.parse(source)
        
        scores[score_id] = score
        cb.record_success()
        
        return {
            "status": "success",
            "score_id": score_id,
            "measures": len(score.parts[0].getElementsByClass('Measure')) if score.parts else 0
        }
    except Exception as e:
        cb.record_failure()
        logger.error(f"Import failed: {e}")
        return {"error": str(e)}

@mcp.tool()
async def analyze_key(score_id: str) -> Dict[str, Any]:
    """Analyze key with circuit breaker protection"""
    
    if not rate_limiter.can_proceed():
        return {"error": "Rate limit exceeded"}
    
    cb = circuit_breakers["analyze"]
    if not cb.can_execute():
        return {"error": f"Circuit breaker open, retry in {cb.timeout}s"}
    
    try:
        if score_id not in scores:
            return {"error": "Score not found"}
        
        score = scores[score_id]
        k = score.analyze('key')
        cb.record_success()
        
        return {
            "score_id": score_id,
            "key": str(k),
            "confidence": k.correlationCoefficient
        }
    except Exception as e:
        cb.record_failure()
        logger.error(f"Analysis failed: {e}")
        return {"error": str(e)}

@mcp.tool()
async def list_scores() -> Dict[str, Any]:
    """List all loaded scores"""
    return {
        "scores": list(scores.keys()),
        "count": len(scores)
    }

@mcp.tool() 
async def cleanup_memory() -> Dict[str, Any]:
    """Force memory cleanup"""
    before = psutil.Process().memory_info().rss / 1024 / 1024
    
    # Clear unused scores
    scores.clear()
    
    # Force garbage collection
    gc.collect()
    gc.collect()  # Second pass for cyclic references
    
    after = psutil.Process().memory_info().rss / 1024 / 1024
    
    return {
        "memory_before_mb": before,
        "memory_after_mb": after,
        "freed_mb": before - after
    }

# Main entry point
if __name__ == "__main__":
    logger.info("Starting basic resilient server...")
    mcp.run()
