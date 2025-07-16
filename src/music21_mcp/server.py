#!/usr/bin/env python3
"""
Music21 MCP Server - Simple Working Implementation
Stripped of over-engineered architecture, focuses on working functionality
"""
import asyncio
import gc
import logging
import psutil
import time
from datetime import datetime
from typing import Dict, Any, Optional

# Try to import MCP
try:
    from mcp.server.fastmcp import FastMCP
    HAS_MCP = True
except ImportError:
    HAS_MCP = False
    class FastMCP:
        def __init__(self, *args, **kwargs):
            raise ImportError("MCP package is not installed. Please install with: pip install mcp")

# Import working tools
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Simple score storage - this is what the tools expect
scores: Dict[str, Any] = {}

# Create FastMCP server
mcp = FastMCP("Music21 MCP Server")

# Simple rate limiter
class SimpleRateLimiter:
    def __init__(self, rate: float, burst: int):
        self.rate = rate
        self.burst = burst
        self.tokens = burst
        self.last_update = time.time()
    
    def acquire(self) -> bool:
        now = time.time()
        elapsed = now - self.last_update
        self.last_update = now
        
        self.tokens = min(self.burst, self.tokens + elapsed * self.rate)
        
        if self.tokens >= 1:
            self.tokens -= 1
            return True
        return False

# Global rate limiter
rate_limiter = SimpleRateLimiter(rate=100/60, burst=10)

def rate_limit_check():
    """Check rate limit and return error if exceeded"""
    if not rate_limiter.acquire():
        return {"error": "Rate limit exceeded, please try again later"}
    return None

# Register tools with FastMCP directly
@mcp.tool(name="import_score")
async def import_score(score_id: str, source: str, source_type: str = "corpus"):
    """Import a score from various sources"""
    rate_check = rate_limit_check()
    if rate_check:
        return rate_check
    
    tool = ImportScoreTool(scores)
    return await tool.execute(score_id=score_id, source=source, source_type=source_type)

@mcp.tool(name="list_scores")
async def list_scores():
    """List all available scores"""
    rate_check = rate_limit_check()
    if rate_check:
        return rate_check
    
    tool = ListScoresTool(scores)
    return await tool.execute()

@mcp.tool(name="key_analysis")
async def key_analysis(score_id: str):
    """Analyze the key signature of a score"""
    rate_check = rate_limit_check()
    if rate_check:
        return rate_check
    
    tool = KeyAnalysisTool(scores)
    return await tool.execute(score_id=score_id)

@mcp.tool(name="chord_analysis")
async def chord_analysis(score_id: str):
    """Analyze chord progressions in a score"""
    rate_check = rate_limit_check()
    if rate_check:
        return rate_check
    
    tool = ChordAnalysisTool(scores)
    return await tool.execute(score_id=score_id)

@mcp.tool(name="score_info")
async def score_info(score_id: str):
    """Get detailed information about a score"""
    rate_check = rate_limit_check()
    if rate_check:
        return rate_check
    
    tool = ScoreInfoTool(scores)
    return await tool.execute(score_id=score_id)

@mcp.tool(name="export_score")
async def export_score(score_id: str, format: str = "musicxml"):
    """Export a score to various formats"""
    rate_check = rate_limit_check()
    if rate_check:
        return rate_check
    
    tool = ExportScoreTool(scores)
    return await tool.execute(score_id=score_id, format=format)

@mcp.tool(name="delete_score")
async def delete_score(score_id: str):
    """Delete a score from storage"""
    rate_check = rate_limit_check()
    if rate_check:
        return rate_check
    
    tool = DeleteScoreTool(scores)
    return await tool.execute(score_id=score_id)

@mcp.tool(name="harmony_analysis")
async def harmony_analysis(score_id: str):
    """Analyze harmonic progressions using Roman numeral notation"""
    rate_check = rate_limit_check()
    if rate_check:
        return rate_check
    
    tool = HarmonyAnalysisTool(scores)
    return await tool.execute(score_id=score_id)

@mcp.tool(name="voice_leading_analysis")
async def voice_leading_analysis(score_id: str):
    """Analyze voice leading patterns in a score"""
    rate_check = rate_limit_check()
    if rate_check:
        return rate_check
    
    tool = VoiceLeadingAnalysisTool(scores)
    return await tool.execute(score_id=score_id)

@mcp.tool(name="pattern_recognition")
async def pattern_recognition(score_id: str):
    """Recognize melodic and rhythmic patterns in a score"""
    rate_check = rate_limit_check()
    if rate_check:
        return rate_check
    
    tool = PatternRecognitionTool(scores)
    return await tool.execute(score_id=score_id)

@mcp.tool(name="harmonization")
async def harmonization(score_id: str, style: str = "basic"):
    """Generate harmonization for a melody"""
    rate_check = rate_limit_check()
    if rate_check:
        return rate_check
    
    tool = HarmonizationTool(scores)
    return await tool.execute(score_id=score_id, style=style)

@mcp.tool(name="counterpoint")
async def counterpoint(score_id: str, species: str = "first"):
    """Generate counterpoint for a given melody"""
    rate_check = rate_limit_check()
    if rate_check:
        return rate_check
    
    tool = CounterpointGeneratorTool(scores)
    return await tool.execute(score_id=score_id, species=species)

@mcp.tool(name="style_imitation")
async def style_imitation(score_id: str, target_style: str):
    """Imitate a specific musical style"""
    rate_check = rate_limit_check()
    if rate_check:
        return rate_check
    
    tool = StyleImitationTool(scores)
    return await tool.execute(score_id=score_id, target_style=target_style)

@mcp.tool(name="health_check")
async def health_check():
    """Check server health and resource usage"""
    process = psutil.Process()
    memory = process.memory_info()
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "uptime_seconds": (
            datetime.now() - datetime.fromtimestamp(process.create_time())
        ).total_seconds(),
        "memory": {
            "used_mb": memory.rss / 1024 / 1024,
            "percent": process.memory_percent(),
        },
        "scores": {
            "count": len(scores),
            "ids": list(scores.keys())
        },
        "architecture": "simple_fastmcp",
        "version": "minimal_working"
    }

@mcp.tool(name="cleanup_memory")
async def cleanup_memory():
    """Force memory cleanup and garbage collection"""
    try:
        before = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Force garbage collection
        for _ in range(3):
            gc.collect()
        
        after = psutil.Process().memory_info().rss / 1024 / 1024
        
        return {
            "status": "success",
            "memory_before_mb": round(before, 2),
            "memory_after_mb": round(after, 2),
            "freed_mb": round(before - after, 2),
            "scores_count": len(scores),
        }
    except Exception as e:
        logger.error(f"Cleanup failed: {str(e)}")
        return {"error": str(e)}

# Add some basic resources for MCP compliance
@mcp.resource("music21://scores/list")
async def get_scores_list():
    """Get list of all available scores"""
    return {
        "scores": [
            {
                "id": score_id,
                "title": getattr(score.metadata, "title", "Untitled") if hasattr(score, "metadata") else "Untitled"
            }
            for score_id, score in scores.items()
        ]
    }

@mcp.resource("music21://scores/{score_id}")
async def get_score_metadata(score_id: str):
    """Get metadata for a specific score"""
    if score_id not in scores:
        return {"error": "Score not found"}
    
    score = scores[score_id]
    if hasattr(score, "metadata"):
        return {
            "id": score_id,
            "title": getattr(score.metadata, "title", "Untitled"),
            "composer": getattr(score.metadata, "composer", "Unknown"),
            "parts": len(score.parts) if hasattr(score, "parts") else 0,
        }
    return {"id": score_id, "title": "Untitled"}

def main():
    """Main entry point"""
    logger.info("🚀 Starting Music21 MCP Server (Simple Working Implementation)")
    logger.info("✅ Architecture: Minimal FastMCP")
    logger.info("✅ Over-engineered abstractions: REMOVED")
    logger.info("✅ Focus: Working functionality")
    
    try:
        # Run the server
        mcp.run()
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {str(e)}")
        raise

if __name__ == "__main__":
    main()