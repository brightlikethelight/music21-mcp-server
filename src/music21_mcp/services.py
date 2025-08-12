#!/usr/bin/env python3
"""
Music Analysis Core Service
Pure music21 analysis service - protocol independent

This core service contains all the valuable music analysis functionality
without any dependencies on MCP, HTTP, or other protocols.
It survives protocol changes and breaking updates.
"""

from typing import Any

from .observability import get_logger, get_metrics, monitor_performance
from .resource_manager import ResourceManager

# Import tools but isolate from protocol concerns
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

logger = get_logger("music21_mcp.services")


class MusicAnalysisService:
    """
    Core music analysis service - completely protocol independent

    This service contains all music21 functionality isolated from:
    - MCP protocol concerns
    - HTTP API specifics
    - Authentication/authorization
    - Networking/transport layers

    It provides pure music analysis value that survives protocol changes.
    """

    def __init__(self, max_memory_mb: int = 512, max_scores: int = 100):
        """Initialize the core music analysis service with resource management"""
        # Initialize resource manager with memory limits and automatic cleanup
        self.resource_manager = ResourceManager(
            max_memory_mb=max_memory_mb,
            max_scores=max_scores,
            score_ttl_seconds=3600  # 1 hour TTL
        )

        # Use managed score storage instead of simple dictionary
        self.scores = self.resource_manager.scores

        # Initialize all analysis tools with managed storage
        self._init_tools()

        logger.info(
            "Music analysis core service initialized with resource management",
            max_memory_mb=max_memory_mb,
            max_scores=max_scores,
            available_tools=len(self.get_available_tools())
        )

    def _init_tools(self):
        """Initialize all music analysis tools"""
        # Core tools - always available
        self.import_tool = ImportScoreTool(self.scores)
        self.list_tool = ListScoresTool(self.scores)
        self.delete_tool = DeleteScoreTool(self.scores)
        self.export_tool = ExportScoreTool(self.scores)
        self.info_tool = ScoreInfoTool(self.scores)

        # Analysis tools
        self.key_tool = KeyAnalysisTool(self.scores)
        self.chord_tool = ChordAnalysisTool(self.scores)
        self.harmony_tool = HarmonyAnalysisTool(self.scores)
        self.voice_leading_tool = VoiceLeadingAnalysisTool(self.scores)
        self.pattern_tool = PatternRecognitionTool(self.scores)

        # Generation tools
        self.harmonization_tool = HarmonizationTool(self.scores)
        self.style_tool = StyleImitationTool(self.scores)
        self.counterpoint_tool = CounterpointGeneratorTool(self.scores)

    # === Core Operations ===

    @monitor_performance("music_analysis.import_score")
    async def import_score(
        self, score_id: str, source: str, source_type: str = "corpus"
    ) -> dict[str, Any]:
        """Import a score from various sources"""
        try:
            return await self.import_tool.execute(
                score_id=score_id, source=source, source_type=source_type
            )
        except Exception as e:
            logger.error(f"Import failed for {score_id}: {e}")
            raise

    @monitor_performance("music_analysis.list_scores")
    async def list_scores(self) -> dict[str, Any]:
        """List all imported scores"""
        return await self.list_tool.execute()

    async def get_score_info(self, score_id: str) -> dict[str, Any]:
        """Get detailed information about a score"""
        return await self.info_tool.execute(score_id=score_id)

    async def export_score(
        self, score_id: str, format: str = "musicxml"
    ) -> dict[str, Any]:
        """Export a score to various formats"""
        return await self.export_tool.execute(score_id=score_id, format=format)

    async def delete_score(self, score_id: str) -> dict[str, Any]:
        """Delete a score from storage"""
        return await self.delete_tool.execute(score_id=score_id)

    # === Analysis Operations ===

    @monitor_performance("music_analysis.analyze_key")
    async def analyze_key(self, score_id: str) -> dict[str, Any]:
        """Analyze the key signature of a score"""
        return await self.key_tool.execute(score_id=score_id)

    @monitor_performance("music_analysis.analyze_chords")
    async def analyze_chords(self, score_id: str) -> dict[str, Any]:
        """Analyze chord progressions in a score"""
        return await self.chord_tool.execute(score_id=score_id)

    async def analyze_harmony(
        self, score_id: str, analysis_type: str = "roman"
    ) -> dict[str, Any]:
        """Perform harmony analysis (roman numeral or functional)"""
        return await self.harmony_tool.execute(
            score_id=score_id, analysis_type=analysis_type
        )

    async def analyze_voice_leading(self, score_id: str) -> dict[str, Any]:
        """Analyze voice leading quality and issues"""
        return await self.voice_leading_tool.execute(score_id=score_id)

    async def recognize_patterns(
        self, score_id: str, pattern_type: str = "melodic"
    ) -> dict[str, Any]:
        """Recognize musical patterns (melodic, rhythmic, harmonic)"""
        return await self.pattern_tool.execute(
            score_id=score_id, pattern_type=pattern_type
        )

    # === Generation Operations ===

    async def harmonize_melody(
        self, score_id: str, style: str = "chorale"
    ) -> dict[str, Any]:
        """Generate harmonization for a melody"""
        return await self.harmonization_tool.execute(score_id=score_id, style=style)

    async def generate_counterpoint(
        self, score_id: str, species: int = 1
    ) -> dict[str, Any]:
        """Generate counterpoint melody"""
        return await self.counterpoint_tool.execute(score_id=score_id, species=species)

    async def imitate_style(self, score_id: str, target_style: str) -> dict[str, Any]:
        """Generate music imitating a specific style"""
        return await self.style_tool.execute(
            score_id=score_id, target_style=target_style
        )

    # === Utility Methods ===

    def get_available_tools(self) -> list[str]:
        """Get list of all available analysis tools"""
        return [
            "import_score",
            "list_scores",
            "get_score_info",
            "export_score",
            "delete_score",
            "analyze_key",
            "analyze_chords",
            "analyze_harmony",
            "analyze_voice_leading",
            "recognize_patterns",
            "harmonize_melody",
            "generate_counterpoint",
            "imitate_style",
        ]

    def get_score_count(self) -> int:
        """Get number of currently loaded scores"""
        return len(self.scores)

    def is_score_loaded(self, score_id: str) -> bool:
        """Check if a score is currently loaded"""
        return score_id in self.scores

    # === Resource Management Methods ===

    def get_resource_stats(self) -> dict[str, Any]:
        """Get comprehensive resource usage statistics"""
        return self.resource_manager.get_system_stats()

    def check_health(self) -> dict[str, Any]:
        """Perform health check and return system status"""
        return self.resource_manager.check_health()

    def cleanup_resources(self) -> dict[str, Any]:
        """Force cleanup of expired resources and return statistics"""
        return self.scores.cleanup()

    def get_memory_usage(self) -> dict[str, Any]:
        """Get current memory usage information"""
        stats = self.get_resource_stats()
        return {
            "storage_memory_mb": stats["storage"]["memory_usage_mb"],
            "storage_utilization_percent": stats["storage"]["memory_utilization_percent"],
            "system_memory_mb": stats["system"]["process_memory_mb"],
            "system_memory_percent": stats["system"]["process_memory_percent"],
            "scores_loaded": stats["storage"]["total_scores"],
            "max_scores": stats["storage"]["max_scores"],
        }

    def is_resource_healthy(self) -> bool:
        """Quick check if resources are in healthy state"""
        health = self.check_health()
        return health["status"] == "healthy"

    # === Observability Methods ===

    def get_performance_metrics(self) -> dict[str, Any]:
        """Get comprehensive performance metrics and statistics"""
        return get_metrics()

    def get_service_status(self) -> dict[str, Any]:
        """Get comprehensive service status including health and metrics"""
        health = self.check_health()
        metrics = self.get_performance_metrics()
        resource_stats = self.get_resource_stats()

        return {
            "service": {
                "name": "music21-mcp-server",
                "version": "1.0.0",
                "status": health["status"],
                "uptime_seconds": metrics.get("metadata", {}).get("uptime_seconds", 0),
            },
            "health": health,
            "resources": resource_stats,
            "performance": {
                "operation_counts": metrics.get("counters", {}),
                "operation_timings": metrics.get("timers", {}),
                "recent_operations": metrics.get("histograms", {}),
            },
            "timestamp": health["timestamp"],
        }

