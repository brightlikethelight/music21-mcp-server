#!/usr/bin/env python3
"""
Music Analysis Core Service
Pure music21 analysis service - protocol independent

This core service contains all the valuable music analysis functionality
without any dependencies on MCP, HTTP, or other protocols.
It survives protocol changes and breaking updates.
"""

from typing import Dict, Any, List, Optional
import logging

# Import tools but isolate from protocol concerns
from .tools import (
    ImportScoreTool,
    ListScoresTool,
    KeyAnalysisTool,
    ScoreInfoTool,
    HarmonyAnalysisTool,
    VoiceLeadingAnalysisTool,
    DeleteScoreTool,
    ExportScoreTool,
    ChordAnalysisTool,
    PatternRecognitionTool,
    HarmonizationTool,
    StyleImitationTool,
    CounterpointGeneratorTool,
)

logger = logging.getLogger(__name__)


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
    
    def __init__(self):
        """Initialize the core music analysis service"""
        # Core score storage - in-memory for simplicity
        self.scores = {}
        
        # Initialize all analysis tools with shared storage
        self._init_tools()
        
        logger.info("Music analysis core service initialized")
    
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
    
    async def import_score(self, score_id: str, source: str, source_type: str = "corpus") -> Dict[str, Any]:
        """Import a score from various sources"""
        try:
            return await self.import_tool.execute(score_id=score_id, source=source, source_type=source_type)
        except Exception as e:
            logger.error(f"Import failed for {score_id}: {e}")
            raise
    
    async def list_scores(self) -> Dict[str, Any]:
        """List all imported scores"""
        return await self.list_tool.execute()
    
    async def get_score_info(self, score_id: str) -> Dict[str, Any]:
        """Get detailed information about a score"""
        return await self.info_tool.execute(score_id=score_id)
    
    async def export_score(self, score_id: str, format: str = "musicxml") -> Dict[str, Any]:
        """Export a score to various formats"""
        return await self.export_tool.execute(score_id=score_id, format=format)
    
    async def delete_score(self, score_id: str) -> Dict[str, Any]:
        """Delete a score from storage"""
        return await self.delete_tool.execute(score_id=score_id)
    
    # === Analysis Operations ===
    
    async def analyze_key(self, score_id: str) -> Dict[str, Any]:
        """Analyze the key signature of a score"""
        return await self.key_tool.execute(score_id=score_id)
    
    async def analyze_chords(self, score_id: str) -> Dict[str, Any]:
        """Analyze chord progressions in a score"""
        return await self.chord_tool.execute(score_id=score_id)
    
    async def analyze_harmony(self, score_id: str, analysis_type: str = "roman") -> Dict[str, Any]:
        """Perform harmony analysis (roman numeral or functional)"""
        return await self.harmony_tool.execute(score_id=score_id, analysis_type=analysis_type)
    
    async def analyze_voice_leading(self, score_id: str) -> Dict[str, Any]:
        """Analyze voice leading quality and issues"""
        return await self.voice_leading_tool.execute(score_id=score_id)
    
    async def recognize_patterns(self, score_id: str, pattern_type: str = "melodic") -> Dict[str, Any]:
        """Recognize musical patterns (melodic, rhythmic, harmonic)"""
        return await self.pattern_tool.execute(score_id=score_id, pattern_type=pattern_type)
    
    # === Generation Operations ===
    
    async def harmonize_melody(self, score_id: str, style: str = "chorale") -> Dict[str, Any]:
        """Generate harmonization for a melody"""
        return await self.harmonization_tool.execute(score_id=score_id, style=style)
    
    async def generate_counterpoint(self, score_id: str, species: int = 1) -> Dict[str, Any]:
        """Generate counterpoint melody"""
        return await self.counterpoint_tool.execute(score_id=score_id, species=species)
    
    async def imitate_style(self, score_id: str, target_style: str) -> Dict[str, Any]:
        """Generate music imitating a specific style"""
        return await self.style_tool.execute(score_id=score_id, target_style=target_style)
    
    # === Utility Methods ===
    
    def get_available_tools(self) -> List[str]:
        """Get list of all available analysis tools"""
        return [
            "import_score", "list_scores", "get_score_info", "export_score", "delete_score",
            "analyze_key", "analyze_chords", "analyze_harmony", "analyze_voice_leading", 
            "recognize_patterns", "harmonize_melody", "generate_counterpoint", "imitate_style"
        ]
    
    def get_score_count(self) -> int:
        """Get number of currently loaded scores"""
        return len(self.scores)
    
    def is_score_loaded(self, score_id: str) -> bool:
        """Check if a score is currently loaded"""
        return score_id in self.scores