"""
Music21 MCP Server - Modular Architecture
Clean separation of concerns with modular tool implementations
"""
import logging
from typing import Any, Dict, Optional

from mcp.server.fastmcp import FastMCP

# Import modular tools
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
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastMCP server instance
mcp = FastMCP("Music21 Server")

# Shared score storage
score_manager: Dict[str, Any] = {}

# Initialize tools
import_tool = ImportScoreTool(score_manager)
list_tool = ListScoresTool(score_manager)
key_tool = KeyAnalysisTool(score_manager)
chord_tool = ChordAnalysisTool(score_manager)
info_tool = ScoreInfoTool(score_manager)
export_tool = ExportScoreTool(score_manager)
delete_tool = DeleteScoreTool(score_manager)
harmony_tool = HarmonyAnalysisTool(score_manager)
voice_leading_tool = VoiceLeadingAnalysisTool(score_manager)
pattern_tool = PatternRecognitionTool(score_manager)
harmonization_tool = HarmonizationTool(score_manager)
counterpoint_tool = CounterpointGeneratorTool(score_manager)
style_tool = StyleImitationTool(score_manager)


@mcp.tool()
async def import_score(
    score_id: str,
    source: str,
    source_type: Optional[str] = "auto"
) -> Dict[str, Any]:
    """
    Import a musical score from various sources.
    
    Args:
        score_id: Unique identifier for the score
        source: File path, corpus path (e.g., 'bach/bwv66.6'), or note sequence
        source_type: Type of source ('file', 'corpus', 'text', 'auto')
    
    Returns:
        Import status with basic metadata
    """
    return await import_tool.execute(score_id=score_id, source=source, source_type=source_type)


@mcp.tool()
async def list_scores() -> Dict[str, Any]:
    """
    List all imported scores with basic metadata.
    
    Returns:
        List of scores with their IDs and basic information
    """
    return await list_tool.execute()


@mcp.tool()
async def analyze_key(
    score_id: str,
    algorithm: str = "all"
) -> Dict[str, Any]:
    """
    Analyze the musical key of a score.
    
    Args:
        score_id: ID of the score to analyze
        algorithm: Algorithm to use ('all', 'krumhansl', 'aarden', 'temperley', 'bellman')
    
    Returns:
        Detected key with confidence score and alternatives
    """
    return await key_tool.execute(score_id=score_id, algorithm=algorithm)


@mcp.tool()
async def analyze_chords(
    score_id: str,
    include_roman_numerals: bool = True
) -> Dict[str, Any]:
    """
    Analyze chord progressions in a score.
    
    Args:
        score_id: ID of the score to analyze
        include_roman_numerals: Include Roman numeral analysis
    
    Returns:
        Chord progression with symbols and optional Roman numerals
    """
    return await chord_tool.execute(
        score_id=score_id,
        include_roman_numerals=include_roman_numerals
    )


@mcp.tool() 
async def get_score_info(score_id: str) -> Dict[str, Any]:
    """
    Get comprehensive information about a score.
    
    Args:
        score_id: ID of the score
    
    Returns:
        Detailed metadata including title, composer, duration, parts, etc.
    """
    return await info_tool.execute(score_id=score_id)


@mcp.tool()
async def export_score(
    score_id: str,
    format: str = "musicxml",
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Export a score to various formats.
    
    Args:
        score_id: ID of the score to export
        format: Output format (midi, musicxml, abc, lilypond, pdf)
        output_path: Optional custom output path
    
    Returns:
        Export status with file path
    """
    return await export_tool.execute(
        score_id=score_id,
        format=format,
        output_path=output_path
    )


@mcp.tool()
async def delete_score(score_id: str) -> Dict[str, Any]:
    """
    Delete a score from memory.
    
    Args:
        score_id: ID of score to delete, or '*' to delete all
    
    Returns:
        Deletion status
    """
    return await delete_tool.execute(score_id=score_id)


@mcp.tool()
async def analyze_harmony(
    score_id: str,
    include_roman_numerals: bool = True,
    include_cadences: bool = True,
    include_non_chord_tones: bool = True
) -> Dict[str, Any]:
    """
    Perform advanced harmonic analysis.
    
    Args:
        score_id: ID of the score to analyze
        include_roman_numerals: Include Roman numeral analysis
        include_cadences: Identify and analyze cadences
        include_non_chord_tones: Analyze non-chord tones
    
    Returns:
        Comprehensive harmonic analysis results
    """
    return await harmony_tool.execute(
        score_id=score_id,
        include_roman_numerals=include_roman_numerals,
        include_cadences=include_cadences,
        include_non_chord_tones=include_non_chord_tones
    )


@mcp.tool()
async def analyze_voice_leading(
    score_id: str,
    check_parallels: bool = True,
    check_voice_crossing: bool = True,
    check_spacing: bool = True,
    style_period: str = "common_practice"
) -> Dict[str, Any]:
    """
    Analyze voice leading and identify potential issues.
    
    Args:
        score_id: ID of the score to analyze
        check_parallels: Check for parallel fifths and octaves
        check_voice_crossing: Check for voice crossing
        check_spacing: Check voice spacing
        style_period: Style period for rules ('common_practice', 'renaissance', 'contemporary')
    
    Returns:
        Voice leading analysis with identified issues
    """
    return await voice_leading_tool.execute(
        score_id=score_id,
        check_parallels=check_parallels,
        check_voice_crossing=check_voice_crossing,
        check_spacing=check_spacing,
        style_period=style_period
    )


@mcp.tool()
async def find_patterns(
    score_id: str,
    pattern_type: str = "both",
    min_pattern_length: int = 3,
    similarity_threshold: float = 0.85,
    include_transformations: bool = True
) -> Dict[str, Any]:
    """
    Find melodic and rhythmic patterns in a score.
    
    Args:
        score_id: ID of the score to analyze
        pattern_type: Type of patterns ('melodic', 'rhythmic', 'both')
        min_pattern_length: Minimum pattern length
        similarity_threshold: Similarity threshold (0-1)
        include_transformations: Include inversions, retrogrades, etc.
    
    Returns:
        Found patterns with occurrences and analysis
    """
    return await pattern_tool.execute(
        score_id=score_id,
        pattern_type=pattern_type,
        min_pattern_length=min_pattern_length,
        similarity_threshold=similarity_threshold,
        include_transformations=include_transformations
    )


@mcp.tool()
async def harmonize_melody(
    score_id: str,
    style: str = "classical",
    constraints: Optional[list] = None,
    include_explanations: bool = True,
    voice_parts: int = 4
) -> Dict[str, Any]:
    """
    Generate intelligent harmonization for a melody.
    
    Args:
        score_id: ID of the melody to harmonize
        style: Harmonization style ('classical', 'jazz', 'pop', 'modal')
        constraints: List of constraints (e.g., ['diatonic_only'])
        include_explanations: Include explanations for choices
        voice_parts: Number of voices (2-4)
    
    Returns:
        Harmonized score with analysis and explanations
    """
    return await harmonization_tool.execute(
        score_id=score_id,
        style=style,
        constraints=constraints,
        include_explanations=include_explanations,
        voice_parts=voice_parts
    )


@mcp.tool()
async def generate_counterpoint(
    score_id: str,
    species: str = "first",
    voice_position: str = "above",
    rule_set: str = "strict",
    custom_rules: Optional[list] = None,
    mode: str = "major"
) -> Dict[str, Any]:
    """
    Generate counterpoint in various species.
    
    Args:
        score_id: ID of the cantus firmus
        species: Species type ('first', 'second', 'third', 'fourth', 'fifth')
        voice_position: Position relative to CF ('above', 'below')
        rule_set: Rule strictness ('strict', 'relaxed', 'custom')
        custom_rules: List of specific rules if rule_set is 'custom'
        mode: Modal context ('major', 'minor', 'dorian', etc.)
    
    Returns:
        Generated counterpoint with rule compliance report
    """
    return await counterpoint_tool.execute(
        score_id=score_id,
        species=species,
        voice_position=voice_position,
        rule_set=rule_set,
        custom_rules=custom_rules,
        mode=mode
    )


@mcp.tool()
async def imitate_style(
    style_source: Optional[str] = None,
    composer: Optional[str] = None,
    generation_length: int = 16,
    starting_note: Optional[str] = None,
    constraints: Optional[list] = None,
    complexity: str = "medium"
) -> Dict[str, Any]:
    """
    Generate music in a specific style.
    
    Args:
        style_source: Score ID to analyze for style
        composer: Pre-defined composer style ('bach', 'mozart', 'chopin', 'debussy')
        generation_length: Number of measures to generate
        starting_note: Starting pitch (e.g., 'C4')
        constraints: List of constraints (e.g., ['key:C', 'range:C3-C6'])
        complexity: Generation complexity ('simple', 'medium', 'complex')
    
    Returns:
        Generated music with style adherence analysis
    """
    return await style_tool.execute(
        style_source=style_source,
        composer=composer,
        generation_length=generation_length,
        starting_note=starting_note,
        constraints=constraints,
        complexity=complexity
    )


@mcp.tool()
async def analyze_style(
    score_id: str,
    detailed: bool = True
) -> Dict[str, Any]:
    """
    Analyze the style characteristics of a score.
    
    Args:
        score_id: ID of score to analyze
        detailed: Include detailed statistics
    
    Returns:
        Style characteristics and comparison to known styles
    """
    return await style_tool.analyze_style(
        score_id=score_id,
        detailed=detailed
    )


def main():
    """Run the MCP server"""
    import asyncio
    
    logger.info("Starting Music21 MCP Server (Modular Architecture)")
    logger.info("Available tools: import_score, list_scores, analyze_key, analyze_chords, get_score_info, export_score, delete_score")
    logger.info("Advanced tools: analyze_harmony, analyze_voice_leading, find_patterns")
    logger.info("Creative tools: harmonize_melody, generate_counterpoint, imitate_style, analyze_style")
    
    # Run the server
    asyncio.run(mcp.run())


if __name__ == "__main__":
    main()