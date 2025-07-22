"""
Pytest configuration and fixtures for music21-mcp-server tests
"""

import asyncio
import sys
from pathlib import Path

import pytest

# Add src directory to Python path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

# Import what we actually have
from music21_mcp.tools import (
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


@pytest.fixture
def clean_score_storage():
    """Provide clean score storage for each test"""
    return {}


@pytest.fixture
def sample_bach_score():
    """Bach chorale from music21 corpus for testing"""
    from music21 import corpus

    try:
        return corpus.parse("bach/bwv66.6")
    except:
        # Fallback if corpus not available
        from music21 import key, note, stream

        s = stream.Stream()
        s.append(key.Key("C"))
        s.append(note.Note("C4", quarterLength=1))
        s.append(note.Note("D4", quarterLength=1))
        s.append(note.Note("E4", quarterLength=1))
        s.append(note.Note("F4", quarterLength=1))
        return s


@pytest.fixture
def all_tool_classes():
    """All available tool classes for testing"""
    return [
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
    ]


@pytest.fixture
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def populated_score_storage(clean_score_storage, sample_bach_score):
    """Score storage with sample Bach chorale loaded"""
    storage = clean_score_storage.copy()
    storage["bach_test"] = sample_bach_score
    return storage
