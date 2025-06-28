"""
Pytest configuration and shared fixtures for music21 MCP server tests
"""
import pytest
import asyncio
from pathlib import Path
from music21 import stream, note, chord, key, meter, tempo
import tempfile
import os

# Test data directory
TEST_DATA_DIR = Path(__file__).parent / "test_data"


@pytest.fixture
def event_loop():
    """Create an instance of the default event loop for each test case."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def simple_score():
    """Create a simple test score in C major"""
    s = stream.Score()
    p = stream.Part()
    
    # Add time signature and key
    p.append(meter.TimeSignature('4/4'))
    p.append(key.KeySignature(0))  # C major
    p.append(tempo.MetronomeMark(number=120))
    
    # Add a simple melody
    notes = ['C4', 'D4', 'E4', 'F4', 'G4', 'A4', 'B4', 'C5']
    for pitch in notes:
        n = note.Note(pitch)
        n.quarterLength = 1
        p.append(n)
    
    s.append(p)
    return s


@pytest.fixture
def chord_progression_score():
    """Create a score with chord progression"""
    s = stream.Score()
    p = stream.Part()
    
    # Add time signature and key
    p.append(meter.TimeSignature('4/4'))
    p.append(key.KeySignature(0))  # C major
    
    # I-IV-V-I progression
    chords_data = [
        ['C4', 'E4', 'G4'],  # I
        ['F4', 'A4', 'C5'],  # IV
        ['G4', 'B4', 'D5'],  # V
        ['C4', 'E4', 'G4'],  # I
    ]
    
    for chord_pitches in chords_data:
        c = chord.Chord(chord_pitches)
        c.quarterLength = 4
        p.append(c)
    
    s.append(p)
    return s


@pytest.fixture
def complex_rhythm_score():
    """Create a score with complex rhythms"""
    s = stream.Score()
    p = stream.Part()
    
    # Add time signature
    p.append(meter.TimeSignature('6/8'))
    
    # Add varied rhythms
    rhythms = [1.5, 0.5, 1, 0.75, 0.25, 2, 0.5, 0.5]
    
    for i, dur in enumerate(rhythms):
        n = note.Note('C4')
        n.quarterLength = dur
        p.append(n)
    
    s.append(p)
    return s


@pytest.fixture
def temp_midi_file(simple_score):
    """Create a temporary MIDI file"""
    with tempfile.NamedTemporaryFile(suffix='.mid', delete=False) as f:
        simple_score.write('midi', fp=f.name)
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def temp_musicxml_file(simple_score):
    """Create a temporary MusicXML file"""
    with tempfile.NamedTemporaryFile(suffix='.xml', delete=False) as f:
        simple_score.write('musicxml', fp=f.name)
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def mock_score_manager():
    """Mock score manager for testing"""
    from src.music21_mcp.server import ScoreManager
    return ScoreManager(max_scores=10)