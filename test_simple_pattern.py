#!/usr/bin/env python3
"""Test pattern recognition with simple ascending sequence"""

from music21 import stream, note, interval
from src.music21_mcp.tools.pattern_recognition_tool import PatternRecognitionTool
from src.music21_mcp.score_manager import ScoreManager
import json
import asyncio

async def test():
    # Create score manager
    score_manager = ScoreManager()
    
    # Create test melody: C4, D4, E4, F4, G4, A4
    score = stream.Score()
    part = stream.Part()
    
    pitches = ['C4', 'D4', 'E4', 'F4', 'G4', 'A4']
    for pitch in pitches:
        n = note.Note(pitch)
        n.quarterLength = 1
        part.append(n)
    
    score.append(part)
    
    # Store the score
    score_manager.store_score("test_ascending", score)
    
    # Create the tool
    tool = PatternRecognitionTool(score_manager)
    
    # Test with correct parameters
    print("Testing pattern recognition with ascending sequence...")
    result = await tool.execute(
        score_id="test_ascending",
        pattern_type="melodic",
        min_pattern_length=2,
        similarity_threshold=0.8,
        include_transformations=False
    )
    
    print("\nResult:")
    print(json.dumps(result, indent=2))
    
    # Extract melodic patterns if successful
    if result.get("status") == "success":
        melodic = result.get("melodic_patterns", {})
        sequences = melodic.get("sequences", [])
        motifs = melodic.get("motifs", [])
        
        print(f"\nFound {len(sequences)} sequences")
        print(f"Found {len(motifs)} motifs")
        
        for i, seq in enumerate(sequences):
            print(f"\nSequence {i+1}:")
            print(f"  Interval pattern: {seq.get('interval_pattern')}")
            print(f"  Type: {seq.get('type')}")
            print(f"  Occurrences: {len(seq.get('occurrences', []))}")

asyncio.run(test())