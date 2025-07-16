#!/usr/bin/env python3
"""Test pattern recognition with simple ascending sequence"""

from music21 import stream, note, interval
from src.music21_mcp.tools.pattern_recognition_tool import PatternRecognitionTool
import json
import asyncio

async def test():
    # Create a simple score manager (just a dict)
    score_manager = {}
    
    # Create test melody: C4, D4, E4, F4, G4, A4
    score = stream.Score()
    part = stream.Part()
    
    pitches = ['C4', 'D4', 'E4', 'F4', 'G4', 'A4']
    for pitch in pitches:
        n = note.Note(pitch)
        n.quarterLength = 1
        part.append(n)
    
    score.append(part)
    
    # Store the score in the dict
    score_manager["test_ascending"] = score
    
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
    
    print("\nResult status:", result.get("status"))
    
    # Extract melodic patterns if successful
    if result.get("status") == "success":
        melodic = result.get("melodic_patterns", {})
        sequences = melodic.get("sequences", [])
        motifs = melodic.get("motifs", [])
        
        print(f"\nFound {len(sequences)} sequences")
        print(f"Found {len(motifs)} motifs")
        
        if sequences:
            for i, seq in enumerate(sequences):
                print(f"\nSequence {i+1}:")
                print(f"  Interval pattern: {seq.get('interval_pattern')}")
                print(f"  Type: {seq.get('type')}")
                print(f"  Occurrences: {len(seq.get('occurrences', []))}")
        else:
            print("\nNo sequences found!")
            
            # Debug: Let's manually check what should be found
            notes = [n for n in part.flatten().notes if hasattr(n, "pitch")]
            print(f"\nDebug - Notes in part: {len(notes)}")
            for i, n in enumerate(notes):
                print(f"  {i}: {n.nameWithOctave}")
            
            # Calculate intervals
            intervals = []
            for i in range(len(notes) - 1):
                intv = interval.Interval(noteStart=notes[i], noteEnd=notes[i + 1])
                intervals.append(intv.semitones)
            print(f"\nDebug - Intervals: {intervals}")
    else:
        print(f"\nError: {result.get('message', 'Unknown error')}")
        if 'error' in result:
            print(f"Details: {result['error']}")

asyncio.run(test())