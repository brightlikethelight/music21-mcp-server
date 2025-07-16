#!/usr/bin/env python3
"""Test pattern recognition tool directly"""

from music21 import stream, note
from src.music21_mcp.tools.pattern_recognition_tool import PatternRecognitionTool
import json
import asyncio

# Create test melody: C4, D4, E4, F4, G4, A4
score = stream.Score()
part = stream.Part()
measure = stream.Measure()

pitches = ['C4', 'D4', 'E4', 'F4', 'G4', 'A4']
for pitch in pitches:
    n = note.Note(pitch)
    n.quarterLength = 1
    measure.append(n)

part.append(measure)
score.append(part)

# Create the tool and test it
tool = PatternRecognitionTool()

async def test():
    # Test the extraction step
    print("Testing melody extraction...")
    melodies = []
    for p in score.parts:
        melody = [n for n in p.flatten().notes if hasattr(n, "pitch")]
        print(f"Part has {len(melody)} notes:")
        for i, n in enumerate(melody):
            print(f"  {i}: {n.nameWithOctave}")
        if melody:
            melodies.append(melody)
    
    print(f"\nTotal melodies found: {len(melodies)}")
    
    if melodies:
        # Test _find_sequences directly
        print("\nTesting _find_sequences directly...")
        sequences = tool._find_sequences(melodies[0], min_length=2)
        print(f"Sequences found: {len(sequences)}")
        for seq in sequences:
            print(json.dumps(seq, indent=2))
    
    # Test the full tool
    print("\n\nTesting full tool...")
    result = await tool.run({
        "music_xml": score.write('musicxml'),
        "pattern_type": "melodic",
        "min_pattern_length": 2,
        "similarity_threshold": 0.8,
        "include_transformations": False
    })
    
    print("\nFull result:")
    print(json.dumps(result, indent=2))

asyncio.run(test())