#!/usr/bin/env python3
"""Verify the pattern recognition fix works with the integration test pattern"""

from music21 import stream, note, metadata
from src.music21_mcp.tools.pattern_recognition_tool import PatternRecognitionTool
import json
import asyncio

async def test():
    # Create a simple score manager (just a dict)
    score_manager = {}
    
    # Recreate the exact test pattern from integration test
    score = stream.Score()
    score.metadata = metadata.Metadata()
    score.metadata.title = "Pattern Test Score"
    part = stream.Part()
    
    # Create a simple melodic sequence
    pattern = [
        note.Note('C4', quarterLength=0.5),
        note.Note('D4', quarterLength=0.5),
        note.Note('E4', quarterLength=0.5),
        note.Note('F4', quarterLength=0.5),
    ]
    
    # Repeat pattern at different pitch levels (sequence)
    for transposition in [0, 2, 4]:  # C, D, E starting notes
        for n in pattern:
            new_note = note.Note()
            new_note.pitch = n.pitch.transpose(transposition)
            new_note.duration = n.duration
            part.append(new_note)
    
    score.append(part)
    score_manager["pattern_test"] = score
    
    # Create the tool
    tool = PatternRecognitionTool(score_manager)
    
    # Test with the corrected parameters
    print("Testing pattern recognition with integration test pattern...")
    result = await tool.execute(
        score_id="pattern_test",
        pattern_type="both",
        min_pattern_length=2
    )
    
    print("\nResult status:", result.get("status"))
    
    if result.get("status") == "success":
        # Test the corrected extraction logic
        melodic_data = result.get("melodic_patterns", {})
        rhythmic_data = result.get("rhythmic_patterns", {})
        
        # Extract sequences and motifs from melodic patterns
        melodic = melodic_data.get("sequences", []) + melodic_data.get("motifs", [])
        rhythmic = rhythmic_data.get("rhythmic_motifs", [])
        
        print(f"\nFound {len(melodic)} melodic patterns")
        print(f"Found {len(rhythmic)} rhythmic patterns")
        
        if melodic:
            print("\nMelodic patterns found:")
            for i, pattern in enumerate(melodic[:3]):  # Show first 3
                print(f"\nPattern {i+1}:")
                if 'interval_pattern' in pattern:
                    print(f"  Interval pattern: {pattern.get('interval_pattern')}")
                if 'type' in pattern:
                    print(f"  Type: {pattern.get('type')}")
                if 'occurrences' in pattern:
                    print(f"  Occurrences: {len(pattern.get('occurrences', []))}")
        
        # Verify we found patterns
        if len(melodic) > 0:
            print("\n✅ SUCCESS: Patterns detected correctly!")
        else:
            print("\n❌ FAIL: No patterns found")
    else:
        print(f"\nError: {result.get('message', 'Unknown error')}")

asyncio.run(test())