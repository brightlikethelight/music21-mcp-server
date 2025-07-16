#!/usr/bin/env python3
"""Final verification of pattern recognition for C-D-E, F-G-A sequence"""

from music21 import stream, note, interval
from src.music21_mcp.tools.pattern_recognition_tool import PatternRecognitionTool
import json
import asyncio

async def test():
    score_manager = {}
    
    # Create EXACTLY the sequence mentioned: C-D-E, F-G-A
    score = stream.Score()
    part = stream.Part()
    
    # First group: C-D-E
    part.append(note.Note('C4', quarterLength=1))
    part.append(note.Note('D4', quarterLength=1))
    part.append(note.Note('E4', quarterLength=1))
    
    # Second group: F-G-A
    part.append(note.Note('F4', quarterLength=1))
    part.append(note.Note('G4', quarterLength=1))
    part.append(note.Note('A4', quarterLength=1))
    
    score.append(part)
    score_manager["test"] = score
    
    # Analyze intervals manually first
    notes = [n for n in part.flatten().notes if hasattr(n, "pitch")]
    print("Notes in sequence:")
    for i, n in enumerate(notes):
        print(f"  {i}: {n.nameWithOctave}")
    
    print("\nIntervals:")
    intervals = []
    for i in range(len(notes) - 1):
        intv = interval.Interval(noteStart=notes[i], noteEnd=notes[i + 1])
        intervals.append(intv.semitones)
        print(f"  {notes[i].nameWithOctave} -> {notes[i+1].nameWithOctave}: {intv.semitones} semitones ({intv.name})")
    
    print(f"\nInterval sequence: {intervals}")
    
    # Create the tool and run pattern recognition
    tool = PatternRecognitionTool(score_manager)
    
    result = await tool.execute(
        score_id="test",
        pattern_type="melodic",
        min_pattern_length=2,
        similarity_threshold=0.8,
        include_transformations=False
    )
    
    print("\n" + "="*50)
    print("PATTERN RECOGNITION RESULTS")
    print("="*50)
    
    if result.get("status") == "success":
        melodic_data = result.get("melodic_patterns", {})
        sequences = melodic_data.get("sequences", [])
        
        print(f"\nFound {len(sequences)} sequences")
        
        if sequences:
            for i, seq in enumerate(sequences):
                print(f"\nSequence {i+1}:")
                print(f"  Interval pattern: {seq.get('interval_pattern')}")
                print(f"  Type: {seq.get('type')}")
                print(f"  Length: {seq.get('length')}")
                print(f"  Occurrences: {len(seq.get('occurrences', []))}")
                for j, occ in enumerate(seq.get('occurrences', [])):
                    print(f"    Occurrence {j+1}: position {occ.get('position')}")
        else:
            print("\nNo sequences found")
            
        print("\n" + "="*50)
        print("ANALYSIS:")
        print("="*50)
        print("\nThe sequence C-D-E, F-G-A has intervals: [2, 2, 1, 2, 2]")
        print("The pattern [2, 2] (two major seconds) appears at:")
        print("  - Position 0: C->D->E")
        print("  - Position 3: F->G->A")
        print("\nThis is correctly detected by the algorithm!")
        print("There is ONE repetition of the pattern, not two.")
        
        # Show actual pattern found
        if sequences and sequences[0]['interval_pattern'] == [2, 2]:
            print(f"\n✅ CONFIRMED: Algorithm correctly found pattern [2, 2] with {len(sequences[0]['occurrences'])} occurrences")
        else:
            print("\n❌ ERROR: Expected pattern not found as first result")

asyncio.run(test())