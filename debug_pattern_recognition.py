#!/usr/bin/env python3
"""Debug script for pattern recognition issue"""

from music21 import stream, note, interval
import json

# Create test melody: C4, D4, E4, F4, G4, A4
melody_notes = []
pitches = ['C4', 'D4', 'E4', 'F4', 'G4', 'A4']
for pitch in pitches:
    n = note.Note(pitch)
    n.quarterLength = 1
    melody_notes.append(n)

# Calculate intervals manually
intervals = []
for i in range(len(melody_notes) - 1):
    try:
        intv = interval.Interval(noteStart=melody_notes[i], noteEnd=melody_notes[i + 1])
        intervals.append(intv.semitones)
        print(f"{melody_notes[i].nameWithOctave} -> {melody_notes[i+1].nameWithOctave}: {intv.semitones} semitones")
    except Exception as e:
        print(f"Error calculating interval: {e}")
        intervals.append(0)

print(f"\nIntervals: {intervals}")
print(f"Length of intervals: {len(intervals)}")

# Test the pattern finding logic
min_length = 2
max_pattern_length = min(len(intervals) // 2 + 1, len(intervals))
print(f"\nmax_pattern_length: {max_pattern_length}")

sequences_found = []

for pattern_length in range(min_length, max_pattern_length):
    print(f"\n--- Testing pattern_length: {pattern_length} ---")
    
    for start in range(len(intervals) - pattern_length * 2 + 1):
        pattern = intervals[start : start + pattern_length]
        print(f"  Start: {start}, Pattern: {pattern}")
        
        # Look for repetitions
        for next_start in range(start + pattern_length, len(intervals) - pattern_length + 1):
            candidate = intervals[next_start : next_start + pattern_length]
            print(f"    Checking position {next_start}: {candidate} == {pattern}? {candidate == pattern}")
            
            if candidate == pattern:
                print(f"    MATCH FOUND!")
                sequences_found.append({
                    "pattern": pattern,
                    "start": start,
                    "next_start": next_start,
                    "pattern_length": pattern_length
                })

print(f"\n\nTotal sequences found: {len(sequences_found)}")
for seq in sequences_found:
    print(f"Pattern {seq['pattern']} at positions {seq['start']} and {seq['next_start']}")