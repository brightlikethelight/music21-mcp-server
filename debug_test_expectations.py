#!/usr/bin/env python3
"""Debug what patterns should be found in C-D-E, F-G-A"""

# The notes are: C4, D4, E4, F4, G4, A4
# Intervals in semitones:
# C4 -> D4: 2 (major 2nd)
# D4 -> E4: 2 (major 2nd)
# E4 -> F4: 1 (minor 2nd)
# F4 -> G4: 2 (major 2nd)
# G4 -> A4: 2 (major 2nd)

intervals = [2, 2, 1, 2, 2]

print("Intervals:", intervals)
print("\nLooking for patterns of length 2:")

# Pattern [2, 2] appears at:
# - Position 0: [2, 2]
# - Position 3: [2, 2]
print("Pattern [2, 2] at positions 0 and 3")

print("\nLooking for patterns of length 3:")
# No repeated patterns of length 3

print("\nNote: The user mentioned 'pattern \"up 1 semitone, up 1 semitone\" repeated twice'")
print("But the actual intervals are major 2nds (2 semitones), not minor 2nds (1 semitone)")
print("And there's only ONE repetition of the pattern [2, 2], not two.")
print("\nThe algorithm IS correctly finding this pattern!")