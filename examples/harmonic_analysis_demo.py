#!/usr/bin/env python
"""
Demonstration of Phase 2 Harmonic Analysis features
"""
import asyncio
from music21 import stream, note, chord, key, meter, tempo

# Add src to path
import sys
sys.path.insert(0, 'src')

from music21_mcp.server import (
    score_manager, import_score, functional_harmony_analysis,
    voice_leading_analysis, jazz_harmony_analysis,
    detect_harmonic_sequences, modulation_analysis,
    comprehensive_analysis
)


async def create_classical_example():
    """Create a classical progression with various harmonic features"""
    s = stream.Score()
    s.metadata.title = "Classical Harmonic Example"
    
    # Create four-part texture
    soprano = stream.Part()
    alto = stream.Part()
    tenor = stream.Part()
    bass = stream.Part()
    
    soprano.partName = "Soprano"
    alto.partName = "Alto"
    tenor.partName = "Tenor"
    bass.partName = "Bass"
    
    # Add key and time signature
    for part in [soprano, alto, tenor, bass]:
        part.append(key.KeySignature(0))  # C major
        part.append(meter.TimeSignature('4/4'))
        part.append(tempo.MetronomeMark(number=72))
    
    # Measure 1-2: Tonic prolongation (I - V6/4 - I)
    soprano.append(note.Note('G4', quarterLength=2))
    soprano.append(note.Note('G4', quarterLength=1))
    soprano.append(note.Note('G4', quarterLength=1))
    
    alto.append(note.Note('E4', quarterLength=2))
    alto.append(note.Note('D4', quarterLength=1))
    alto.append(note.Note('E4', quarterLength=1))
    
    tenor.append(note.Note('C4', quarterLength=2))
    tenor.append(note.Note('B3', quarterLength=1))
    tenor.append(note.Note('C4', quarterLength=1))
    
    bass.append(note.Note('C3', quarterLength=2))
    bass.append(note.Note('G2', quarterLength=1))
    bass.append(note.Note('C3', quarterLength=1))
    
    # Measure 3-4: Pre-dominant to dominant (ii6 - V7)
    soprano.append(note.Note('A4', quarterLength=2))
    soprano.append(note.Note('B4', quarterLength=2))
    
    alto.append(note.Note('F4', quarterLength=2))
    alto.append(note.Note('F4', quarterLength=2))
    
    tenor.append(note.Note('D4', quarterLength=2))
    tenor.append(note.Note('D4', quarterLength=2))
    
    bass.append(note.Note('F3', quarterLength=2))
    bass.append(note.Note('G3', quarterLength=2))
    
    # Measure 5-6: Deceptive resolution (V - vi) then authentic (V - I)
    soprano.append(note.Note('B4', quarterLength=1))
    soprano.append(note.Note('C5', quarterLength=1))
    soprano.append(note.Note('B4', quarterLength=1))
    soprano.append(note.Note('C5', quarterLength=1))
    
    alto.append(note.Note('F4', quarterLength=1))
    alto.append(note.Note('E4', quarterLength=1))
    alto.append(note.Note('D4', quarterLength=1))
    alto.append(note.Note('E4', quarterLength=1))
    
    tenor.append(note.Note('D4', quarterLength=1))
    tenor.append(note.Note('C4', quarterLength=1))
    tenor.append(note.Note('G3', quarterLength=1))
    tenor.append(note.Note('G3', quarterLength=1))
    
    bass.append(note.Note('G3', quarterLength=1))
    bass.append(note.Note('A3', quarterLength=1))
    bass.append(note.Note('G3', quarterLength=1))
    bass.append(note.Note('C3', quarterLength=1))
    
    # Add parts to score
    s.insert(0, soprano)
    s.insert(0, alto)
    s.insert(0, tenor)
    s.insert(0, bass)
    
    # Import to server
    score_manager.add_score("classical_example", s, {})
    print("Created classical example score")
    return "classical_example"


async def create_jazz_example():
    """Create a jazz progression with extended harmonies"""
    s = stream.Score()
    s.metadata.title = "Jazz Harmonic Example"
    
    p = stream.Part()
    p.append(key.KeySignature(-2))  # Bb major
    p.append(meter.TimeSignature('4/4'))
    p.append(tempo.MetronomeMark('Medium Swing', number=120))
    
    # Create a ii-V-I with extensions and alterations
    chords_data = [
        # Cm7 (ii7)
        (['C3', 'E-3', 'G3', 'B-3'], 4),
        # F7alt (V7 with alterations)
        (['F3', 'A3', 'C4', 'E-4', 'G#4'], 4),
        # BbMaj7 (IMaj7)
        (['B-2', 'D3', 'F3', 'A3'], 4),
        # G7 (VI7 - secondary dominant)
        (['G3', 'B3', 'D4', 'F4'], 4),
        # Cm7 (ii7 again)
        (['C3', 'E-3', 'G3', 'B-3'], 4),
        # B7 (tritone sub for F7)
        (['B2', 'D#3', 'F#3', 'A3'], 4),
        # BbMaj7 (IMaj7)
        (['B-2', 'D3', 'F3', 'A3'], 8),
    ]
    
    for pitches, duration in chords_data:
        ch = chord.Chord(pitches)
        ch.quarterLength = duration
        p.append(ch)
    
    s.append(p)
    
    # Import to server
    score_manager.add_score("jazz_example", s, {})
    print("Created jazz example score")
    return "jazz_example"


async def create_modulation_example():
    """Create a score with clear modulation"""
    s = stream.Score()
    s.metadata.title = "Modulation Example"
    
    p = stream.Part()
    p.append(meter.TimeSignature('4/4'))
    
    # Start in C major
    p.append(key.KeySignature(0))
    
    # Establish C major (4 measures)
    c_major_chords = [
        chord.Chord(['C4', 'E4', 'G4']),    # I
        chord.Chord(['G3', 'B3', 'D4']),    # V
        chord.Chord(['A3', 'C4', 'E4']),    # vi
        chord.Chord(['F3', 'A3', 'C4']),    # IV
        chord.Chord(['G3', 'B3', 'D4']),    # V
        chord.Chord(['C4', 'E4', 'G4']),    # I
    ]
    
    for ch in c_major_chords:
        ch.quarterLength = 2
        p.append(ch)
    
    # Pivot chord (D major = V/V in C, V in G)
    pivot = chord.Chord(['D4', 'F#4', 'A4'])
    pivot.quarterLength = 4
    p.append(pivot)
    
    # Establish G major (4 measures)
    # Change key signature
    p.append(key.KeySignature(1))
    
    g_major_chords = [
        chord.Chord(['G4', 'B4', 'D5']),    # I (in G)
        chord.Chord(['D4', 'F#4', 'A4']),   # V
        chord.Chord(['E4', 'G4', 'B4']),    # vi
        chord.Chord(['C4', 'E4', 'G4']),    # IV
        chord.Chord(['D4', 'F#4', 'A4']),   # V
        chord.Chord(['G4', 'B4', 'D5']),    # I
    ]
    
    for ch in g_major_chords:
        ch.quarterLength = 2
        p.append(ch)
    
    s.append(p)
    
    # Import to server
    score_manager.add_score("modulation_example", s, {})
    print("Created modulation example score")
    return "modulation_example"


async def demonstrate_harmonic_analysis():
    """Run all harmonic analyses on example scores"""
    print("\n=== HARMONIC ANALYSIS DEMONSTRATION ===\n")
    
    # Create example scores
    classical_id = await create_classical_example()
    jazz_id = await create_jazz_example()
    modulation_id = await create_modulation_example()
    
    print("\n--- Classical Example Analysis ---")
    
    # Functional harmony
    print("\n1. Functional Harmony Analysis:")
    functional_result = await functional_harmony_analysis(classical_id)
    if functional_result['status'] == 'success':
        print(f"   Roman numerals: {functional_result['roman_numerals']}")
        print(f"   Phrase model: {functional_result['phrase_model']}")
        print(f"   Tonal strength: {functional_result['tonal_strength']:.2f}")
        if functional_result['deceptive_resolutions']:
            print(f"   Deceptive resolutions: {functional_result['deceptive_resolutions']}")
    
    # Voice leading
    print("\n2. Voice Leading Analysis:")
    voice_result = await voice_leading_analysis(classical_id, strict=True)
    if voice_result['status'] == 'success':
        print(f"   Smoothness score: {voice_result['smoothness_score']:.2f}")
        print(f"   Independence score: {voice_result['independence_score']:.2f}")
        if voice_result['errors']:
            print(f"   Voice leading errors: {len(voice_result['errors'])}")
            for err in voice_result['errors'][:3]:
                print(f"      - {err['type']} in voices {err['voices']}")
    
    print("\n--- Jazz Example Analysis ---")
    
    # Jazz harmony
    print("\n3. Jazz Harmony Analysis:")
    jazz_result = await jazz_harmony_analysis(jazz_id)
    if jazz_result['status'] == 'success':
        print(f"   Chord symbols: {jazz_result['chord_symbols']}")
        print(f"   Extended chords: {len(jazz_result['extended_chords'])}")
        if jazz_result['substitutions']:
            print(f"   Substitutions detected:")
            for sub in jazz_result['substitutions']:
                print(f"      - {sub['type']}: {sub.get('original', '?')} → {sub.get('substitute', '?')}")
    
    print("\n--- Modulation Example Analysis ---")
    
    # Modulation analysis
    print("\n4. Modulation Analysis:")
    mod_result = await modulation_analysis(modulation_id, sensitivity=0.6)
    if mod_result['status'] == 'success':
        print(f"   Key areas: {len(mod_result['key_areas'])}")
        for area in mod_result['key_areas']:
            print(f"      - mm. {area['start_measure']}-{area['end_measure']}: {area['key']}")
        
        if mod_result['modulations']:
            print(f"   Modulations detected:")
            for mod in mod_result['modulations']:
                print(f"      - m. {mod['measure']}: {mod['from_key']} → {mod['to_key']} ({mod['type']})")
    
    # Harmonic sequences (on classical example)
    print("\n5. Harmonic Sequence Detection:")
    seq_result = await detect_harmonic_sequences(classical_id)
    if seq_result['status'] == 'success':
        print(f"   Total sequences found: {seq_result['total_sequences']}")
        for seq in seq_result['sequences']:
            print(f"      - Pattern: {seq['pattern']}")
            print(f"        Type: {seq['sequence_type']}, Direction: {seq['direction']}")
    
    print("\n--- Comprehensive Analysis (Classical) ---")
    
    # Run comprehensive analysis with new features
    comp_result = await comprehensive_analysis(classical_id, include_advanced=True)
    if comp_result['status'] == 'success':
        print(f"\n6. All analyses completed:")
        for analysis_type, result in comp_result['analyses'].items():
            if 'error' not in result:
                print(f"   ✓ {analysis_type}")
            else:
                print(f"   ✗ {analysis_type}: {result['error']}")
    
    print("\n=== DEMONSTRATION COMPLETE ===")


async def main():
    """Run the demonstration"""
    try:
        await demonstrate_harmonic_analysis()
    except Exception as e:
        print(f"Error in demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())