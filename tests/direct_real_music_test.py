#!/usr/bin/env python3
"""
Direct testing with real music - bypassing server complexity
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from music21 import corpus, converter, stream, note, chord, key, meter, tempo, interval
from music21_mcp.core.theory_analyzer import TheoryAnalyzer
from music21_mcp.core.harmonic_analyzer import HarmonicAnalyzer
from music21_mcp.core.melodic_analyzer import MelodicAnalyzer
from music21_mcp.core.rhythm_analyzer import RhythmAnalyzer
import asyncio
import time


class DirectMusicTester:
    """Direct testing of analyzers with real music"""
    
    def __init__(self):
        self.theory_analyzer = TheoryAnalyzer()
        self.harmonic_analyzer = HarmonicAnalyzer()
        self.melodic_analyzer = MelodicAnalyzer()
        self.rhythm_analyzer = RhythmAnalyzer()
        self.results = []
    
    async def test_bach_invention(self):
        """Test Bach Invention No. 1 in C Major"""
        print("\n🎼 Bach Invention No. 1 in C Major (BWV 772)")
        print("-" * 50)
        
        try:
            # Load Bach invention
            bach = corpus.parse('bach/bwv772')
            
            # Test 1: Key Detection
            print("\n1. Key Detection:")
            start = time.time()
            key_result = await self.theory_analyzer.detect_key(bach)
            duration = time.time() - start
            
            print(f"   Detected: {key_result.tonic} {key_result.mode}")
            print(f"   Confidence: {key_result.confidence:.2f}")
            print(f"   Time: {duration:.2f}s")
            
            # Expected: C major
            is_correct = str(key_result.tonic) == "C" and key_result.mode == "major"
            print(f"   {'✅' if is_correct else '❌'} Expected C major")
            self.results.append(("Bach key detection", is_correct))
            
            # Test 2: Harmonic Analysis
            print("\n2. Harmonic Analysis:")
            start = time.time()
            harmony = await self.harmonic_analyzer.analyze_harmony(bach)
            duration = time.time() - start
            
            print(f"   Total chords: {len(harmony.chords)}")
            print(f"   Unique chords: {len(harmony.chord_histogram)}")
            print(f"   Time: {duration:.2f}s")
            
            if harmony.roman_numerals[:5]:
                print(f"   First 5 chords: {' → '.join(harmony.roman_numerals[:5])}")
            
            # Bach should have clear tonic/dominant relationships
            has_tonic_dominant = 'I' in harmony.roman_numerals and 'V' in harmony.roman_numerals
            print(f"   {'✅' if has_tonic_dominant else '❌'} Contains I-V progression")
            self.results.append(("Bach harmony", has_tonic_dominant))
            
            # Test 3: Contrapuntal Features
            print("\n3. Contrapuntal Features:")
            if len(bach.parts) >= 2:
                # Check voice independence
                part1_notes = list(bach.parts[0].flatten().notes)
                part2_notes = list(bach.parts[1].flatten().notes)
                
                # Check for contrary motion
                contrary_count = 0
                parallel_count = 0
                
                for i in range(min(len(part1_notes), len(part2_notes)) - 1):
                    if i + 1 < len(part1_notes) and i + 1 < len(part2_notes):
                        interval1 = part1_notes[i+1].pitch.midi - part1_notes[i].pitch.midi
                        interval2 = part2_notes[i+1].pitch.midi - part2_notes[i].pitch.midi
                        
                        if interval1 * interval2 < 0:  # Opposite direction
                            contrary_count += 1
                        elif interval1 == interval2 and interval1 != 0:
                            parallel_count += 1
                
                total_motions = contrary_count + parallel_count
                if total_motions > 0:
                    contrary_ratio = contrary_count / total_motions
                    print(f"   Contrary motion: {contrary_ratio:.2%}")
                    print(f"   {'✅' if contrary_ratio > 0.3 else '❌'} Good voice independence")
                    self.results.append(("Bach counterpoint", contrary_ratio > 0.3))
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    async def test_mozart_sonata(self):
        """Test Mozart Sonata K. 331 - Theme and Variations"""
        print("\n🎼 Mozart Piano Sonata K. 331, 1st Movement")
        print("-" * 50)
        
        try:
            # Load Mozart sonata
            mozart = corpus.parse('mozart/k331')
            # Get first movement
            if mozart.parts:
                first_movement = mozart.parts[0].measures(1, 32)  # First theme
            else:
                first_movement = mozart
            
            # Test 1: Key Detection
            print("\n1. Key Detection:")
            key_result = await self.theory_analyzer.detect_key(first_movement)
            print(f"   Detected: {key_result.tonic} {key_result.mode}")
            
            # Expected: A major
            is_correct = str(key_result.tonic) == "A" and key_result.mode == "major"
            print(f"   {'✅' if is_correct else '❌'} Expected A major")
            self.results.append(("Mozart key detection", is_correct))
            
            # Test 2: Melodic Analysis
            print("\n2. Melodic Analysis:")
            melody_result = await self.melodic_analyzer.analyze_melodic_contour(first_movement)
            print(f"   Contour type: {melody_result.overall_contour}")
            print(f"   Arch points: {len(melody_result.arch_points)}")
            print(f"   Smoothness: {melody_result.smoothness_score:.2f}")
            
            # Mozart should have clear melodic structure
            has_structure = len(melody_result.arch_points) > 2
            print(f"   {'✅' if has_structure else '❌'} Clear melodic structure")
            self.results.append(("Mozart melody", has_structure))
            
            # Test 3: Phrase Structure
            print("\n3. Phrase Structure:")
            # Simple phrase detection based on rests and cadences
            measures = list(first_movement.getElementsByClass('Measure'))
            phrase_endings = []
            
            for i, m in enumerate(measures):
                # Check for phrase endings (rests, long notes, cadences)
                if any(isinstance(el, note.Rest) for el in m.elements):
                    phrase_endings.append(i)
                elif any(el.duration.quarterLength >= 2 for el in m.notes):
                    phrase_endings.append(i)
            
            print(f"   Potential phrase endings at measures: {phrase_endings[:5]}")
            # Classical style should have regular phrases
            has_phrases = len(phrase_endings) >= 2
            print(f"   {'✅' if has_phrases else '❌'} Regular phrase structure")
            self.results.append(("Mozart phrases", has_phrases))
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    async def test_chopin_prelude(self):
        """Test Chopin Prelude Op. 28 No. 4 in E minor"""
        print("\n🎼 Chopin Prelude Op. 28 No. 4 in E minor")
        print("-" * 50)
        
        try:
            # Create a simplified version of Chopin's prelude
            chopin = stream.Score()
            part = stream.Part()
            
            # E minor key signature
            part.append(key.KeySignature(1))  # 1 sharp = G major/E minor
            part.append(tempo.MetronomeMark(number=60))
            part.append(meter.TimeSignature('2/2'))
            
            # Characteristic descending chromatic line
            chromatic_line = ['E4', 'D#4', 'D4', 'C#4', 'C4', 'B3']
            for p in chromatic_line:
                n = note.Note(p, quarterLength=1)
                part.append(n)
            
            # Add some harmony
            part.append(chord.Chord(['E3', 'G3', 'B3'], quarterLength=4))  # i chord
            
            chopin.append(part)
            
            # Test 1: Chromatic Detection
            print("\n1. Chromatic Content:")
            chromatic_result = await self.theory_analyzer.analyze_chromaticism(chopin)
            chromatic_ratio = chromatic_result.get('chromatic_note_ratio', 0)
            print(f"   Chromatic ratio: {chromatic_ratio:.2%}")
            
            # Should have significant chromaticism
            is_chromatic = chromatic_ratio > 0.3
            print(f"   {'✅' if is_chromatic else '❌'} Significant chromaticism")
            self.results.append(("Chopin chromaticism", is_chromatic))
            
            # Test 2: Key Detection with chromaticism
            print("\n2. Key Detection (with chromaticism):")
            key_result = await self.theory_analyzer.detect_key(chopin)
            print(f"   Detected: {key_result.tonic} {key_result.mode}")
            
            # Should still detect E minor despite chromaticism
            is_e_minor = (str(key_result.tonic) == "E" and key_result.mode == "minor") or \
                        (str(key_result.tonic) == "G" and key_result.mode == "major")
            print(f"   {'✅' if is_e_minor else '❌'} E minor or relative major")
            self.results.append(("Chopin key with chromaticism", is_e_minor))
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    async def test_jazz_standards(self):
        """Test Jazz Standards - Complex harmony"""
        print("\n🎼 Jazz Standard: 'All The Things You Are' Changes")
        print("-" * 50)
        
        try:
            # Create "All The Things You Are" chord changes (first 8 bars)
            jazz = stream.Score()
            part = stream.Part()
            
            # Key centers: Ab major -> C major -> Eb major
            changes = [
                ('F3', 'A3', 'C4', 'E4'),      # Fm7
                ('Bb2', 'D3', 'F3', 'Ab3'),     # Bbm7
                ('Eb3', 'G3', 'Bb3', 'Db4'),    # Eb7
                ('Ab2', 'C3', 'Eb3', 'G3'),     # Abmaj7
                ('Db3', 'F3', 'Ab3', 'C4'),     # Dbmaj7
                ('G3', 'B3', 'D4', 'F4'),       # G7
                ('C3', 'E3', 'G3', 'B3'),       # Cmaj7
                ('C3', 'E3', 'G3', 'B3'),       # Cmaj7
            ]
            
            for pitches in changes:
                ch = chord.Chord(pitches, quarterLength=2)
                part.append(ch)
            
            jazz.append(part)
            
            # Test 1: Jazz Harmony Detection
            print("\n1. Jazz Harmony Analysis:")
            harmony = await self.harmonic_analyzer.analyze_harmony(jazz)
            
            # Count extended chords
            extended_count = 0
            for ch in harmony.chords:
                if any(q in str(ch) for q in ['7', '9', '11', '13']):
                    extended_count += 1
            
            print(f"   Extended chords: {extended_count}/{len(harmony.chords)}")
            has_extensions = extended_count > len(harmony.chords) * 0.5
            print(f"   {'✅' if has_extensions else '❌'} Majority extended chords")
            self.results.append(("Jazz extensions", has_extensions))
            
            # Test 2: ii-V-I Detection
            print("\n2. ii-V-I Progression Detection:")
            # Check for ii-V-I patterns in the progression
            progressions = []
            for i in range(len(harmony.roman_numerals) - 2):
                prog = harmony.roman_numerals[i:i+3]
                if len(prog) == 3:
                    progressions.append(' '.join(prog))
            
            has_ii_v_i = any('ii' in p and 'V' in p for p in progressions)
            print(f"   {'✅' if has_ii_v_i else '❌'} Contains ii-V progression")
            self.results.append(("Jazz ii-V-I", has_ii_v_i))
            
            # Test 3: Modulation Detection
            print("\n3. Key Centers:")
            # The piece modulates through several keys
            key_result = await self.theory_analyzer.detect_key(jazz)
            print(f"   Primary key: {key_result.tonic} {key_result.mode}")
            
            # Should detect multiple key areas
            if hasattr(key_result, 'alternative_keys'):
                print(f"   Alternative keys: {len(key_result.alternative_keys)}")
                has_modulation = len(key_result.alternative_keys) > 1
            else:
                has_modulation = key_result.confidence < 0.7  # Low confidence suggests modulation
            
            print(f"   {'✅' if has_modulation else '❌'} Multiple key centers detected")
            self.results.append(("Jazz modulation", has_modulation))
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    async def test_performance(self):
        """Test performance with different score sizes"""
        print("\n⚡ Performance Testing")
        print("-" * 50)
        
        sizes = [
            ("Small (16 bars)", 16),
            ("Medium (64 bars)", 64),
            ("Large (256 bars)", 256)
        ]
        
        for name, num_bars in sizes:
            # Create test score
            score = stream.Score()
            part = stream.Part()
            
            for i in range(num_bars):
                m = stream.Measure(number=i+1)
                # 4 quarter notes per bar
                for beat in range(4):
                    m.append(note.Note('C4', quarterLength=1))
                part.append(m)
            
            score.append(part)
            
            # Time analysis
            start = time.time()
            
            # Run multiple analyses
            key_result = await self.theory_analyzer.detect_key(score)
            harmony = await self.harmonic_analyzer.analyze_harmony(score)
            rhythm = await self.rhythm_analyzer.analyze_rhythm(score)
            
            duration = time.time() - start
            
            print(f"\n{name}:")
            print(f"   Total analysis time: {duration:.2f}s")
            print(f"   Key: {key_result.tonic} {key_result.mode}")
            print(f"   Chords analyzed: {len(harmony.chords)}")
            print(f"   Tempo: {rhythm.tempo_bpm} BPM")
            
            # Performance criteria
            expected_time = 5 if num_bars < 100 else 30
            passed = duration < expected_time
            print(f"   {'✅' if passed else '❌'} Under {expected_time}s threshold")
            self.results.append((f"Performance {name}", passed))
    
    def print_summary(self):
        """Print test summary"""
        print("\n" + "="*60)
        print("📊 REAL MUSIC VALIDATION SUMMARY")
        print("="*60)
        
        passed = sum(1 for _, p in self.results if p)
        total = len(self.results)
        
        for test_name, passed_test in self.results:
            print(f"{'✅' if passed_test else '❌'} {test_name}")
        
        print(f"\nOVERALL: {passed}/{total} tests passed ({passed/total*100:.0f}%)")
        
        if passed == total:
            print("\n🎉 All tests passed! Excellent accuracy.")
        elif passed >= total * 0.8:
            print("\n✅ Good accuracy - most tests passed.")
        elif passed >= total * 0.6:
            print("\n⚠️ Moderate accuracy - some issues to address.")
        else:
            print("\n🚨 Poor accuracy - significant issues found.")


async def main():
    """Run all direct tests"""
    print("🎵 Direct Real Music Testing")
    print("="*60)
    
    tester = DirectMusicTester()
    
    # Run all tests
    await tester.test_bach_invention()
    await tester.test_mozart_sonata()
    await tester.test_chopin_prelude()
    await tester.test_jazz_standards()
    await tester.test_performance()
    
    # Summary
    tester.print_summary()


if __name__ == "__main__":
    asyncio.run(main())