#!/usr/bin/env python3
"""
Real Music Data Validation with Accuracy Testing
Tests with actual musical compositions and validates against known correct analyses
"""
import asyncio
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Any
import json

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from music21 import corpus, converter, stream, note, chord, key, roman, tempo
from music21_mcp.server import (
    score_manager, import_score, analyze_key, analyze_chord_progressions, 
    analyze_rhythm, analyze_melodic_contour, detect_melodic_motives,
    check_voice_leading, analyze_bach_chorale_style, analyze_fugue,
    jazz_harmony_analysis, detect_harmonic_sequences, export_score,
    functional_harmony_analysis, voice_leading_analysis, phrase_structure
)


class RealMusicAccuracyTester:
    """Comprehensive testing with real musical works and accuracy validation"""
    
    def __init__(self):
        self.test_results = []
        self.performance_metrics = []
        self.known_analyses = self._load_known_analyses()
    
    def _load_known_analyses(self) -> Dict[str, Any]:
        """Load known correct analyses for validation"""
        return {
            "bach/bwv846": {  # WTC Book 1, Prelude 1 in C Major
                "key": "C major",
                "time_signature": "4/4",
                "first_chord": "C major",
                "characteristics": ["arpeggiated", "consistent rhythm", "harmonic progression"]
            },
            "bach/bwv66.6": {  # Chorale "Christ unser Herr zum Jordan kam"
                "key": "F# minor",
                "cadences": ["authentic", "plagal", "half"],
                "voice_count": 4,
                "characteristics": ["four-part", "homophonic", "voice leading"]
            },
            "mozart/k331": {  # Piano Sonata No. 11, 1st movement
                "key": "A major",
                "form": "theme and variations",
                "time_signature": "6/8",
                "tempo_marking": "Andante grazioso"
            },
            "mozart/k545": {  # Piano Sonata No. 16 in C Major
                "key": "C major",
                "form": "sonata",
                "characteristics": ["alberti bass", "scalar passages", "clear phrases"]
            },
            "chopin/opus28no15": {  # Prelude "Raindrop"
                "key": "Db major",
                "middle_section_key": "C# minor",
                "characteristics": ["repeated note", "ABA form", "sustained pedal"]
            },
            "schubert/opus94no3": {  # Moment Musical No. 3
                "key": "F minor",
                "time_signature": "2/4",
                "characteristics": ["staccato", "dance-like", "contrasting dynamics"]
            }
        }
    
    async def test_bach_wtc_prelude(self):
        """Test Bach WTC Prelude in C Major - known for clear harmony"""
        print("\n🎼 Testing Bach WTC Book 1, Prelude No. 1 in C Major...")
        
        try:
            # Load from corpus
            bach_score = corpus.parse('bach/bwv846')
            
            # Import into our system
            start_time = time.time()
            result = await import_score(
                score_id="bach_wtc_c_major",
                content=bach_score,
                format="stream"
            )
            import_time = time.time() - start_time
            
            if result['status'] == 'success':
                print(f"   ✅ Import successful ({import_time:.2f}s)")
                self.performance_metrics.append(("Bach WTC import", import_time))
                
                # Test 1: Key Analysis (Should be C major)
                start_time = time.time()
                key_result = await analyze_key("bach_wtc_c_major")
                key_time = time.time() - start_time
                
                if key_result['status'] == 'success':
                    detected_key = key_result['key']
                    confidence = key_result['confidence']
                    expected_key = "C major"
                    
                    is_correct = detected_key == expected_key
                    print(f"   {'✅' if is_correct else '❌'} Key detection: {detected_key} (confidence: {confidence:.2f})")
                    print(f"      Expected: {expected_key}, Time: {key_time:.2f}s")
                    
                    self.test_results.append({
                        "test": "Bach WTC key detection",
                        "passed": is_correct,
                        "expected": expected_key,
                        "actual": detected_key,
                        "confidence": confidence
                    })
                    self.performance_metrics.append(("Bach WTC key analysis", key_time))
                
                # Test 2: Harmony Analysis (Should find clear progression)
                start_time = time.time()
                harmony_result = await analyze_chord_progressions("bach_wtc_c_major")
                harmony_time = time.time() - start_time
                
                if harmony_result['status'] == 'success':
                    total_chords = harmony_result['total_chords']
                    print(f"   ✅ Harmony analysis: {total_chords} chords detected ({harmony_time:.2f}s)")
                    
                    # Check for expected progression patterns
                    if harmony_result['chord_progressions']:
                        first_prog = harmony_result['chord_progressions'][0]
                        print(f"      First progression: {' → '.join(first_prog['progression'][:5])}")
                        
                        # Bach prelude should have many I-V-I patterns
                        has_tonic_dominant = any('I' in p and 'V' in p for p in first_prog['progression'])
                        self.test_results.append({
                            "test": "Bach WTC tonic-dominant pattern",
                            "passed": has_tonic_dominant,
                            "details": "Contains I-V progression"
                        })
                    
                    self.performance_metrics.append(("Bach WTC harmony analysis", harmony_time))
                
                # Test 3: Rhythm Analysis (Should be consistent 16th notes)
                start_time = time.time()
                rhythm_result = await analyze_rhythm("bach_wtc_c_major")
                rhythm_time = time.time() - start_time
                
                if rhythm_result['status'] == 'success':
                    consistency = rhythm_result.get('rhythmic_consistency', 0)
                    print(f"   ✅ Rhythm analysis: consistency score {consistency:.2f} ({rhythm_time:.2f}s)")
                    
                    # Bach prelude has very consistent rhythm
                    self.test_results.append({
                        "test": "Bach WTC rhythmic consistency",
                        "passed": consistency > 0.8,
                        "actual": consistency,
                        "expected": ">0.8"
                    })
                    self.performance_metrics.append(("Bach WTC rhythm analysis", rhythm_time))
                    
        except Exception as e:
            print(f"   ❌ Error testing Bach WTC: {e}")
            traceback.print_exc()
    
    async def test_mozart_sonata(self):
        """Test Mozart Sonata K. 545 - clear form and phrases"""
        print("\n🎼 Testing Mozart Piano Sonata K. 545 in C Major...")
        
        try:
            # Load from corpus
            mozart_score = corpus.parse('mozart/k545')
            
            # Get just the first movement
            first_movement = mozart_score.parts[0]
            
            # Import
            start_time = time.time()
            result = await import_score(
                score_id="mozart_k545",
                content=first_movement,
                format="stream"
            )
            import_time = time.time() - start_time
            
            if result['status'] == 'success':
                print(f"   ✅ Import successful ({import_time:.2f}s)")
                
                # Test 1: Key Analysis (Should be C major)
                key_result = await analyze_key("mozart_k545")
                if key_result['status'] == 'success':
                    detected_key = key_result['key']
                    is_correct = detected_key == "C major"
                    print(f"   {'✅' if is_correct else '❌'} Key detection: {detected_key}")
                    
                    self.test_results.append({
                        "test": "Mozart K545 key detection",
                        "passed": is_correct,
                        "expected": "C major",
                        "actual": detected_key
                    })
                
                # Test 2: Phrase Detection
                phrase_result = await phrase_structure("mozart_k545")
                if phrase_result['status'] == 'success':
                    num_phrases = phrase_result['total_phrases']
                    print(f"   ✅ Phrase detection: {num_phrases} phrases found")
                    
                    # Classical sonatas have clear phrase structure
                    self.test_results.append({
                        "test": "Mozart phrase structure",
                        "passed": num_phrases >= 4,  # Should have multiple clear phrases
                        "actual": num_phrases,
                        "expected": "≥4"
                    })
                
                # Test 3: Melodic Analysis (Should have clear melodic contour)
                contour_result = await analyze_melodic_contour("mozart_k545")
                if contour_result['status'] == 'success':
                    overall_contour = contour_result['overall_contour']
                    arch_points = len(contour_result.get('arch_points', []))
                    print(f"   ✅ Melodic contour: {overall_contour} with {arch_points} arch points")
                    
                    self.test_results.append({
                        "test": "Mozart melodic structure",
                        "passed": arch_points > 0,  # Should have clear melodic peaks
                        "details": f"{arch_points} arch points found"
                    })
                    
        except Exception as e:
            print(f"   ❌ Error testing Mozart: {e}")
    
    async def test_chopin_nocturne(self):
        """Test Chopin Nocturne - complex harmony and melody"""
        print("\n🎼 Testing Chopin Prelude Op. 28 No. 15 'Raindrop'...")
        
        try:
            # Create a simplified version for testing
            chopin = stream.Score()
            
            # A section in Db major
            part = stream.Part()
            part.append(key.KeySignature(5))  # 5 flats = Db major
            part.append(tempo.MetronomeMark(number=60, referent=note.Note(type='quarter')))
            
            # Characteristic repeated Ab notes (the "raindrop")
            for _ in range(8):
                part.append(note.Note('Ab4', quarterLength=0.5))
            
            # Add some Db major harmony
            part.append(chord.Chord(['Db4', 'F4', 'Ab4'], quarterLength=2))
            part.append(chord.Chord(['Gb4', 'Bb4', 'Db5'], quarterLength=2))
            
            chopin.append(part)
            
            # Import
            result = await import_score(
                score_id="chopin_raindrop",
                content=chopin,
                format="stream"
            )
            
            if result['status'] == 'success':
                print("   ✅ Import successful")
                
                # Test key detection
                key_result = await analyze_key("chopin_raindrop")
                if key_result['status'] == 'success':
                    detected_key = key_result['key']
                    # Should detect Db major or related key
                    is_db_related = "Db" in detected_key or "C#" in detected_key or "flat" in detected_key
                    print(f"   {'✅' if is_db_related else '❌'} Key detection: {detected_key}")
                    
                    self.test_results.append({
                        "test": "Chopin key detection",
                        "passed": is_db_related,
                        "expected": "Db major or related",
                        "actual": detected_key
                    })
                
        except Exception as e:
            print(f"   ❌ Error testing Chopin: {e}")
    
    async def test_jazz_standard(self):
        """Test Jazz Standard - complex chords and progressions"""
        print("\n🎼 Testing Jazz Standard Progression (ii-V-I in C)...")
        
        try:
            # Create a jazz standard progression
            jazz = stream.Score()
            part = stream.Part()
            
            # Classic ii-V-I in C with extensions
            jazz_chords = [
                chord.Chord(['D3', 'F3', 'A3', 'C4', 'E4']),     # Dm9
                chord.Chord(['G3', 'B3', 'D4', 'F4', 'A4']),     # G13
                chord.Chord(['C3', 'E3', 'G3', 'B3', 'D4'])      # Cmaj9
            ]
            
            for ch in jazz_chords:
                ch.quarterLength = 4
                part.append(ch)
            
            jazz.append(part)
            
            # Import
            result = await import_score(
                score_id="jazz_standard",
                content=jazz,
                format="stream"
            )
            
            if result['status'] == 'success':
                print("   ✅ Import successful")
                
                # Test jazz harmony analysis
                jazz_result = await jazz_harmony_analysis("jazz_standard")
                if jazz_result['status'] == 'success':
                    extended_chords = jazz_result.get('extended_chords', [])
                    print(f"   ✅ Jazz harmony: {len(extended_chords)} extended chords found")
                    
                    # Should detect the extended chords
                    found_extensions = any('9' in str(ch) or '13' in str(ch) for ch in extended_chords)
                    self.test_results.append({
                        "test": "Jazz extended chord detection",
                        "passed": found_extensions,
                        "details": f"Found {len(extended_chords)} extended chords"
                    })
                    
                    # Check for ii-V-I detection
                    if 'common_progressions' in jazz_result:
                        has_ii_v_i = any('ii' in prog and 'V' in prog and 'I' in prog 
                                       for prog in jazz_result['common_progressions'])
                        self.test_results.append({
                            "test": "Jazz ii-V-I detection",
                            "passed": has_ii_v_i,
                            "details": "Classic jazz progression"
                        })
                
        except Exception as e:
            print(f"   ❌ Error testing jazz: {e}")
    
    async def test_bach_chorale_voices(self):
        """Test Bach Chorale - voice leading and part writing"""
        print("\n🎼 Testing Bach Chorale BWV 66.6 - Voice Leading...")
        
        try:
            # Load actual Bach chorale
            chorale = corpus.parse('bach/bwv66.6')
            
            # Import
            result = await import_score(
                score_id="bach_chorale_66",
                content=chorale,
                format="stream"
            )
            
            if result['status'] == 'success':
                print("   ✅ Import successful")
                
                # Test 1: Voice leading analysis
                vl_result = await check_voice_leading("bach_chorale_66")
                if vl_result['status'] == 'success':
                    parallel_fifths = vl_result.get('parallel_fifths', 0)
                    parallel_octaves = vl_result.get('parallel_octaves', 0)
                    smoothness = vl_result.get('smoothness_score', 0)
                    
                    print(f"   ✅ Voice leading: {parallel_fifths} ||5ths, {parallel_octaves} ||8ves, smoothness: {smoothness:.2f}")
                    
                    # Bach chorales should have excellent voice leading
                    self.test_results.append({
                        "test": "Bach chorale voice leading",
                        "passed": parallel_fifths < 3 and parallel_octaves < 3 and smoothness > 0.7,
                        "details": f"||5ths: {parallel_fifths}, ||8ves: {parallel_octaves}, smooth: {smoothness:.2f}"
                    })
                
                # Test 2: Chorale style analysis
                style_result = await analyze_bach_chorale_style("bach_chorale_66")
                if style_result['status'] == 'success':
                    voice_count = len(style_result.get('voice_ranges', {}))
                    style_score = style_result.get('style_conformance_score', 0)
                    
                    print(f"   ✅ Chorale style: {voice_count} voices, conformance: {style_score:.2f}")
                    
                    # Should have 4 voices (SATB)
                    self.test_results.append({
                        "test": "Bach chorale SATB structure",
                        "passed": voice_count == 4,
                        "expected": 4,
                        "actual": voice_count
                    })
                
        except Exception as e:
            print(f"   ❌ Error testing Bach chorale: {e}")
    
    async def test_performance_benchmarks(self):
        """Test performance with various score sizes"""
        print("\n⚡ Performance Benchmark Tests...")
        
        test_cases = [
            ("Small piece (32 bars)", 32, 4),
            ("Medium piece (128 bars)", 128, 4),
            ("Large piece (256 bars)", 256, 8),
        ]
        
        for name, num_measures, num_parts in test_cases:
            try:
                # Create test score
                test_score = stream.Score()
                
                for p in range(num_parts):
                    part = stream.Part()
                    for m in range(num_measures):
                        measure = stream.Measure(number=m+1)
                        # Add 4 quarter notes per measure
                        for beat in range(4):
                            measure.append(note.Note('C4', quarterLength=1))
                        part.append(measure)
                    test_score.append(part)
                
                # Time the import
                start_time = time.time()
                result = await import_score(
                    score_id=f"perf_test_{num_measures}",
                    content=test_score,
                    format="stream"
                )
                import_time = time.time() - start_time
                
                if result['status'] == 'success':
                    # Time analysis
                    start_time = time.time()
                    await analyze_key(f"perf_test_{num_measures}")
                    analysis_time = time.time() - start_time
                    
                    total_time = import_time + analysis_time
                    print(f"   ✅ {name}: {total_time:.2f}s (import: {import_time:.2f}s, analysis: {analysis_time:.2f}s)")
                    
                    # Performance criteria
                    expected_time = 5 if num_measures < 100 else 30
                    self.test_results.append({
                        "test": f"Performance - {name}",
                        "passed": total_time < expected_time,
                        "actual_time": total_time,
                        "expected_time": expected_time
                    })
                    
                    self.performance_metrics.append((name, total_time))
                
            except Exception as e:
                print(f"   ❌ Error in performance test {name}: {e}")
    
    async def test_round_trip_accuracy(self):
        """Test that round-trip conversion preserves musical content"""
        print("\n🔄 Testing Round-Trip Conversion Accuracy...")
        
        try:
            # Create a complex test score with various elements
            original = stream.Score()
            part = stream.Part()
            
            # Add various musical elements
            part.append(key.KeySignature(2))  # D major
            part.append(meter.TimeSignature('6/8'))
            part.append(tempo.MetronomeMark(number=120))
            
            # Add notes with various articulations
            n1 = note.Note('D4', quarterLength=1)
            n1.articulations.append(articulations.Staccato())
            part.append(n1)
            
            # Add chord
            ch = chord.Chord(['F#4', 'A4', 'D5'], quarterLength=2)
            part.append(ch)
            
            # Add dynamic
            part.append(dynamics.Dynamic('mf'))
            
            # More notes
            for pitch in ['E4', 'F#4', 'G4', 'A4']:
                part.append(note.Note(pitch, quarterLength=0.5))
            
            original.append(part)
            
            # Import original
            await import_score("round_trip_test", content=original, format="stream")
            
            # Export as MusicXML
            export_result = await export_score("round_trip_test", format="musicxml")
            if export_result['status'] == 'success':
                xml_path = export_result['file_path']
                
                # Re-import the exported file
                reimport_result = await import_score(
                    "round_trip_reimport",
                    file_path=xml_path,
                    format="musicxml"
                )
                
                if reimport_result['status'] == 'success':
                    # Compare key signatures
                    key1 = await analyze_key("round_trip_test")
                    key2 = await analyze_key("round_trip_reimport")
                    
                    keys_match = key1.get('key') == key2.get('key')
                    print(f"   {'✅' if keys_match else '❌'} Key preservation: {key1.get('key')} → {key2.get('key')}")
                    
                    # Compare note counts
                    notes_match = (reimport_result.get('num_notes') == 
                                 export_result.get('metadata', {}).get('num_notes', 7))
                    print(f"   {'✅' if notes_match else '❌'} Note count preservation")
                    
                    self.test_results.append({
                        "test": "Round-trip conversion accuracy",
                        "passed": keys_match and notes_match,
                        "details": "Key and note preservation"
                    })
                    
        except Exception as e:
            print(f"   ❌ Error in round-trip test: {e}")
    
    def print_accuracy_report(self):
        """Print comprehensive accuracy report"""
        print("\n" + "="*60)
        print("📊 ACCURACY VALIDATION REPORT")
        print("="*60)
        
        # Group results by category
        categories = {
            "Key Detection": [],
            "Harmony Analysis": [],
            "Voice Leading": [],
            "Performance": [],
            "Other": []
        }
        
        for result in self.test_results:
            test_name = result['test']
            if 'key' in test_name.lower():
                categories["Key Detection"].append(result)
            elif 'harmony' in test_name.lower() or 'chord' in test_name.lower():
                categories["Harmony Analysis"].append(result)
            elif 'voice' in test_name.lower():
                categories["Voice Leading"].append(result)
            elif 'performance' in test_name.lower():
                categories["Performance"].append(result)
            else:
                categories["Other"].append(result)
        
        # Print results by category
        for category, results in categories.items():
            if results:
                print(f"\n{category}:")
                passed = sum(1 for r in results if r.get('passed', False))
                total = len(results)
                print(f"  Overall: {passed}/{total} passed ({passed/total*100:.0f}%)")
                
                for result in results:
                    status = "✅" if result.get('passed', False) else "❌"
                    print(f"  {status} {result['test']}")
                    if 'expected' in result and 'actual' in result:
                        print(f"     Expected: {result['expected']}, Actual: {result['actual']}")
                    if 'details' in result:
                        print(f"     {result['details']}")
        
        # Performance summary
        if self.performance_metrics:
            print("\n⚡ Performance Metrics:")
            for metric_name, time_taken in self.performance_metrics:
                print(f"  {metric_name}: {time_taken:.2f}s")
            
            avg_time = sum(t for _, t in self.performance_metrics) / len(self.performance_metrics)
            print(f"  Average operation time: {avg_time:.2f}s")
        
        # Overall summary
        total_passed = sum(1 for r in self.test_results if r.get('passed', False))
        total_tests = len(self.test_results)
        success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
        
        print("\n" + "="*60)
        print(f"OVERALL SUCCESS RATE: {success_rate:.0f}% ({total_passed}/{total_tests})")
        print("="*60)
        
        if success_rate >= 90:
            print("🎉 Excellent accuracy! System is performing well.")
        elif success_rate >= 70:
            print("⚠️ Good accuracy but some improvements needed.")
        else:
            print("🚨 Significant accuracy issues detected. Review failed tests.")


async def main():
    """Run comprehensive accuracy validation"""
    print("🎵 Real Music Data Accuracy Validation")
    print("="*60)
    
    tester = RealMusicAccuracyTester()
    
    # Run all tests
    await tester.test_bach_wtc_prelude()
    await tester.test_mozart_sonata()
    await tester.test_chopin_nocturne()
    await tester.test_jazz_standard()
    await tester.test_bach_chorale_voices()
    await tester.test_performance_benchmarks()
    await tester.test_round_trip_accuracy()
    
    # Print comprehensive report
    tester.print_accuracy_report()


if __name__ == "__main__":
    asyncio.run(main())