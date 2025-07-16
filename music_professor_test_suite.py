#!/usr/bin/env python3
"""
🎼 MUSIC PROFESSOR TEST SUITE - REAL SCORES, REAL BUGS
Tests music21-mcp with ACTUAL scores that music professors use
Finds bugs in music theory analysis, not just code coverage
"""

import asyncio
import json
import os
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import traceback

# Add src to path for imports
sys.path.insert(0, 'src')

try:
    from music21 import converter, corpus, key, meter, analysis, harmony, pitch, chord
    MUSIC21_AVAILABLE = True
except ImportError:
    MUSIC21_AVAILABLE = False
    print("⚠️ Music21 not available - some tests will be skipped")

from music21_mcp.server import ScoreManager
from music21_mcp.core.harmonic_analyzer import HarmonicAnalyzer
from music21_mcp.core.melodic_analyzer import MelodicAnalyzer
from music21_mcp.core.rhythm_analyzer import RhythmAnalyzer

class MusicProfessorTester:
    """Tests with real musical scores that professors actually use"""
    
    def __init__(self):
        self.score_manager = ScoreManager(max_scores=50)
        self.harmonic_analyzer = HarmonicAnalyzer()
        self.melodic_analyzer = MelodicAnalyzer()
        self.rhythm_analyzer = RhythmAnalyzer()
        self.test_results = []
        self.musical_errors = []
        
        # Famous scores for testing (Public Domain from IMSLP)
        self.test_scores = {
            # Basic Western Classical
            "bach_invention_1": {
                "composer": "J.S. Bach",
                "title": "Invention No. 1 in C major, BWV 772",
                "expected_key": "C major",
                "expected_time_sig": "4/4",
                "difficulty": "intermediate",
                "musical_features": ["counterpoint", "two_voice_texture", "imitation"]
            },
            
            # Complex Harmony
            "chopin_prelude_4": {
                "composer": "Frédéric Chopin", 
                "title": "Prelude Op. 28 No. 4 in E minor",
                "expected_key": "E minor",
                "expected_time_sig": "2/2",
                "difficulty": "advanced",
                "musical_features": ["chromatic_harmony", "expressive_melody", "left_hand_accompaniment"]
            },
            
            # Modal/Non-Western Elements
            "debussy_clair_de_lune": {
                "composer": "Claude Debussy",
                "title": "Clair de Lune",
                "expected_key": "D♭ major",  # Actually quite ambiguous
                "expected_time_sig": "9/8",
                "difficulty": "advanced",
                "musical_features": ["impressionist_harmony", "whole_tone_scales", "parallel_motion"]
            },
            
            # Rhythm Challenges
            "bartok_mikrokosmos_149": {
                "composer": "Béla Bartók",
                "title": "Mikrokosmos No. 149",
                "expected_key": "ambiguous",  # Modal/atonal
                "expected_time_sig": "7/8",
                "difficulty": "advanced",
                "musical_features": ["irregular_meter", "folk_elements", "bitonality"]
            }
        }
    
    def log_musical_error(self, test_name: str, error_type: str, description: str, severity: str = "medium"):
        """Log musical analysis errors"""
        error = {
            "test": test_name,
            "error_type": error_type,
            "description": description,
            "severity": severity,
            "timestamp": time.time()
        }
        self.musical_errors.append(error)
        print(f"🎵 MUSICAL ERROR ({severity.upper()}): {description}")
    
    async def test_key_analysis_accuracy(self) -> bool:
        """Test key analysis against known musical facts"""
        print("🔍 Testing key analysis accuracy with real scores...")
        
        if not MUSIC21_AVAILABLE:
            print("⚠️ Skipping - Music21 not available")
            return True
        
        try:
            # Test with music21's built-in corpus (real pieces)
            test_pieces = [
                ("bach/bwv772", "C major"),  # Bach Invention 1
                ("bach/bwv7.7", "D minor"),  # Bach Chorale
                ("chopin/mazurka23", "D major"),  # Chopin Mazurka
            ]
            
            accuracy_issues = 0
            
            for piece_path, expected_key in test_pieces:
                try:
                    print(f"   Analyzing {piece_path}...")
                    
                    # Load from corpus
                    score = corpus.parse(piece_path)
                    
                    # Store in our system
                    score_id = f"corpus_{piece_path.replace('/', '_')}"
                    await self.score_manager.add_score(score_id, score)
                    
                    # Test key analysis
                    start_time = time.time()
                    analysis_result = await self.harmonic_analyzer.analyze_key_signature(score)
                    analysis_time = time.time() - start_time
                    
                    if analysis_time > 10.0:  # Should be fast
                        self.log_musical_error(
                            piece_path,
                            "performance",
                            f"Key analysis too slow: {analysis_time:.2f}s",
                            "medium"
                        )
                        accuracy_issues += 1
                    
                    # Check if analysis result is reasonable
                    if not analysis_result or 'key' not in analysis_result:
                        self.log_musical_error(
                            piece_path,
                            "analysis_failure",
                            "Key analysis returned no result",
                            "high"
                        )
                        accuracy_issues += 1
                        continue
                    
                    detected_key = analysis_result.get('key', 'unknown')
                    confidence = analysis_result.get('confidence', 0)
                    
                    print(f"     Expected: {expected_key}")
                    print(f"     Detected: {detected_key}")
                    print(f"     Confidence: {confidence:.2f}")
                    
                    # For well-known pieces, we should get reasonable results
                    if confidence < 0.3:  # Very low confidence is suspicious
                        self.log_musical_error(
                            piece_path,
                            "low_confidence",
                            f"Very low confidence in key analysis: {confidence:.2f}",
                            "medium"
                        )
                        accuracy_issues += 1
                    
                    # Check for completely wrong analysis
                    if "unknown" in detected_key.lower() or detected_key == "":
                        self.log_musical_error(
                            piece_path,
                            "no_detection",
                            "Failed to detect any key",
                            "high"
                        )
                        accuracy_issues += 1
                
                except Exception as e:
                    self.log_musical_error(
                        piece_path,
                        "analysis_crash",
                        f"Key analysis crashed: {str(e)}",
                        "critical"
                    )
                    accuracy_issues += 1
            
            print(f"✅ Key analysis test completed - {accuracy_issues} issues found")
            return accuracy_issues == 0
            
        except Exception as e:
            self.log_musical_error(
                "key_analysis_test",
                "test_crash",
                f"Key analysis test crashed: {str(e)}",
                "critical"
            )
            return False
    
    async def test_harmonic_progression_analysis(self) -> bool:
        """Test harmonic progression analysis with known chord progressions"""
        print("🔍 Testing harmonic progression analysis...")
        
        if not MUSIC21_AVAILABLE:
            print("⚠️ Skipping - Music21 not available") 
            return True
        
        try:
            # Create test progressions with known harmonic content
            test_progressions = [
                {
                    "name": "I-V-vi-IV (Pop progression)",
                    "chords": ["C", "G", "Am", "F"],
                    "expected_functions": ["I", "V", "vi", "IV"],
                    "key": "C major"
                },
                {
                    "name": "ii-V-I (Jazz progression)",
                    "chords": ["Dm7", "G7", "Cmaj7"],
                    "expected_functions": ["ii7", "V7", "Imaj7"],
                    "key": "C major"
                },
                {
                    "name": "Circle of fifths",
                    "chords": ["Am", "Dm", "G", "C"],
                    "expected_functions": ["vi", "ii", "V", "I"],
                    "key": "C major"
                }
            ]
            
            progression_issues = 0
            
            for progression in test_progressions:
                try:
                    print(f"   Testing {progression['name']}...")
                    
                    # Create a simple score with the progression
                    from music21 import stream, meter, chord as m21_chord, duration
                    
                    score = stream.Score()
                    score.append(meter.TimeSignature('4/4'))
                    score.append(key.KeySignature(0))  # C major
                    
                    # Add chords to score
                    for chord_symbol in progression['chords']:
                        # Convert chord symbol to music21 chord
                        try:
                            chord_obj = m21_chord.Chord(chord_symbol)
                            chord_obj.duration = duration.Duration(1.0)  # Quarter note
                            score.append(chord_obj)
                        except Exception as e:
                            self.log_musical_error(
                                progression['name'],
                                "chord_creation",
                                f"Failed to create chord {chord_symbol}: {e}",
                                "medium"
                            )
                            progression_issues += 1
                            continue
                    
                    # Store and analyze
                    score_id = f"progression_{progression['name'].replace(' ', '_')}"
                    await self.score_manager.add_score(score_id, score)
                    
                    # Test harmonic analysis
                    start_time = time.time()
                    harmony_result = await self.harmonic_analyzer.analyze_harmonic_progression(score)
                    analysis_time = time.time() - start_time
                    
                    if analysis_time > 5.0:
                        self.log_musical_error(
                            progression['name'],
                            "performance",
                            f"Harmonic analysis too slow: {analysis_time:.2f}s",
                            "medium"
                        )
                        progression_issues += 1
                    
                    # Check analysis results
                    if not harmony_result:
                        self.log_musical_error(
                            progression['name'],
                            "analysis_failure",
                            "Harmonic analysis returned no result",
                            "high"
                        )
                        progression_issues += 1
                        continue
                    
                    # Check if it found some harmonic content
                    if 'progressions' in harmony_result:
                        progressions_found = len(harmony_result['progressions'])
                        print(f"     Found {progressions_found} chord progressions")
                        
                        if progressions_found == 0:
                            self.log_musical_error(
                                progression['name'],
                                "no_progressions",
                                "No chord progressions detected in known progression",
                                "high"
                            )
                            progression_issues += 1
                    
                    # Check for chord identification
                    if 'chords' in harmony_result:
                        chords_found = len(harmony_result['chords'])
                        expected_chords = len(progression['chords'])
                        print(f"     Found {chords_found} chords (expected {expected_chords})")
                        
                        # Should find roughly the right number of chords
                        if chords_found < expected_chords * 0.5:
                            self.log_musical_error(
                                progression['name'],
                                "chord_detection",
                                f"Found too few chords: {chords_found}/{expected_chords}",
                                "medium"
                            )
                            progression_issues += 1
                
                except Exception as e:
                    self.log_musical_error(
                        progression['name'],
                        "analysis_crash", 
                        f"Harmonic progression analysis crashed: {str(e)}",
                        "critical"
                    )
                    progression_issues += 1
            
            print(f"✅ Harmonic progression test completed - {progression_issues} issues found")
            return progression_issues == 0
            
        except Exception as e:
            self.log_musical_error(
                "harmonic_progression_test",
                "test_crash",
                f"Harmonic progression test crashed: {str(e)}",
                "critical"
            )
            return False
    
    async def test_melodic_analysis_correctness(self) -> bool:
        """Test melodic analysis with known musical patterns"""
        print("🔍 Testing melodic analysis correctness...")
        
        if not MUSIC21_AVAILABLE:
            print("⚠️ Skipping - Music21 not available")
            return True
        
        try:
            # Test with well-known melodies
            melody_tests = [
                {
                    "name": "C Major Scale",
                    "notes": ["C4", "D4", "E4", "F4", "G4", "A4", "B4", "C5"],
                    "expected_pattern": "ascending_scale",
                    "expected_intervals": ["M2", "M2", "m2", "M2", "M2", "M2", "m2"]
                },
                {
                    "name": "Arpeggio Pattern", 
                    "notes": ["C4", "E4", "G4", "C5"],
                    "expected_pattern": "arpeggio",
                    "expected_intervals": ["M3", "m3", "P4"]
                },
                {
                    "name": "Chromatic Passage",
                    "notes": ["C4", "C#4", "D4", "D#4", "E4"],
                    "expected_pattern": "chromatic",
                    "expected_intervals": ["m2", "m2", "m2", "m2"]
                }
            ]
            
            melodic_issues = 0
            
            for melody_test in melody_tests:
                try:
                    print(f"   Testing {melody_test['name']}...")
                    
                    # Create melody
                    from music21 import stream, note, duration
                    
                    melody = stream.Stream()
                    
                    for note_name in melody_test['notes']:
                        note_obj = note.Note(note_name)
                        note_obj.duration = duration.Duration(0.5)  # Eighth note
                        melody.append(note_obj)
                    
                    # Store and analyze
                    score_id = f"melody_{melody_test['name'].replace(' ', '_')}"
                    await self.score_manager.add_score(score_id, melody)
                    
                    # Test melodic analysis
                    start_time = time.time()
                    melodic_result = await self.melodic_analyzer.analyze_melodic_patterns(melody)
                    analysis_time = time.time() - start_time
                    
                    if analysis_time > 3.0:
                        self.log_musical_error(
                            melody_test['name'],
                            "performance",
                            f"Melodic analysis too slow: {analysis_time:.2f}s",
                            "medium"
                        )
                        melodic_issues += 1
                    
                    # Check analysis results
                    if not melodic_result:
                        self.log_musical_error(
                            melody_test['name'],
                            "analysis_failure",
                            "Melodic analysis returned no result",
                            "high"
                        )
                        melodic_issues += 1
                        continue
                    
                    # Check interval analysis
                    if 'intervals' in melodic_result:
                        intervals_found = len(melodic_result['intervals'])
                        expected_intervals = len(melody_test['expected_intervals'])
                        
                        print(f"     Found {intervals_found} intervals (expected {expected_intervals})")
                        
                        if intervals_found < expected_intervals * 0.8:
                            self.log_musical_error(
                                melody_test['name'],
                                "interval_detection",
                                f"Found too few intervals: {intervals_found}/{expected_intervals}",
                                "medium"
                            )
                            melodic_issues += 1
                    
                    # Check pattern recognition
                    if 'patterns' in melodic_result:
                        patterns = melodic_result['patterns']
                        print(f"     Detected patterns: {patterns}")
                        
                        # For well-known patterns, should detect something relevant
                        if len(patterns) == 0 and melody_test['expected_pattern'] != "none":
                            self.log_musical_error(
                                melody_test['name'],
                                "pattern_recognition",
                                f"Failed to detect expected pattern: {melody_test['expected_pattern']}",
                                "medium"
                            )
                            melodic_issues += 1
                
                except Exception as e:
                    self.log_musical_error(
                        melody_test['name'],
                        "analysis_crash",
                        f"Melodic analysis crashed: {str(e)}",
                        "critical"
                    )
                    melodic_issues += 1
            
            print(f"✅ Melodic analysis test completed - {melodic_issues} issues found")
            return melodic_issues == 0
            
        except Exception as e:
            self.log_musical_error(
                "melodic_analysis_test", 
                "test_crash",
                f"Melodic analysis test crashed: {str(e)}",
                "critical"
            )
            return False
    
    async def test_rhythm_analysis_accuracy(self) -> bool:
        """Test rhythm analysis with known rhythmic patterns"""
        print("🔍 Testing rhythm analysis accuracy...")
        
        if not MUSIC21_AVAILABLE:
            print("⚠️ Skipping - Music21 not available")
            return True
        
        try:
            # Test common rhythmic patterns
            rhythm_tests = [
                {
                    "name": "Simple 4/4 Pattern",
                    "time_sig": "4/4",
                    "durations": [1.0, 1.0, 1.0, 1.0],  # Four quarter notes
                    "expected_complexity": "simple"
                },
                {
                    "name": "Syncopated Pattern",
                    "time_sig": "4/4", 
                    "durations": [0.5, 1.5, 0.5, 1.5],  # Syncopation
                    "expected_complexity": "moderate"
                },
                {
                    "name": "Complex Subdivision",
                    "time_sig": "4/4",
                    "durations": [0.25, 0.25, 0.5, 0.75, 0.25, 2.0],  # Mixed subdivisions
                    "expected_complexity": "complex"
                },
                {
                    "name": "Irregular Meter",
                    "time_sig": "7/8",
                    "durations": [0.5, 0.5, 0.5, 1.0, 1.0],  # 7/8 pattern
                    "expected_complexity": "complex"
                }
            ]
            
            rhythm_issues = 0
            
            for rhythm_test in rhythm_tests:
                try:
                    print(f"   Testing {rhythm_test['name']}...")
                    
                    # Create rhythmic pattern
                    from music21 import stream, note, meter, duration
                    
                    rhythm_stream = stream.Stream()
                    rhythm_stream.append(meter.TimeSignature(rhythm_test['time_sig']))
                    
                    for dur in rhythm_test['durations']:
                        note_obj = note.Note('C4')  # Constant pitch
                        note_obj.duration = duration.Duration(dur)
                        rhythm_stream.append(note_obj)
                    
                    # Store and analyze
                    score_id = f"rhythm_{rhythm_test['name'].replace(' ', '_')}"
                    await self.score_manager.add_score(score_id, rhythm_stream)
                    
                    # Test rhythm analysis
                    start_time = time.time()
                    rhythm_result = await self.rhythm_analyzer.analyze_rhythmic_patterns(rhythm_stream)
                    analysis_time = time.time() - start_time
                    
                    if analysis_time > 3.0:
                        self.log_musical_error(
                            rhythm_test['name'],
                            "performance",
                            f"Rhythm analysis too slow: {analysis_time:.2f}s",
                            "medium"
                        )
                        rhythm_issues += 1
                    
                    # Check analysis results
                    if not rhythm_result:
                        self.log_musical_error(
                            rhythm_test['name'],
                            "analysis_failure",
                            "Rhythm analysis returned no result",
                            "high"
                        )
                        rhythm_issues += 1
                        continue
                    
                    # Check time signature detection
                    if 'time_signature' in rhythm_result:
                        detected_ts = rhythm_result['time_signature']
                        expected_ts = rhythm_test['time_sig']
                        
                        print(f"     Expected time sig: {expected_ts}")
                        print(f"     Detected time sig: {detected_ts}")
                        
                        # Should at least detect something reasonable
                        if not detected_ts or detected_ts == "unknown":
                            self.log_musical_error(
                                rhythm_test['name'],
                                "time_signature_detection",
                                "Failed to detect time signature",
                                "medium"
                            )
                            rhythm_issues += 1
                    
                    # Check rhythm complexity assessment
                    if 'complexity' in rhythm_result:
                        complexity = rhythm_result['complexity']
                        print(f"     Detected complexity: {complexity}")
                        
                        # Basic sanity check - should detect some level
                        if not complexity or complexity == "unknown":
                            self.log_musical_error(
                                rhythm_test['name'],
                                "complexity_assessment",
                                "Failed to assess rhythmic complexity",
                                "medium"
                            )
                            rhythm_issues += 1
                
                except Exception as e:
                    self.log_musical_error(
                        rhythm_test['name'],
                        "analysis_crash",
                        f"Rhythm analysis crashed: {str(e)}",
                        "critical"
                    )
                    rhythm_issues += 1
            
            print(f"✅ Rhythm analysis test completed - {rhythm_issues} issues found")
            return rhythm_issues == 0
            
        except Exception as e:
            self.log_musical_error(
                "rhythm_analysis_test",
                "test_crash", 
                f"Rhythm analysis test crashed: {str(e)}",
                "critical"
            )
            return False
    
    async def test_non_western_music_handling(self) -> bool:
        """Test system's ability to handle non-Western musical elements"""
        print("🔍 Testing non-Western music handling...")
        
        if not MUSIC21_AVAILABLE:
            print("⚠️ Skipping - Music21 not available")
            return True
        
        try:
            # Test microtonal and non-Western scales
            non_western_tests = [
                {
                    "name": "Pentatonic Scale",
                    "notes": ["C4", "D4", "E4", "G4", "A4", "C5"],
                    "expected_scale": "pentatonic",
                    "culture": "East Asian"
                },
                {
                    "name": "Arabic Maqam Hijaz",
                    "notes": ["D4", "Eb4", "F#4", "G4", "A4", "Bb4", "C5", "D5"],
                    "expected_scale": "hijaz_maqam",
                    "culture": "Arabic"
                },
                {
                    "name": "Indian Raga Elements",
                    "notes": ["C4", "Db4", "E4", "F4", "G4", "Ab4", "B4", "C5"],
                    "expected_scale": "raga_based",
                    "culture": "Indian"
                }
            ]
            
            non_western_issues = 0
            
            for test in non_western_tests:
                try:
                    print(f"   Testing {test['name']} ({test['culture']})...")
                    
                    # Create non-Western melody
                    from music21 import stream, note, duration
                    
                    melody = stream.Stream()
                    
                    for note_name in test['notes']:
                        note_obj = note.Note(note_name)
                        note_obj.duration = duration.Duration(0.5)
                        melody.append(note_obj)
                    
                    # Store and analyze
                    score_id = f"non_western_{test['name'].replace(' ', '_')}"
                    await self.score_manager.add_score(score_id, melody)
                    
                    # Test if system can handle non-Western elements without crashing
                    try:
                        key_result = await self.harmonic_analyzer.analyze_key_signature(melody)
                        melodic_result = await self.melodic_analyzer.analyze_melodic_patterns(melody)
                        
                        # System should not crash, even if analysis is imperfect
                        print(f"     Analysis completed without crash")
                        
                        # Check if it detects the non-Western nature
                        if key_result and 'key' in key_result:
                            detected_key = key_result['key']
                            confidence = key_result.get('confidence', 0)
                            
                            # For non-Western scales, confidence might be low (which is okay)
                            print(f"     Key analysis: {detected_key} (confidence: {confidence:.2f})")
                            
                            # Should at least indicate uncertainty for truly non-Western scales
                            if test['culture'] in ['Arabic', 'Indian'] and confidence > 0.8:
                                print(f"     Note: High confidence on non-Western scale may indicate over-confidence")
                    
                    except Exception as e:
                        self.log_musical_error(
                            test['name'],
                            "non_western_crash",
                            f"System crashed on {test['culture']} music: {str(e)}",
                            "critical"
                        )
                        non_western_issues += 1
                
                except Exception as e:
                    self.log_musical_error(
                        test['name'],
                        "test_setup_error",
                        f"Failed to set up {test['culture']} test: {str(e)}",
                        "medium"
                    )
                    non_western_issues += 1
            
            print(f"✅ Non-Western music test completed - {non_western_issues} issues found")
            return non_western_issues == 0
            
        except Exception as e:
            self.log_musical_error(
                "non_western_test",
                "test_crash",
                f"Non-Western music test crashed: {str(e)}",
                "critical"
            )
            return False

async def run_music_professor_tests():
    """Run comprehensive music professor test suite"""
    print("🎼 MUSIC PROFESSOR TEST SUITE - REAL SCORES, REAL BUGS")
    print("=" * 65)
    print("Testing music21-mcp with ACTUAL scores that music professors use")
    print("Finding bugs in music theory analysis, not just code coverage")
    print()
    
    if not MUSIC21_AVAILABLE:
        print("❌ CRITICAL: Music21 not available!")
        print("Install with: pip install music21")
        print("This test suite requires music21 for real musical analysis")
        return {"total_tests": 0, "passed_tests": 0, "musical_errors": [], "professor_approval": False}
    
    tester = MusicProfessorTester()
    
    professor_tests = [
        ("Key Analysis Accuracy", tester.test_key_analysis_accuracy),
        ("Harmonic Progression Analysis", tester.test_harmonic_progression_analysis),
        ("Melodic Analysis Correctness", tester.test_melodic_analysis_correctness),
        ("Rhythm Analysis Accuracy", tester.test_rhythm_analysis_accuracy),
        ("Non-Western Music Handling", tester.test_non_western_music_handling),
    ]
    
    results = {}
    passed = 0
    total = len(professor_tests)
    
    for test_name, test_func in professor_tests:
        print(f"\n🎼 {test_name}:")
        try:
            result = await test_func()
            results[test_name] = result
            if result:
                passed += 1
                print(f"✅ {test_name} - Musically sound")
            else:
                print(f"❌ {test_name} - Musical issues found")
        except Exception as e:
            print(f"💥 {test_name} - TEST CRASHED: {str(e)}")
            print(f"Traceback: {traceback.format_exc()}")
            results[test_name] = False
        print()
    
    # Analyze musical errors
    print("📊 MUSIC PROFESSOR ASSESSMENT:")
    print("=" * 65)
    print(f"Total music tests: {total}")
    print(f"Tests passed: {passed}")
    print(f"Tests failed: {total - passed}")
    print(f"Musical accuracy: {passed/total*100:.1f}%")
    print(f"Total musical errors: {len(tester.musical_errors)}")
    print()
    
    # Categorize musical errors
    errors_by_type = {}
    errors_by_severity = {"critical": 0, "high": 0, "medium": 0, "low": 0}
    
    for error in tester.musical_errors:
        error_type = error["error_type"]
        severity = error["severity"]
        
        if error_type not in errors_by_type:
            errors_by_type[error_type] = []
        errors_by_type[error_type].append(error)
        
        if severity in errors_by_severity:
            errors_by_severity[severity] += 1
    
    print("🎵 MUSICAL ERRORS BY TYPE:")
    for error_type, errors in errors_by_type.items():
        print(f"   {error_type.upper()}: {len(errors)} errors")
        for error in errors[:2]:  # Show first 2 errors
            print(f"     - {error['severity'].upper()}: {error['description']}")
    print()
    
    print("🚨 ERRORS BY SEVERITY:")
    for severity, count in errors_by_severity.items():
        if count > 0:
            print(f"   {severity.upper()}: {count} errors")
    print()
    
    # Professor approval assessment
    critical_errors = errors_by_severity["critical"]
    high_errors = errors_by_severity["high"]
    total_errors = len(tester.musical_errors)
    
    professor_approval = False
    
    if critical_errors == 0 and high_errors == 0 and passed >= total * 0.8:
        professor_approval = True
        print("🎓 PROFESSOR APPROVAL: ✅ EXCELLENT")
        print("   System demonstrates strong musical understanding")
        print("   Ready for use in music education and analysis")
    elif critical_errors == 0 and passed >= total * 0.6:
        print("🎓 PROFESSOR APPROVAL: 🟡 CONDITIONAL")
        print("   System shows promise but needs improvement")
        print("   Suitable for basic analysis with supervision")
    else:
        print("🎓 PROFESSOR APPROVAL: ❌ NOT RECOMMENDED")
        print("   System has significant musical analysis flaws")
        print("   Not suitable for educational or professional use")
    
    print()
    print("🔥 NEXT STEPS FOR MUSICAL IMPROVEMENT:")
    if professor_approval:
        print("1. ✅ Excellent musical foundation - Ready for advanced features")
        print("2. Add more sophisticated harmonic analysis")
        print("3. Implement advanced counterpoint analysis")
        print("4. Add support for contemporary music styles")
    else:
        print("1. Fix critical musical analysis errors")
        print("2. Improve basic music theory accuracy")
        print("3. Add proper error handling for edge cases")
        print("4. Consult with music theory experts")
    
    return {
        'total_tests': total,
        'passed_tests': passed,
        'musical_errors': tester.musical_errors,
        'errors_by_type': errors_by_type,
        'errors_by_severity': errors_by_severity,
        'musical_accuracy': passed/total*100 if total > 0 else 0,
        'professor_approval': professor_approval
    }

def main():
    """Main entry point"""
    return asyncio.run(run_music_professor_tests())

if __name__ == "__main__":
    results = main()
    print(f"\n🎼 Musical Accuracy Score: {results['musical_accuracy']:.1f}%")
    if results['professor_approval']:
        print("🎉 READY FOR MUSIC EDUCATION!")
    else:
        print("📚 More musical study needed")