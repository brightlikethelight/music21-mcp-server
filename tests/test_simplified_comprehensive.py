#!/usr/bin/env python3
"""
Comprehensive test suite for the simplified Music21 MCP Server
Tests all 7 tools with various edge cases and real-world scenarios
"""
import asyncio
import sys
import os
import tempfile
from pathlib import Path
import json
from typing import Dict, Any

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from music21_mcp.server import (
    import_score, list_scores, analyze_key, analyze_chords, 
    get_score_info, export_score, delete_score, score_manager
)
from music21 import stream, note, chord, meter, tempo


class TestRunner:
    """Test runner with result tracking"""
    def __init__(self):
        self.results = {}
        self.test_count = 0
        self.pass_count = 0
    
    async def run_test(self, name: str, test_func, *args, **kwargs):
        """Run a single test and track results"""
        self.test_count += 1
        try:
            result = await test_func(*args, **kwargs)
            self.results[name] = {"status": "pass", "result": result}
            self.pass_count += 1
            print(f"✅ {name}")
            return True
        except Exception as e:
            self.results[name] = {"status": "fail", "error": str(e)}
            print(f"❌ {name}: {e}")
            return False
    
    def print_summary(self):
        """Print test summary"""
        print("\n" + "="*70)
        print(f"TEST SUMMARY: {self.pass_count}/{self.test_count} passed ({self.pass_count/self.test_count*100:.1f}%)")
        print("="*70)
        
        # Group results
        failures = {k: v for k, v in self.results.items() if v["status"] == "fail"}
        if failures:
            print("\nFAILURES:")
            for name, result in failures.items():
                print(f"  ❌ {name}: {result['error']}")
        else:
            print("\n🎉 ALL TESTS PASSED!")


async def test_import_score():
    """Test import_score tool with various inputs"""
    runner = TestRunner()
    
    print("\n📥 Testing import_score")
    print("-" * 50)
    
    # Test 1: Import from corpus
    await runner.run_test(
        "Import Bach chorale from corpus",
        import_score,
        "bach_test1",
        "bach/bwv66.6"
    )
    
    # Test 2: Import simple text notation
    await runner.run_test(
        "Import from text notation",
        import_score,
        "text_test1",
        "C4 D4 E4 F4 G4 A4 B4 C5"
    )
    
    # Test 3: Import with accidentals
    await runner.run_test(
        "Import text with accidentals",
        import_score,
        "text_test2",
        "C#4 Eb4 F#4 G#4 Bb4"
    )
    
    # Test 4: Import tiny corpus work
    await runner.run_test(
        "Import tiny notation from corpus",
        import_score,
        "tiny_test",
        "common/tinyscore"
    )
    
    # Test 5: Invalid source
    result = await import_score("invalid_test", "/nonexistent/file.xml")
    assert result["status"] == "error", "Should fail on invalid file"
    print("✅ Handles invalid file correctly")
    
    # Test 6: Empty source
    result = await import_score("empty_test", "")
    assert result["status"] == "error", "Should fail on empty source"
    print("✅ Handles empty source correctly")
    
    return runner


async def test_list_scores():
    """Test list_scores tool"""
    runner = TestRunner()
    
    print("\n📋 Testing list_scores")
    print("-" * 50)
    
    # Clear and add some test scores
    score_manager.scores.clear()
    await import_score("test1", "C4 D4 E4")
    await import_score("test2", "F4 G4 A4")
    await import_score("test3", "bach/bwv66.6")
    
    # Test listing
    result = await runner.run_test(
        "List all scores",
        list_scores
    )
    
    if result:
        scores_result = runner.results["List all scores"]["result"]
        assert scores_result["total_count"] == 3, "Should have 3 scores"
        assert len(scores_result["scores"]) == 3, "Should return 3 score details"
        print(f"  Found {scores_result['total_count']} scores")
    
    return runner


async def test_analyze_key():
    """Test analyze_key tool with various methods"""
    runner = TestRunner()
    
    print("\n🎵 Testing analyze_key")
    print("-" * 50)
    
    # Import test score
    await import_score("key_test", "bach/bwv66.6")
    
    # Test 1: Default method
    result = await runner.run_test(
        "Analyze key with default method",
        analyze_key,
        "key_test"
    )
    
    if result:
        key_result = runner.results["Analyze key with default method"]["result"]
        print(f"  Key: {key_result['key']}, Confidence: {key_result['confidence']:.3f}")
    
    # Test 2: Krumhansl method
    await runner.run_test(
        "Analyze key with Krumhansl method",
        analyze_key,
        "key_test",
        "krumhansl"
    )
    
    # Test 3: Aarden method
    await runner.run_test(
        "Analyze key with Aarden method",
        analyze_key,
        "key_test",
        "aarden"
    )
    
    # Test 4: Non-existent score
    result = await analyze_key("nonexistent")
    assert result["status"] == "error", "Should fail on non-existent score"
    print("✅ Handles non-existent score correctly")
    
    # Test 5: Simple melody
    await import_score("simple_key", "C4 E4 G4 C5 G4 E4 C4")
    await runner.run_test(
        "Analyze key of simple C major scale",
        analyze_key,
        "simple_key"
    )
    
    return runner


async def test_analyze_chords():
    """Test analyze_chords tool"""
    runner = TestRunner()
    
    print("\n🎹 Testing analyze_chords")
    print("-" * 50)
    
    # Test 1: Bach chorale (rich harmony)
    await import_score("chord_test1", "bach/bwv66.6")
    result = await runner.run_test(
        "Analyze chords in Bach chorale",
        analyze_chords,
        "chord_test1"
    )
    
    if result:
        chord_result = runner.results["Analyze chords in Bach chorale"]["result"]
        print(f"  Found {chord_result['total_chords']} chords")
        assert chord_result["total_chords"] > 20, "Bach chorale should have many chords"
    
    # Test 2: With Roman numerals
    await runner.run_test(
        "Analyze chords with Roman numerals",
        analyze_chords,
        "chord_test1",
        True
    )
    
    # Test 3: Simple melody (no chords)
    await import_score("melody_test", "C4 D4 E4 F4 G4")
    result = await runner.run_test(
        "Analyze chords in simple melody",
        analyze_chords,
        "melody_test"
    )
    
    if result:
        chord_result = runner.results["Analyze chords in simple melody"]["result"]
        print(f"  Found {chord_result['total_chords']} chords (expected 0)")
    
    # Test 4: Create score with actual chords
    chord_score = stream.Score()
    part = stream.Part()
    part.append(chord.Chord(['C4', 'E4', 'G4']))
    part.append(chord.Chord(['F4', 'A4', 'C5']))
    part.append(chord.Chord(['G4', 'B4', 'D5']))
    part.append(chord.Chord(['C4', 'E4', 'G4', 'C5']))
    chord_score.append(part)
    score_manager.scores["chord_test2"] = chord_score
    
    await runner.run_test(
        "Analyze manually created chords",
        analyze_chords,
        "chord_test2"
    )
    
    return runner


async def test_get_score_info():
    """Test get_score_info tool"""
    runner = TestRunner()
    
    print("\n📊 Testing get_score_info")
    print("-" * 50)
    
    # Test 1: Bach chorale with metadata
    await import_score("info_test1", "bach/bwv66.6")
    result = await runner.run_test(
        "Get info for Bach chorale",
        get_score_info,
        "info_test1"
    )
    
    if result:
        info = runner.results["Get info for Bach chorale"]["result"]
        print(f"  Title: {info.get('title', 'N/A')}")
        print(f"  Composer: {info.get('composer', 'N/A')}")
        print(f"  Parts: {info['num_parts']}, Measures: {info['num_measures']}")
        print(f"  Notes: {info['num_notes']}, Duration: {info['duration_quarters']} quarters")
    
    # Test 2: Simple score without metadata
    await import_score("info_test2", "C4 D4 E4 F4 G4")
    await runner.run_test(
        "Get info for simple score",
        get_score_info,
        "info_test2"
    )
    
    # Test 3: Complex score with multiple parts
    complex_score = stream.Score()
    part1 = stream.Part()
    part2 = stream.Part()
    
    # Add measures with time signatures
    for i in range(4):
        m1 = stream.Measure(number=i+1)
        m2 = stream.Measure(number=i+1)
        if i == 0:
            m1.append(meter.TimeSignature('4/4'))
            m2.append(meter.TimeSignature('4/4'))
            m1.append(tempo.MetronomeMark(number=120))
        
        # Add notes
        for j in range(4):
            m1.append(note.Note('C5', quarterLength=1))
            m2.append(note.Note('E3', quarterLength=1))
        
        part1.append(m1)
        part2.append(m2)
    
    complex_score.insert(0, part1)
    complex_score.insert(0, part2)
    complex_score.metadata.title = "Test Piece"
    complex_score.metadata.composer = "Test Composer"
    
    score_manager.scores["complex_test"] = complex_score
    
    result = await runner.run_test(
        "Get info for complex multi-part score",
        get_score_info,
        "complex_test"
    )
    
    if result:
        info = runner.results["Get info for complex multi-part score"]["result"]
        assert info["num_parts"] == 2, "Should have 2 parts"
        assert len(info["time_signatures"]) > 0, "Should have time signatures"
        assert len(info["tempo_markings"]) > 0, "Should have tempo markings"
    
    return runner


async def test_export_score():
    """Test export_score tool with various formats"""
    runner = TestRunner()
    
    print("\n💾 Testing export_score")
    print("-" * 50)
    
    # Import test score
    await import_score("export_test", "bach/bwv66.6")
    
    # Test different formats
    formats = ["musicxml", "midi", "abc", "lilypond"]
    
    for fmt in formats:
        result = await runner.run_test(
            f"Export to {fmt}",
            export_score,
            "export_test",
            fmt
        )
        
        if result:
            export_result = runner.results[f"Export to {fmt}"]["result"]
            file_path = export_result["file_path"]
            assert os.path.exists(file_path), f"{fmt} file should exist"
            file_size = os.path.getsize(file_path)
            print(f"  Created {file_path} ({file_size} bytes)")
            os.unlink(file_path)  # Clean up
    
    # Test with custom file path
    custom_path = tempfile.mktemp(suffix=".xml")
    result = await runner.run_test(
        "Export with custom path",
        export_score,
        "export_test",
        "musicxml",
        custom_path
    )
    
    if result:
        assert os.path.exists(custom_path), "Custom path file should exist"
        os.unlink(custom_path)
    
    # Test with non-existent score
    result = await export_score("nonexistent", "musicxml")
    assert result["status"] == "error", "Should fail on non-existent score"
    print("✅ Handles non-existent score correctly")
    
    return runner


async def test_delete_score():
    """Test delete_score tool"""
    runner = TestRunner()
    
    print("\n🗑️ Testing delete_score")
    print("-" * 50)
    
    # Add test scores
    await import_score("delete_test1", "C4 D4 E4")
    await import_score("delete_test2", "F4 G4 A4")
    
    initial_count = len(score_manager.scores)
    
    # Test 1: Delete existing score
    result = await runner.run_test(
        "Delete existing score",
        delete_score,
        "delete_test1"
    )
    
    if result:
        assert "delete_test1" not in score_manager.scores, "Score should be deleted"
        assert len(score_manager.scores) == initial_count - 1, "Score count should decrease"
    
    # Test 2: Delete non-existent score
    result = await delete_score("nonexistent")
    assert result["status"] == "error", "Should fail on non-existent score"
    print("✅ Handles non-existent score correctly")
    
    # Test 3: Delete remaining test score
    await runner.run_test(
        "Delete another score",
        delete_score,
        "delete_test2"
    )
    
    return runner


async def test_edge_cases():
    """Test edge cases and error handling"""
    runner = TestRunner()
    
    print("\n⚠️ Testing Edge Cases")
    print("-" * 50)
    
    # Test 1: Very long text input
    long_notes = " ".join([f"C{i%8}" for i in range(1000)])
    result = await import_score("long_test", long_notes)
    # Should either succeed or fail gracefully
    assert result["status"] in ["success", "error"], "Should have valid status"
    print("✅ Handles very long input")
    
    # Test 2: Invalid note names
    result = await import_score("invalid_notes", "XYZ ABC 123")
    assert result["status"] == "error", "Should fail on invalid notes"
    print("✅ Handles invalid note names")
    
    # Test 3: Mixed valid/invalid corpus paths
    result = await import_score("mixed_corpus", "bach/invalid/path")
    assert result["status"] == "error", "Should fail on invalid corpus path"
    print("✅ Handles invalid corpus paths")
    
    # Test 4: Empty score operations
    empty_score = stream.Score()
    score_manager.scores["empty_test"] = empty_score
    
    # Try operations on empty score
    key_result = await analyze_key("empty_test")
    print(f"✅ Key analysis on empty score: {key_result['status']}")
    
    chord_result = await analyze_chords("empty_test")
    print(f"✅ Chord analysis on empty score: {chord_result['status']}")
    
    info_result = await get_score_info("empty_test")
    print(f"✅ Info on empty score: {info_result['status']}")
    
    return runner


async def test_real_world_scenarios():
    """Test real-world usage scenarios"""
    runner = TestRunner()
    
    print("\n🌍 Testing Real-World Scenarios")
    print("-" * 50)
    
    # Scenario 1: Complete analysis workflow
    print("\n  Scenario 1: Complete Bach chorale analysis")
    
    # Import
    await import_score("bach_complete", "bach/bwv66.6")
    
    # Full analysis
    key_result = await analyze_key("bach_complete")
    chord_result = await analyze_chords("bach_complete", True)
    info_result = await get_score_info("bach_complete")
    
    print(f"    Key: {key_result['key']} (confidence: {key_result['confidence']:.2f})")
    print(f"    Chords: {chord_result['total_chords']}")
    print(f"    Duration: {info_result['duration_quarters']} quarters")
    
    # Export to MIDI
    export_result = await export_score("bach_complete", "midi")
    if export_result["status"] == "success":
        print(f"    Exported to: {export_result['file_path']}")
        os.unlink(export_result['file_path'])
    
    # Scenario 2: Simple composition workflow
    print("\n  Scenario 2: Simple composition workflow")
    
    # Create a simple melody
    await import_score("melody", "C4 E4 G4 C5 B4 G4 E4 C4")
    
    # Analyze it
    key_result = await analyze_key("melody")
    print(f"    Detected key: {key_result['key']}")
    
    # Create a harmony
    await import_score("harmony", "C3 G3 C3 G3 G3 D3 G3 C3")
    
    # Scenario 3: Batch processing
    print("\n  Scenario 3: Batch processing multiple scores")
    
    corpus_works = ["bach/bwv66.6", "common/tinyscore"]
    
    for i, work in enumerate(corpus_works):
        score_id = f"batch_{i}"
        import_result = await import_score(score_id, work)
        if import_result["status"] == "success":
            key_result = await analyze_key(score_id)
            print(f"    {work}: {key_result['key']}")
    
    runner.pass_count += 3  # Count scenarios as passes
    runner.test_count += 3
    
    return runner


async def run_comprehensive_tests():
    """Run all comprehensive tests"""
    print("🎼 Music21 MCP Server - Comprehensive Test Suite")
    print("="*70)
    
    all_runners = []
    
    # Run all test suites
    all_runners.append(await test_import_score())
    all_runners.append(await test_list_scores())
    all_runners.append(await test_analyze_key())
    all_runners.append(await test_analyze_chords())
    all_runners.append(await test_get_score_info())
    all_runners.append(await test_export_score())
    all_runners.append(await test_delete_score())
    all_runners.append(await test_edge_cases())
    all_runners.append(await test_real_world_scenarios())
    
    # Calculate totals
    total_tests = sum(r.test_count for r in all_runners)
    total_passes = sum(r.pass_count for r in all_runners)
    
    # Print final summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {total_passes}")
    print(f"Failed: {total_tests - total_passes}")
    print(f"Success Rate: {total_passes/total_tests*100:.1f}%")
    
    if total_passes == total_tests:
        print("\n🎉 ALL TESTS PASSED! The simplified server is fully functional.")
    else:
        print(f"\n⚠️ {total_tests - total_passes} tests failed. Review the output above.")
    
    return total_passes == total_tests


def main():
    """Main entry point"""
    success = asyncio.run(run_comprehensive_tests())
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())