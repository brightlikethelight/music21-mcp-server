#!/usr/bin/env python3
"""
Critical Fixes Validation - Test that all critical issues are resolved
"""
import asyncio
import sys
from pathlib import Path
import json
import time
import subprocess

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from music21 import corpus, stream, note, chord, key, meter
from music21_mcp.server import score_manager


class CriticalFixValidator:
    """Validates that all critical fixes are working"""
    
    def __init__(self):
        self.results = {}
        self.start_time = time.time()
    
    async def test_mcp_server_startup(self):
        """Test 1: MCP Server STDIO Mode (was 0% working)"""
        print("\n🔧 TEST 1: MCP Server STDIO Mode")
        print("-" * 50)
        
        # Test if server can start in STDIO mode
        try:
            # Try to run the server with --help
            result = subprocess.run(
                [sys.executable, "-m", "music21_mcp.server", "--help"],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                print("✅ Server module loads without errors")
                self.results['mcp_startup'] = True
            else:
                print(f"❌ Server startup failed: {result.stderr}")
                self.results['mcp_startup'] = False
                
        except Exception as e:
            print(f"❌ Server startup error: {e}")
            self.results['mcp_startup'] = False
    
    async def test_chord_analysis_fix(self):
        """Test 2: Chord Analysis (was returning 0 chords)"""
        print("\n🔧 TEST 2: Chord Analysis Fix")
        print("-" * 50)
        
        try:
            # Load Bach chorale
            bach_chorale = corpus.parse('bach/bwv66.6')
            score_manager.add_score("bach_test", bach_chorale, {})
            
            # Import and test
            from music21_mcp.server import analyze_chord_progressions
            
            result = await analyze_chord_progressions("bach_test")
            
            if result.get('status') == 'success':
                chord_count = result.get('total_chords', 0)
                print(f"✅ Chord analysis works: {chord_count} chords detected")
                
                # Bach chorale should have 30-50 chords
                if 30 <= chord_count <= 50:
                    print(f"✅ Chord count is reasonable (expected 30-50)")
                    self.results['chord_analysis'] = True
                    
                    # Check if chords have proper data
                    if result.get('chord_progressions'):
                        prog = result['chord_progressions'][0]
                        print(f"   First progression: {' → '.join(prog['progression'][:5])}")
                else:
                    print(f"❌ Unexpected chord count: {chord_count} (expected 30-50)")
                    self.results['chord_analysis'] = False
            else:
                print(f"❌ Chord analysis failed: {result}")
                self.results['chord_analysis'] = False
                
        except Exception as e:
            print(f"❌ Chord analysis error: {e}")
            import traceback
            traceback.print_exc()
            self.results['chord_analysis'] = False
    
    async def test_text_import_fix(self):
        """Test 3: Text Import (was rejecting all text input)"""
        print("\n🔧 TEST 3: Text Import Fix")
        print("-" * 50)
        
        try:
            from music21_mcp.server import import_score
            
            # Test simple text notation
            test_cases = [
                ("C4 D4 E4 F4 G4", "text", "Simple note sequence"),
                ("X:1\nT:Test\nK:C\nCDEF|GABc|", "abc", "ABC notation"),
                ("4/4 c4 d e f g1", "text", "TinyNotation"),
            ]
            
            passed = 0
            for content, format_type, description in test_cases:
                try:
                    result = await import_score(
                        score_id=f"text_test_{passed}",
                        source=content,
                        source_type=format_type
                    )
                    
                    if result.get('status') == 'success':
                        print(f"✅ {description}: Import successful")
                        passed += 1
                    else:
                        print(f"❌ {description}: Import failed - {result.get('message')}")
                except Exception as e:
                    print(f"❌ {description}: Error - {e}")
            
            self.results['text_import'] = passed >= 2  # At least 2 formats should work
            
        except Exception as e:
            print(f"❌ Text import error: {e}")
            self.results['text_import'] = False
    
    async def test_key_detection_accuracy(self):
        """Test 4: Key Detection Accuracy (was 31% confidence)"""
        print("\n🔧 TEST 4: Key Detection Accuracy")
        print("-" * 50)
        
        try:
            from music21_mcp.server import analyze_key
            
            # Test with known pieces
            test_pieces = [
                ('bach/bwv66.6', 'f# minor', 'Bach Chorale BWV 66.6'),
                ('bach/bwv7.7', 'e minor', 'Bach Chorale BWV 7.7'),
            ]
            
            passed = 0
            total_confidence = 0
            
            for corpus_path, expected_key, name in test_pieces:
                try:
                    # Load and analyze
                    score = corpus.parse(corpus_path)
                    score_manager.add_score(corpus_path.replace('/', '_'), score, {})
                    
                    result = await analyze_key(
                        score_id=corpus_path.replace('/', '_'),
                        method="hybrid"  # Use improved hybrid method
                    )
                    
                    if result.get('status') == 'success':
                        detected_key = result.get('key', '').lower()
                        confidence = result.get('confidence', 0)
                        
                        # Check if key is correct
                        key_correct = expected_key in detected_key or detected_key in expected_key
                        
                        print(f"\n{name}:")
                        print(f"   Expected: {expected_key}")
                        print(f"   Detected: {detected_key}")
                        print(f"   Confidence: {confidence:.2%}")
                        print(f"   {'✅' if key_correct else '❌'} Key {'matches' if key_correct else 'mismatch'}")
                        print(f"   {'✅' if confidence > 0.7 else '❌'} Confidence {'good' if confidence > 0.7 else 'low'}")
                        
                        if key_correct and confidence > 0.7:
                            passed += 1
                        total_confidence += confidence
                        
                except Exception as e:
                    print(f"❌ Error analyzing {name}: {e}")
            
            avg_confidence = total_confidence / len(test_pieces) if test_pieces else 0
            print(f"\nAverage confidence: {avg_confidence:.2%}")
            
            # Should have high confidence (>70%) for clear tonal pieces
            self.results['key_accuracy'] = passed == len(test_pieces) and avg_confidence > 0.7
            
        except Exception as e:
            print(f"❌ Key detection error: {e}")
            self.results['key_accuracy'] = False
    
    async def test_real_music_comprehensive(self):
        """Test 5: Comprehensive Real Music Test"""
        print("\n🔧 TEST 5: Comprehensive Real Music Analysis")
        print("-" * 50)
        
        try:
            # Test complete workflow with a real piece
            bach_score = corpus.parse('bach/bwv66.6')
            
            from music21_mcp.server import (
                import_score, analyze_key, analyze_chord_progressions,
                analyze_rhythm, export_score
            )
            
            # 1. Import as stream object
            import_result = await import_score(
                score_id="comprehensive_test",
                source=bach_score,
                source_type="stream"
            )
            
            if import_result.get('status') != 'success':
                print(f"❌ Import failed: {import_result}")
                self.results['comprehensive'] = False
                return
            
            print("✅ Import successful")
            
            # 2. Analyze key
            key_result = await analyze_key("comprehensive_test")
            print(f"✅ Key: {key_result.get('key')} ({key_result.get('confidence', 0):.2%} confidence)")
            
            # 3. Analyze chords
            chord_result = await analyze_chord_progressions("comprehensive_test")
            print(f"✅ Chords: {chord_result.get('total_chords', 0)} found")
            
            # 4. Analyze rhythm
            rhythm_result = await analyze_rhythm("comprehensive_test")
            print(f"✅ Rhythm: {rhythm_result.get('time_signature')} time signature")
            
            # 5. Export test
            export_result = await export_score("comprehensive_test", format="musicxml")
            if export_result.get('status') == 'success':
                print("✅ Export successful")
            
            # All tests passed
            self.results['comprehensive'] = all([
                import_result.get('status') == 'success',
                key_result.get('confidence', 0) > 0.5,
                chord_result.get('total_chords', 0) > 20,
                rhythm_result.get('status') == 'success',
                export_result.get('status') == 'success'
            ])
            
        except Exception as e:
            print(f"❌ Comprehensive test error: {e}")
            import traceback
            traceback.print_exc()
            self.results['comprehensive'] = False
    
    def print_summary(self):
        """Print validation summary"""
        elapsed = time.time() - self.start_time
        
        print("\n" + "="*60)
        print("🏁 CRITICAL FIXES VALIDATION SUMMARY")
        print("="*60)
        
        # Status of each critical issue
        fixes = [
            ("MCP Server STDIO Mode", 'mcp_startup', "Can start without crashes"),
            ("Chord Analysis", 'chord_analysis', "Detects 30-50 chords in Bach"),
            ("Text Import", 'text_import', "Accepts text/ABC notation"),
            ("Key Detection Accuracy", 'key_accuracy', ">70% confidence on clear keys"),
            ("Comprehensive Workflow", 'comprehensive', "Full import→analyze→export")
        ]
        
        passed = 0
        for name, key, description in fixes:
            status = self.results.get(key, False)
            if status:
                passed += 1
            print(f"{'✅' if status else '❌'} {name}: {description}")
        
        print(f"\nTotal: {passed}/{len(fixes)} critical fixes working")
        print(f"Time: {elapsed:.1f} seconds")
        
        # Overall verdict
        if passed == len(fixes):
            print("\n🎉 ALL CRITICAL FIXES SUCCESSFUL!")
            print("The server is now ready for further development.")
        elif passed >= 3:
            print("\n⚠️ PARTIAL SUCCESS")
            print("Most critical issues fixed, but some problems remain.")
        else:
            print("\n🚨 CRITICAL FIXES FAILED")
            print("Major issues still present. Do not proceed.")
        
        # Save results
        with open('critical_fixes_validation.json', 'w') as f:
            json.dump({
                'results': self.results,
                'passed': passed,
                'total': len(fixes),
                'elapsed_seconds': elapsed,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }, f, indent=2)
        
        return passed == len(fixes)


async def main():
    """Run critical fixes validation"""
    print("🚨 CRITICAL FIXES VALIDATION")
    print("Testing all urgent repairs...")
    
    validator = CriticalFixValidator()
    
    # Run all tests
    await validator.test_mcp_server_startup()
    await validator.test_chord_analysis_fix()
    await validator.test_text_import_fix()
    await validator.test_key_detection_accuracy()
    await validator.test_real_music_comprehensive()
    
    # Summary
    success = validator.print_summary()
    
    if success:
        print("\n✅ Safe to proceed with Phase 3 development!")
    else:
        print("\n❌ Fix remaining issues before continuing!")
    
    return success


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)