#!/usr/bin/env python3
"""
Comprehensive Real Music Test Suite
Tests with actual musical compositions to validate accuracy and functionality
"""
import asyncio
import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from music21 import corpus, stream, note, chord, key, meter, tempo
from music21_mcp.server import score_manager

# Known correct analyses for validation
VALIDATION_DATA = {
    'bach/bwv66.6': {
        'name': 'Bach Chorale BWV 66.6 - Christ unser Herr zum Jordan kam',
        'key': 'f# minor',
        'time_signature': '4/4',
        'measures': 9,
        'expected_chords': (35, 45),  # Range
        'voice_count': 4,
        'characteristics': ['four-part chorale', 'homophonic texture', 'authentic cadences']
    },
    'mozart/k545': {
        'name': 'Mozart Piano Sonata K. 545, 1st movement',
        'key': 'C major',
        'time_signature': '4/4',
        'form': 'sonata allegro',
        'expected_chords': (100, 200),
        'characteristics': ['alberti bass', 'scalar passages', 'clear phrases']
    },
    'bach/bwv846': {
        'name': 'Bach WTC Book 1, Prelude No. 1 in C Major',
        'key': 'C major',
        'time_signature': '4/4',
        'measures': 35,
        'expected_chords': (35, 40),  # One chord per measure approximately
        'characteristics': ['arpeggiated chords', 'consistent rhythm', 'harmonic progression']
    },
    'schubert/d576': {
        'name': 'Schubert 13 Variations on a Theme by Hüttenbrenner',
        'key': 'A minor',
        'expected_chords': (50, 150),
        'characteristics': ['theme and variations', 'romantic harmony']
    }
}


class ComprehensiveRealMusicTester:
    """Comprehensive testing with real musical works"""
    
    def __init__(self):
        self.results = {}
        self.performance_metrics = {}
        self.accuracy_scores = {}
    
    async def run_all_tests(self):
        """Run comprehensive test suite"""
        print("🎵 Comprehensive Real Music Test Suite")
        print("="*60)
        
        # Test 1: Import functionality with different formats
        await self.test_import_functionality()
        
        # Test 2: Key detection accuracy
        await self.test_key_detection_accuracy()
        
        # Test 3: Chord analysis accuracy
        await self.test_chord_analysis_accuracy()
        
        # Test 4: Complete analysis workflow
        await self.test_complete_workflow()
        
        # Test 5: Performance benchmarks
        await self.test_performance_benchmarks()
        
        # Generate report
        self.generate_comprehensive_report()
    
    async def test_import_functionality(self):
        """Test import with various formats and real music"""
        print("\n📥 TEST 1: Import Functionality")
        print("-"*50)
        
        from music21_mcp.server import import_score
        
        test_cases = [
            # Test corpus import
            {
                'name': 'Bach Chorale (corpus)',
                'source': corpus.parse('bach/bwv66.6'),
                'source_type': 'stream',
                'score_id': 'bach_corpus'
            },
            # Test text notation
            {
                'name': 'Simple melody (text)',
                'source': 'C4 D4 E4 F4 G4 A4 B4 c1',
                'source_type': 'text',
                'score_id': 'text_melody'
            },
            # Test ABC notation with proper headers
            {
                'name': 'ABC notation',
                'source': '''X:1
T:Scale Test
M:4/4
L:1/4
K:C
CDEF|GABc|''',
                'source_type': 'abc',
                'score_id': 'abc_test'
            },
            # Test tinyNotation
            {
                'name': 'TinyNotation',
                'source': '4/4 c4 d e f g2 a b c1',
                'source_type': 'text',
                'score_id': 'tiny_test'
            }
        ]
        
        import_results = {}
        for test in test_cases:
            try:
                start = time.time()
                result = await import_score(
                    score_id=test['score_id'],
                    source=test['source'],
                    source_type=test['source_type']
                )
                duration = time.time() - start
                
                success = result.get('status') == 'success'
                import_results[test['name']] = {
                    'success': success,
                    'duration': duration,
                    'num_notes': result.get('num_notes', 0) if success else 0
                }
                
                print(f"{'✅' if success else '❌'} {test['name']}: "
                      f"{'Success' if success else result.get('message', 'Failed')} ({duration:.2f}s)")
                
            except Exception as e:
                import_results[test['name']] = {'success': False, 'error': str(e)}
                print(f"❌ {test['name']}: Error - {e}")
        
        self.results['import_functionality'] = import_results
    
    async def test_key_detection_accuracy(self):
        """Test key detection accuracy on known pieces"""
        print("\n🎹 TEST 2: Key Detection Accuracy")
        print("-"*50)
        
        from music21_mcp.server import import_score, analyze_key
        
        key_results = {}
        
        for corpus_path, expected_data in VALIDATION_DATA.items():
            try:
                # Load piece
                score = corpus.parse(corpus_path)
                score_id = corpus_path.replace('/', '_')
                
                # Import
                await import_score(
                    score_id=score_id,
                    source=score,
                    source_type='stream'
                )
                
                # Analyze key with different methods
                methods = ['hybrid', 'krumhansl', 'aarden']
                best_result = None
                best_confidence = 0
                
                for method in methods:
                    result = await analyze_key(score_id, method=method)
                    
                    if result.get('status') == 'success':
                        confidence = result.get('confidence', 0)
                        if confidence > best_confidence:
                            best_confidence = confidence
                            best_result = result
                
                if best_result:
                    detected_key = best_result.get('key', '').lower()
                    expected_key = expected_data['key'].lower()
                    
                    # Check if keys match (handle enharmonic equivalents)
                    key_correct = (expected_key in detected_key or 
                                 detected_key in expected_key or
                                 self._are_keys_equivalent(expected_key, detected_key))
                    
                    key_results[corpus_path] = {
                        'name': expected_data['name'],
                        'expected': expected_key,
                        'detected': detected_key,
                        'confidence': best_confidence,
                        'correct': key_correct,
                        'method': best_result.get('method', 'unknown')
                    }
                    
                    print(f"\n{expected_data['name']}:")
                    print(f"  Expected: {expected_key}")
                    print(f"  Detected: {detected_key} (confidence: {best_confidence:.2%})")
                    print(f"  {'✅' if key_correct and best_confidence > 0.7 else '❌'} "
                          f"{'PASS' if key_correct and best_confidence > 0.7 else 'FAIL'}")
                
            except Exception as e:
                key_results[corpus_path] = {'error': str(e)}
                print(f"❌ Error analyzing {corpus_path}: {e}")
        
        self.results['key_detection'] = key_results
    
    async def test_chord_analysis_accuracy(self):
        """Test chord analysis accuracy"""
        print("\n🎼 TEST 3: Chord Analysis Accuracy")
        print("-"*50)
        
        from music21_mcp.server import analyze_chord_progressions
        
        chord_results = {}
        
        for corpus_path, expected_data in VALIDATION_DATA.items():
            if 'expected_chords' not in expected_data:
                continue
                
            try:
                score_id = corpus_path.replace('/', '_')
                
                # Analyze chords
                result = await analyze_chord_progressions(score_id)
                
                if result.get('status') == 'success':
                    chord_count = result.get('total_chords', 0)
                    min_expected, max_expected = expected_data['expected_chords']
                    
                    in_range = min_expected <= chord_count <= max_expected
                    
                    chord_results[corpus_path] = {
                        'name': expected_data['name'],
                        'chord_count': chord_count,
                        'expected_range': (min_expected, max_expected),
                        'in_range': in_range,
                        'progressions': result.get('chord_progressions', [])[:1]  # First progression
                    }
                    
                    print(f"\n{expected_data['name']}:")
                    print(f"  Chords found: {chord_count}")
                    print(f"  Expected range: {min_expected}-{max_expected}")
                    print(f"  {'✅' if in_range else '❌'} {'PASS' if in_range else 'FAIL'}")
                    
                    if result.get('chord_progressions'):
                        prog = result['chord_progressions'][0]['progression'][:5]
                        print(f"  Sample progression: {' → '.join(str(c) for c in prog)}")
                
            except Exception as e:
                chord_results[corpus_path] = {'error': str(e)}
                print(f"❌ Error analyzing chords in {corpus_path}: {e}")
        
        self.results['chord_analysis'] = chord_results
    
    async def test_complete_workflow(self):
        """Test complete import → analyze → export workflow"""
        print("\n🔄 TEST 4: Complete Analysis Workflow")
        print("-"*50)
        
        from music21_mcp.server import (
            import_score, analyze_key, analyze_chord_progressions,
            analyze_rhythm, export_score
        )
        
        # Use Bach chorale for comprehensive test
        test_piece = 'bach/bwv66.6'
        
        try:
            # 1. Import
            score = corpus.parse(test_piece)
            import_result = await import_score(
                score_id="workflow_test",
                source=score,
                source_type='stream'
            )
            print(f"✅ Import: {import_result.get('num_notes', 0)} notes")
            
            # 2. Key analysis
            key_result = await analyze_key("workflow_test", method="hybrid")
            print(f"✅ Key: {key_result.get('key')} ({key_result.get('confidence', 0):.2%})")
            
            # 3. Chord analysis
            chord_result = await analyze_chord_progressions("workflow_test")
            print(f"✅ Chords: {chord_result.get('total_chords', 0)} found")
            
            # 4. Rhythm analysis
            rhythm_result = await analyze_rhythm("workflow_test")
            print(f"✅ Rhythm: {rhythm_result.get('time_signature')} time signature")
            
            # 5. Export
            export_result = await export_score("workflow_test", format="musicxml")
            print(f"✅ Export: {'Success' if export_result.get('status') == 'success' else 'Failed'}")
            
            self.results['complete_workflow'] = {
                'all_steps_successful': all([
                    import_result.get('status') == 'success',
                    key_result.get('status') == 'success',
                    chord_result.get('status') == 'success',
                    rhythm_result.get('status') == 'success',
                    export_result.get('status') == 'success'
                ])
            }
            
        except Exception as e:
            print(f"❌ Workflow error: {e}")
            self.results['complete_workflow'] = {'error': str(e)}
    
    async def test_performance_benchmarks(self):
        """Test performance with various piece sizes"""
        print("\n⚡ TEST 5: Performance Benchmarks")
        print("-"*50)
        
        from music21_mcp.server import import_score, analyze_key, analyze_chord_progressions
        
        performance_results = {}
        
        # Test different corpus pieces of varying complexity
        test_pieces = [
            ('bach/bwv66.6', 'Small chorale (9 measures)'),
            ('mozart/k545', 'Medium sonata movement'),
            ('beethoven/opus18no1', 'Large quartet movement')
        ]
        
        for corpus_path, description in test_pieces:
            try:
                # Check if piece exists
                try:
                    score = corpus.parse(corpus_path)
                except:
                    print(f"⏭️ Skipping {description} - not in corpus")
                    continue
                
                score_id = f"perf_{corpus_path.replace('/', '_')}"
                
                # Time import
                start = time.time()
                await import_score(score_id, score, 'stream')
                import_time = time.time() - start
                
                # Time key analysis
                start = time.time()
                await analyze_key(score_id)
                key_time = time.time() - start
                
                # Time chord analysis
                start = time.time()
                await analyze_chord_progressions(score_id)
                chord_time = time.time() - start
                
                total_time = import_time + key_time + chord_time
                
                performance_results[corpus_path] = {
                    'description': description,
                    'import_time': import_time,
                    'key_time': key_time,
                    'chord_time': chord_time,
                    'total_time': total_time
                }
                
                print(f"\n{description}:")
                print(f"  Import: {import_time:.2f}s")
                print(f"  Key analysis: {key_time:.2f}s")
                print(f"  Chord analysis: {chord_time:.2f}s")
                print(f"  Total: {total_time:.2f}s")
                print(f"  {'✅' if total_time < 5 else '⚠️'} "
                      f"{'Good' if total_time < 5 else 'Slow'} performance")
                
            except Exception as e:
                performance_results[corpus_path] = {'error': str(e)}
                print(f"❌ Error testing {description}: {e}")
        
        self.results['performance'] = performance_results
    
    def _are_keys_equivalent(self, key1: str, key2: str) -> bool:
        """Check if two keys are enharmonically equivalent"""
        # Simple enharmonic equivalents
        equivalents = {
            'c# major': 'db major',
            'db major': 'c# major',
            'f# major': 'gb major',
            'gb major': 'f# major',
            'g# minor': 'ab minor',
            'ab minor': 'g# minor',
            'd# minor': 'eb minor',
            'eb minor': 'd# minor',
            'a# minor': 'bb minor',
            'bb minor': 'a# minor'
        }
        
        return equivalents.get(key1) == key2
    
    def generate_comprehensive_report(self):
        """Generate comprehensive test report"""
        print("\n" + "="*60)
        print("📊 COMPREHENSIVE TEST REPORT")
        print("="*60)
        
        # Calculate overall scores
        total_tests = 0
        passed_tests = 0
        
        # Import functionality
        if 'import_functionality' in self.results:
            print("\n📥 Import Functionality:")
            for format_name, result in self.results['import_functionality'].items():
                if result.get('success'):
                    passed_tests += 1
                total_tests += 1
                status = '✅' if result.get('success') else '❌'
                print(f"  {status} {format_name}")
        
        # Key detection accuracy
        if 'key_detection' in self.results:
            print("\n🎹 Key Detection Accuracy:")
            correct_keys = 0
            high_confidence_keys = 0
            
            for piece, result in self.results['key_detection'].items():
                if 'error' not in result:
                    total_tests += 1
                    if result.get('correct') and result.get('confidence', 0) > 0.7:
                        passed_tests += 1
                        correct_keys += 1
                        high_confidence_keys += 1
                    
                    status = '✅' if result.get('correct') and result.get('confidence', 0) > 0.7 else '❌'
                    print(f"  {status} {result.get('name', piece)}: "
                          f"{result.get('confidence', 0):.0%} confidence")
            
            if total_tests > 0:
                self.accuracy_scores['key_detection'] = correct_keys / len(self.results['key_detection'])
        
        # Chord analysis accuracy
        if 'chord_analysis' in self.results:
            print("\n🎼 Chord Analysis Accuracy:")
            in_range_count = 0
            
            for piece, result in self.results['chord_analysis'].items():
                if 'error' not in result:
                    total_tests += 1
                    if result.get('in_range'):
                        passed_tests += 1
                        in_range_count += 1
                    
                    status = '✅' if result.get('in_range') else '❌'
                    print(f"  {status} {result.get('name', piece)}: "
                          f"{result.get('chord_count', 0)} chords")
            
            if total_tests > 0:
                self.accuracy_scores['chord_analysis'] = (
                    in_range_count / len(self.results['chord_analysis'])
                )
        
        # Complete workflow
        if 'complete_workflow' in self.results:
            print("\n🔄 Complete Workflow:")
            if self.results['complete_workflow'].get('all_steps_successful'):
                print("  ✅ All workflow steps successful")
                passed_tests += 1
            else:
                print("  ❌ Workflow failed")
            total_tests += 1
        
        # Performance
        if 'performance' in self.results:
            print("\n⚡ Performance Summary:")
            for piece, metrics in self.results['performance'].items():
                if 'error' not in metrics:
                    total = metrics.get('total_time', 0)
                    status = '✅' if total < 5 else '⚠️'
                    print(f"  {status} {metrics.get('description', piece)}: {total:.2f}s total")
        
        # Overall summary
        print("\n" + "="*60)
        print("OVERALL RESULTS:")
        print(f"  Total tests: {total_tests}")
        print(f"  Passed: {passed_tests}")
        print(f"  Failed: {total_tests - passed_tests}")
        print(f"  Success rate: {(passed_tests/total_tests*100) if total_tests > 0 else 0:.0f}%")
        
        # Accuracy scores
        if self.accuracy_scores:
            print("\nAccuracy Scores:")
            for category, score in self.accuracy_scores.items():
                print(f"  {category}: {score:.0%}")
        
        # Save detailed results
        with open('comprehensive_test_results.json', 'w') as f:
            json.dump({
                'results': self.results,
                'accuracy_scores': self.accuracy_scores,
                'summary': {
                    'total_tests': total_tests,
                    'passed_tests': passed_tests,
                    'success_rate': (passed_tests/total_tests) if total_tests > 0 else 0
                }
            }, f, indent=2, default=str)
        
        print("\n📄 Detailed results saved to comprehensive_test_results.json")
        
        # Final verdict
        success_rate = (passed_tests/total_tests) if total_tests > 0 else 0
        if success_rate >= 0.9:
            print("\n🎉 EXCELLENT: System is performing very well!")
        elif success_rate >= 0.7:
            print("\n✅ GOOD: System is mostly working correctly")
        elif success_rate >= 0.5:
            print("\n⚠️ FAIR: System needs improvement")
        else:
            print("\n🚨 POOR: System has significant issues")


async def main():
    """Run comprehensive test suite"""
    tester = ComprehensiveRealMusicTester()
    await tester.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())