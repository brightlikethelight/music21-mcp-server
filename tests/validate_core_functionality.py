#!/usr/bin/env python3
"""
Core functionality validation - Test what actually works
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import asyncio
from music21 import stream, note, chord, key, meter, tempo, corpus
from music21_mcp.server import score_manager
import time
import json


async def validate_basic_functionality():
    """Test the most basic functionality"""
    print("🔍 Core Functionality Validation")
    print("="*60)
    
    results = {}
    
    # Test 1: Can we create and store a score?
    print("\n1. Score Creation and Storage:")
    try:
        # Create simple score
        test_score = stream.Score()
        part = stream.Part()
        part.append(meter.TimeSignature('4/4'))
        part.append(key.KeySignature(0))  # C major
        
        # Add C major scale
        for pitch in ['C4', 'D4', 'E4', 'F4', 'G4', 'A4', 'B4', 'C5']:
            part.append(note.Note(pitch, quarterLength=1))
        
        test_score.append(part)
        
        # Store in score manager
        score_manager.add_score("test_scale", test_score, {"title": "C Major Scale"})
        
        # Retrieve it
        retrieved = score_manager.get_score("test_scale")
        if retrieved:
            print("   ✅ Score storage works")
            results['score_storage'] = True
        else:
            print("   ❌ Score retrieval failed")
            results['score_storage'] = False
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
        results['score_storage'] = False
    
    # Test 2: Import function from server
    print("\n2. Import Function Test:")
    try:
        from music21_mcp.server import import_score
        
        # Test text import (simplest case)
        result = await import_score(
            score_id="text_import_test",
            source="C D E F G",
            source_type="text"
        )
        
        if result.get('status') == 'success':
            print("   ✅ Text import works")
            results['text_import'] = True
        else:
            print(f"   ❌ Import failed: {result}")
            results['text_import'] = False
            
    except Exception as e:
        print(f"   ❌ Import error: {e}")
        results['text_import'] = False
    
    # Test 3: Basic analysis functions
    print("\n3. Analysis Functions:")
    
    # Key analysis
    try:
        from music21_mcp.server import analyze_key
        key_result = await analyze_key("test_scale")
        
        if key_result.get('status') == 'success':
            detected_key = key_result.get('key', 'Unknown')
            print(f"   ✅ Key analysis works: {detected_key}")
            results['key_analysis'] = True
        else:
            print(f"   ❌ Key analysis failed: {key_result}")
            results['key_analysis'] = False
            
    except Exception as e:
        print(f"   ❌ Key analysis error: {e}")
        results['key_analysis'] = False
    
    # Chord progression analysis
    try:
        from music21_mcp.server import analyze_chord_progressions
        
        # Create a score with chords
        chord_score = stream.Score()
        part = stream.Part()
        
        # Simple I-IV-V-I progression in C
        chords = [
            chord.Chord(['C4', 'E4', 'G4']),     # I
            chord.Chord(['F4', 'A4', 'C5']),     # IV
            chord.Chord(['G4', 'B4', 'D5']),     # V
            chord.Chord(['C4', 'E4', 'G4']),     # I
        ]
        
        for ch in chords:
            ch.quarterLength = 2
            part.append(ch)
        
        chord_score.append(part)
        score_manager.add_score("chord_test", chord_score, {})
        
        chord_result = await analyze_chord_progressions("chord_test")
        
        if chord_result.get('status') == 'success':
            print(f"   ✅ Chord analysis works: {chord_result.get('total_chords', 0)} chords")
            results['chord_analysis'] = True
        else:
            print(f"   ❌ Chord analysis failed")
            results['chord_analysis'] = False
            
    except Exception as e:
        print(f"   ❌ Chord analysis error: {e}")
        results['chord_analysis'] = False
    
    # Test 4: Check available corpus works
    print("\n4. Music21 Corpus Check:")
    try:
        # Get some Bach works
        bach_works = corpus.search('bach', 'composer')
        if bach_works:
            print(f"   ✅ Found {len(bach_works)} Bach works")
            # Try to parse one
            first_work = bach_works[0]
            parsed = first_work.parse()
            print(f"   ✅ Successfully parsed: {first_work.metadata.title}")
            results['corpus_access'] = True
        else:
            print("   ❌ No Bach works found")
            results['corpus_access'] = False
            
    except Exception as e:
        print(f"   ❌ Corpus error: {e}")
        results['corpus_access'] = False
    
    # Test 5: Performance check
    print("\n5. Performance Test:")
    try:
        # Create larger score
        large_score = stream.Score()
        part = stream.Part()
        
        # 100 measures of notes
        for m in range(100):
            measure = stream.Measure(number=m+1)
            for beat in range(4):
                measure.append(note.Note('C4', quarterLength=1))
            part.append(measure)
        
        large_score.append(part)
        score_manager.add_score("large_test", large_score, {})
        
        # Time analysis
        start = time.time()
        key_result = await analyze_key("large_test")
        duration = time.time() - start
        
        print(f"   Key analysis of 100 measures: {duration:.2f}s")
        if duration < 5:
            print("   ✅ Good performance")
            results['performance'] = True
        else:
            print("   ⚠️ Slow performance")
            results['performance'] = False
            
    except Exception as e:
        print(f"   ❌ Performance test error: {e}")
        results['performance'] = False
    
    # Summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY:")
    print("="*60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test, passed_test in results.items():
        print(f"{'✅' if passed_test else '❌'} {test}")
    
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.0f}%)")
    
    # Save results
    with open('validation_results.json', 'w') as f:
        json.dump({
            'results': results,
            'passed': passed,
            'total': total,
            'percentage': passed/total*100
        }, f, indent=2)
    
    return results


async def test_real_music_if_available():
    """Try to test with real music if corpus is accessible"""
    print("\n\n📚 Real Music Testing (if available)")
    print("="*60)
    
    # Try different corpus paths
    test_pieces = [
        ('bwv66.6', 'Bach Chorale'),
        ('bach/bwv66.6', 'Bach Chorale (alt path)'),
        ('opus74no1', 'Haydn String Quartet'),
        ('mozart/k545', 'Mozart Sonata'),
    ]
    
    found_pieces = []
    
    for path, name in test_pieces:
        try:
            score = corpus.parse(path)
            print(f"✅ Found: {name} ({path})")
            found_pieces.append((path, name, score))
        except:
            # Try search
            try:
                results = corpus.search(path)
                if results:
                    score = results[0].parse()
                    print(f"✅ Found via search: {name}")
                    found_pieces.append((path, name, score))
            except:
                print(f"❌ Not found: {name} ({path})")
    
    # Test any found pieces
    for path, name, score in found_pieces[:2]:  # Test first 2
        print(f"\n🎼 Testing {name}:")
        
        # Store it
        score_id = path.replace('/', '_')
        score_manager.add_score(score_id, score, {"title": name})
        
        # Analyze
        try:
            from music21_mcp.server import analyze_key, analyze_chord_progressions
            
            key_result = await analyze_key(score_id)
            if key_result.get('status') == 'success':
                print(f"   Key: {key_result.get('key')} (confidence: {key_result.get('confidence', 0):.2f})")
            
            chord_result = await analyze_chord_progressions(score_id)
            if chord_result.get('status') == 'success':
                print(f"   Chords: {chord_result.get('total_chords', 0)}")
                if chord_result.get('chord_progressions'):
                    prog = chord_result['chord_progressions'][0]
                    print(f"   First progression: {' → '.join(prog['progression'][:5])}")
                    
        except Exception as e:
            print(f"   Error analyzing: {e}")


async def main():
    """Run all validations"""
    # Basic validation
    results = await validate_basic_functionality()
    
    # Try real music if possible
    await test_real_music_if_available()
    
    print("\n✅ Validation complete. Check validation_results.json for details.")


if __name__ == "__main__":
    asyncio.run(main())