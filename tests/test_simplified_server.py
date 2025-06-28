#!/usr/bin/env python3
"""
Test the simplified server to ensure basic functionality works
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from music21_mcp.server_simple import (
    import_score, analyze_key, analyze_chords, 
    get_score_info, export_score, scores
)


async def test_basic_functionality():
    """Test all basic functions work without crashes"""
    print("🧪 Testing Simplified Server")
    print("="*50)
    
    all_passed = True
    
    # Test 1: Import corpus file
    print("\n1. Testing corpus import...")
    try:
        result = await import_score("bach_test", "bach/bwv66.6")
        if result['status'] == 'success':
            print(f"   ✅ Import successful: {result['num_notes']} notes, {result['num_measures']} measures")
        else:
            print(f"   ❌ Import failed: {result}")
            all_passed = False
    except Exception as e:
        print(f"   ❌ Import crashed: {e}")
        all_passed = False
    
    # Test 2: Key analysis
    print("\n2. Testing key analysis...")
    try:
        result = await analyze_key("bach_test")
        if result['status'] == 'success':
            print(f"   ✅ Key detected: {result['key']} (confidence: {result['confidence']:.2f})")
            if result['confidence'] < 0.3:
                print(f"   ⚠️  Warning: Low confidence")
        else:
            print(f"   ❌ Key analysis failed: {result}")
            all_passed = False
    except Exception as e:
        print(f"   ❌ Key analysis crashed: {e}")
        all_passed = False
    
    # Test 3: Chord analysis
    print("\n3. Testing chord analysis...")
    try:
        result = await analyze_chords("bach_test")
        if result['status'] == 'success':
            print(f"   ✅ Chords found: {result['chord_count']}")
            if result['chord_count'] < 10:
                print(f"   ⚠️  Warning: Few chords found")
                all_passed = False
            if result.get('sample_progression'):
                # Handle list of lists for chord progression
                try:
                    prog_str = []
                    for chord_notes in result['sample_progression'][:3]:
                        if isinstance(chord_notes, list):
                            prog_str.append('[' + ','.join(chord_notes) + ']')
                        else:
                            prog_str.append(str(chord_notes))
                    print(f"   Sample: {' -> '.join(prog_str)}")
                except:
                    pass
        else:
            print(f"   ❌ Chord analysis failed: {result}")
            all_passed = False
    except Exception as e:
        print(f"   ❌ Chord analysis crashed: {e}")
        all_passed = False
    
    # Test 4: Score info
    print("\n4. Testing score info...")
    try:
        result = await get_score_info("bach_test")
        if result['status'] == 'success':
            print(f"   ✅ Score info retrieved:")
            print(f"      Title: {result['title']}")
            print(f"      Composer: {result['composer']}")
            print(f"      Parts: {result['num_parts']}")
        else:
            print(f"   ❌ Score info failed: {result}")
            all_passed = False
    except Exception as e:
        print(f"   ❌ Score info crashed: {e}")
        all_passed = False
    
    # Test 5: Export
    print("\n5. Testing export...")
    try:
        result = await export_score("bach_test", format="musicxml")
        if result['status'] == 'success':
            print(f"   ✅ Export successful: {result['file_path']}")
            # Check if file exists
            import os
            if os.path.exists(result['file_path']):
                file_size = os.path.getsize(result['file_path'])
                print(f"      File size: {file_size} bytes")
                os.unlink(result['file_path'])  # Clean up
            else:
                print(f"   ❌ Export file not created")
                all_passed = False
        else:
            print(f"   ❌ Export failed: {result}")
            all_passed = False
    except Exception as e:
        print(f"   ❌ Export crashed: {e}")
        all_passed = False
    
    # Test 6: Text import
    print("\n6. Testing text import...")
    try:
        result = await import_score("text_test", "C4 D4 E4 F4 G4")
        if result['status'] == 'success':
            print(f"   ✅ Text import successful: {result['num_notes']} notes")
        else:
            print(f"   ❌ Text import failed: {result}")
            all_passed = False
    except Exception as e:
        print(f"   ❌ Text import crashed: {e}")
        all_passed = False
    
    # Summary
    print("\n" + "="*50)
    if all_passed:
        print("✅ ALL TESTS PASSED! Basic functionality is working.")
    else:
        print("❌ SOME TESTS FAILED. Check the errors above.")
    
    return all_passed


async def test_edge_cases():
    """Test edge cases and error handling"""
    print("\n\n🔧 Testing Edge Cases")
    print("="*50)
    
    # Test non-existent score
    print("\n1. Testing non-existent score...")
    result = await analyze_key("nonexistent")
    if result['status'] == 'error':
        print("   ✅ Correctly handled missing score")
    else:
        print("   ❌ Should have returned error")
    
    # Test invalid import
    print("\n2. Testing invalid import...")
    result = await import_score("bad_import", "this_file_does_not_exist.xyz")
    if result['status'] == 'error':
        print("   ✅ Correctly handled invalid import")
    else:
        print("   ❌ Should have returned error")
    
    # Test empty score name
    print("\n3. Testing empty score analysis...")
    # Create an empty score
    from music21 import stream
    empty_score = stream.Score()
    scores["empty"] = empty_score
    
    result = await analyze_chords("empty")
    print(f"   Empty score chords: {result.get('chord_count', 'N/A')}")
    
    result = await analyze_key("empty")
    print(f"   Empty score key: {result.get('key', 'N/A')}")


async def test_multiple_pieces():
    """Test with multiple different pieces"""
    print("\n\n🎼 Testing Multiple Pieces")
    print("="*50)
    
    test_pieces = [
        ("bach/bwv7.7", "Bach Chorale BWV 7.7"),
        ("mozart/k545", "Mozart K. 545"),
        ("schoenberg/opus19/movement2", "Schoenberg Op. 19"),
    ]
    
    for corpus_path, name in test_pieces:
        print(f"\nTesting {name}...")
        try:
            # Import
            result = await import_score(corpus_path.replace('/', '_'), corpus_path)
            if result['status'] != 'success':
                print(f"   ⏭️  Skipping - not in corpus")
                continue
            
            # Analyze
            key_result = await analyze_key(corpus_path.replace('/', '_'))
            chord_result = await analyze_chords(corpus_path.replace('/', '_'))
            
            print(f"   Key: {key_result.get('key', 'Unknown')} "
                  f"(confidence: {key_result.get('confidence', 0):.2f})")
            print(f"   Chords: {chord_result.get('chord_count', 0)}")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")


async def main():
    """Run all tests"""
    # Basic tests
    basic_passed = await test_basic_functionality()
    
    # Edge cases
    await test_edge_cases()
    
    # Multiple pieces
    await test_multiple_pieces()
    
    print("\n" + "="*50)
    print("🏁 Testing Complete")
    
    if basic_passed:
        print("\n✅ Core functionality is stable. Safe to proceed.")
        return 0
    else:
        print("\n❌ Core functionality has issues. Fix before proceeding.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)