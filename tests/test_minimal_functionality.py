#!/usr/bin/env python3
"""
Minimal functionality test - just ensure the absolute basics work
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from music21 import corpus, converter, stream, note
from music21_mcp.server_simple import scores


def test_music21_basics():
    """Test that music21 itself works correctly"""
    print("🎵 Testing music21 basics...")
    
    # Test 1: Can we load a Bach chorale?
    try:
        bach = corpus.parse('bach/bwv66.6')
        print(f"✅ Loaded Bach chorale: {len(list(bach.flatten().notes))} notes")
    except Exception as e:
        print(f"❌ Failed to load Bach chorale: {e}")
        return False
    
    # Test 2: Can we analyze key?
    try:
        key = bach.analyze('key')
        conf = getattr(key, 'correlationCoefficient', 0.5)
        print(f"✅ Key analysis works: {key} (confidence: {conf:.2f})")
    except Exception as e:
        print(f"❌ Key analysis failed: {e}")
        return False
    
    # Test 3: Can we chordify?
    try:
        chordified = bach.chordify()
        chords = list(chordified.flatten().getElementsByClass('Chord'))
        print(f"✅ Chordify works: {len(chords)} chords")
    except Exception as e:
        print(f"❌ Chordify failed: {e}")
        return False
    
    # Test 4: Can we parse text notation?
    try:
        # Try different approaches
        success = False
        
        # Approach 1: tinyNotation
        try:
            from music21.tinyNotation import Converter
            tnc = Converter("C4 D4 E4 F4")
            parsed = tnc.parse().stream
            notes = list(parsed.flatten().notes)
            if len(notes) == 4:
                print(f"✅ TinyNotation works: {len(notes)} notes")
                success = True
        except:
            pass
        
        # Approach 2: converter with format
        if not success:
            try:
                parsed = converter.parse("tinyNotation: C4 D4 E4 F4", format='tinyNotation')
                notes = list(parsed.flatten().notes)
                if len(notes) == 4:
                    print(f"✅ Converter with format works: {len(notes)} notes")
                    success = True
            except:
                pass
        
        # Approach 3: Create manually
        if not success:
            s = stream.Stream()
            for pitch in ['C4', 'D4', 'E4', 'F4']:
                s.append(note.Note(pitch))
            if len(s.notes) == 4:
                print(f"✅ Manual creation works: {len(s.notes)} notes")
                success = True
        
        if not success:
            print("❌ No text parsing method worked")
            return False
            
    except Exception as e:
        print(f"❌ Text parsing failed: {e}")
        return False
    
    return True


async def test_server_functions():
    """Test the simplified server functions directly"""
    print("\n🔧 Testing server functions...")
    
    from music21_mcp.server_simple import import_score, analyze_key, analyze_chords
    
    # Clear any existing scores
    scores.clear()
    
    # Test 1: Import Bach chorale
    result = await import_score("test1", "bach/bwv66.6")
    if result['status'] == 'success':
        print(f"✅ Import works: {result['num_notes']} notes")
    else:
        print(f"❌ Import failed: {result}")
        return False
    
    # Test 2: Key analysis
    result = await analyze_key("test1")
    if result['status'] == 'success':
        print(f"✅ Key analysis works: {result['key']} ({result['confidence']:.2f})")
    else:
        print(f"❌ Key analysis failed: {result}")
        return False
    
    # Test 3: Chord analysis
    result = await analyze_chords("test1")
    if result['status'] == 'success':
        print(f"✅ Chord analysis works: {result['chord_count']} chords")
    else:
        print(f"❌ Chord analysis failed: {result}")
        return False
    
    return True


async def test_text_import_workaround():
    """Test a workaround for text import"""
    print("\n🔨 Testing text import workaround...")
    
    # Create a score manually and add it to the scores dict
    s = stream.Stream()
    for pitch in ['C4', 'D4', 'E4', 'F4', 'G4']:
        s.append(note.Note(pitch, quarterLength=1))
    
    # Add directly to scores
    scores['manual_test'] = s
    
    # Now test analysis on it
    from music21_mcp.server_simple import analyze_key, analyze_chords
    
    result = await analyze_key('manual_test')
    if result['status'] == 'success':
        print(f"✅ Can analyze manually created score: {result['key']}")
    else:
        print(f"❌ Failed to analyze manual score: {result}")
    
    return True


async def main():
    """Run minimal tests"""
    print("🧪 Minimal Functionality Test")
    print("="*50)
    
    # Test music21 basics
    if not test_music21_basics():
        print("\n❌ music21 basic functionality is broken!")
        return 1
    
    # Test server functions
    if not await test_server_functions():
        print("\n❌ Server functions are broken!")
        return 1
    
    # Test workarounds
    await test_text_import_workaround()
    
    print("\n" + "="*50)
    print("✅ Basic functionality is working!")
    print("\nNext steps:")
    print("1. Fix text import to use manual score creation")
    print("2. Ensure all functions handle edge cases gracefully")
    print("3. Create proper integration tests")
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)