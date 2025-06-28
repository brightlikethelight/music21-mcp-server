#!/usr/bin/env python3
"""
Basic Usage Example - Music21 MCP Server

This example demonstrates the fundamental workflow:
1. Import a score
2. Analyze key and chords
3. Get metadata
4. Export to different formats

Perfect for getting started with the server!
"""

import asyncio
import sys
from pathlib import Path

# Add src to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# Import the server tools
from music21_mcp.server import (
    import_score, analyze_key, analyze_chords, 
    get_score_info, export_score, delete_score
)


async def basic_workflow_example():
    """Demonstrate basic analysis workflow"""
    
    print("🎼 Music21 MCP Server - Basic Usage Example")
    print("=" * 50)
    
    # Step 1: Import a Bach chorale from the music21 corpus
    print("\n📥 Step 1: Importing Bach Chorale...")
    
    result = await import_score(
        score_id="bach_example",
        source="bach/bwv66.6"  # Famous Bach chorale
    )
    
    if result["status"] == "success":
        print(f"✅ Successfully imported Bach chorale")
        print(f"   Notes: {result['num_notes']}")
        print(f"   Measures: {result['num_measures']}")
        print(f"   Parts: {result['num_parts']}")
    else:
        print(f"❌ Import failed: {result['message']}")
        return
    
    # Step 2: Analyze the musical key
    print("\n🔑 Step 2: Analyzing Musical Key...")
    
    key_result = await analyze_key("bach_example")
    
    if key_result["status"] == "success":
        print(f"✅ Key detected: {key_result['key']}")
        print(f"   Confidence: {key_result['confidence']:.2%}")
        
        # Show alternatives if available
        if key_result.get('alternatives'):
            print("   Alternative keys:")
            for alt in key_result['alternatives'][:2]:
                print(f"     - {alt['key']} ({alt['confidence']:.2%})")
    else:
        print(f"❌ Key analysis failed: {key_result['message']}")
    
    # Step 3: Analyze chord progression
    print("\n🎹 Step 3: Analyzing Chord Progression...")
    
    chord_result = await analyze_chords(
        "bach_example", 
        include_roman_numerals=True
    )
    
    if chord_result["status"] == "success":
        print(f"✅ Found {chord_result['total_chords']} chords")
        print("   First few chords:")
        
        for i, chord in enumerate(chord_result['chord_progression'][:5]):
            pitches = ", ".join(chord['pitches'])
            roman = chord.get('roman_numeral', 'N/A')
            print(f"     {i+1}. {pitches} (Roman: {roman})")
    else:
        print(f"❌ Chord analysis failed: {chord_result['message']}")
    
    # Step 4: Get detailed score information
    print("\n📊 Step 4: Getting Score Metadata...")
    
    info_result = await get_score_info("bach_example")
    
    if info_result["status"] == "success":
        print(f"✅ Score Information:")
        print(f"   Title: {info_result.get('title', 'Unknown')}")
        print(f"   Composer: {info_result.get('composer', 'Unknown')}")
        print(f"   Duration: {info_result['duration_quarters']} quarter notes")
        
        # Show time signatures if present
        if info_result.get('time_signatures'):
            print(f"   Time Signature: {info_result['time_signatures'][0]['signature']}")
        
        # Show tempo if present
        if info_result.get('tempo_markings'):
            tempo = info_result['tempo_markings'][0]
            print(f"   Tempo: {tempo['bpm']} BPM")
    else:
        print(f"❌ Score info failed: {info_result['message']}")
    
    # Step 5: Export to different formats
    print("\n💾 Step 5: Exporting to Different Formats...")
    
    formats = ["midi", "musicxml"]
    
    for fmt in formats:
        export_result = await export_score("bach_example", fmt)
        
        if export_result["status"] == "success":
            file_path = export_result["file_path"]
            file_size = export_result["file_size"]
            print(f"✅ Exported {fmt.upper()}: {file_path} ({file_size} bytes)")
        else:
            print(f"❌ Export to {fmt} failed: {export_result['message']}")
    
    # Step 6: Clean up
    print("\n🧹 Step 6: Cleaning Up...")
    
    delete_result = await delete_score("bach_example")
    if delete_result["status"] == "success":
        print(f"✅ {delete_result['message']}")
    
    print("\n🎉 Basic workflow complete!")
    print("\nThis example showed:")
    print("  - How to import scores from the music21 corpus")
    print("  - Basic key detection and confidence scoring")
    print("  - Chord progression analysis with Roman numerals")
    print("  - Extracting score metadata and structure")
    print("  - Exporting to multiple formats")
    print("  - Proper cleanup and memory management")


async def text_import_example():
    """Demonstrate importing from text notation"""
    
    print("\n" + "=" * 50)
    print("📝 Bonus: Text Notation Import Example")
    print("=" * 50)
    
    # Create a simple C major scale
    scale_text = "C4 D4 E4 F4 G4 A4 B4 C5"
    
    print(f"\n📥 Importing text: '{scale_text}'")
    
    result = await import_score("scale_example", scale_text)
    
    if result["status"] == "success":
        print(f"✅ Successfully created scale with {result['num_notes']} notes")
        
        # Analyze the scale
        key_result = await analyze_key("scale_example")
        if key_result["status"] == "success":
            print(f"🔑 Detected key: {key_result['key']} (confidence: {key_result['confidence']:.2%})")
        
        # Export to MIDI for playback
        export_result = await export_score("scale_example", "midi")
        if export_result["status"] == "success":
            print(f"🎵 Exported playable MIDI: {export_result['file_path']}")
        
        # Cleanup
        await delete_score("scale_example")
        print("✅ Cleaned up scale example")
    else:
        print(f"❌ Text import failed: {result['message']}")


async def main():
    """Run all examples"""
    try:
        # Run the basic workflow
        await basic_workflow_example()
        
        # Run the text import example
        await text_import_example()
        
        print(f"\n🎯 Next Steps:")
        print(f"  - Try other examples in this directory")
        print(f"  - Experiment with your own MIDI/XML files")
        print(f"  - Check out the comprehensive API docs")
        print(f"  - Integrate with Claude Desktop for AI-powered analysis")
        
    except Exception as e:
        print(f"\n❌ Example failed with error: {e}")
        print(f"   Make sure music21 is installed: pip install music21")
        print(f"   Make sure the server package is installed: pip install -e .")
        return 1
    
    return 0


if __name__ == "__main__":
    # Run the example
    exit_code = asyncio.run(main())
    sys.exit(exit_code)