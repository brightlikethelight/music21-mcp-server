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

import sys
from pathlib import Path

# Add src to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# Import the server tools using the adapter
from music21_mcp.adapters import create_sync_analyzer


def basic_workflow_example():
    """Demonstrate basic analysis workflow"""
    
    print("🎼 Music21 MCP Server - Basic Usage Example")
    print("=" * 50)
    
    # Create a synchronous analyzer
    analyzer = create_sync_analyzer()
    
    # Step 1: Import a Bach chorale from the music21 corpus
    print("\n📥 Step 1: Importing Bach Chorale...")
    
    result = analyzer.import_score(
        score_id="bach_example",
        source="bach/bwv66.6",  # Famous Bach chorale
        source_type="corpus"
    )
    
    if result["status"] == "success":
        print("✅ Successfully imported Bach chorale")
        print(f"   Notes: {result['num_notes']}")
        print(f"   Measures: {result['num_measures']}")
        print(f"   Parts: {result['num_parts']}")
    else:
        print(f"❌ Import failed: {result['message']}")
        return
    
    # Step 2: Analyze the musical key
    print("\n🔑 Step 2: Analyzing Musical Key...")
    
    key_result = analyzer.analyze_key("bach_example")
    
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
    
    chord_result = analyzer.analyze_chords("bach_example")
    
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
    
    info_result = analyzer.get_score_info("bach_example")
    
    if info_result["status"] == "success":
        print("✅ Score Information:")
        print(f"   Title: {info_result.get('title', 'Unknown')}")
        print(f"   Composer: {info_result.get('composer', 'Unknown')}")
        print(f"   Time Signature: {info_result.get('time_signature', 'Unknown')}")
        print(f"   Duration: {info_result.get('duration_seconds', 0):.1f} seconds")
        print(f"   Ambitus: {info_result.get('lowest_note', '?')} - {info_result.get('highest_note', '?')}")
    else:
        print(f"❌ Info retrieval failed: {info_result['message']}")
    
    # Step 5: Export to MusicXML
    print("\n💾 Step 5: Exporting to MusicXML...")
    
    export_result = analyzer.export_score(
        "bach_example",
        format="musicxml"
    )
    
    if export_result["status"] == "success":
        print(f"✅ Exported to: {export_result['file_path']}")
        print(f"   Format: {export_result['format']}")
    else:
        print(f"❌ Export failed: {export_result['message']}")
    
    # Step 6: Clean up - delete the score from memory
    print("\n🧹 Step 6: Cleaning up...")
    
    delete_result = analyzer.delete_score("bach_example")
    
    if delete_result["status"] == "success":
        print("✅ Score removed from memory")
    else:
        print(f"❌ Cleanup failed: {delete_result['message']}")
    
    print("\n" + "=" * 50)
    print("✨ Basic workflow complete!")
    print("\nNext steps:")
    print("  - Try analyzing your own MIDI/MusicXML files")
    print("  - Explore harmony analysis and voice leading")
    print("  - Generate harmonizations and counterpoint")


# Run the example
if __name__ == "__main__":
    basic_workflow_example()