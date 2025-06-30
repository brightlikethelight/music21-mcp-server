#!/usr/bin/env python3
"""
Complete workflow example for Music21 MCP Server
Demonstrates all core functionality with a real musical piece
"""
import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from music21_mcp.server import (
    analyze_chords,
    analyze_key,
    delete_score,
    export_score,
    get_score_info,
    import_score,
    list_scores,
)


async def main():
    """Complete workflow demonstration"""
    print("🎵 Music21 MCP Server - Complete Workflow Example")
    print("="*60)
    
    # 1. Import multiple pieces
    print("\n1️⃣ Importing musical pieces...")
    pieces = {
        "bach_chorale": "bach/bwv66.6",
        "mozart_sonata": "mozart/k331/movement1",
        "simple_melody": "C4 E4 G4 C5 E5 G5 E5 C5"
    }
    
    for score_id, source in pieces.items():
        result = await import_score(score_id, source)
        if result["status"] == "success":
            print(f"   ✅ {score_id}: {result['num_notes']} notes")
        else:
            print(f"   ❌ {score_id}: {result['message']}")
    
    # 2. List all scores
    print("\n2️⃣ Listing all imported scores...")
    list_result = await list_scores()
    print(f"   Found {list_result['count']} scores:")
    for score in list_result["scores"]:
        print(f"   - {score['id']}: {score['measures']} measures, {score['parts']} parts")
    
    # 3. Analyze keys
    print("\n3️⃣ Analyzing musical keys...")
    for score_id in pieces.keys():
        key_result = await analyze_key(score_id)
        if key_result["status"] == "success":
            print(f"   {score_id}: {key_result['key']} (confidence: {key_result['confidence']:.1%})")
    
    # 4. Analyze chords (Bach chorale has the richest harmony)
    print("\n4️⃣ Analyzing chord progressions...")
    chord_result = await analyze_chords("bach_chorale")
    if chord_result["status"] == "success":
        print(f"   Total chords: {chord_result['total_chords']}")
        print(f"   First 5 chords: {', '.join(chord_result['chord_progression'][:5])}")
        if chord_result["roman_numerals"]:
            print(f"   Roman numerals: {', '.join(chord_result['roman_numerals'][:5])}")
    
    # 5. Get detailed information
    print("\n5️⃣ Getting detailed score information...")
    info_result = await get_score_info("mozart_sonata")
    if info_result["status"] == "success":
        print(f"   Title: {info_result['metadata'].get('movementName', 'Unknown')}")
        print(f"   Composer: {info_result['metadata'].get('composer', 'Unknown')}")
        print(f"   Duration: {info_result['duration_seconds']:.1f} seconds")
        print(f"   Tempo: {info_result['tempo_bpm']} BPM")
        print(f"   Time Signature: {info_result['time_signatures'][0] if info_result['time_signatures'] else 'Unknown'}")
    
    # 6. Export in different formats
    print("\n6️⃣ Exporting scores...")
    formats = ["midi", "musicxml", "abc"]
    for fmt in formats:
        export_result = await export_score("simple_melody", format=fmt)
        if export_result["status"] == "success":
            print(f"   ✅ Exported to {fmt}: {Path(export_result['file_path']).name}")
            # Clean up the file
            Path(export_result['file_path']).unlink()
    
    # 7. Demonstrate error handling
    print("\n7️⃣ Demonstrating error handling...")
    
    # Try to analyze non-existent score
    error_result = await analyze_key("does_not_exist")
    print(f"   Non-existent score: {error_result['message']}")
    
    # Try invalid format
    error_result = await export_score("bach_chorale", format="invalid")
    print(f"   Invalid format: {error_result['message']}")
    
    # 8. Clean up
    print("\n8️⃣ Cleaning up...")
    
    # Delete individual score
    delete_result = await delete_score("simple_melody")
    print(f"   Deleted simple_melody: {delete_result['status']}")
    
    # Delete all remaining scores
    delete_result = await delete_score("*")
    print(f"   Deleted {delete_result['deleted_count']} remaining scores")
    
    # Verify cleanup
    list_result = await list_scores()
    print(f"   Scores remaining: {list_result['count']}")
    
    print("\n✅ Workflow complete!")
    print("\nThis example demonstrated:")
    print("- Importing from corpus, files, and text notation")
    print("- Listing and managing multiple scores")
    print("- Analyzing keys and chord progressions")
    print("- Extracting detailed metadata")
    print("- Exporting to multiple formats")
    print("- Proper error handling")
    print("- Memory management with cleanup")


if __name__ == "__main__":
    asyncio.run(main())