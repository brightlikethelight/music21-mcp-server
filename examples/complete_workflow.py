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

# Import the adapter for proper async usage
from music21_mcp.adapters import create_music_analyzer


async def main():
    """Complete workflow demonstration"""
    print("🎵 Music21 MCP Server - Complete Workflow Example")
    print("="*60)
    
    # Create the music analyzer
    analyzer = create_music_analyzer()
    
    try:
        # 1. Import multiple pieces
        print("\n1️⃣ Importing musical pieces...")
        pieces = {
            "bach_chorale": ("bach/bwv66.6", "corpus"),
            "beethoven_sonata": ("beethoven/opus27no1", "corpus"),
            "simple_melody": ("C4 E4 G4 C5 E5 G5 E5 C5", "text")
        }
        
        for score_id, (source, source_type) in pieces.items():
            result = await analyzer.import_score(score_id, source, source_type)
            if result["status"] == "success":
                print(f"   ✅ {score_id}: {result.get('num_notes', 0)} notes")
            else:
                print(f"   ❌ {score_id}: {result.get('message', 'Import failed')}")
        
        # 2. List all scores
        print("\n2️⃣ Listing all imported scores...")
        list_result = await analyzer.list_scores()
        if list_result["status"] == "success":
            scores = list_result.get('scores', [])
            print(f"   Found {len(scores)} scores:")
            for score in scores:
                # Handle different possible formats
                score_id = score.get('id') or score.get('score_id', 'Unknown')
                measures = score.get('measures', 0)
                parts = score.get('parts', 0)
                title = score.get('title', 'Untitled')
                print(f"   - {score_id}: {measures} measures, {parts} parts ({title})")
        else:
            print(f"   ❌ Failed to list scores: {list_result.get('message', 'Unknown error')}")
        
        # 3. Analyze keys
        print("\n3️⃣ Analyzing musical keys...")
        for score_id in pieces.keys():
            key_result = await analyzer.analyze_key(score_id)
            if key_result["status"] == "success":
                key = key_result.get('key', 'Unknown')
                confidence = key_result.get('confidence', 0)
                print(f"   {score_id}: {key} (confidence: {confidence:.1%})")
            else:
                print(f"   ❌ {score_id}: {key_result.get('message', 'Analysis failed')}")
        
        # 4. Analyze chords (Bach chorale has the richest harmony)
        print("\n4️⃣ Analyzing chord progressions...")
        chord_result = await analyzer.analyze_chords("bach_chorale")
        if chord_result["status"] == "success":
            total_chords = chord_result.get('total_chords', 0)
            chord_progression = chord_result.get('chord_progression', [])
            print(f"   Total chords: {total_chords}")
            
            # Show first few chords with better formatting
            if chord_progression:
                print("   First 5 chords:")
                for i, chord in enumerate(chord_progression[:5], 1):
                    if isinstance(chord, dict):
                        symbol = chord.get('symbol', 'Unknown')
                        roman = chord.get('roman_numeral', '?')
                        print(f"     {i}. {symbol} (Roman: {roman})")
                    else:
                        print(f"     {i}. {str(chord)}")
        else:
            print(f"   ❌ Chord analysis failed: {chord_result.get('message', 'Unknown error')}")
        
        # 5. Get detailed information
        print("\n5️⃣ Getting detailed score information...")
        info_result = await analyzer.get_score_info("bach_chorale")
        if info_result["status"] == "success":
            title = info_result.get('title', 'Unknown')
            composer = info_result.get('composer', 'Unknown')
            duration = info_result.get('duration_seconds', 0)
            time_sig = info_result.get('time_signature', 'Unknown')
            print(f"   Title: {title}")
            print(f"   Composer: {composer}")
            print(f"   Duration: {duration:.1f} seconds")
            print(f"   Time Signature: {time_sig}")
        else:
            print(f"   ❌ Info retrieval failed: {info_result.get('message', 'Unknown error')}")
        
        # 6. Export in different formats
        print("\n6️⃣ Exporting scores...")
        formats = ["midi", "musicxml", "abc"]
        for fmt in formats:
            export_result = await analyzer.export_score("bach_chorale", format=fmt)
            if export_result["status"] == "success":
                file_path = export_result.get('file_path', '')
                print(f"   ✅ Exported to {fmt}: {Path(file_path).name if file_path else 'success'}")
                # Clean up the file if it exists
                if file_path and Path(file_path).exists():
                    Path(file_path).unlink()
            else:
                print(f"   ❌ Export to {fmt} failed: {export_result.get('message', 'Unknown error')}")
        
        # 7. Demonstrate error handling
        print("\n7️⃣ Demonstrating error handling...")
        
        # Try to analyze non-existent score
        error_result = await analyzer.analyze_key("does_not_exist")
        print(f"   Non-existent score: {error_result.get('message', 'Error occurred')}")
        
        # Try invalid format
        error_result = await analyzer.export_score("bach_chorale", format="invalid")
        print(f"   Invalid format: {error_result.get('message', 'Error occurred')}")
        
        # 8. Clean up
        print("\n8️⃣ Cleaning up...")
        
        # Delete individual scores
        for score_id in pieces.keys():
            delete_result = await analyzer.delete_score(score_id)
            status = delete_result.get('status', 'error')
            print(f"   Deleted {score_id}: {status}")
        
        # Verify cleanup
        list_result = await analyzer.list_scores()
        if list_result["status"] == "success":
            scores = list_result.get('scores', [])
            print(f"   Scores remaining: {len(scores)}")
        
    except Exception as e:
        print(f"\n❌ Workflow failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False
    
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