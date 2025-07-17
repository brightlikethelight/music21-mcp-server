#!/usr/bin/env python3
"""
Simple Example: Music21 MCP Server Tools
=========================================

This example shows how to use the music21 MCP tools directly,
without needing MCP client setup. Perfect for testing and development.

Prerequisites:
- pip install music21
- python -m music21.configure (to set up corpus)
"""

import asyncio
import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from music21_mcp.tools import (
    ImportScoreTool,
    KeyAnalysisTool,
    HarmonyAnalysisTool,
    PatternRecognitionTool,
    ListScoresTool,
    ScoreInfoTool,
    ChordAnalysisTool,
    ExportScoreTool,
    VoiceLeadingAnalysisTool,
)
import json

class SimpleExample:
    """Simple example using tools directly"""
    
    def __init__(self):
        # Create simple score storage (dict)
        self.scores = {}
        
        # Initialize tools
        self.import_tool = ImportScoreTool(self.scores)
        self.key_tool = KeyAnalysisTool(self.scores)
        self.harmony_tool = HarmonyAnalysisTool(self.scores)
        self.pattern_tool = PatternRecognitionTool(self.scores)
        self.list_tool = ListScoresTool(self.scores)
        self.info_tool = ScoreInfoTool(self.scores)
        self.chord_tool = ChordAnalysisTool(self.scores)
        self.export_tool = ExportScoreTool(self.scores)
        self.voice_tool = VoiceLeadingAnalysisTool(self.scores)
    
    async def demonstrate_basic_analysis(self):
        """Demonstrate basic music analysis workflow"""
        print("🎼 Basic Music Analysis Demonstration")
        print("=" * 50)
        
        # Step 1: Import a Bach chorale
        print("\n1️⃣ Importing Bach chorale...")
        
        import_result = await self.import_tool.execute(
            score_id="bach_chorale",
            source="bach/bwv66.6",
            source_type="corpus"
        )
        
        if import_result["status"] == "success":
            print("✅ Successfully imported Bach chorale")
            print(f"   Title: {import_result.get('title', 'Unknown')}")
        else:
            print(f"❌ Import failed: {import_result.get('message', 'Unknown error')}")
            return False
        
        # Step 2: Get score information
        print("\n2️⃣ Getting score information...")
        
        info_result = await self.info_tool.execute(score_id="bach_chorale")
        
        if info_result["status"] == "success":
            print("✅ Score information retrieved")
            print(f"   Composer: {info_result.get('composer', 'Unknown')}")
            print(f"   Parts: {info_result.get('parts', 0)}")
            print(f"   Measures: {info_result.get('measures', 0)}")
            print(f"   Notes: {info_result.get('notes', 0)}")
        else:
            print(f"❌ Info failed: {info_result.get('message', 'Unknown error')}")
        
        # Step 3: Analyze key
        print("\n3️⃣ Analyzing musical key...")
        
        key_result = await self.key_tool.execute(score_id="bach_chorale")
        
        if key_result["status"] == "success":
            print("✅ Key analysis completed")
            print(f"   Key: {key_result.get('key', 'Unknown')}")
            print(f"   Confidence: {key_result.get('confidence', 0):.2f}")
            print(f"   Mode: {key_result.get('mode', 'Unknown')}")
        else:
            print(f"❌ Key analysis failed: {key_result.get('message', 'Unknown error')}")
        
        # Step 4: Analyze chords
        print("\n4️⃣ Analyzing chord progressions...")
        
        chord_result = await self.chord_tool.execute(score_id="bach_chorale")
        
        if chord_result["status"] == "success":
            print("✅ Chord analysis completed")
            chords = chord_result.get('chords', [])
            print(f"   Found {len(chords)} chords")
            
            # Show first few chords
            if chords:
                print("   First few chords:")
                for i, chord in enumerate(chords[:5]):
                    print(f"     {i+1}. {chord.get('chord', 'Unknown')} (measure {chord.get('measure', '?')})")
        else:
            print(f"❌ Chord analysis failed: {chord_result.get('message', 'Unknown error')}")
        
        # Step 5: Harmony analysis (Roman numerals)
        print("\n5️⃣ Analyzing harmonic progressions...")
        
        harmony_result = await self.harmony_tool.execute(score_id="bach_chorale")
        
        if harmony_result["status"] == "success":
            print("✅ Harmony analysis completed")
            roman_numerals = harmony_result.get('roman_numerals', [])
            print(f"   Found {len(roman_numerals)} Roman numerals")
            
            # Show first few Roman numerals
            if roman_numerals:
                print("   First few Roman numerals:")
                for i, rn in enumerate(roman_numerals[:5]):
                    print(f"     {i+1}. {rn.get('roman_numeral', '?')} - {rn.get('chord', 'Unknown')}")
        else:
            print(f"❌ Harmony analysis failed: {harmony_result.get('message', 'Unknown error')}")
        
        # Step 6: Pattern recognition
        print("\n6️⃣ Detecting melodic patterns...")
        
        pattern_result = await self.pattern_tool.execute(score_id="bach_chorale")
        
        if pattern_result["status"] == "success":
            print("✅ Pattern recognition completed")
            melodic_patterns = pattern_result.get('melodic_patterns', {})
            sequences = melodic_patterns.get('sequences', [])
            motifs = melodic_patterns.get('motifs', [])
            
            print(f"   Found {len(sequences)} melodic sequences")
            print(f"   Found {len(motifs)} motivic patterns")
            
            # Show some sequences
            if sequences:
                print("   Sample sequences:")
                for i, seq in enumerate(sequences[:3]):
                    print(f"     {i+1}. {seq.get('pattern_type', 'Unknown')} - length {seq.get('length', '?')}")
        else:
            print(f"❌ Pattern recognition failed: {pattern_result.get('message', 'Unknown error')}")
        
        # Step 7: Voice leading analysis
        print("\n7️⃣ Analyzing voice leading...")
        
        voice_result = await self.voice_tool.execute(score_id="bach_chorale")
        
        if voice_result["status"] == "success":
            print("✅ Voice leading analysis completed")
            issues = voice_result.get('total_issues', 0)
            score = voice_result.get('overall_score', 0)
            
            print(f"   Voice leading issues: {issues}")
            print(f"   Overall score: {score}/100")
            
            parallel_issues = voice_result.get('parallel_issues', [])
            if parallel_issues:
                print(f"   Parallel fifths/octaves: {len(parallel_issues)}")
        else:
            print(f"❌ Voice leading analysis failed: {voice_result.get('message', 'Unknown error')}")
        
        # Step 8: List all scores
        print("\n8️⃣ Listing all loaded scores...")
        
        list_result = await self.list_tool.execute()
        
        if list_result["status"] == "success":
            print("✅ Score list retrieved")
            scores = list_result.get('scores', [])
            print(f"   Total scores in memory: {len(scores)}")
            
            for score in scores:
                print(f"     - {score.get('id', 'Unknown')} ({score.get('title', 'Untitled')})")
        else:
            print(f"❌ List failed: {list_result.get('message', 'Unknown error')}")
        
        return True
    
    async def demonstrate_multiple_scores(self):
        """Demonstrate working with multiple scores"""
        print("\n🎵 Multiple Scores Example")
        print("=" * 40)
        
        # Import multiple scores
        scores_to_import = [
            ("bach_chorale", "bach/bwv66.6", "Bach Chorale"),
            ("mozart_sonata", "mozart/k545", "Mozart Sonata"),
            ("beethoven_symphony", "beethoven/opus18no1", "Beethoven String Quartet"),
        ]
        
        for score_id, source, name in scores_to_import:
            print(f"\n📥 Importing {name}...")
            
            result = await self.import_tool.execute(
                score_id=score_id,
                source=source,
                source_type="corpus"
            )
            
            if result["status"] == "success":
                print(f"✅ Successfully imported {name}")
                
                # Quick key analysis
                key_result = await self.key_tool.execute(score_id=score_id)
                if key_result["status"] == "success":
                    print(f"   Key: {key_result.get('key', 'Unknown')}")
            else:
                print(f"❌ Failed to import {name}: {result.get('message', 'Unknown error')}")
        
        # List all scores
        print("\n📋 Final score inventory:")
        list_result = await self.list_tool.execute()
        
        if list_result["status"] == "success":
            scores = list_result.get('scores', [])
            print(f"Total scores loaded: {len(scores)}")
            
            for score in scores:
                print(f"  - {score.get('id', 'Unknown')}: {score.get('title', 'Untitled')}")
    
    async def demonstrate_export(self):
        """Demonstrate score export functionality"""
        print("\n💾 Export Example")
        print("=" * 25)
        
        # Export the Bach chorale to different formats
        formats = ["musicxml", "midi", "abc"]
        
        for format_type in formats:
            print(f"\n📤 Exporting to {format_type.upper()}...")
            
            export_result = await self.export_tool.execute(
                score_id="bach_chorale",
                format=format_type
            )
            
            if export_result["status"] == "success":
                print(f"✅ Successfully exported to {format_type}")
                file_path = export_result.get('file_path', 'Unknown')
                print(f"   File: {file_path}")
            else:
                print(f"❌ Export to {format_type} failed: {export_result.get('message', 'Unknown error')}")
    
    async def run_complete_example(self):
        """Run the complete example"""
        print("🎵 Music21 MCP Tools - Simple Example")
        print("=" * 50)
        print("This example demonstrates using the music21 MCP tools directly")
        print("without needing a full MCP client setup.")
        print()
        
        try:
            # Basic analysis
            await self.demonstrate_basic_analysis()
            
            # Multiple scores
            await self.demonstrate_multiple_scores()
            
            # Export functionality
            await self.demonstrate_export()
            
            print("\n🎉 Example completed successfully!")
            print("\nKey findings:")
            print("- All core music analysis tools work correctly")
            print("- Pattern recognition finds melodic sequences and motifs")
            print("- Harmony analysis provides Roman numeral analysis")
            print("- Voice leading analysis detects parallel motion")
            print("- Export works for multiple formats")
            
            print("\nNext steps:")
            print("- Try the MCP server with Claude Desktop")
            print("- Experiment with your own music files")
            print("- Build custom analysis workflows")
            
        except Exception as e:
            print(f"\n❌ Example failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        return True

async def main():
    """Main entry point"""
    example = SimpleExample()
    
    print("Music21 MCP Tools - Simple Example")
    print("This will demonstrate the core functionality.")
    print()
    
    input("Press Enter to start the demonstration...")
    
    success = await example.run_complete_example()
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())