#!/usr/bin/env python3
"""
Complete End-to-End Example: Music21 MCP Server
===============================================

This example demonstrates comprehensive music analysis using the Music21 MCP Server.
It shows all major features working together in a realistic scenario.

Prerequisites:
- pip install music21
- python -m music21.configure (to set up corpus)

For Claude Desktop integration, add to your MCP config:
{
  "music21": {
    "command": "python",
    "args": ["-m", "music21_mcp.server_minimal"],
    "env": {}
  }
}
"""

import asyncio
import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Use the direct adapter approach instead of MCP client
from music21_mcp.adapters import create_music_analyzer
import json
import tempfile
import time

class MusicAnalysisExample:
    """Complete example of using the Music21 MCP Server"""
    
    def __init__(self):
        self.analyzer = None
        
    async def initialize(self):
        """Initialize the music analyzer"""
        print("🎵 Initializing Music21 Analysis Service...")
        self.analyzer = create_music_analyzer()
        print("✅ Analysis service ready!")
        print(f"✅ Available tools: {len(self.analyzer.get_available_tools())}")
        return True
    
    async def list_available_tools(self):
        """List all available tools"""
        print("\n📋 Available Music Analysis Tools:")
        print("=" * 50)
        
        tools = self.analyzer.get_available_tools()
        
        tool_descriptions = {
            "import_score": "Import musical scores from various sources",
            "list_scores": "List all imported scores with metadata",
            "get_score_info": "Get detailed information about a score",
            "analyze_key": "Analyze the key signature of a musical piece",
            "analyze_chords": "Identify chord progressions in a score",
            "analyze_harmony": "Perform Roman numeral harmonic analysis",
            "analyze_voice_leading": "Check voice leading rules and quality",
            "recognize_patterns": "Find melodic and rhythmic patterns",
            "export_score": "Export scores to various formats",
            "delete_score": "Remove scores from memory",
            "harmonize_melody": "Generate harmonizations for melodies",
            "generate_counterpoint": "Create counterpoint for given melodies",
            "imitate_style": "Generate music in the style of composers"
        }
        
        for tool in tools:
            description = tool_descriptions.get(tool, "Advanced music analysis tool")
            print(f"🔧 {tool}")
            print(f"   Description: {description}")
            print()
        
        return tools
    
    async def demonstrate_music_analysis(self):
        """Demonstrate complete music analysis workflow"""
        print("\n🎼 Complete Music Analysis Demonstration")
        print("=" * 60)
        
        # Step 1: Import multiple scores
        print("\n1️⃣ Importing multiple musical pieces...")
        
        scores_to_import = [
            ("bach_chorale", "bach/bwv66.6", "corpus", "Bach Chorale BWV 66.6"),
            ("beethoven_sonata", "beethoven/opus27no1", "corpus", "Beethoven Moonlight Sonata"),
            ("simple_scale", "C4 D4 E4 F4 G4 A4 B4 C5", "text", "C Major Scale"),
            ("chord_progression", "C4 E4 G4 C5 | F3 A3 C4 F4 | G3 B3 D4 G4 | C4 E4 G4 C5", "text", "I-IV-V-I Progression")
        ]
        
        imported_scores = []
        for score_id, source, source_type, description in scores_to_import:
            print(f"\n   📥 Importing {description}...")
            
            result = await self.analyzer.import_score(score_id, source, source_type)
            if result["status"] == "success":
                print(f"   ✅ Success: {result.get('num_notes', 0)} notes, {result.get('num_parts', 0)} parts")
                imported_scores.append(score_id)
            else:
                print(f"   ❌ Failed: {result.get('message', 'Unknown error')}")
        
        print(f"\n✅ Successfully imported {len(imported_scores)} out of {len(scores_to_import)} scores")
        
        # Step 2: Comprehensive analysis of Bach chorale
        if "bach_chorale" in imported_scores:
            print("\n2️⃣ Comprehensive analysis of Bach chorale...")
            
            # Key analysis
            print("\n   🔑 Key Analysis:")
            key_result = await self.analyzer.analyze_key("bach_chorale")
            if key_result["status"] == "success":
                key = key_result.get('key', 'Unknown')
                confidence = key_result.get('confidence', 0)
                print(f"   Key: {key} (confidence: {confidence:.1%})")
            
            # Chord analysis  
            print("\n   🎵 Chord Analysis:")
            chord_result = await self.analyzer.analyze_chords("bach_chorale")
            if chord_result["status"] == "success":
                total_chords = chord_result.get('total_chords', 0)
                print(f"   Found {total_chords} chord changes")
                
                chord_progression = chord_result.get('chord_progression', [])
                if chord_progression:
                    print("   First 5 chords:")
                    for i, chord in enumerate(chord_progression[:5], 1):
                        if isinstance(chord, dict):
                            symbol = chord.get('symbol', 'Unknown')
                            roman = chord.get('roman_numeral', '?')
                            print(f"     {i}. {symbol} (Roman: {roman})")
            
            # Harmony analysis
            print("\n   🎼 Harmonic Analysis:")
            harmony_result = await self.analyzer.analyze_harmony("bach_chorale", "roman")
            if harmony_result["status"] == "success":
                roman_numerals = harmony_result.get('roman_numerals', [])
                print(f"   Found {len(roman_numerals)} harmonic progressions")
                
                if roman_numerals:
                    print("   Roman numeral progression:")
                    rn_sequence = []
                    for rn in roman_numerals[:10]:
                        if isinstance(rn, dict):
                            rn_sequence.append(rn.get('roman_numeral', '?'))
                        else:
                            rn_sequence.append(str(rn))
                    print(f"   {' - '.join(rn_sequence)}")
            
            # Voice leading analysis
            print("\n   🎶 Voice Leading Analysis:")
            voice_result = await self.analyzer.analyze_voice_leading("bach_chorale")
            if voice_result["status"] == "success":
                issues = voice_result.get('total_issues', 0)
                score = voice_result.get('overall_score', 0)
                parallel_issues = voice_result.get('parallel_issues', [])
                
                print(f"   Voice leading quality: {score}/100")
                print(f"   Total issues found: {issues}")
                print(f"   Parallel motion violations: {len(parallel_issues)}")
            
            # Pattern recognition
            print("\n   🔍 Pattern Recognition:")
            pattern_result = await self.analyzer.recognize_patterns("bach_chorale", "melodic")
            if pattern_result["status"] == "success":
                melodic_patterns = pattern_result.get('melodic_patterns', {})
                sequences = melodic_patterns.get('sequences', [])
                motifs = melodic_patterns.get('motifs', [])
                
                print(f"   Melodic sequences: {len(sequences)}")
                print(f"   Motivic patterns: {len(motifs)}")
                
                # Show some patterns
                if sequences:
                    print("   Sample sequences:")
                    for i, seq in enumerate(sequences[:3], 1):
                        pattern_type = seq.get('pattern_type', 'Unknown')
                        length = seq.get('length', '?')
                        print(f"     {i}. {pattern_type} (length: {length})")
        
        # Step 3: Comparative key analysis
        print("\n3️⃣ Comparative key analysis across all pieces...")
        
        key_results = {}
        for score_id in imported_scores:
            key_result = await self.analyzer.analyze_key(score_id)
            if key_result["status"] == "success":
                key_results[score_id] = key_result.get('key', 'Unknown')
        
        print("   Key signatures detected:")
        for score_id, key in key_results.items():
            print(f"   - {score_id}: {key}")
        
        # Step 4: Export demonstrations
        print("\n4️⃣ Export format demonstrations...")
        
        export_formats = ["musicxml", "midi", "abc"]
        
        if "simple_scale" in imported_scores:
            for fmt in export_formats:
                print(f"\n   📤 Exporting C major scale to {fmt.upper()}...")
                export_result = await self.analyzer.export_score("simple_scale", format=fmt)
                
                if export_result["status"] == "success":
                    file_path = export_result.get('file_path', '')
                    file_name = Path(file_path).name if file_path else 'generated_file'
                    print(f"   ✅ Created: {file_name}")
                    
                    # Show a preview of the content for text formats
                    if fmt == "abc" and file_path and Path(file_path).exists():
                        try:
                            with open(file_path, 'r') as f:
                                content = f.read()[:200]  # First 200 chars
                                print(f"   Preview: {content}...")
                        except:
                            pass
                    
                    # Clean up the file
                    if file_path and Path(file_path).exists():
                        Path(file_path).unlink()
                else:
                    print(f"   ❌ Failed: {export_result.get('message', 'Unknown error')}")
        
        return True
    
    async def demonstrate_service_status(self):
        """Demonstrate service status and capabilities"""
        print("\n🏥 Service Status Check")
        print("=" * 40)
        
        # Get service status
        status = self.analyzer.get_status()
        
        print("✅ Service is healthy")
        print(f"   Service: {status.get('service', 'Unknown')}")
        print(f"   Adapter: {status.get('adapter', 'Unknown')}")
        print(f"   Tools available: {status.get('tools_available', 0)}")
        print(f"   Scores loaded: {status.get('scores_loaded', 0)}")
        print(f"   Status: {status.get('status', 'unknown')}")
    
    async def demonstrate_batch_processing(self):
        """Demonstrate batch processing capabilities"""
        print("\n⚡ Batch Processing Demonstration")
        print("=" * 50)
        
        # Prepare multiple scores for batch import
        batch_scores = [
            {"score_id": "scale_c", "source": "C4 D4 E4 F4 G4 A4 B4 C5", "source_type": "text"},
            {"score_id": "scale_g", "source": "G4 A4 B4 C5 D5 E5 F#5 G5", "source_type": "text"},
            {"score_id": "scale_f", "source": "F4 G4 A4 Bb4 C5 D5 E5 F5", "source_type": "text"},
        ]
        
        print(f"\n   📦 Batch importing {len(batch_scores)} scales...")
        
        batch_result = await self.analyzer.batch_import(batch_scores)
        
        print(f"   ✅ Batch completed: {batch_result.get('successful', 0)}/{batch_result.get('total_scores', 0)} successful")
        
        # Analyze all keys in batch
        print("\n   🔑 Analyzing keys for all scales...")
        
        for score_data in batch_scores:
            score_id = score_data["score_id"]
            key_result = await self.analyzer.analyze_key(score_id)
            if key_result["status"] == "success":
                key = key_result.get('key', 'Unknown')
                print(f"   - {score_id}: {key}")
        
        # Clean up batch scores
        print("\n   🧹 Cleaning up batch scores...")
        for score_data in batch_scores:
            await self.analyzer.delete_score(score_data["score_id"])
        
        print("   ✅ Batch cleanup completed")
    
    async def cleanup(self):
        """Clean up resources"""
        print("\n🧹 Final cleanup...")
        
        # List remaining scores and clean them up
        list_result = await self.analyzer.list_scores()
        if list_result["status"] == "success":
            scores = list_result.get('scores', [])
            for score in scores:
                score_id = score.get('id') or score.get('score_id', 'unknown')
                await self.analyzer.delete_score(score_id)
            
            print(f"   Cleaned up {len(scores)} remaining scores")
        
        print("✅ Cleanup completed")
    
    async def run_complete_example(self):
        """Run the complete example"""
        print("🎵 Music21 MCP Server - Complete End-to-End Example")
        print("=" * 65)
        print("This example demonstrates comprehensive music analysis capabilities")
        print("including advanced pattern recognition, voice leading analysis,")
        print("and batch processing features.")
        print()
        
        try:
            # Initialize service
            await self.initialize()
            
            # List available tools
            await self.list_available_tools()
            
            # Demonstrate comprehensive music analysis
            await self.demonstrate_music_analysis()
            
            # Demonstrate batch processing
            await self.demonstrate_batch_processing()
            
            # Check service status
            await self.demonstrate_service_status()
            
            print("\n🎉 Complete example finished successfully!")
            print("\n" + "=" * 65)
            print("🌟 What you can do next:")
            print("- Use this server with Claude Desktop for AI-assisted music analysis")
            print("- Integrate with Jupyter notebooks for research workflows")
            print("- Build web applications using the HTTP adapter")
            print("- Create command-line tools using the CLI adapter")
            print("- Deploy as a microservice in production environments")
            print("- Extend with custom analysis tools for specific research needs")
            
        except Exception as e:
            print(f"\n❌ Example failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        finally:
            await self.cleanup()
        
        return True

def create_claude_config():
    """Create Claude Desktop configuration"""
    config = {
        "mcpServers": {
            "music21": {
                "command": "python",
                "args": ["-m", "music21_mcp.server_minimal"],
                "env": {}
            }
        }
    }
    
    print("\n📝 Claude Desktop Configuration")
    print("=" * 40)
    print("Add this to your Claude Desktop MCP configuration:")
    print()
    print(json.dumps(config, indent=2))
    print()
    print("Configuration file locations:")
    print("- macOS: ~/Library/Application Support/Claude/claude_desktop_config.json")
    print("- Windows: %APPDATA%/Claude/claude_desktop_config.json") 
    print("- Linux: ~/.config/claude_desktop_config.json")
    print("\nAfter adding this configuration, restart Claude Desktop and you'll be able")
    print("to ask Claude to analyze your music files using natural language!")

async def main():
    """Main entry point"""
    example = MusicAnalysisExample()
    
    print("🎵 Music21 MCP Server - Complete Example")
    print("Choose what you'd like to see:")
    print("1. Run complete analysis demonstration (recommended)")
    print("2. Show Claude Desktop configuration only")
    print("3. Both demonstration and configuration")
    
    try:
        choice = input("\nEnter your choice (1-3): ").strip()
    except (EOFError, KeyboardInterrupt):
        # Handle automated testing where input isn't available
        print("Running complete demonstration (automated mode)")
        choice = "1"
    
    if choice in ["1", "3"]:
        success = await example.run_complete_example()
        if not success:
            sys.exit(1)
    
    if choice in ["2", "3"]:
        create_claude_config()

if __name__ == "__main__":
    asyncio.run(main())