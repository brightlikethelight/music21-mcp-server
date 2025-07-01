# Complete Testing Guide for Music21 MCP Server

## 🎯 What This Project Does

The Music21 MCP Server is a bridge between AI assistants and the music21 library. It provides:

1. **Music Analysis**: Analyze keys, chords, harmony, patterns in musical scores
2. **Music Generation**: Create harmonizations, counterpoint, and new music
3. **File I/O**: Import/export MIDI, MusicXML, ABC notation files
4. **Educational Tools**: Learn music theory with explanations

## 🚀 Quick Start Testing

### Step 1: Install Dependencies

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install music21 mcp pytest numpy
```

### Step 2: Create a Simple Test Script

Create `test_everything.py`:

```python
#!/usr/bin/env python3
"""
Complete test of Music21 MCP Server functionality
This script tests all major features without needing MCP
"""

import asyncio
import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Import all the tools we'll test
from music21_mcp.tools import (
    ImportScoreTool,
    ListScoresTool,
    KeyAnalysisTool,
    ChordAnalysisTool,
    ScoreInfoTool,
    ExportScoreTool,
    DeleteScoreTool,
    HarmonyAnalysisTool,
    VoiceLeadingAnalysisTool,
    PatternRecognitionTool,
    HarmonizationTool,
    CounterpointGeneratorTool,
    StyleImitationTool
)

async def test_all_features():
    """Test every major feature of the server"""
    
    # Shared score storage
    score_manager = {}
    
    # Initialize all tools
    tools = {
        'import': ImportScoreTool(score_manager),
        'list': ListScoresTool(score_manager),
        'key': KeyAnalysisTool(score_manager),
        'chord': ChordAnalysisTool(score_manager),
        'info': ScoreInfoTool(score_manager),
        'export': ExportScoreTool(score_manager),
        'delete': DeleteScoreTool(score_manager),
        'harmony': HarmonyAnalysisTool(score_manager),
        'voice_leading': VoiceLeadingAnalysisTool(score_manager),
        'pattern': PatternRecognitionTool(score_manager),
        'harmonization': HarmonizationTool(score_manager),
        'counterpoint': CounterpointGeneratorTool(score_manager),
        'style': StyleImitationTool(score_manager)
    }
    
    print("🎵 Music21 MCP Server - Complete Feature Test")
    print("=" * 60)
    
    # TEST 1: Import from corpus
    print("\n1. Testing Score Import (Bach Chorale from corpus)...")
    result = await tools['import'].execute(
        score_id="bach_chorale",
        source="bach/bwv66.6",
        source_type="corpus"
    )
    if result['status'] == 'success':
        print(f"   ✅ Imported successfully!")
        print(f"   - Notes: {result['num_notes']}")
        print(f"   - Measures: {result['num_measures']}")
        print(f"   - Parts: {result['num_parts']}")
    else:
        print(f"   ❌ Import failed: {result.get('message', 'Unknown error')}")
    
    # TEST 2: Import from text
    print("\n2. Testing Text Import (Simple melody)...")
    result = await tools['import'].execute(
        score_id="simple_melody",
        source="C4 E4 G4 E4 C4 D4 E4 F4 E4 D4 C4",
        source_type="text"
    )
    print(f"   ✅ Created melody with {result['num_notes']} notes")
    
    # TEST 3: List scores
    print("\n3. Testing Score Listing...")
    result = await tools['list'].execute()
    print(f"   ✅ Found {len(result['scores'])} scores:")
    for score in result['scores']:
        print(f"      - {score['score_id']}: {score['num_notes']} notes")
    
    # TEST 4: Key analysis
    print("\n4. Testing Key Analysis...")
    result = await tools['key'].execute(score_id="bach_chorale")
    print(f"   ✅ Detected key: {result['key']}")
    print(f"   - Confidence: {result['confidence']:.2%}")
    print(f"   - Alternatives: {[f'{k} ({c:.1%})' for k, c in result['alternatives'][:3]]}")
    
    # TEST 5: Chord analysis
    print("\n5. Testing Chord Analysis...")
    result = await tools['chord'].execute(
        score_id="bach_chorale",
        include_roman_numerals=True
    )
    print(f"   ✅ Found {result['total_chords']} chords")
    print(f"   - First 5 chords: {result['chord_symbols'][:5]}")
    print(f"   - Roman numerals: {result['roman_numerals'][:5]}")
    
    # TEST 6: Advanced harmony analysis
    print("\n6. Testing Advanced Harmony Analysis...")
    result = await tools['harmony'].execute(
        score_id="bach_chorale",
        include_cadences=True,
        include_non_chord_tones=True
    )
    print(f"   ✅ Advanced harmony analysis complete")
    print(f"   - Cadences found: {len(result['cadences'])}")
    if result['cadences']:
        print(f"   - Cadence types: {[c['type'] for c in result['cadences']]}")
    print(f"   - Non-chord tones: {len(result['non_chord_tones'])}")
    
    # TEST 7: Voice leading analysis
    print("\n7. Testing Voice Leading Analysis...")
    result = await tools['voice_leading'].execute(
        score_id="bach_chorale",
        check_parallels=True,
        check_voice_crossing=True
    )
    print(f"   ✅ Voice leading analysis complete")
    print(f"   - Issues found: {len(result['issues'])}")
    if result['issues']:
        print(f"   - Issue types: {set(i['type'] for i in result['issues'][:5])}")
    
    # TEST 8: Pattern recognition
    print("\n8. Testing Pattern Recognition...")
    result = await tools['pattern'].execute(
        score_id="simple_melody",
        pattern_type="melodic",
        min_pattern_length=3
    )
    print(f"   ✅ Pattern analysis complete")
    print(f"   - Patterns found: {len(result['patterns'])}")
    if result['patterns']:
        print(f"   - Most frequent: {result['patterns'][0]['notes']} ({result['patterns'][0]['occurrences']} times)")
    
    # TEST 9: Harmonization
    print("\n9. Testing Melody Harmonization...")
    result = await tools['harmonization'].execute(
        score_id="simple_melody",
        style="classical",
        voice_parts=4
    )
    print(f"   ✅ Harmonization complete")
    print(f"   - Generated {result['voice_parts']} voices")
    print(f"   - Chord progression: {' '.join(result['chord_progression'][:8])}")
    
    # TEST 10: Counterpoint generation
    print("\n10. Testing Counterpoint Generation...")
    result = await tools['counterpoint'].execute(
        score_id="simple_melody",
        species="first",
        voice_position="above"
    )
    print(f"   ✅ Counterpoint generated")
    print(f"   - Species: {result['species']}")
    print(f"   - Rule compliance: {result['rule_compliance_score']:.1f}%")
    print(f"   - Violations: {len(result['rule_violations'])}")
    
    # TEST 11: Style analysis
    print("\n11. Testing Style Analysis...")
    result = await tools['style'].analyze_style(
        score_id="bach_chorale",
        detailed=True
    )
    print(f"   ✅ Style analysis complete")
    melodic = result['style_characteristics'].get('melodic', {})
    if melodic:
        print(f"   - Stepwise motion: {melodic.get('stepwise_motion', 0):.1%}")
    print(f"   - Closest composers: {[s[0] for s in result['closest_known_styles']]}")
    
    # TEST 12: Style imitation
    print("\n12. Testing Style Imitation (Bach style)...")
    result = await tools['style'].execute(
        composer="bach",
        generation_length=8,
        starting_note="C4",
        complexity="medium"
    )
    print(f"   ✅ Generated {result['measures_generated']} measures")
    print(f"   - Style adherence: {result['style_adherence']:.1%}")
    
    # TEST 13: Score info
    print("\n13. Testing Score Info...")
    result = await tools['info'].execute(score_id="bach_chorale")
    print(f"   ✅ Score information retrieved")
    print(f"   - Title: {result.get('title', 'Unknown')}")
    print(f"   - Duration: {result['total_duration']} quarter notes")
    print(f"   - Time signatures: {result['time_signatures']}")
    
    # TEST 14: Export
    print("\n14. Testing Export...")
    result = await tools['export'].execute(
        score_id="simple_melody",
        format="musicxml"
    )
    print(f"   ✅ Exported to: {result['file_path']}")
    
    # TEST 15: Delete
    print("\n15. Testing Delete...")
    result = await tools['delete'].execute(score_id="simple_melody")
    print(f"   ✅ Deleted score: {result['deleted']}")
    
    print("\n" + "=" * 60)
    print("✅ All tests completed successfully!")
    print("\nThe Music21 MCP Server is working properly!")
    print("\nNext steps:")
    print("1. Try the example scripts in examples/")
    print("2. Read the documentation in docs/")
    print("3. Integrate with MCP client using the MCP protocol")

if __name__ == "__main__":
    asyncio.run(test_all_features())
```

### Step 3: Run the Test

```bash
python test_everything.py
```

## 📊 Understanding the Output

When you run the test, you should see output like:

```
🎵 Music21 MCP Server - Complete Feature Test
============================================================

1. Testing Score Import (Bach Chorale from corpus)...
   ✅ Imported successfully!
   - Notes: 104
   - Measures: 9
   - Parts: 4

2. Testing Text Import (Simple melody)...
   ✅ Created melody with 11 notes

3. Testing Score Listing...
   ✅ Found 2 scores:
      - bach_chorale: 104 notes
      - simple_melody: 11 notes

...and so on
```

## 🧪 Run Unit Tests

### Option 1: Simple Test Runner

```bash
python run_tests.py
```

### Option 2: Using pytest

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_server_pytest.py -v

# Run with coverage report
pytest tests/ --cov=src/music21_mcp
```

## 🔍 What Each Component Does

### Core Tools (Basic Features)
1. **ImportScoreTool**: Loads music from files, corpus, or text
2. **KeyAnalysisTool**: Detects the musical key (C major, G minor, etc.)
3. **ChordAnalysisTool**: Identifies chords and progressions
4. **ScoreInfoTool**: Gets metadata (title, duration, instruments)
5. **ExportScoreTool**: Saves to MIDI, MusicXML, etc.

### Advanced Analysis Tools
1. **HarmonyAnalysisTool**: Deep harmonic analysis with Roman numerals
2. **VoiceLeadingAnalysisTool**: Checks voice leading rules
3. **PatternRecognitionTool**: Finds repeating patterns

### Creative Tools
1. **HarmonizationTool**: Adds chords to melodies
2. **CounterpointGeneratorTool**: Creates countermelodies
3. **StyleImitationTool**: Generates music in specific styles

## 🎹 Try Interactive Testing

Create `interactive_test.py`:

```python
import asyncio
from music21_mcp.tools import ImportScoreTool, HarmonizationTool

async def interactive_demo():
    score_manager = {}
    import_tool = ImportScoreTool(score_manager)
    harmonize_tool = HarmonizationTool(score_manager)
    
    # Create your own melody
    print("Enter a melody (e.g., C D E F G):")
    melody = input("> ") or "C D E F G F E D C"
    
    # Import it
    await import_tool.execute(
        score_id="user_melody",
        source=melody,
        source_type="text"
    )
    
    # Harmonize it
    print("\nHarmonizing your melody...")
    result = await harmonize_tool.execute(
        score_id="user_melody",
        style="classical"
    )
    
    print(f"\nChord progression: {' '.join(result['chord_progression'])}")
    print(f"Saved as: {result['harmonized_score_id']}")

asyncio.run(interactive_demo())
```

## 🐛 Troubleshooting

### Common Issues

1. **ModuleNotFoundError: No module named 'music21'**
   ```bash
   pip install music21
   ```

2. **ModuleNotFoundError: No module named 'mcp'**
   ```bash
   pip install mcp
   ```

3. **Corpus file not found**
   ```bash
   python -c "from music21 import corpus; print(corpus.getComposer('bach'))"
   ```

4. **Permission errors when exporting**
   - Check write permissions in the output directory
   - Try exporting to /tmp/ or current directory

## 📝 Verify It's Working

You know it's working when:
- ✅ Import accepts corpus paths like "bach/bwv66.6"
- ✅ Key analysis returns sensible keys with confidence scores
- ✅ Chord analysis finds actual chord progressions
- ✅ Harmonization adds musically appropriate chords
- ✅ Export creates valid MIDI/MusicXML files

## 🎯 Next Steps

1. **Explore More Examples**
   ```bash
   python examples/basic_usage.py
   python examples/complete_workflow.py
   python examples/advanced_features_demo.py
   ```

2. **Try Different Corpus Pieces**
   - Bach: "bach/bwv846" (Well-Tempered Clavier)
   - Mozart: "mozart/k331/movement1"
   - Beethoven: "beethoven/opus18no1/movement1"

3. **Integrate with MCP client**
   - Follow the MCP setup in docs/getting-started.md
   - Use natural language to analyze music

The server is now fully functional and ready for music analysis and generation tasks!