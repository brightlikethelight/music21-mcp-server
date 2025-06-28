# Music21 MCP Server Examples

This directory contains practical examples showing how to use the Music21 MCP Server in real-world scenarios.

## 📁 Examples Overview

| File | Description | Complexity |
|------|-------------|------------|
| [`basic_usage.py`](basic_usage.py) | Simple import, analyze, export workflow | ⭐ Beginner |
| [`text_to_midi.py`](text_to_midi.py) | Convert text notation to MIDI files | ⭐ Beginner |
| [`batch_analysis.py`](batch_analysis.py) | Analyze multiple scores in batch | ⭐⭐ Intermediate |
| [`key_detection_comparison.py`](key_detection_comparison.py) | Compare different key detection methods | ⭐⭐ Intermediate |
| [`chord_progression_analysis.py`](chord_progression_analysis.py) | Detailed chord analysis with Roman numerals | ⭐⭐ Intermediate |
| [`corpus_explorer.py`](corpus_explorer.py) | Explore the music21 corpus systematically | ⭐⭐⭐ Advanced |
| [`claude_desktop_demo.py`](claude_desktop_demo.py) | Integration example with Claude Desktop | ⭐⭐⭐ Advanced |

## 🚀 Quick Start

### Running Examples

```bash
# Make sure you have the server installed
cd music21-mcp-server
pip install -e .

# Run any example
python examples/basic_usage.py
python examples/text_to_midi.py
```

### Using with Claude Desktop

1. Set up the MCP server in Claude Desktop config
2. Use the examples as templates for your own analysis
3. Ask Claude to help you analyze your music!

## 📚 Learning Path

### 1. Start Here: Basic Usage
- Import scores from different sources
- Perform key and chord analysis
- Export results

### 2. Text Processing
- Create melodies from text
- Convert to playable MIDI
- Analyze your creations

### 3. Batch Operations
- Process multiple files
- Compare results across pieces
- Generate reports

### 4. Advanced Analysis
- Detailed harmonic analysis
- Method comparisons
- Custom workflows

## 🎵 Music Theory Background

### Key Detection
The server uses several algorithms:
- **Krumhansl-Schmuckler**: Based on pitch-class profiles
- **Aarden**: Improved profiles for better accuracy
- **Temperley**: Optimized for classical music

### Chord Analysis
- **Chordify**: Groups simultaneous notes into chords
- **Roman Numerals**: Functional harmony analysis
- **Root Detection**: Identifies chord roots and qualities

### Supported Formats
- **Input**: MIDI, MusicXML, ABC, Kern, text notation
- **Output**: MIDI, MusicXML, ABC, LilyPond, PDF

## 💡 Tips for Success

### Best Practices
1. **Start Simple**: Begin with corpus examples
2. **Test Edge Cases**: Try empty scores, single notes
3. **Check Results**: Verify analysis makes musical sense
4. **Document Your Work**: Add comments to your code

### Common Pitfalls
1. **Missing Scores**: Always check import success
2. **Wrong Score IDs**: Use descriptive, unique identifiers
3. **Memory Management**: Delete scores when done
4. **File Paths**: Use absolute paths for reliability

### Performance Tips
1. **Reuse Scores**: Import once, analyze multiple times
2. **Batch Operations**: Process multiple scores together
3. **Clean Up**: Delete unused scores to free memory
4. **Cache Results**: Save analysis results for reuse

## 🔧 Customization Ideas

### Custom Analysis Workflows
```python
async def my_analysis_workflow(score_id):
    """Custom analysis combining multiple tools"""
    key = await analyze_key(score_id)
    chords = await analyze_chords(score_id, include_roman_numerals=True)
    info = await get_score_info(score_id)
    
    return {
        "piece": info['title'],
        "key": key['key'],
        "confidence": key['confidence'],
        "total_chords": chords['total_chords'],
        "analysis_quality": "high" if key['confidence'] > 0.8 else "medium"
    }
```

### Educational Tools
```python
async def chord_progression_quiz(score_id):
    """Generate quiz questions from chord analysis"""
    chords = await analyze_chords(score_id, include_roman_numerals=True)
    
    questions = []
    for chord in chords['chord_progression'][:5]:
        questions.append({
            "question": f"What is the Roman numeral for {chord['pitches']}?",
            "answer": chord['roman_numeral']
        })
    
    return questions
```

### Data Export
```python
async def export_analysis_csv(score_ids, filename):
    """Export analysis results to CSV"""
    import csv
    
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Score ID', 'Key', 'Confidence', 'Chord Count'])
        
        for score_id in score_ids:
            key = await analyze_key(score_id)
            chords = await analyze_chords(score_id)
            writer.writerow([
                score_id,
                key['key'],
                key['confidence'],
                chords['total_chords']
            ])
```

## 📖 Additional Resources

### Music21 Documentation
- [music21 User's Guide](https://web.mit.edu/music21/doc/usersGuide/index.html)
- [music21 API Documentation](https://web.mit.edu/music21/doc/moduleReference/index.html)
- [music21 Corpus](https://web.mit.edu/music21/doc/usersGuide/usersGuide_09_corpus.html)

### Music Theory Resources
- [Tonal Harmony Basics](https://musictheory.net/)
- [Roman Numeral Analysis](https://musictheory.net/lessons/57)
- [Chord Progressions](https://musictheory.net/lessons/58)

### MCP Protocol
- [Model Context Protocol Docs](https://modelcontextprotocol.io/)
- [Claude Desktop Integration](https://docs.anthropic.com/claude/desktop/mcp)

## 🆘 Getting Help

### Common Issues
1. **Import Errors**: Check file paths and formats
2. **Analysis Failures**: Verify score has musical content
3. **Export Problems**: Check target directory permissions

### Support Channels
- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and tips
- **Documentation**: Complete API reference in `docs/`

## 🎯 Contributing Examples

Have a great example? We'd love to include it! See our [Contributing Guide](../CONTRIBUTING.md) for:

- Example submission guidelines
- Code style requirements
- Documentation standards
- Testing requirements

Happy music analysis! 🎵