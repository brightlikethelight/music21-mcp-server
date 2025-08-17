# Music21 MCP Server Examples

This directory contains working example scripts demonstrating how to use the Music21 MCP Server.

## ✅ Working Examples

All examples have been tested and are fully functional:

### 1. basic_usage.py ⭐ **Recommended for beginners**
Simple workflow showing core functionality:
- Import a Bach chorale from the music21 corpus
- Analyze key signature and chord progressions
- Get score metadata and information
- Export to different formats (MusicXML, MIDI, ABC)
- Proper error handling and cleanup

### 2. complete_workflow.py
End-to-end workflow demonstrating:
- Import multiple scores from different sources
- Comprehensive analysis pipeline
- Error handling for failed imports
- Export demonstrations
- Memory management and cleanup

### 3. simple_example.py
Direct tool usage without MCP protocol:
- Uses tools directly for development/testing
- Comprehensive analysis of Bach chorale
- Multiple score import and analysis
- Export functionality demonstration

### 4. basic_working.py
Minimal working example:
- Shows basic tool instantiation
- Simple import and analysis workflow
- Perfect for understanding the basics

### 5. complete_example.py 🚀 **Most comprehensive**
Full-featured demonstration:
- Complete music analysis workflow
- Batch processing capabilities
- Service status monitoring
- All available analysis tools
- Production-ready patterns

## Prerequisites

1. **Install music21 and configure corpus:**
   ```bash
   pip install music21
   python -m music21.configure
   ```

2. **Install the MCP server:**
   ```bash
   pip install -e .  # If running from source
   # OR
   pip install music21-mcp-server  # If installed from PyPI
   ```

## Running the Examples

```bash
# Start with the basic example
python examples/basic_usage.py

# Try the comprehensive workflow
python examples/complete_workflow.py

# Explore all features
python examples/complete_example.py

# For development and testing
python examples/simple_example.py
```

## Understanding the Output

Each example provides:
- ✅ Success indicators for completed operations
- ❌ Error messages with helpful explanations
- 📊 Analysis results with confidence scores
- 🎵 Musical insights (keys, chords, patterns)
- 📁 File paths for exported content

## Example Output

```
🎼 Music21 MCP Server - Basic Usage Example
==================================================

📥 Step 1: Importing Bach Chorale...
✅ Successfully imported Bach chorale
   Notes: 165
   Measures: 0
   Parts: 4

🔑 Step 2: Analyzing Musical Key...
✅ Key detected: F# minor
   Confidence: 87.31%

🎹 Step 3: Analyzing Chord Progression...
✅ Found 51 chords
   First few chords:
     1. A3, E4, C#5 (Roman: III)
     2. G#3, B3, E4, B4 (Roman: VII)
```

## Claude Desktop Integration

To use with Claude Desktop, add this to your MCP configuration:

```json
{
  "mcpServers": {
    "music21": {
      "command": "python",
      "args": ["-m", "music21_mcp.server_minimal"],
      "env": {}
    }
  }
}
```

**Configuration file locations:**
- macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
- Windows: `%APPDATA%/Claude/claude_desktop_config.json`
- Linux: `~/.config/claude_desktop_config.json`

## Troubleshooting

### Common Issues

1. **ModuleNotFoundError: music21**
   ```bash
   pip install music21
   python -m music21.configure
   ```

2. **"Could not find work" errors**
   - The music21 corpus has specific entry names
   - Use `bach/bwv66.6` instead of `bach/bwv66`
   - Check available corpus with `python -c "from music21 import corpus; print(corpus.corpora.CoreCorpus().getPaths())"`

3. **Import errors from src/**
   - Examples automatically add src to Python path
   - Ensure you're running from the project root directory

### Getting Help

- Read the error messages - they contain specific guidance
- Check the logs for detailed debugging information
- All examples include comprehensive error handling
- Open an issue with the full error traceback

## Creating Your Own Examples

Feel free to:
- Copy and modify existing examples
- Use the `create_sync_analyzer()` pattern for simple scripts
- Use the `create_music_analyzer()` pattern for async workflows
- Follow the error handling patterns shown in the examples

## API Reference

**Synchronous API (easiest):**
```python
from music21_mcp.adapters import create_sync_analyzer

analyzer = create_sync_analyzer()
result = analyzer.import_score("my_score", "bach/bwv66.6", "corpus")
key_result = analyzer.analyze_key("my_score")
```

**Async API (more powerful):**
```python
from music21_mcp.adapters import create_music_analyzer

analyzer = create_music_analyzer()
result = await analyzer.import_score("my_score", "bach/bwv66.6", "corpus")
key_result = await analyzer.analyze_key("my_score")
```

## Next Steps

1. 🎯 Run `basic_usage.py` to see everything working
2. 🔬 Try `complete_example.py` for advanced features
3. 🤖 Set up Claude Desktop integration
4. 🎵 Start analyzing your own music files!

Happy music analysis! 🎼