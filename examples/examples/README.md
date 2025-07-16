# Music21 MCP Server - Examples

This directory contains working examples that demonstrate the complete functionality of the Music21 MCP Server.

## 🚀 Quick Start (5 minutes)

### 1. Install the Server

```bash
# Clone the repository
git clone https://github.com/brightlikethelight/music21-mcp-server.git
cd music21-mcp-server

# Install with Poetry (recommended)
poetry install

# Or install with pip
pip install -e .

# Configure music21 corpus (required for examples)
python -m music21.configure
```

### 2. Run the Simple Example

The simple example demonstrates all core functionality without needing MCP client setup:

```bash
cd examples
python simple_example.py
```

**Expected output:**
- ✅ Imports Bach chorale from corpus
- ✅ Analyzes musical key (F# minor, 87% confidence)
- ✅ Finds 9 melodic sequences using pattern recognition
- ✅ Analyzes voice leading (detects parallel motion)
- ✅ Exports to MusicXML, MIDI, and ABC formats

### 3. Try with Claude Desktop

Add this to your Claude Desktop MCP configuration:

```json
{
  "mcpServers": {
    "music21": {
      "command": "python",
      "args": ["-m", "music21_mcp.server"],
      "env": {}
    }
  }
}
```

**Configuration file locations:**
- macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
- Windows: `%APPDATA%/Claude/claude_desktop_config.json`
- Linux: `~/.config/claude_desktop_config.json`

## 📁 Available Examples

### `simple_example.py` - Direct Tool Usage
- **Best for**: Testing, development, understanding the tools
- **Requirements**: Just Python and music21
- **Time**: 2-3 minutes to run
- **What it does**: Demonstrates all 13 music analysis tools directly

### `complete_example.py` - Full MCP Integration
- **Best for**: Understanding MCP client-server communication
- **Requirements**: MCP client libraries
- **Time**: 5-10 minutes to run
- **What it does**: Shows complete MCP workflow with client-server setup

## 🎵 What These Examples Demonstrate

### Core Music Analysis
- **Import scores** from music21 corpus, local files, or text notation
- **Key analysis** with confidence scores and multiple algorithms
- **Chord progression analysis** with Roman numeral notation
- **Pattern recognition** finding melodic sequences and motifs
- **Voice leading analysis** detecting parallel motion and rule violations

### Advanced Features
- **Multiple score management** - work with several pieces simultaneously
- **Export to multiple formats** - MusicXML, MIDI, ABC notation
- **Real-time analysis** - fast processing suitable for interactive use
- **Memory management** - efficient handling of large musical datasets

### Production Features
- **Health monitoring** - server status and resource usage
- **Error handling** - graceful failure with helpful messages
- **Rate limiting** - prevents server overload
- **Clean APIs** - consistent input/output formats

## 🔧 Available Tools

| Tool | Description | Input | Output |
|------|-------------|-------|--------|
| `import_score` | Import from corpus, file, or text | score_id, source, source_type | Success/error, metadata |
| `key_analysis` | Detect musical key | score_id | Key, confidence, mode |
| `chord_analysis` | Analyze chord progressions | score_id | Chord list with positions |
| `harmony_analysis` | Roman numeral analysis | score_id | Roman numerals, progressions |
| `pattern_recognition` | Find melodic patterns | score_id, options | Sequences, motifs, contours |
| `voice_leading_analysis` | Check voice leading rules | score_id | Issues, score, smoothness |
| `score_info` | Get score metadata | score_id | Title, composer, parts, measures |
| `list_scores` | List all loaded scores | - | Score inventory |
| `export_score` | Export to file format | score_id, format | File path |
| `delete_score` | Remove from memory | score_id | Success/error |
| `health_check` | Server status | - | Memory, uptime, status |
| `cleanup_memory` | Force garbage collection | - | Memory freed |

## 🎼 Sample Music Files

The examples use pieces from the music21 corpus:

- **Bach Chorale (bwv66.6)** - Perfect for harmony analysis
- **Mozart Sonata (k545)** - Great for pattern recognition
- **Beethoven String Quartet (opus18no1)** - Excellent for voice leading analysis

## 🐛 Troubleshooting

### "No module named 'music21'"
```bash
pip install music21
python -m music21.configure
```

### "Could not find a work that met this criterion"
```bash
# Configure the music21 corpus
python -m music21.configure
# Accept the default settings
```

### "MCP package not found"
```bash
pip install mcp
```

### "Server won't start"
```bash
# Check dependencies
poetry install
# Or
pip install -e .
```

## 📚 Next Steps

1. **Try the examples** - Start with `simple_example.py`
2. **Test with Claude** - Add to your Claude Desktop configuration
3. **Experiment with your music** - Import your own files
4. **Build custom workflows** - Combine tools for specific analysis needs
5. **Deploy to production** - Use Docker for consistent deployment

## 🤝 Need Help?

- **Documentation**: Check the main README.md
- **Issues**: Report bugs on GitHub
- **Examples**: All examples are fully self-contained
- **Support**: Contact the maintainers

---

**Ready to analyze music with AI?** Start with `python simple_example.py` and see the magic happen\! 🎵
EOF < /dev/null