# Claude Desktop Integration Guide

This guide shows how to integrate the Music21 MCP Server with Claude Desktop.

## Prerequisites

1. **Claude Desktop** installed on your system
2. **Python 3.8+** with the music21 package
3. **Music21 MCP Server** installed

## Installation

```bash
# Install the server
pip install -e .

# Test the server works
python -m music21_mcp.server_minimal --help
```

## Configuration

### 1. Locate Claude Desktop Config

**macOS:**
```bash
~/Library/Application Support/Claude/claude_desktop_config.json
```

**Windows:**
```bash
%APPDATA%\Claude\claude_desktop_config.json
```

**Linux:**
```bash
~/.config/Claude/claude_desktop_config.json
```

### 2. Add MCP Server Configuration

Edit the Claude Desktop config file and add:

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

### 3. Restart Claude Desktop

Close and reopen Claude Desktop for the configuration to take effect.

## Usage Examples

### Basic Music Analysis

```
Hi Claude! I want to analyze some music. Can you help me import a Bach chorale and analyze its key signature?
```

Claude will use the MCP server to:
1. Import the score using `import_score`
2. Analyze the key using `key_analysis`
3. Provide results and insights

### Chord Analysis Workflow

```
Let's analyze the harmonic structure of a piece. Please:
1. Import Bach's BWV 66.6
2. Analyze the chord progressions
3. Identify any interesting harmonic patterns
```

### Pattern Recognition

```
I'm studying melodic patterns. Can you import a Mozart sonata and find recurring melodic motifs?
```

## Available Tools

The server provides these tools to Claude:

1. **import_score** - Import music from various sources
2. **list_scores** - List all available scores
3. **score_info** - Get detailed score information
4. **export_score** - Export scores to different formats
5. **delete_score** - Remove scores from memory
6. **key_analysis** - Analyze key signatures
7. **chord_analysis** - Analyze chord progressions
8. **harmony_analysis** - Roman numeral analysis
9. **voice_leading_analysis** - Analyze voice leading
10. **pattern_recognition** - Find musical patterns
11. **harmonization** - Generate harmonizations
12. **counterpoint_generation** - Generate counterpoint
13. **style_imitation** - Imitate composer styles

## Troubleshooting

### Server Not Starting

```bash
# Check if MCP package is installed
pip list | grep mcp

# Install if missing
pip install mcp

# Test server directly
python -m music21_mcp.server_minimal
```

### Claude Desktop Not Seeing Server

1. Check the config file path is correct
2. Verify JSON syntax is valid
3. Restart Claude Desktop completely
4. Check Claude Desktop logs for error messages

### Import Errors

```bash
# Install music21 if missing
pip install music21

# Test music21 works
python -c "import music21; print('OK')"
```

## Advanced Configuration

### Custom Music Corpus

```json
{
  "mcpServers": {
    "music21": {
      "command": "python",
      "args": ["-m", "music21_mcp.server_minimal"],
      "env": {
        "MUSIC21_CORPUS_PATH": "/path/to/your/music/files"
      }
    }
  }
}
```

### Development Mode

```json
{
  "mcpServers": {
    "music21": {
      "command": "python",
      "args": ["-m", "music21_mcp.server_minimal", "--debug"],
      "env": {
        "PYTHONPATH": "/path/to/music21-mcp-server/src"
      }
    }
  }
}
```

## Example Claude Conversations

### Quick Analysis

**You:** "Analyze the key and harmony of Bach's BWV 66.6"

**Claude:** "I'll analyze that Bach chorale for you. Let me import it and examine its key signature and harmonic structure."

*Claude uses the MCP server to import and analyze the piece*

**Claude:** "Bach's BWV 66.6 is in F# minor with a confidence of 0.87. The harmonic analysis shows..."

### Comparative Analysis

**You:** "Compare the harmonic complexity of two different Bach chorales"

**Claude:** "I'll import both pieces and compare their harmonic structures. Let me analyze BWV 66.6 and BWV 60.5..."

This integration makes music analysis as easy as having a conversation with Claude!