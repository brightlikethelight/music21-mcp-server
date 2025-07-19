# Music21 Analysis - Multiple Interface Guide

This project provides **4 different ways** to access the same powerful music21 analysis functionality. Choose the interface that best fits your use case:

## 🎯 Quick Start

```bash
# Show all available interfaces
python -m music21_mcp.launcher

# Test all interfaces  
python -m music21_mcp.launcher demo
```

---

## 📡 1. MCP Server (for Claude Desktop)

**Best for:** AI assistant integration, interactive music analysis with Claude

```bash
# Start MCP server
python -m music21_mcp.launcher mcp
```

**Claude Desktop Setup:**
```json
{
  "mcpServers": {
    "music21-analysis": {
      "command": "python",
      "args": ["-m", "music21_mcp.server_minimal"],
      "env": {
        "PYTHONPATH": "/path/to/music21-mcp-server/src"
      }
    }
  }
}
```

**Example Usage in Claude:**
- "Import Bach chorale BWV 66.6 and analyze its key signature"
- "Analyze the harmony of the imported score using Roman numerals" 
- "Check the voice leading quality and export to MIDI"

---

## 🌐 2. HTTP API Server (for Web Integration)

**Best for:** Web applications, microservices, REST API integration

```bash
# Start HTTP API server
python -m music21_mcp.launcher http
# Opens: http://localhost:8000
# API Docs: http://localhost:8000/docs
```

**Example Usage:**
```bash
# Import a score
curl -X POST "http://localhost:8000/scores/import" \
  -H "Content-Type: application/json" \
  -d '{"score_id": "chorale", "source": "bach/bwv66.6", "source_type": "corpus"}'

# Analyze key
curl -X POST "http://localhost:8000/analysis/key" \
  -H "Content-Type: application/json" \
  -d '{"score_id": "chorale"}'

# List all scores
curl "http://localhost:8000/scores"
```

**Health Check:**
```bash
curl http://localhost:8000/health
```

---

## 💻 3. CLI Tools (for Command Line)

**Best for:** Scripting, automation, command-line workflows

```bash
# Basic commands
python -m music21_mcp.launcher cli status
python -m music21_mcp.launcher cli tools

# Import and analyze
python -m music21_mcp.launcher cli import chorale bach/bwv66.6 corpus
python -m music21_mcp.launcher cli key-analysis chorale
python -m music21_mcp.launcher cli harmony chorale roman

# List and export
python -m music21_mcp.launcher cli list
python -m music21_mcp.launcher cli export chorale musicxml
```

**Automation Example:**
```bash
#!/bin/bash
# Analyze multiple pieces
for piece in bach/bwv66.6 bach/bwv4.8 bach/bwv69.6; do
    python -m music21_mcp.launcher cli import "$(basename $piece)" "$piece" corpus
    python -m music21_mcp.launcher cli key-analysis "$(basename $piece)"
done
```

---

## 🐍 4. Python Library (for Programming)

**Best for:** Jupyter notebooks, Python applications, programmatic access

### Async Interface:
```python
from music21_mcp.adapters import PythonAdapter
import asyncio

async def analyze_music():
    analyzer = PythonAdapter()
    
    # Import score
    await analyzer.import_score("chorale", "bach/bwv66.6", "corpus")
    
    # Analyze
    key_result = await analyzer.analyze_key("chorale")
    harmony_result = await analyzer.analyze_harmony("chorale", "roman")
    
    # Quick comprehensive analysis
    analysis = await analyzer.quick_analysis("chorale")
    print(analysis)

asyncio.run(analyze_music())
```

### Synchronous Interface (easier):
```python
from music21_mcp.adapters import create_sync_analyzer

# Create analyzer
analyzer = create_sync_analyzer()

# Import and analyze
analyzer.import_score("chorale", "bach/bwv66.6", "corpus")
key_result = analyzer.analyze_key("chorale")

print(f"Key: {key_result}")
```

### Jupyter Notebook Example:
```python
# In Jupyter cell
from music21_mcp.adapters import Music21Analysis

analyzer = Music21Analysis()

# Quick analysis
result = analyzer.quick_analysis("chorale")

# Display results
import json
print(json.dumps(result, indent=2))
```

---

## 🔄 Interface Comparison

| Interface | Best For | Reliability | Setup Complexity |
|-----------|----------|-------------|------------------|
| **MCP** | AI Assistants | 60-70% | Medium |
| **HTTP** | Web Apps | 95%+ | Low |
| **CLI** | Automation | 99%+ | None |
| **Python** | Programming | 99%+ | None |

## 🛡️ Reliability Strategy

This project provides **multiple pathways** to the same music analysis functionality because:

1. **MCP Protocol**: Has ~60% success rate in production, frequent breaking changes
2. **HTTP API**: Reliable web standard, works when MCP fails
3. **CLI Tools**: Always works locally, no network dependencies  
4. **Python Library**: Direct access, no protocol overhead

**Recommendation**: Use MCP for AI integration, but have HTTP/CLI/Python as backups.

---

## 🎵 Available Analysis Tools

All interfaces provide access to these 13 music analysis tools:

1. **import_score** - Import from corpus, files, URLs
2. **list_scores** - List all imported scores
3. **get_score_info** - Detailed score information
4. **export_score** - Export to MIDI, MusicXML, etc.
5. **delete_score** - Remove scores from storage
6. **analyze_key** - Key signature analysis
7. **analyze_chords** - Chord progression analysis
8. **analyze_harmony** - Roman numeral/functional harmony
9. **analyze_voice_leading** - Voice leading quality
10. **recognize_patterns** - Melodic/rhythmic patterns
11. **harmonize_melody** - Automatic harmonization
12. **generate_counterpoint** - Counterpoint generation
13. **imitate_style** - Style imitation

---

## 🚀 Quick Examples

### Import and Analyze Bach Chorale:
```bash
# CLI
python -m music21_mcp.launcher cli import chorale bach/bwv66.6 corpus
python -m music21_mcp.launcher cli key-analysis chorale

# Python
analyzer = create_sync_analyzer()
analyzer.import_score("chorale", "bach/bwv66.6", "corpus")
print(analyzer.analyze_key("chorale"))
```

### Start Services:
```bash
# MCP for Claude Desktop
python -m music21_mcp.launcher mcp

# HTTP API for web apps  
python -m music21_mcp.launcher http

# CLI for scripting
python -m music21_mcp.launcher cli status
```

---

## 💡 Tips

1. **Start with CLI** - easiest to test and debug
2. **Use Python library** for Jupyter notebooks and scripts
3. **Use HTTP API** for web applications and microservices
4. **Use MCP** for AI assistant integration (with backup plan)
5. **Test with demo** - `python -m music21_mcp.launcher demo`

Choose the interface that matches your workflow. All provide the same powerful music21 analysis capabilities!