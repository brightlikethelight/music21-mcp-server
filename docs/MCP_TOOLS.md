# MCP Tools Documentation - Music21 Analysis Server

## Overview

The Music21 MCP Server provides 13 comprehensive music analysis and generation tools built on the powerful music21 library. All tools follow MCP standards and provide detailed error handling, progress reporting, and structured responses.

## Tool Categories

### 📂 Score Management
- [import_score](#import_score) - Import scores from various sources
- [list_scores](#list_scores) - List all loaded scores
- [score_info](#score_info) - Get detailed score information
- [export_score](#export_score) - Export scores to different formats
- [delete_score](#delete_score) - Remove scores from memory

### 🎵 Analysis Tools
- [key_analysis](#key_analysis) - Musical key detection with multiple algorithms
- [chord_analysis](#chord_analysis) - Chord progression analysis
- [harmony_analysis](#harmony_analysis) - Roman numeral and functional harmony analysis
- [voice_leading_analysis](#voice_leading_analysis) - Voice leading and part-writing analysis
- [pattern_recognition](#pattern_recognition) - Musical pattern identification

### 🎼 Generation Tools
- [harmonization](#harmonization) - Add harmonic accompaniment to melodies
- [counterpoint_generation](#counterpoint_generation) - Generate species counterpoint
- [style_imitation](#style_imitation) - Analyze and generate music in specific styles

### 🩺 System Tools
- [health_check](#health_check) - Server health and status monitoring

---

## Detailed Tool Documentation

### import_score

**Description**: Import musical scores from various sources including local files, URLs, music21 corpus, and direct text input.

**Parameters**:
- `score_id` (string, required): Unique identifier for the imported score
- `source` (string, required): Source path, URL, or corpus identifier
- `source_type` (string, optional): Type of source - "file", "url", "corpus", or "text" (default: "corpus")

**Supported Formats**: MusicXML, MIDI, ABC notation, Lilypond, MEI, Kern, music21 pickle

**Example Usage**:
```json
{
  "tool": "import_score",
  "params": {
    "score_id": "bach_chorale",
    "source": "bach/bwv66.6",
    "source_type": "corpus"
  }
}
```

**Response Format**:
```json
{
  "success": true,
  "data": {
    "score_id": "bach_chorale",
    "title": "BWV 66.6",
    "composer": "J.S. Bach",
    "format": "musicxml",
    "measures": 32,
    "parts": 4,
    "metadata": {...}
  }
}
```

### list_scores

**Description**: Retrieve a list of all currently loaded scores with basic metadata.

**Parameters**: None

**Example Usage**:
```json
{
  "tool": "list_scores",
  "params": {}
}
```

**Response Format**:
```json
{
  "success": true,
  "data": {
    "scores": ["bach_chorale", "mozart_sonata"],
    "total_count": 2,
    "memory_usage": "4.2 MB"
  }
}
```

### score_info

**Description**: Get comprehensive information about a specific score including structure, metadata, and basic statistics.

**Parameters**:
- `score_id` (string, required): ID of the score to analyze

**Example Usage**:
```json
{
  "tool": "score_info",
  "params": {
    "score_id": "bach_chorale"
  }
}
```

**Response Format**:
```json
{
  "success": true,
  "data": {
    "basic_info": {
      "title": "BWV 66.6",
      "composer": "J.S. Bach",
      "key_signature": "A major",
      "time_signature": "4/4",
      "tempo": 120
    },
    "structure": {
      "measures": 32,
      "parts": 4,
      "voices": ["Soprano", "Alto", "Tenor", "Bass"]
    },
    "statistics": {
      "note_count": 256,
      "duration": "2:15",
      "range": {
        "highest_note": "G5",
        "lowest_note": "D3"
      }
    }
  }
}
```

### export_score

**Description**: Export a score to various formats for use in other applications.

**Parameters**:
- `score_id` (string, required): ID of the score to export
- `format` (string, optional): Export format - "musicxml", "midi", "abc", "lilypond", "png", "text" (default: "musicxml")

**Example Usage**:
```json
{
  "tool": "export_score",
  "params": {
    "score_id": "bach_chorale",
    "format": "midi"
  }
}
```

**Response Format**:
```json
{
  "success": true,
  "data": {
    "format": "midi",
    "file_path": "/tmp/bach_chorale.mid",
    "file_size": "2.1 KB",
    "encoding": "binary"
  }
}
```

### delete_score

**Description**: Remove a score from the server's memory to free up resources.

**Parameters**:
- `score_id` (string, required): ID of the score to delete

**Example Usage**:
```json
{
  "tool": "delete_score",
  "params": {
    "score_id": "bach_chorale"
  }
}
```

### key_analysis

**Description**: Analyze the musical key and tonal center using multiple algorithms with confidence scoring.

**Parameters**:
- `score_id` (string, required): ID of the score to analyze
- `algorithm` (string, optional): Algorithm to use - "all", "krumhansl", "aarden", "temperley", "bellman" (default: "all")

**Algorithms**:
- **Krumhansl-Schmuckler**: Traditional key-finding algorithm based on pitch-class profiles
- **Aarden-Essen**: Statistical approach using large corpus analysis
- **Temperley-Kostka-Payne**: Bayesian approach with chord progressions
- **Bellman-Budge**: Dynamic programming approach

**Example Usage**:
```json
{
  "tool": "key_analysis",
  "params": {
    "score_id": "bach_chorale",
    "algorithm": "all"
  }
}
```

**Response Format**:
```json
{
  "success": true,
  "data": {
    "primary_key": "A major",
    "confidence": 0.95,
    "algorithms": {
      "krumhansl": {"key": "A major", "confidence": 0.92},
      "aarden": {"key": "A major", "confidence": 0.89},
      "temperley": {"key": "A major", "confidence": 0.97},
      "bellman": {"key": "A major", "confidence": 0.94}
    },
    "key_changes": [
      {"measure": 16, "key": "E major", "confidence": 0.78}
    ]
  }
}
```

### chord_analysis

**Description**: Analyze chord progressions and harmonic content throughout the score.

**Parameters**:
- `score_id` (string, required): ID of the score to analyze

**Example Usage**:
```json
{
  "tool": "chord_analysis",
  "params": {
    "score_id": "bach_chorale"
  }
}
```

**Response Format**:
```json
{
  "success": true,
  "data": {
    "key": "A major",
    "chord_progression": [
      {"measure": 1, "beat": 1, "chord": "A", "quality": "major", "inversion": "root"},
      {"measure": 1, "beat": 3, "chord": "D", "quality": "major", "inversion": "root"},
      {"measure": 2, "beat": 1, "chord": "E", "quality": "major", "inversion": "first"}
    ],
    "statistics": {
      "total_chords": 64,
      "unique_chords": 12,
      "most_common": [
        {"chord": "A major", "count": 16},
        {"chord": "D major", "count": 12},
        {"chord": "E major", "count": 10}
      ]
    }
  }
}
```

### harmony_analysis

**Description**: Perform detailed harmonic analysis including Roman numeral analysis and functional harmony identification.

**Parameters**:
- `score_id` (string, required): ID of the score to analyze
- `analysis_type` (string, optional): Type of analysis - "roman", "functional", "both" (default: "roman")

**Example Usage**:
```json
{
  "tool": "harmony_analysis",
  "params": {
    "score_id": "bach_chorale",
    "analysis_type": "both"
  }
}
```

**Response Format**:
```json
{
  "success": true,
  "data": {
    "key": "A major",
    "roman_numeral_analysis": [
      {"measure": 1, "beat": 1, "roman": "I", "chord": "A major"},
      {"measure": 1, "beat": 3, "roman": "IV", "chord": "D major"},
      {"measure": 2, "beat": 1, "roman": "V", "chord": "E major"}
    ],
    "functional_analysis": [
      {"measure": 1, "beat": 1, "function": "tonic", "chord": "A major"},
      {"measure": 1, "beat": 3, "function": "subdominant", "chord": "D major"},
      {"measure": 2, "beat": 1, "function": "dominant", "chord": "E major"}
    ],
    "cadences": [
      {"measure": 8, "type": "authentic", "strength": "perfect"},
      {"measure": 16, "type": "half", "strength": "strong"}
    ]
  }
}
```

### voice_leading_analysis

**Description**: Analyze voice leading patterns, parallel motion, and voice crossing in multi-part music.

**Parameters**:
- `score_id` (string, required): ID of the score to analyze

**Example Usage**:
```json
{
  "tool": "voice_leading_analysis",
  "params": {
    "score_id": "bach_chorale"
  }
}
```

**Response Format**:
```json
{
  "success": true,
  "data": {
    "parallel_motion": [
      {"measures": "3-4", "voices": ["soprano", "alto"], "interval": "third", "type": "parallel"}
    ],
    "voice_crossings": [
      {"measure": 12, "voices": ["alto", "tenor"], "severity": "minor"}
    ],
    "voice_leading_quality": {
      "overall_score": 8.5,
      "smoothness": 9.0,
      "independence": 8.0,
      "range_usage": 8.5
    },
    "statistics": {
      "average_voice_range": {
        "soprano": "D4-G5",
        "alto": "G3-D5",
        "tenor": "C3-G4",
        "bass": "F2-C4"
      }
    }
  }
}
```

### pattern_recognition

**Description**: Identify and analyze musical patterns including melodic motifs, rhythmic patterns, and harmonic sequences.

**Parameters**:
- `score_id` (string, required): ID of the score to analyze
- `pattern_type` (string, optional): Type of pattern - "melodic", "rhythmic", "harmonic", "all" (default: "melodic")

**Example Usage**:
```json
{
  "tool": "pattern_recognition",
  "params": {
    "score_id": "bach_chorale",
    "pattern_type": "all"
  }
}
```

**Response Format**:
```json
{
  "success": true,
  "data": {
    "melodic_patterns": [
      {
        "pattern": "C-D-E-F",
        "occurrences": 3,
        "locations": ["m.1", "m.5", "m.9"],
        "variations": ["transposed", "inverted"]
      }
    ],
    "rhythmic_patterns": [
      {
        "pattern": "quarter-eighth-eighth-quarter",
        "occurrences": 8,
        "locations": ["m.2", "m.4", "m.6"]
      }
    ],
    "harmonic_patterns": [
      {
        "pattern": "I-vi-IV-V",
        "occurrences": 2,
        "locations": ["m.1-4", "m.9-12"]
      }
    ]
  }
}
```

### harmonization

**Description**: Generate harmonic accompaniment for melodies in various musical styles.

**Parameters**:
- `score_id` (string, required): ID of the melody to harmonize
- `style` (string, optional): Harmonization style - "bach", "jazz", "pop", "classical" (default: "bach")

**Example Usage**:
```json
{
  "tool": "harmonization",
  "params": {
    "score_id": "melody",
    "style": "bach"
  }
}
```

**Response Format**:
```json
{
  "success": true,
  "data": {
    "style": "bach",
    "harmonized_score_id": "melody_harmonized",
    "voice_count": 4,
    "chord_progression": ["I", "vi", "IV", "V", "I"],
    "voice_leading_score": 8.7,
    "style_authenticity": 0.89
  }
}
```

### counterpoint_generation

**Description**: Generate counterpoint lines following species counterpoint rules (1st through 5th species).

**Parameters**:
- `score_id` (string, required): ID of the cantus firmus
- `species` (integer, optional): Species type 1-5 (default: 1)

**Species Types**:
- **1st Species**: Note against note
- **2nd Species**: Two notes against one
- **3rd Species**: Four notes against one
- **4th Species**: Syncopation
- **5th Species**: Florid counterpoint (mixed species)

**Example Usage**:
```json
{
  "tool": "counterpoint_generation",
  "params": {
    "score_id": "cantus_firmus",
    "species": 1
  }
}
```

### style_imitation

**Description**: Analyze musical style characteristics and generate new music following those patterns.

**Parameters**:
- `score_id` (string, required): ID of the reference score for style analysis
- `target_style` (string, optional): Target composer or style - "bach", "mozart", "chopin", "beethoven", "jazz", "blues", "folk" (default: "bach")

**Example Usage**:
```json
{
  "tool": "style_imitation",
  "params": {
    "score_id": "reference_piece",
    "target_style": "bach"
  }
}
```

### health_check

**Description**: Check the health and operational status of the music analysis server.

**Parameters**: None

**Example Usage**:
```json
{
  "tool": "health_check",
  "params": {}
}
```

**Response Format**:
```json
{
  "success": true,
  "data": {
    "status": "healthy",
    "server": "Music21 MCP Server - Minimal",
    "adapter_version": "1.0.0",
    "tools_available": 13,
    "core_service_healthy": true,
    "memory_usage": "24.5 MB",
    "uptime": "2h 15m"
  }
}
```

## Error Handling

All tools provide consistent error handling with detailed error messages:

```json
{
  "success": false,
  "error": {
    "type": "ValidationError",
    "message": "Score with ID 'nonexistent' not found",
    "code": "SCORE_NOT_FOUND",
    "details": {
      "available_scores": ["bach_chorale", "mozart_sonata"]
    }
  }
}
```

## Security Considerations

- **No Code Execution**: The server only analyzes existing music files and does not execute arbitrary code
- **Read-Only File Access**: File system access is limited to reading music files
- **No Network Access**: The server does not make external network requests
- **Input Validation**: All inputs are validated and sanitized before processing
- **Resource Limits**: Memory and processing time limits prevent abuse

## Performance Notes

- **Parallel Processing**: Complex analyses use parallel processing when possible
- **Caching**: Frequently accessed scores and analysis results are cached
- **Memory Management**: Automatic cleanup of unused scores to prevent memory leaks
- **Progress Reporting**: Long-running analyses provide progress updates
- **Timeout Protection**: All operations have reasonable timeout limits

## Integration Examples

See the main documentation for complete integration examples with Claude Desktop, VS Code, and other MCP-compatible applications.