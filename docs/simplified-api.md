# Music21 MCP Server - Simplified API Documentation

## Overview

The Music21 MCP Server provides music analysis capabilities through a simplified, stable API. This server has been streamlined to focus on core functionality with 100% reliability.

## Quick Start

```python
# Import a score
result = await import_score("my_score", "bach/bwv66.6")

# Analyze the key
key_result = await analyze_key("my_score")
print(f"Key: {key_result['key']}, Confidence: {key_result['confidence']}")

# Get chord analysis
chord_result = await analyze_chords("my_score")
print(f"Found {chord_result['total_chords']} chords")

# Export to MIDI
export_result = await export_score("my_score", "midi")
```

## Available Tools

### 1. import_score

Import musical scores from various sources.

**Parameters:**
- `score_id` (str): Unique identifier for the score
- `source` (str): Source of the score
  - File path: `/path/to/file.xml`
  - Corpus path: `bach/bwv66.6`
  - Text notation: `C4 D4 E4 F4 G4`
- `source_type` (str, optional): Type of source ('file', 'corpus', 'text', 'auto')

**Returns:**
```json
{
  "status": "success",
  "score_id": "my_score",
  "num_notes": 156,
  "num_measures": 20,
  "num_parts": 4,
  "source_type": "corpus"
}
```

**Examples:**
```python
# Import from music21 corpus
await import_score("bach_chorale", "bach/bwv66.6")

# Import from file
await import_score("my_piece", "/Users/me/scores/piece.xml")

# Import from text notation
await import_score("melody", "C4 E4 G4 C5 B4 G4 E4 C4")

# Import with explicit type
await import_score("abc_tune", "X:1\nT:My Tune\nK:C\nCDEF|GABc|", "text")
```

### 2. list_scores

List all imported scores with basic information.

**Parameters:** None

**Returns:**
```json
{
  "status": "success",
  "scores": [
    {
      "score_id": "bach_chorale",
      "num_notes": 156,
      "num_parts": 4
    },
    {
      "score_id": "melody",
      "num_notes": 8,
      "num_parts": 1
    }
  ],
  "total_count": 2
}
```

**Example:**
```python
result = await list_scores()
for score in result['scores']:
    print(f"{score['score_id']}: {score['num_notes']} notes")
```

### 3. analyze_key

Analyze the key of a score using various methods.

**Parameters:**
- `score_id` (str): ID of the score to analyze
- `method` (str, optional): Analysis method
  - `"default"`: Music21's default key detection
  - `"krumhansl"`: Krumhansl-Schmuckler algorithm
  - `"aarden"`: Aarden key profiles
  - `"temperley"`: Temperley-Kostka-Payne algorithm

**Returns:**
```json
{
  "status": "success",
  "score_id": "bach_chorale",
  "key": "f# minor",
  "confidence": 0.941,
  "method": "default",
  "alternatives": [
    {"key": "A major", "confidence": 0.823},
    {"key": "D major", "confidence": 0.756}
  ]
}
```

**Examples:**
```python
# Default analysis
result = await analyze_key("bach_chorale")

# Specific method
result = await analyze_key("bach_chorale", "krumhansl")

# Check confidence
if result['confidence'] > 0.8:
    print(f"Confident: {result['key']}")
else:
    print(f"Uncertain, alternatives: {result['alternatives']}")
```

### 4. analyze_chords

Extract and analyze chords from a score.

**Parameters:**
- `score_id` (str): ID of the score to analyze
- `include_roman_numerals` (bool, optional): Include Roman numeral analysis

**Returns:**
```json
{
  "status": "success",
  "score_id": "bach_chorale",
  "total_chords": 51,
  "chord_progression": [
    {
      "index": 0,
      "pitches": ["F#3", "A3", "C#4", "F#4"],
      "root": "F#",
      "quality": "minor",
      "measure": 1,
      "roman_numeral": "i"
    },
    ...
  ],
  "includes_roman_numerals": true
}
```

**Examples:**
```python
# Basic chord analysis
result = await analyze_chords("bach_chorale")
print(f"Total chords: {result['total_chords']}")

# With Roman numerals
result = await analyze_chords("bach_chorale", True)
for chord in result['chord_progression'][:5]:
    print(f"{chord['roman_numeral']}: {chord['pitches']}")
```

### 5. get_score_info

Get comprehensive metadata about a score.

**Parameters:**
- `score_id` (str): ID of the score

**Returns:**
```json
{
  "status": "success",
  "score_id": "bach_chorale",
  "title": "Chorale BWV 66.6",
  "composer": "J.S. Bach",
  "date": "1735",
  "num_parts": 4,
  "num_measures": 20,
  "num_notes": 156,
  "duration_quarters": 40.0,
  "time_signatures": [
    {"measure": 1, "signature": "4/4"}
  ],
  "key_signatures": [
    {"measure": 1, "sharps": 2}
  ],
  "tempo_markings": [
    {"measure": 1, "bpm": 72, "unit": "quarter"}
  ]
}
```

**Example:**
```python
info = await get_score_info("bach_chorale")
print(f"Title: {info['title']} by {info['composer']}")
print(f"Duration: {info['duration_quarters']} quarter notes")
```

### 6. export_score

Export a score to various formats.

**Parameters:**
- `score_id` (str): ID of the score to export
- `format` (str): Export format
  - `"musicxml"`: MusicXML format (.xml)
  - `"midi"`: MIDI format (.mid)
  - `"lilypond"`: LilyPond format (.ly)
  - `"abc"`: ABC notation (.abc)
  - `"pdf"`: PDF score (requires LilyPond)
- `file_path` (str, optional): Output path (auto-generated if not provided)

**Returns:**
```json
{
  "status": "success",
  "score_id": "bach_chorale",
  "format": "midi",
  "file_path": "/tmp/tmpx1y2z3.mid",
  "file_size": 2458
}
```

**Examples:**
```python
# Export to MIDI with auto-generated path
result = await export_score("bach_chorale", "midi")
print(f"Exported to: {result['file_path']}")

# Export to specific path
result = await export_score("bach_chorale", "musicxml", "/Users/me/bach.xml")

# Export to multiple formats
for fmt in ["musicxml", "midi", "abc"]:
    result = await export_score("bach_chorale", fmt)
    print(f"{fmt}: {result['file_path']}")
```

### 7. delete_score

Remove a score from memory.

**Parameters:**
- `score_id` (str): ID of the score to delete

**Returns:**
```json
{
  "status": "success",
  "message": "Score 'bach_chorale' deleted",
  "remaining_scores": 3
}
```

**Example:**
```python
# Delete a score
result = await delete_score("bach_chorale")
print(result['message'])

# Clean up all scores
scores_list = await list_scores()
for score in scores_list['scores']:
    await delete_score(score['score_id'])
```

## Error Handling

All tools return consistent error responses:

```json
{
  "status": "error",
  "message": "Detailed error description"
}
```

Always check the status before using results:

```python
result = await analyze_key("my_score")
if result["status"] == "success":
    print(f"Key: {result['key']}")
else:
    print(f"Error: {result['message']}")
```

## Common Workflows

### 1. Complete Analysis Pipeline

```python
# Import a Bach chorale
await import_score("bach", "bach/bwv66.6")

# Perform full analysis
key = await analyze_key("bach")
chords = await analyze_chords("bach", include_roman_numerals=True)
info = await get_score_info("bach")

print(f"Analysis of {info['title']}:")
print(f"  Key: {key['key']} (confidence: {key['confidence']:.2f})")
print(f"  Chords: {chords['total_chords']}")
print(f"  Duration: {info['duration_quarters']} quarters")

# Export results
await export_score("bach", "midi", "bach_chorale.mid")
await export_score("bach", "musicxml", "bach_chorale.xml")
```

### 2. Batch Processing

```python
# Process multiple corpus works
corpus_works = [
    "bach/bwv66.6",
    "mozart/k155/movement1",
    "schubert/impromptu1",
    "common/tinyscore"
]

results = []
for i, work in enumerate(corpus_works):
    score_id = f"score_{i}"
    
    # Import
    import_result = await import_score(score_id, work)
    if import_result["status"] != "success":
        continue
    
    # Analyze
    key_result = await analyze_key(score_id)
    
    results.append({
        "work": work,
        "key": key_result["key"],
        "confidence": key_result["confidence"]
    })

# Print results
for r in results:
    print(f"{r['work']}: {r['key']} ({r['confidence']:.2f})")
```

### 3. Text-to-MIDI Conversion

```python
# Create a simple melody
melody = "C4 E4 G4 C5 B4 G4 E4 C4"
await import_score("my_melody", melody)

# Check what we created
info = await get_score_info("my_melody")
print(f"Created melody with {info['num_notes']} notes")

# Export to MIDI
result = await export_score("my_melody", "midi", "my_melody.mid")
print(f"Saved to: {result['file_path']}")
```

## Performance Considerations

1. **Score Storage**: Scores are kept in memory. The server supports up to 100 scores by default.

2. **Analysis Caching**: Analysis results are cached for 1 hour to improve performance on repeated operations.

3. **File Operations**: Export operations create temporary files if no path is specified. Clean up files when done.

4. **Corpus Access**: First access to corpus files may be slower as music21 downloads them.

## Limitations

1. **Memory**: Large scores (orchestral works) consume significant memory
2. **Formats**: Not all music21 formats are supported for import/export
3. **Analysis**: Complex analytical features from the full server are not available
4. **Real-time**: Not suitable for real-time audio processing

## Migration from Complex Server

If migrating from the complex server, note these changes:

1. **Simplified tool names**: No more nested analysis methods
2. **Fewer parameters**: Most tools use sensible defaults
3. **Consistent returns**: All tools return `status` field
4. **No advanced features**: Counterpoint, voice leading, etc. not available
5. **Stable API**: This API will remain stable for backward compatibility

## Examples Repository

Find more examples at: https://github.com/Bright-L01/music21-mcp-server/tree/main/examples

## Support

For issues or questions:
- GitHub Issues: https://github.com/Bright-L01/music21-mcp-server/issues
- Documentation: https://github.com/Bright-L01/music21-mcp-server/wiki