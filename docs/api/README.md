# Music21 MCP Server API Documentation

## Overview

The Music21 MCP Server provides a comprehensive set of tools for music analysis, generation, and manipulation through the Model Context Protocol (MCP). All tools follow consistent patterns for input/output and error handling.

## Table of Contents

1. [Core Concepts](#core-concepts)
2. [Import & Export Tools](#import--export-tools)
3. [Analysis Tools](#analysis-tools)
4. [Generation Tools](#generation-tools)
5. [Utility Tools](#utility-tools)
6. [Error Handling](#error-handling)
7. [Rate Limits & Quotas](#rate-limits--quotas)

## Core Concepts

### Score Management
All operations work with "scores" - musical pieces stored in memory with unique identifiers. Scores must be imported before they can be analyzed or manipulated.

### Tool Response Format
All tools return JSON responses with consistent structure:
```json
{
  "status": "success",
  "data": { ... },
  "metadata": { ... }
}
```

Error responses:
```json
{
  "error": "Error message",
  "error_type": "ValueError",
  "details": { ... }
}
```

## Import & Export Tools

### import_score
Import a musical score from various sources.

**Parameters:**
- `score_id` (string, required): Unique identifier for the score
- `source` (string, required): File path, corpus path, or note sequence
- `source_type` (string, optional): "file", "corpus", "text", or "auto" (default: "auto")

**Example:**
```json
{
  "tool": "import_score",
  "arguments": {
    "score_id": "my_bach_piece",
    "source": "bach/bwv66.6",
    "source_type": "corpus"
  }
}
```

**Response:**
```json
{
  "status": "success",
  "score_id": "my_bach_piece",
  "source_type": "corpus",
  "parts": 4,
  "measures": 40,
  "title": "Chorale 'Christ unser Herr zum Jordan kam'",
  "composer": "J.S. Bach"
}
```

### export_score
Export a score to various formats.

**Parameters:**
- `score_id` (string, required): Score identifier
- `format` (string, required): "musicxml", "midi", "lilypond", "abc", or "text"
- `output_path` (string, optional): Output file path (default: temp directory)

**Example:**
```json
{
  "tool": "export_score",
  "arguments": {
    "score_id": "my_bach_piece",
    "format": "midi",
    "output_path": "/path/to/output.mid"
  }
}
```

## Analysis Tools

### analyze_key
Determine the key of a musical piece using various algorithms.

**Parameters:**
- `score_id` (string, required): Score identifier
- `algorithm` (string, optional): "krumhansl", "aarden", "bellman-budge", or "temperley" (default: "krumhansl")

**Response:**
```json
{
  "score_id": "my_bach_piece",
  "key": "G major",
  "tonic": "G",
  "mode": "major",
  "confidence": 0.891,
  "algorithm": "krumhansl",
  "alternative_keys": [
    {"key": "E minor", "confidence": 0.623},
    {"key": "D major", "confidence": 0.412}
  ]
}
```

### analyze_harmony
Analyze harmonic content and progressions.

**Parameters:**
- `score_id` (string, required): Score identifier
- `include_roman` (boolean, optional): Include Roman numeral analysis (default: true)
- `include_functions` (boolean, optional): Include functional harmony (default: false)

**Response:**
```json
{
  "score_id": "my_bach_piece",
  "key_context": "G major",
  "harmonic_rhythm": 2.5,
  "progression_analysis": {
    "most_common": ["I-V", "ii-V", "V-I"],
    "cadences": [
      {"measure": 8, "type": "perfect_authentic"},
      {"measure": 16, "type": "half"}
    ]
  },
  "modulations": [
    {"measure": 24, "from_key": "G major", "to_key": "D major"}
  ]
}
```

### analyze_voice_leading
Analyze voice leading and part writing.

**Parameters:**
- `score_id` (string, required): Score identifier
- `check_parallels` (boolean, optional): Check for parallel fifths/octaves (default: true)
- `check_spacing` (boolean, optional): Check voice spacing (default: true)

**Response:**
```json
{
  "score_id": "my_bach_piece",
  "voice_count": 4,
  "parallel_intervals": {
    "fifths": [{"measure": 12, "voices": [1, 2]}],
    "octaves": []
  },
  "voice_crossings": [{"measure": 8, "voices": [2, 3]}],
  "smooth_voice_leading_score": 0.89,
  "average_leap_size": 2.3
}
```

### recognize_patterns
Identify musical patterns and motifs.

**Parameters:**
- `score_id` (string, required): Score identifier
- `pattern_types` (array, optional): ["melodic", "rhythmic", "harmonic"] (default: all)
- `min_occurrences` (integer, optional): Minimum pattern occurrences (default: 2)

**Response:**
```json
{
  "patterns": {
    "melodic": [
      {
        "pattern": "C-D-E-F",
        "occurrences": 5,
        "measures": [1, 5, 12, 20, 28]
      }
    ],
    "rhythmic": [
      {
        "pattern": "quarter-eighth-eighth-quarter",
        "occurrences": 8
      }
    ]
  }
}
```

## Generation Tools

### harmonize
Generate harmonization for a melody.

**Parameters:**
- `melody_score_id` (string, required): Melody score identifier
- `output_score_id` (string, required): Output score identifier
- `style` (string, optional): "bach_chorale", "jazz", "classical" (default: "bach_chorale")
- `voices` (integer, optional): Number of voices (default: 4)

**Response:**
```json
{
  "status": "success",
  "output_score_id": "harmonized_melody",
  "style": "bach_chorale",
  "parts_created": 4,
  "harmony_stats": {
    "chord_vocabulary": 12,
    "predominant_progression": "I-IV-V-I"
  }
}
```

### generate_counterpoint
Create counterpoint following classical rules.

**Parameters:**
- `cantus_firmus_id` (string, required): Cantus firmus score identifier
- `output_score_id` (string, required): Output score identifier
- `species` (integer, required): Species number (1-5)
- `mode` (string, optional): "major", "minor", "dorian", etc. (default: "major")
- `voice_position` (string, optional): "above" or "below" (default: "above")

**Response:**
```json
{
  "status": "success",
  "output_score_id": "counterpoint_result",
  "species": 1,
  "mode": "major",
  "voices_created": 1,
  "rule_violations": [],
  "intervals_used": ["P8", "P5", "M3", "m3", "M6", "m6"]
}
```

### imitate_style
Generate music imitating the style of an input piece.

**Parameters:**
- `source_score_id` (string, required): Source score for style analysis
- `output_score_id` (string, required): Output score identifier
- `measures` (integer, optional): Number of measures to generate (default: 16)
- `preserve_harmony` (boolean, optional): Maintain harmonic structure (default: false)

**Response:**
```json
{
  "status": "success",
  "output_score_id": "style_imitation",
  "measures_generated": 16,
  "style_characteristics": {
    "average_pitch": "E4",
    "rhythm_complexity": 0.65,
    "harmonic_rhythm": 2.0,
    "texture": "homophonic"
  }
}
```

## Utility Tools

### list_scores
List all loaded scores.

**Parameters:** None

**Response:**
```json
{
  "scores": [
    {
      "id": "bach_invention",
      "title": "Invention No. 1",
      "last_accessed": "2024-01-15T10:30:00Z"
    }
  ],
  "count": 1,
  "max_allowed": 100
}
```

### get_score_info
Get detailed information about a score.

**Parameters:**
- `score_id` (string, required): Score identifier

**Response:**
```json
{
  "score_id": "bach_invention",
  "title": "Invention No. 1 in C major",
  "composer": "Johann Sebastian Bach",
  "parts": 2,
  "measures": 22,
  "duration_seconds": 45.2,
  "time_signatures": ["4/4"],
  "key_signatures": ["C major"],
  "tempo_markings": [{"bpm": 120, "text": "Allegro"}],
  "instruments": ["Piano"]
}
```

### delete_score
Remove a score from memory.

**Parameters:**
- `score_id` (string, required): Score identifier

**Response:**
```json
{
  "status": "success",
  "message": "Score 'bach_invention' deleted"
}
```

### health_check
Check server health and resource usage.

**Parameters:** None

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "uptime_seconds": 3600,
  "memory": {
    "used_mb": 512.3,
    "limit_mb": 2048,
    "percent": 25.0
  },
  "scores": {
    "count": 15,
    "max": 100
  },
  "circuit_breakers": {
    "import": "closed",
    "analysis": "closed",
    "export": "closed",
    "generation": "closed"
  }
}
```

### cleanup_memory
Force memory cleanup and garbage collection.

**Parameters:** None

**Response:**
```json
{
  "status": "success",
  "memory_before_mb": 812.5,
  "memory_after_mb": 423.1,
  "freed_mb": 389.4,
  "scores_remaining": 10
}
```

## Error Handling

### Common Error Types

1. **Score Not Found**
```json
{
  "error": "Score 'unknown_id' not found",
  "error_type": "NotFoundError"
}
```

2. **Rate Limit Exceeded**
```json
{
  "error": "Rate limit exceeded, please try again later",
  "error_type": "RateLimitError",
  "retry_after_seconds": 60
}
```

3. **Circuit Breaker Open**
```json
{
  "error": "Service temporarily unavailable, retry in 60s",
  "error_type": "CircuitBreakerError",
  "retry_after_seconds": 60
}
```

4. **Invalid Parameters**
```json
{
  "error": "Invalid species: 6. Must be between 1-5",
  "error_type": "ValidationError",
  "parameter": "species"
}
```

## Rate Limits & Quotas

### Default Limits
- **Requests per minute**: 100
- **Max scores in memory**: 100
- **Max score size**: 50 MB
- **Memory limit**: 2048 MB

### Circuit Breaker Settings
- **Failure threshold**: 5 consecutive failures
- **Recovery timeout**: 60 seconds
- **Half-open test requests**: 1

### Rate Limiter Settings
- **Token bucket rate**: 100 requests/minute
- **Burst capacity**: 10 requests
- **Refill interval**: 0.6 seconds

## Best Practices

1. **Score Lifecycle Management**
   - Import scores once and reuse the score_id
   - Delete scores when no longer needed
   - Use meaningful score_id values

2. **Error Handling**
   - Always check for error responses
   - Implement exponential backoff for retries
   - Respect rate limits and circuit breaker states

3. **Performance Optimization**
   - Batch related operations when possible
   - Use appropriate analysis algorithms for your use case
   - Monitor memory usage with health_check

4. **Generation Parameters**
   - Start with default parameters
   - Adjust based on output quality
   - Use style-appropriate settings