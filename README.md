# music21-mcp-server

A Model Context Protocol (MCP) server that provides comprehensive music analysis and composition capabilities through the music21 library.

## Overview

This MCP server exposes music21's powerful music analysis features through a standardized API, enabling AI assistants and other tools to perform sophisticated musical analysis, including:

- **Score I/O**: Import/export support for MIDI, MusicXML, ABC notation, Kern, and more
- **Music Theory Analysis**: Key detection, scale analysis, chord progressions, interval calculations, chromatic analysis
- **Advanced Harmony**: Secondary dominants, phrase structure, interval vectors, chromatic functions
- **Rhythm Analysis**: Tempo detection, meter analysis, rhythmic pattern recognition, syncopation analysis, beat strength
- **Integrated Tools**: Comprehensive analysis, batch processing, formatted report generation

## Installation

```bash
# Clone the repository
git clone https://github.com/Bright-L01/music21-mcp-server.git
cd music21-mcp-server

# Install dependencies
pip install -e .
```

## Usage

### Starting the Server

```bash
python -m music21_mcp.server
```

The server will start on `http://localhost:8000`.

### Core Features

#### 1. Score Import/Export

Import scores from various formats:
```python
# Import a MIDI file
import_score(score_id="my_score", source="path/to/file.mid")

# Import MusicXML with validation
import_score(score_id="my_score", source="path/to/score.xml", validate=True)

# Export to different formats
export_score(score_id="my_score", format="musicxml")
export_score(score_id="my_score", format="midi", options={"velocity_map": "expressive"})
```

#### 2. Music Theory Analysis

Basic analysis tools:
```python
# Key detection with multiple algorithms
analyze_key(score_id="my_score", method="hybrid", window_size=4)

# Scale analysis
analyze_scale(score_id="my_score", include_modes=True, include_exotic=True)

# Chord progression analysis
analyze_chord_progressions(score_id="my_score", analysis_type="roman")
```

Advanced theory tools:
```python
# Identify scales in passages
identify_scale(score_id="my_score", start_measure=1, end_measure=8)

# Calculate interval vectors
interval_vector(score_id="my_score", start_measure=1, end_measure=16)

# Chromatic analysis
chromatic_analysis(score_id="my_score", include_voice_leading=True)

# Detect secondary dominants
secondary_dominants(score_id="my_score")

# Analyze phrase structure
phrase_structure(score_id="my_score", include_motives=True)
```

#### 3. Rhythm Analysis

Comprehensive rhythm tools:
```python
# Full rhythm analysis
analyze_rhythm(score_id="my_score", include_patterns=True)

# Tempo analysis
analyze_tempo(score_id="my_score")

# Find specific rhythmic patterns
find_rhythmic_patterns(score_id="my_score", min_length=4, pattern_type="ostinato")

# Beat strength analysis
beat_strength(score_id="my_score")
```

#### 4. Integrated Analysis

Unified analysis tools:
```python
# Run all analyses on a score
comprehensive_analysis(score_id="my_score", include_advanced=True)

# Batch process multiple scores
batch_analysis(score_ids=["score1", "score2"], analysis_types=["key", "rhythm"])

# Generate formatted reports
generate_report(score_id="my_score", report_format="educational")
```

## Architecture

The project follows a modular architecture:

```
src/music21_mcp/
├── server.py              # Main MCP server and endpoints
├── core/
│   ├── score_parser.py    # Multi-format parsing with error recovery
│   ├── theory_analyzer.py # Music theory analysis engine
│   ├── rhythm_analyzer.py # Rhythm and tempo analysis
│   └── advanced_theory.py # Advanced harmony and phrase analysis
├── analysis/             # Advanced analysis modules (Phase 2)
├── visualization/        # Score rendering (Phase 2)
└── utils/               # Utilities and helpers
```

## Development

### Running Tests

```bash
pytest tests/
```

### Code Quality

```bash
# Format code
black src/

# Check types
mypy src/

# Lint
flake8 src/
```

## API Reference

See the [API documentation](docs/api.md) for detailed endpoint descriptions.

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please read our [contributing guidelines](CONTRIBUTING.md) before submitting PRs.

## Acknowledgments

Built on top of the excellent [music21](https://github.com/cuthbertLab/music21) library by Michael Scott Cuthbert and cuthbertLab.