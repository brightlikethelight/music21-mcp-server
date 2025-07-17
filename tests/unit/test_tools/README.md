# Unit Tests for Music21 MCP Server Tools

This directory contains comprehensive unit tests for the critical tools in the music21-mcp-server project.

## Test Coverage

The following tools have comprehensive test suites:

### 1. ImportScoreTool (`test_import_tool.py`)
- **Test Cases**: 30+
- **Coverage Areas**:
  - File import (MusicXML, MIDI, ABC, etc.)
  - Corpus import from music21
  - Text notation parsing
  - Auto-detection of source types
  - Error handling for invalid inputs
  - Progress reporting
  - Metadata extraction
  - Edge cases (empty scores, single notes, etc.)

### 2. KeyAnalysisTool (`test_key_analysis_tool.py`)
- **Test Cases**: 25+
- **Coverage Areas**:
  - Key detection with multiple algorithms (Krumhansl, Aarden, Temperley, Bellman)
  - Consensus finding across algorithms
  - Major and minor key detection
  - Confidence scoring
  - Multi-part score handling
  - Chromatic/ambiguous score analysis
  - Edge cases and error handling
  - Real corpus piece testing

### 3. PatternRecognitionTool (`test_pattern_recognition_tool.py`)
- **Test Cases**: 25+
- **Coverage Areas**:
  - Melodic sequence detection
  - Rhythmic pattern finding
  - Motivic analysis with transformations
  - Contour analysis
  - Interval pattern recognition
  - Syncopation detection
  - Phrase structure analysis
  - Cross-rhythm detection
  - Similarity threshold testing
  - Pattern deduplication

### 4. HarmonyAnalysisTool (`test_harmony_analysis_tool.py`)
- **Test Cases**: 20+
- **Coverage Areas**:
  - Roman numeral analysis
  - Chord progression detection (I-IV-V-I, ii-V-I, etc.)
  - Functional harmony categorization
  - Harmonic rhythm analysis
  - Chord quality detection
  - Inversion detection
  - Major and minor key handling
  - Mixed content handling (notes + chords)

## Running the Tests

### Run all tool tests with coverage:
```bash
pytest tests/unit/test_tools/ -v --cov=src/music21_mcp/tools --cov-report=html
```

### Run tests for a specific tool:
```bash
pytest tests/unit/test_tools/test_key_analysis_tool.py -v
```

### Run with coverage for a specific tool:
```bash
pytest tests/unit/test_tools/test_pattern_recognition_tool.py -v \
  --cov=src/music21_mcp/tools/pattern_recognition_tool \
  --cov-report=term-missing
```

### Run the test summary script:
```bash
python tests/unit/test_tools_coverage_summary.py
```

## Test Structure

Each test file follows a consistent structure:

1. **Fixtures**: Reusable test data (scores, progressions, etc.)
2. **Success Cases**: Testing normal operation
3. **Error Cases**: Testing error handling
4. **Edge Cases**: Testing boundary conditions
5. **Integration**: Testing with real music21 corpus data

## Writing New Tests

When adding tests for new tools:

1. Create a new file: `test_<tool_name>_tool.py`
2. Import the tool class and necessary music21 modules
3. Create fixtures for common test data
4. Test the following areas:
   - Basic instantiation
   - Successful execution with valid inputs
   - All parameters and options
   - Error handling for invalid inputs
   - Edge cases (empty data, single elements, etc.)
   - Progress reporting
   - Real corpus data when applicable

## Coverage Goals

- Each critical tool should have **90%+ code coverage**
- All public methods should be tested
- All error paths should be tested
- Edge cases should be thoroughly covered

## Continuous Integration

These tests are designed to run in CI/CD pipelines. They:
- Use only built-in music21 corpus data
- Create temporary files when needed
- Clean up after themselves
- Run independently without external dependencies