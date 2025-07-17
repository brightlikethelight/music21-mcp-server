# Musical Accuracy Validation Tests

This directory contains comprehensive tests that validate the musical correctness of the music21-mcp-server analysis tools. Unlike unit tests that verify code functionality, these tests ensure that the musical analysis results are accurate according to established music theory principles.

## Overview

The validation suite tests the following musical analysis capabilities:

1. **Key Detection Accuracy** - Validates key analysis against known compositions
2. **Chord Progression Analysis** - Tests chord identification and Roman numeral analysis
3. **Pattern Recognition** - Validates melodic and rhythmic pattern detection
4. **Voice Leading Analysis** - Checks for proper voice leading and forbidden parallels
5. **Harmony Analysis** - Tests functional harmony and non-chord tone detection
6. **Counterpoint Generation** - Validates species counterpoint rules
7. **Style Imitation** - Tests style-specific characteristic detection

## Test Files

### `test_musical_accuracy.py`
Core validation tests using music21 corpus examples and synthetic test cases:
- Bach chorale key detection with known keys
- Common progression analysis (ii-V-I, etc.)
- Melodic sequence recognition
- Voice leading smoothness validation
- Counterpoint species rules checking
- Style-specific feature detection

### `test_known_compositions.py`
Tests using specific well-known compositions:
- Bach: WTC Prelude in C Major (BWV 846)
- Mozart: Piano Sonata K. 545
- Beethoven: Moonlight Sonata (Op. 27 No. 2)
- Chopin: Prelude Op. 28 No. 4 in E minor
- Pachelbel: Canon in D
- Debussy: Impressionistic harmony examples
- Schoenberg: Twelve-tone technique

### `test_config.py`
Configuration and test data including:
- Expected results for known pieces
- Music theory rules and constraints
- Style characteristics definitions
- Chord progression templates

## Running the Tests

### Run all validation tests:
```bash
python tests/validation/run_validation_tests.py
```

### Run specific test file:
```bash
pytest tests/validation/test_musical_accuracy.py -v
```

### Run with coverage:
```bash
pytest tests/validation/ --cov=music21_mcp.tools --cov-report=html
```

## Test Data Sources

The tests use:
1. **music21 Corpus** - Bach chorales, Mozart sonatas, etc.
2. **Synthetic Examples** - Carefully constructed musical examples
3. **Known Progressions** - Standard chord progressions in various styles

## Validation Criteria

### Key Detection
- Correct key identification with >80% confidence for tonal music
- Low confidence (<50%) for atonal music
- Proper detection of modulations

### Chord Analysis
- Accurate chord symbol identification
- Correct Roman numeral analysis
- Proper inversion detection
- Extended harmony recognition (7ths, 9ths, etc.)

### Pattern Recognition
- Detection of exact repetitions
- Transposed pattern recognition
- Inverted/retrograde pattern detection
- Motivic development tracking

### Voice Leading
- No parallel fifths/octaves
- Smooth voice motion (minimal leaps)
- Proper resolution of dissonances
- Voice range compliance

### Counterpoint
- Species-specific rule compliance
- Proper consonance/dissonance treatment
- Correct cadential formulas
- Independent voice motion

### Style Imitation
- Period-appropriate harmony
- Characteristic rhythmic patterns
- Proper texture and voicing
- Style-specific ornamentations

## Adding New Tests

To add new validation tests:

1. **Choose Test Category** - Determine if it's accuracy or known composition test
2. **Select Test Data** - Use corpus examples or create synthetic examples
3. **Define Expected Results** - Document the correct musical analysis
4. **Write Test Case** - Follow existing patterns for consistency
5. **Update Documentation** - Add to this README and test_config.py

Example test structure:
```python
@pytest.mark.asyncio
async def test_new_musical_feature(self):
    """Test description"""
    # Load or create test score
    score = self.load_corpus_score("composer/piece", "test_id")
    
    # Run analysis tool
    tool = AnalysisTool()
    result = await tool.execute(score_id="test_id", parameters=...)
    
    # Validate results against music theory
    assert result["success"]
    assert expected_value == actual_value
```

## Interpreting Results

### Success Criteria
- All tests passing indicates accurate musical analysis
- Individual test failures point to specific accuracy issues
- Coverage report shows which tools are thoroughly validated

### Common Issues
1. **Key Detection Failures** - May indicate algorithm tuning needed
2. **Chord Misidentification** - Could need enharmonic handling
3. **Pattern False Positives** - Might require stricter matching criteria
4. **Voice Leading Errors** - May need rule refinement

## Future Enhancements

1. **Expand Corpus Coverage** - Add more diverse musical examples
2. **Machine Learning Validation** - Compare against ML model predictions
3. **User Study Integration** - Validate against human expert analysis
4. **Performance Metrics** - Add timing and efficiency measurements
5. **Cross-Cultural Music** - Add non-Western music validation

## References

The validation tests are based on:
- Aldwell & Schachter: "Harmony and Voice Leading"
- Kostka & Payne: "Tonal Harmony"
- Gauldin: "A Practical Approach to 18th Century Counterpoint"
- Various music21 documentation and examples