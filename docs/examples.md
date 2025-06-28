# Music21 MCP Server - Usage Examples

This guide provides practical examples of using the music21 MCP server for various music analysis tasks.

## Table of Contents
1. [Getting Started](#getting-started)
2. [Score Import and Export](#score-import-and-export)
3. [Basic Music Theory Analysis](#basic-music-theory-analysis)
4. [Advanced Theory Analysis](#advanced-theory-analysis)
5. [Rhythm Analysis](#rhythm-analysis)
6. [Integrated Analysis Workflows](#integrated-analysis-workflows)
7. [Real-World Use Cases](#real-world-use-cases)

## Getting Started

First, start the MCP server:

```bash
python -m music21_mcp.server
```

The server will be available at `http://localhost:8000`.

## Score Import and Export

### Import a MIDI File

```python
# Import a MIDI file from disk
response = await import_score(
    score_id="beethoven_5",
    source="/path/to/beethoven_symphony_5.mid",
    validate=True
)

# Response includes metadata
print(f"Imported {response['metadata']['title']}")
print(f"Duration: {response['metadata']['duration_quarters']} quarters")
print(f"Parts: {response['metadata']['part_count']}")
```

### Import MusicXML with Auto-Detection

```python
# Server auto-detects format
response = await import_score(
    score_id="bach_invention",
    source="/path/to/bach_invention.xml",
    source_type="auto",  # Auto-detect format
    encoding="auto"      # Auto-detect encoding
)
```

### Import ABC Notation from String

```python
abc_content = """
X:1
T:Scarborough Fair
M:3/4
K:Am
A2A|c2c|B2G|A3|
c3|e2e|d2B|c3|
"""

response = await import_score(
    score_id="scarborough_fair",
    source={"content": abc_content},
    source_type="abc"
)
```

### Export to Different Formats

```python
# Export to compressed MusicXML
response = await export_score(
    score_id="beethoven_5",
    format="musicxml",
    options={"compress": True, "pretty_print": True}
)

# Export to MIDI with expressive velocity
response = await export_score(
    score_id="bach_invention",
    format="midi",
    options={"velocity_map": "expressive"}
)

# Export to PNG image (requires LilyPond)
response = await export_score(
    score_id="scarborough_fair",
    format="png",
    options={"dpi": 300, "transparent": True}
)
```

## Basic Music Theory Analysis

### Key Detection

```python
# Detect key using hybrid method
response = await analyze_key(
    score_id="beethoven_5",
    method="hybrid",  # Combines multiple algorithms
    window_size=8,    # Analyze in 8-measure windows
    confidence_threshold=0.6
)

print(f"Key: {response['key']}")
print(f"Confidence: {response['confidence']:.2%}")

# Check for modulations
if response['modulations']:
    for mod in response['modulations']:
        print(f"Modulation at measure {mod['measure']}: "
              f"{mod['from_key']} → {mod['to_key']}")
```

### Scale Analysis

```python
# Analyze scales including modes and exotic scales
response = await analyze_scale(
    score_id="debussy_prelude",
    include_modes=True,
    include_exotic=True
)

# Find whole-tone or pentatonic scales
for scale in response['possible_scales']:
    if scale['match_score'] > 0.8:
        print(f"{scale['scale']}: {scale['match_score']:.2%} match")
```

### Chord Progression Analysis

```python
# Roman numeral analysis
response = await analyze_chord_progressions(
    score_id="jazz_standard",
    analysis_type="roman",
    include_inversions=True
)

# Print progression
for chord in response['progression']:
    print(f"m{chord['measure']}: {chord['roman_numeral']} "
          f"({chord['function']})")

# Common progression patterns
print("\nCommon patterns:")
for pattern, count in response['common_patterns']:
    print(f"  {pattern}: {count} times")
```

## Advanced Theory Analysis

### Identify Scales in Specific Passages

```python
# Analyze measures 17-24 for scale content
response = await identify_scale(
    score_id="ravel_bolero",
    start_measure=17,
    end_measure=24,
    confidence_threshold=0.75
)

for scale in response['detected_scales']:
    print(f"{scale['scale']}: {scale['match_score']:.2%}")
    print(f"  Tonic: {scale['tonic']}, Type: {scale['type']}")
```

### Interval Vector Analysis

```python
# Calculate interval class vector for atonal analysis
response = await interval_vector(
    score_id="schoenberg_op11",
    start_measure=1,
    end_measure=12
)

print(f"Interval Vector: {response['interval_vector']}")
print(f"Consonance Ratio: {response['consonance_ratio']:.2%}")
print(f"Tritones: {response['tritone_count']}")

if response['z_relation']:
    print(f"Z-related to: {response['z_relation']}")
```

### Chromatic Analysis

```python
# Analyze chromatic passages
response = await chromatic_analysis(
    score_id="wagner_tristan",
    include_voice_leading=True
)

print(f"Chromatic Density: {response['chromatic_density']:.2%}")

# Chromatic functions
print("\nChromatic Functions:")
for func, count in response['chromatic_functions'].items():
    print(f"  {func}: {count}")

# Modal mixture chords
if response['modal_mixture_chords']:
    print("\nModal Mixture:")
    for chord in response['modal_mixture_chords']:
        print(f"  m{chord['measure']}: {chord['function_in_parallel']}")
```

### Secondary Dominant Detection

```python
response = await secondary_dominants(
    score_id="beethoven_sonata"
)

print(f"Found {response['count']} secondary dominants")

for sd in response['secondary_dominants']:
    print(f"m{sd['measure']}: {sd['symbol']} → "
          f"{sd['resolution']} (resolves in m{sd['resolution_measure']})")
```

### Phrase Structure Analysis

```python
response = await phrase_structure(
    score_id="mozart_sonata",
    include_motives=True
)

print(f"Phrase Type: {response['phrase_type']}")
print(f"Phrase Lengths: {response['phrase_lengths']}")

# Cadences
for cadence in response['cadences']:
    print(f"m{cadence['phrase_end']}: {cadence['type']} cadence "
          f"(strength: {cadence['strength']:.2f})")

# Motivic content
if response['motivic_analysis']:
    for motive in response['motivic_analysis']['melodic_motives']:
        print(f"Motive: {motive['interval_pattern']} "
              f"({motive['occurrences']} times)")
```

## Rhythm Analysis

### Comprehensive Rhythm Analysis

```python
response = await analyze_rhythm(
    score_id="stravinsky_rite",
    include_patterns=True,
    pattern_min_length=3,
    pattern_min_occurrences=4
)

# Tempo information
tempo = response['tempo']
print(f"Tempo: {tempo['primary_bpm']} BPM ({tempo['character']})")
print(f"Stability: {tempo['stability']:.2%}")
print(f"Rubato Likelihood: {tempo['rubato_likelihood']:.2%}")

# Meter analysis
meter = response['meter']
print(f"\nMeter: {meter['primary']}")
print(f"Complexity: {meter['complexity']}")
if meter['is_mixed']:
    print("Contains mixed meters")

# Rhythmic patterns
print("\nRhythmic Patterns:")
for pattern in response['patterns'][:5]:
    print(f"  Pattern: {pattern['pattern']}")
    print(f"  Type: {pattern['type']}, Occurrences: {pattern['occurrences']}")
    if pattern['is_ostinato']:
        print("  (Ostinato)")
```

### Beat Strength Analysis

```python
response = await beat_strength(
    score_id="sousa_march"
)

print(f"Time Signature: {response['time_signature']}")

# Analyze first few measures
for measure in response['measure_analysis'][:4]:
    print(f"\nMeasure {measure['measure']}:")
    for beat in measure['beats']:
        print(f"  Beat {beat['beat']}: strength={beat['strength']:.1f}, "
              f"notes={beat['notes']}, accent={beat['has_accent']}")
```

### Find Specific Rhythmic Patterns

```python
# Search for syncopated patterns
response = await find_rhythmic_patterns(
    score_id="jazz_tune",
    min_length=4,
    min_occurrences=3,
    pattern_type="syncopated"
)

for pattern in response['patterns']:
    print(f"Pattern: {pattern['pattern_notation']}")
    print(f"Found in measures: {pattern['measures'][:10]}...")
    print(f"Confidence: {pattern['confidence']:.2%}")
```

## Integrated Analysis Workflows

### Comprehensive Analysis

```python
# Analyze everything at once
response = await comprehensive_analysis(
    score_id="brahms_intermezzo",
    include_advanced=True
)

# Summary of all analyses
print("=== Comprehensive Analysis ===")
print(f"Key: {response['analyses']['key']['key']}")
print(f"Time Signature: {response['metadata']['time_signatures'][0]['signature']}")
print(f"Tempo: {response['analyses']['rhythm']['tempo']['primary_bpm']} BPM")
print(f"Complexity: {response['analyses']['rhythm']['complexity']}")
print(f"Chromatic Density: {response['analyses']['chromatic']['chromatic_density']:.2%}")
```

### Batch Analysis

```python
# Analyze multiple scores
response = await batch_analysis(
    score_ids=["bach_fugue1", "bach_fugue2", "bach_fugue3"],
    analysis_types=["key", "scale", "rhythm", "harmony"]
)

# Compare results
print("Batch Analysis Summary:")
for score_id in response['analyses']:
    analyses = response['analyses'][score_id]
    print(f"\n{score_id}:")
    print(f"  Key: {analyses['key']['key']}")
    print(f"  Complexity: {analyses['rhythm']['complexity']}")
```

### Generate Reports

```python
# Educational report
response = await generate_report(
    score_id="chopin_nocturne",
    report_format="educational"
)

print(response['explanations']['key'])
print(response['explanations']['tempo'])

# Detailed technical report
response = await generate_report(
    score_id="chopin_nocturne",
    report_format="detailed"
)

# Access all analysis details
print(json.dumps(response['details'], indent=2))
```

## Real-World Use Cases

### 1. Music Education - Homework Checker

```python
async def check_counterpoint_exercise(student_file):
    """Check a student's counterpoint exercise"""
    
    # Import the student's work
    result = await import_score(
        score_id="student_exercise",
        source=student_file,
        validate=True
    )
    
    if result['status'] != 'success':
        return {"error": "Invalid file format"}
    
    # Analyze for parallel fifths/octaves
    harmony = await analyze_chord_progressions(
        score_id="student_exercise",
        analysis_type="roman"
    )
    
    # Check voice leading
    advanced = await chromatic_analysis(
        score_id="student_exercise",
        include_voice_leading=True
    )
    
    # Generate educational feedback
    report = await generate_report(
        score_id="student_exercise",
        report_format="educational"
    )
    
    return report
```

### 2. Music Research - Style Analysis

```python
async def analyze_composer_style(composer_scores):
    """Analyze stylistic traits across multiple works"""
    
    results = {
        'keys': [],
        'chromatic_density': [],
        'phrase_types': [],
        'tempo_characters': []
    }
    
    for score_file in composer_scores:
        # Import score
        score_id = os.path.basename(score_file)
        await import_score(score_id=score_id, source=score_file)
        
        # Run comprehensive analysis
        analysis = await comprehensive_analysis(
            score_id=score_id,
            include_advanced=True
        )
        
        # Collect stylistic data
        results['keys'].append(analysis['analyses']['key']['key'])
        results['chromatic_density'].append(
            analysis['analyses']['chromatic']['chromatic_density']
        )
        results['phrase_types'].append(
            analysis['analyses']['phrase_structure']['phrase_type']
        )
        results['tempo_characters'].append(
            analysis['analyses']['rhythm']['tempo']['character']
        )
    
    # Statistical analysis
    from collections import Counter
    
    return {
        'preferred_keys': Counter(results['keys']).most_common(3),
        'avg_chromatic_density': sum(results['chromatic_density']) / len(results['chromatic_density']),
        'common_phrase_types': Counter(results['phrase_types']).most_common(2),
        'tempo_preferences': Counter(results['tempo_characters']).most_common(3)
    }
```

### 3. Performance Preparation

```python
async def prepare_performance_notes(score_file):
    """Generate performance notes for musicians"""
    
    # Import and analyze
    await import_score(score_id="performance", source=score_file)
    
    # Structural analysis
    structure = await phrase_structure(
        score_id="performance",
        include_motives=True
    )
    
    # Tempo and rhythm
    rhythm = await analyze_rhythm(
        score_id="performance",
        include_patterns=True
    )
    
    # Key areas for modulations
    keys = await analyze_key(
        score_id="performance",
        window_size=4
    )
    
    # Generate performance notes
    notes = {
        'structure': f"Form: {structure['phrase_type']} with phrases of {structure['phrase_lengths']}",
        'tempo': f"{rhythm['tempo']['primary_bpm']} BPM ({rhythm['tempo']['character']})",
        'key_areas': []
    }
    
    # Add modulation warnings
    for mod in keys['modulations']:
        notes['key_areas'].append(
            f"Watch for modulation at m.{mod['measure']}: "
            f"{mod['from_key']} → {mod['to_key']}"
        )
    
    # Add cadence points
    for cadence in structure['cadences']:
        notes['key_areas'].append(
            f"Important cadence at m.{cadence['phrase_end']} ({cadence['type']})"
        )
    
    return notes
```

### 4. Composition Assistant

```python
async def analyze_harmonic_language(score_file):
    """Analyze harmonic language for compositional reference"""
    
    await import_score(score_id="reference", source=score_file)
    
    # Get chord vocabulary
    harmony = await analyze_chord_progressions(
        score_id="reference",
        analysis_type="jazz"
    )
    
    # Find special harmonies
    secondary = await secondary_dominants(score_id="reference")
    chromatic = await chromatic_analysis(score_id="reference")
    
    # Extract harmonic patterns
    chord_types = Counter()
    progressions = []
    
    for chord in harmony['progression']:
        chord_types[chord['quality']] += 1
        
    # Create harmonic palette
    return {
        'common_chords': chord_types.most_common(10),
        'chromatic_techniques': chromatic['chromatic_functions'],
        'secondary_dominants_used': secondary['tonicized_degrees'],
        'progression_formulas': harmony['common_patterns']
    }
```

## Tips and Best Practices

1. **Use appropriate analysis methods**: Different musical styles benefit from different analysis approaches. Use jazz analysis for jazz, Roman numerals for classical, etc.

2. **Set confidence thresholds**: When using detection algorithms, adjust confidence thresholds based on the musical style and complexity.

3. **Combine analyses**: Use comprehensive_analysis for a complete picture, then drill down into specific areas of interest.

4. **Cache results**: For large scores or repeated analyses, the server automatically caches results for better performance.

5. **Validate imports**: Always use `validate=True` when importing scores to catch potential issues early.

6. **Use batch processing**: When analyzing multiple scores, use batch_analysis for better performance.

7. **Export in appropriate formats**: Use MusicXML for maximum compatibility, MIDI for audio applications, and PNG/SVG for visual documentation.