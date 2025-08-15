# MCP Usage Examples - Music21 Analysis Server

## Overview

This document provides comprehensive examples of using the Music21 MCP Server across different applications and workflows. Each example includes step-by-step instructions and expected outcomes.

## Table of Contents

1. [Getting Started Examples](#getting-started-examples)
2. [Educational Workflows](#educational-workflows)
3. [Research Applications](#research-applications)
4. [Composition Assistance](#composition-assistance)
5. [Advanced Analysis](#advanced-analysis)
6. [Integration Examples](#integration-examples)
7. [Troubleshooting Common Issues](#troubleshooting-common-issues)

---

## Getting Started Examples

### Example 1: Basic Score Analysis

**Scenario**: Analyze a Bach chorale for key and harmonic content.

**Steps**:
1. Import a score from the music21 corpus
2. Analyze the key signature  
3. Perform harmonic analysis
4. Export results

**Claude Desktop Conversation**:
```
User: I want to analyze Bach's BWV 66.6 chorale. Can you help me import it and analyze its key and harmony?

Assistant: I'll help you analyze Bach's BWV 66.6 chorale using the Music21 analysis tools. Let me start by importing the score and then perform key and harmonic analysis.

*Uses import_score tool*

*Uses key_analysis tool*

*Uses harmony_analysis tool*

The analysis shows this chorale is in A major with a clear I-vi-IV-V-I progression in the opening phrase. The Roman numeral analysis reveals typical Bach voice leading with strong cadential patterns at measures 8, 16, and 32.
```

**Expected Results**:
- Key: A major (confidence: 0.95)
- Chord progression: I-vi-IV-V-I pattern
- Cadences: Authentic cadences at phrase endings
- Voice leading: Smooth with minimal parallel motion

### Example 2: Quick Score Information

**Scenario**: Get basic information about a musical score.

**Workflow**:
```json
{
  "tool": "import_score",
  "params": {
    "score_id": "chorale",
    "source": "bach/bwv66.6",
    "source_type": "corpus"
  }
}

{
  "tool": "score_info",
  "params": {
    "score_id": "chorale"
  }
}
```

**Response**:
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
    }
  }
}
```

### Example 3: Export to Different Formats

**Scenario**: Export a analyzed score to MIDI and MusicXML formats.

**Steps**:
1. Import score
2. Export to MIDI
3. Export to MusicXML

**Workflow**:
```json
[
  {
    "tool": "import_score",
    "params": {
      "score_id": "export_demo",
      "source": "mozart/k331_1st.xml",
      "source_type": "corpus"
    }
  },
  {
    "tool": "export_score",
    "params": {
      "score_id": "export_demo",
      "format": "midi"
    }
  },
  {
    "tool": "export_score",
    "params": {
      "score_id": "export_demo",
      "format": "musicxml"
    }
  }
]
```

---

## Educational Workflows

### Example 4: Classroom Harmonic Analysis

**Scenario**: A music theory teacher wants to demonstrate Roman numeral analysis to students.

**Teaching Workflow**:

**Phase 1: Setup**
```
Teacher: "Today we're analyzing Bach chorales. Let's start with BWV 66.6."

*Imports score and displays basic information*
```

**Phase 2: Key Analysis**
```
Teacher: "First, let's determine the key. What do you think it is?"

*Uses key_analysis tool to confirm student guesses*

Result: A major (92% confidence with Krumhansl algorithm)
```

**Phase 3: Chord-by-Chord Analysis**
```
Teacher: "Now let's analyze the chord progressions."

*Uses harmony_analysis tool with roman numeral setting*

Students can see: I - vi - IV - V - I progression clearly labeled
```

**Phase 4: Voice Leading Discussion**
```
Teacher: "How does Bach handle voice leading here?"

*Uses voice_leading_analysis tool*

Results show: Smooth voice leading, minimal parallel motion, proper resolution of tendency tones
```

### Example 5: Student Homework Assistant

**Scenario**: A student needs help analyzing a piece for homework.

**Student Workflow**:
```
Student: I need to analyze this Mozart sonata movement for my theory class. Can you help me understand the harmonic structure?

Assistant: I'll help you analyze the Mozart sonata. Let me import it and break down the analysis step by step.

1. *Import the score*
2. *Analyze the key structure*
3. *Identify main chord progressions*
4. *Point out important cadences*
5. *Discuss formal structure*

This way you'll understand the analytical process, not just get the answers.
```

### Example 6: Comparative Analysis Exercise

**Scenario**: Comparing two different composers' approaches to harmony.

**Workflow**:
```python
# Bach chorale analysis
bach_result = [
  import_score("bach_sample", "bach/bwv66.6", "corpus"),
  harmony_analysis("bach_sample", "roman")
]

# Mozart sonata analysis  
mozart_result = [
  import_score("mozart_sample", "mozart/k331_1st.xml", "corpus"),
  harmony_analysis("mozart_sample", "roman")
]

# Compare results
Comparison shows:
- Bach: More complex voice leading, frequent secondary dominants
- Mozart: Clearer phrase structure, simpler harmonic rhythm
```

---

## Research Applications

### Example 7: Large-Scale Corpus Analysis

**Scenario**: A musicologist studying harmonic patterns in Bach chorales.

**Research Workflow**:

**Phase 1: Data Collection**
```python
chorale_ids = ["bwv66.6", "bwv84.5", "bwv86.6", "bwv96.6"]
results = []

for chorale_id in chorale_ids:
    # Import each chorale
    import_score(f"chorale_{i}", f"bach/{chorale_id}", "corpus")
    
    # Analyze harmonies
    harmony_data = harmony_analysis(f"chorale_{i}", "roman")
    
    # Analyze voice leading
    voice_data = voice_leading_analysis(f"chorale_{i}")
    
    results.append({
        "id": chorale_id,
        "harmony": harmony_data,
        "voice_leading": voice_data
    })
```

**Phase 2: Pattern Analysis**
```python
# Analyze common progressions across all chorales
for chorale in results:
    patterns = pattern_recognition(chorale["id"], "harmonic")
    # Identify frequently used progressions
```

**Expected Research Outcomes**:
- Common harmonic progressions in Bach chorales
- Voice leading patterns and their frequency
- Cadential formulas and their variations
- Statistical analysis of harmonic rhythm

### Example 8: Style Analysis Research

**Scenario**: Comparing compositional styles across different periods.

**Research Question**: How do harmonic progressions differ between Baroque and Classical periods?

**Methodology**:
```python
# Baroque sample (Bach)
baroque_samples = [
    ("bach1", "bach/bwv66.6"),
    ("bach2", "bach/bwv84.5"),
    ("bach3", "bach/bwv86.6")
]

# Classical sample (Mozart)
classical_samples = [
    ("mozart1", "mozart/k331_1st.xml"),
    ("mozart2", "mozart/k332_1st.xml"),
    ("mozart3", "mozart/k333_1st.xml")
]

# Analyze each sample
for sample_id, source in baroque_samples + classical_samples:
    import_score(sample_id, source, "corpus")
    harmony_analysis(sample_id, "roman")
    chord_analysis(sample_id)
    pattern_recognition(sample_id, "harmonic")
```

**Analysis Results**:
- Baroque: More frequent use of secondary dominants
- Classical: Clearer tonal areas and phrase structures
- Statistical significance testing on harmonic content

---

## Composition Assistance

### Example 9: Melody Harmonization

**Scenario**: A composer wants to harmonize a melody in Bach style.

**Composition Workflow**:

**Step 1: Import Melody**
```json
{
  "tool": "import_score",
  "params": {
    "score_id": "my_melody",
    "source": "path/to/melody.musicxml",
    "source_type": "file"
  }
}
```

**Step 2: Generate Harmonization**
```json
{
  "tool": "harmonization",
  "params": {
    "score_id": "my_melody",
    "style": "bach"
  }
}
```

**Step 3: Review and Refine**
```json
{
  "tool": "voice_leading_analysis",
  "params": {
    "score_id": "my_melody"
  }
}
```

**Expected Output**:
- Four-part harmonization in Bach style
- Voice leading score: 8.5/10
- Proper resolution of tendency tones
- Authentic cadences at phrase endings

### Example 10: Counterpoint Generation

**Scenario**: Creating a counterpoint exercise for students.

**Educational Application**:
```json
[
  {
    "tool": "import_score",
    "params": {
      "score_id": "cantus_firmus",
      "source": "simple_cantus_firmus.musicxml",
      "source_type": "file"
    }
  },
  {
    "tool": "counterpoint_generation",
    "params": {
      "score_id": "cantus_firmus",
      "species": 1
    }
  }
]
```

**Results**:
- First species counterpoint (note against note)
- Follows traditional species rules
- Proper intervals and motion
- Good melodic contour

### Example 11: Style Imitation

**Scenario**: Learning to compose in the style of a specific composer.

**Learning Workflow**:

**Phase 1: Style Analysis**
```json
{
  "tool": "style_imitation",
  "params": {
    "score_id": "bach_reference",
    "target_style": "bach"
  }
}
```

**Phase 2: Apply Style Characteristics**
- Analyze harmonic patterns in reference piece
- Generate new material following those patterns
- Compare results with original style

**Phase 3: Validation**
```json
{
  "tool": "harmony_analysis",
  "params": {
    "score_id": "generated_piece",
    "analysis_type": "both"
  }
}
```

---

## Advanced Analysis

### Example 12: Multi-Algorithm Key Analysis

**Scenario**: Detailed key analysis using multiple algorithms for research accuracy.

**Advanced Workflow**:
```json
{
  "tool": "key_analysis",
  "params": {
    "score_id": "complex_piece",
    "algorithm": "all"
  }
}
```

**Detailed Results**:
```json
{
  "success": true,
  "data": {
    "primary_key": "F# minor",
    "confidence": 0.87,
    "algorithms": {
      "krumhansl": {"key": "F# minor", "confidence": 0.89},
      "aarden": {"key": "F# minor", "confidence": 0.82},
      "temperley": {"key": "F# minor", "confidence": 0.91},
      "bellman": {"key": "F# minor", "confidence": 0.85}
    },
    "key_changes": [
      {"measure": 16, "key": "A major", "confidence": 0.78},
      {"measure": 32, "key": "C# minor", "confidence": 0.73}
    ],
    "ambiguous_regions": [
      {"measures": "24-28", "reason": "chromatic_harmony"}
    ]
  }
}
```

### Example 13: Comprehensive Pattern Analysis

**Scenario**: Deep pattern analysis for musicological research.

**Research Workflow**:
```json
[
  {
    "tool": "pattern_recognition",
    "params": {
      "score_id": "research_piece",
      "pattern_type": "melodic"
    }
  },
  {
    "tool": "pattern_recognition",
    "params": {
      "score_id": "research_piece",
      "pattern_type": "rhythmic"
    }
  },
  {
    "tool": "pattern_recognition",
    "params": {
      "score_id": "research_piece",
      "pattern_type": "harmonic"
    }
  }
]
```

**Comprehensive Results**:
- Melodic: Motivic development patterns
- Rhythmic: Syncopation patterns and metric modulation
- Harmonic: Sequence patterns and harmonic rhythm

---

## Integration Examples

### Example 14: VS Code Integration

**Scenario**: Using the MCP server within VS Code for music analysis while coding.

**VS Code Setup**:
1. Install MCP extension
2. Configure music21 server in settings.json
3. Use Copilot Chat with MCP integration

**Usage in VS Code**:
```
@mcp-music21 Analyze the harmonic progression in this MusicXML file I'm working with.

*Automatically imports file from workspace*
*Provides harmonic analysis*
*Suggests improvements to chord progressions*
```

### Example 15: Cursor IDE Integration

**Scenario**: Composition assistance while writing music notation code.

**Cursor Workflow**:
```python
# Writing music21 code in Cursor
from music21 import stream, note, chord

# Ask MCP for harmonic analysis
# @mcp What chord progression would work well here?

s = stream.Stream()
# MCP suggests: I-vi-IV-V-I progression in C major
s.append(chord.Chord(['C', 'E', 'G']))
s.append(chord.Chord(['A', 'C', 'E']))
s.append(chord.Chord(['F', 'A', 'C']))
s.append(chord.Chord(['G', 'B', 'D']))
s.append(chord.Chord(['C', 'E', 'G']))
```

### Example 16: Jupyter Notebook Research

**Scenario**: Interactive music analysis in Jupyter notebooks.

**Notebook Cell 1: Setup**
```python
# MCP integration in Jupyter
import json
import subprocess

def call_mcp_tool(tool_name, params):
    mcp_request = {
        "jsonrpc": "2.0",
        "method": "tools/call",
        "params": {
            "name": tool_name,
            "arguments": params
        },
        "id": 1
    }
    # Call MCP server
    result = subprocess.run(
        ["python", "-m", "music21_mcp.server_minimal"],
        input=json.dumps(mcp_request),
        capture_output=True,
        text=True
    )
    return json.loads(result.stdout)
```

**Notebook Cell 2: Analysis**
```python
# Import and analyze Bach chorale
result = call_mcp_tool("import_score", {
    "score_id": "bach_analysis",
    "source": "bach/bwv66.6",
    "source_type": "corpus"
})

harmony_result = call_mcp_tool("harmony_analysis", {
    "score_id": "bach_analysis",
    "analysis_type": "roman"
})

# Visualize results
import matplotlib.pyplot as plt
# Plot harmonic progression over time
```

---

## Troubleshooting Common Issues

### Issue 1: Score Import Failures

**Problem**: "Score not found" error when importing.

**Solutions**:
```json
// Check available corpus pieces
{
  "tool": "list_scores",
  "params": {}
}

// Try alternative source types
{
  "tool": "import_score",
  "params": {
    "score_id": "test_score",
    "source": "path/to/file.xml",
    "source_type": "file"  // instead of "corpus"
  }
}
```

### Issue 2: Analysis Returning Empty Results

**Problem**: Harmony analysis returns no chords.

**Diagnosis**:
```json
{
  "tool": "score_info",
  "params": {
    "score_id": "problematic_score"
  }
}
```

**Common Causes**:
- Score has no harmonic content (single line)
- Score is corrupted or malformed
- Analysis parameters need adjustment

### Issue 3: Performance Issues with Large Scores

**Problem**: Analysis takes too long or times out.

**Solutions**:
```json
// Check score size first
{
  "tool": "score_info",
  "params": {
    "score_id": "large_score"
  }
}

// Use health check to monitor resources
{
  "tool": "health_check",
  "params": {}
}

// Delete unused scores to free memory
{
  "tool": "delete_score",
  "params": {
    "score_id": "unused_score"
  }
}
```

### Issue 4: Export Format Problems

**Problem**: Exported files are corrupted or unreadable.

**Solutions**:
```json
// Try different export formats
{
  "tool": "export_score",
  "params": {
    "score_id": "my_score",
    "format": "musicxml"  // most compatible
  }
}

// Check score integrity before export
{
  "tool": "score_info",
  "params": {
    "score_id": "my_score"
  }
}
```

### Issue 5: Inconsistent Analysis Results

**Problem**: Different algorithms give very different results.

**Investigation**:
```json
// Use multi-algorithm analysis
{
  "tool": "key_analysis",
  "params": {
    "score_id": "ambiguous_piece",
    "algorithm": "all"
  }
}

// Check for key changes or modulations
{
  "tool": "harmony_analysis",
  "params": {
    "score_id": "ambiguous_piece",
    "analysis_type": "both"
  }
}
```

**Understanding Discrepancies**:
- Atonal or highly chromatic music
- Frequent modulations
- Mixed modal characteristics
- Short excerpts with limited harmonic content

---

## Best Practices

### Workflow Optimization
1. **Check Server Health**: Start sessions with `health_check`
2. **Import Once**: Reuse imported scores for multiple analyses
3. **Clean Up**: Delete unused scores to manage memory
4. **Validate Inputs**: Use `score_info` to verify imports
5. **Progress Monitoring**: Use health checks for long analyses

### Educational Use
1. **Scaffolded Learning**: Start with simple analyses, build complexity
2. **Compare Results**: Use multiple algorithms to show different perspectives
3. **Interactive Discovery**: Let students predict before revealing results
4. **Visual Integration**: Combine with score display software

### Research Applications
1. **Systematic Methodology**: Document all analysis parameters
2. **Reproducible Results**: Save analysis configurations
3. **Statistical Validation**: Use multiple pieces for generalizable results
4. **Cross-Validation**: Compare with other analysis tools

### Performance Tips
1. **Batch Processing**: Group similar analyses together
2. **Memory Management**: Monitor and clean up regularly
3. **Algorithm Selection**: Choose appropriate algorithms for your needs
4. **Format Optimization**: Use efficient import/export formats

---

## Conclusion

The Music21 MCP Server provides powerful music analysis capabilities across multiple applications and use cases. These examples demonstrate the flexibility and depth of analysis possible, from simple educational exercises to advanced research applications.

For more detailed documentation:
- [MCP Tools Reference](MCP_TOOLS.md)
- [Installation Guide](MCP_INSTALLATION.md)
- [API Documentation](API.md)

For support and community:
- [GitHub Issues](https://github.com/brightlikethelight/music21-mcp-server/issues)
- [GitHub Discussions](https://github.com/brightlikethelight/music21-mcp-server/discussions)
- Email: brightliu@college.harvard.edu