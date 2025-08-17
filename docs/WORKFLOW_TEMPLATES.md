# Music21 MCP Server - Workflow Templates

These workflow templates demonstrate how to chain multiple tools together for powerful music analysis and generation. Each workflow includes example prompts for Claude Desktop.

## 📚 Table of Contents
1. [Bach Chorale Analysis Workflow](#bach-chorale-analysis-workflow)
2. [Melody Harmonization Workflow](#melody-harmonization-workflow)
3. [Counterpoint Generation Workflow](#counterpoint-generation-workflow)
4. [Style Analysis & Imitation Workflow](#style-analysis--imitation-workflow)
5. [Music Theory Homework Helper](#music-theory-homework-helper)
6. [Composition Assistant Workflow](#composition-assistant-workflow)
7. [Corpus Analysis Research Workflow](#corpus-analysis-research-workflow)
8. [Performance Preparation Workflow](#performance-preparation-workflow)

---

## 🎼 Bach Chorale Analysis Workflow

**Goal**: Analyze a Bach chorale for teaching music theory concepts

### Claude Desktop Prompt:
```
Please analyze Bach chorale BWV 66.6:
1. Import the chorale from the corpus
2. Analyze the key and show confidence levels
3. Identify all chords with Roman numerals
4. Analyze voice leading quality
5. Find recurring harmonic patterns
6. Export to MusicXML for notation software
```

### Step-by-Step Commands:
```python
# Step 1: Import
import_score("bach_chorale", "bach/bwv66.6", "corpus")

# Step 2: Key Analysis
key_analysis("bach_chorale")
# Returns: Key of F major, 95% confidence

# Step 3: Chord Analysis
chord_analysis("bach_chorale")
# Returns: Complete progression with Roman numerals

# Step 4: Voice Leading
voice_leading_analysis("bach_chorale")
# Returns: Smoothness score, parallel fifths/octaves detection

# Step 5: Pattern Recognition
pattern_recognition("bach_chorale", "harmonic")
# Returns: Common progressions like I-IV-V-I

# Step 6: Export
export_score("bach_chorale", "musicxml")
```

---

## 🎹 Melody Harmonization Workflow

**Goal**: Generate harmonizations in multiple styles for composition students

### Claude Desktop Prompt:
```
I have a simple melody. Please:
1. Import my melody file (melody.mid)
2. Create a classical 4-part harmonization
3. Create a jazz harmonization with extended chords
4. Create a pop harmonization with guitar-friendly voicings
5. Compare the different harmonization styles
6. Export each version as MIDI
```

### Step-by-Step Commands:
```python
# Step 1: Import melody
import_score("my_melody", "melody.mid", "file")

# Step 2: Classical harmonization
harmonize_melody("my_melody", style="classical", voice_parts=4)
# Creates "my_melody_harmonized"

# Step 3: Jazz harmonization  
harmonize_melody("my_melody", style="jazz", voice_parts=2)
# Creates "my_melody_jazz"

# Step 4: Pop harmonization
harmonize_melody("my_melody", style="pop", voice_parts=3)
# Creates "my_melody_pop"

# Step 5: Analyze each version
chord_analysis("my_melody_harmonized")
chord_analysis("my_melody_jazz")
chord_analysis("my_melody_pop")

# Step 6: Export all versions
export_score("my_melody_harmonized", "midi")
export_score("my_melody_jazz", "midi")
export_score("my_melody_pop", "midi")
```

---

## 🎯 Counterpoint Generation Workflow

**Goal**: Generate species counterpoint for educational exercises

### Claude Desktop Prompt:
```
Help me practice counterpoint:
1. Import a cantus firmus melody
2. Generate first species counterpoint above
3. Generate second species counterpoint below
4. Check for voice leading errors
5. Export the exercises as PDF
```

### Step-by-Step Commands:
```python
# Step 1: Import cantus firmus
import_score("cantus", "exercises/fux_cf1.xml", "file")

# Step 2: First species above
generate_counterpoint("cantus", species=1, voice_position="above")
# Creates "cantus_counterpoint"

# Step 3: Second species below
generate_counterpoint("cantus", species=2, voice_position="below")
# Creates "cantus_counterpoint_2"

# Step 4: Check voice leading
voice_leading_analysis("cantus_counterpoint")
voice_leading_analysis("cantus_counterpoint_2")

# Step 5: Export as PDF
export_score("cantus_counterpoint", "pdf")
export_score("cantus_counterpoint_2", "pdf")
```

---

## 🎨 Style Analysis & Imitation Workflow

**Goal**: Learn composer styles and generate new music in those styles

### Claude Desktop Prompt:
```
Analyze Mozart's style and create new music:
1. Import several Mozart sonatas
2. Analyze their harmonic and melodic patterns
3. Generate a new 16-bar piece in Mozart's style
4. Compare the generated piece with the originals
```

### Step-by-Step Commands:
```python
# Step 1: Import Mozart pieces
import_score("mozart1", "mozart/k331-1", "corpus")
import_score("mozart2", "mozart/k333-1", "corpus")

# Step 2: Analyze patterns
pattern_recognition("mozart1", "melodic")
pattern_recognition("mozart1", "harmonic")
harmony_analysis("mozart1")

# Step 3: Generate in Mozart style
imitate_style(composer="mozart", generation_length=16, complexity="medium")
# Creates "generated_style"

# Step 4: Compare with originals
chord_analysis("generated_style")
key_analysis("generated_style")
pattern_recognition("generated_style", "melodic")
```

---

## 📖 Music Theory Homework Helper

**Goal**: Complete music theory assignments with explanations

### Claude Desktop Prompt:
```
Help with my music theory homework:
1. Import the score my professor provided
2. Label all chords with Roman numerals
3. Identify all non-chord tones
4. Find and label all cadences
5. Explain the harmonic rhythm
6. Create a harmonic reduction
```

### Step-by-Step Commands:
```python
# Step 1: Import assignment
import_score("homework", "assignment1.xml", "file")

# Step 2: Roman numeral analysis
harmony_analysis("homework", analysis_type="roman")
# Returns complete Roman numeral analysis

# Step 3: Full harmonic analysis
harmony_analysis("homework", analysis_type="functional")
# Identifies non-chord tones

# Step 4: Pattern recognition for cadences
pattern_recognition("homework", "harmonic")
# Finds PAC, IAC, HC, DC patterns

# Step 5: Get detailed info
score_info("homework")
# Shows harmonic rhythm statistics

# Step 6: Export simplified version
export_score("homework", "musicxml")
```

---

## 🎵 Composition Assistant Workflow

**Goal**: Help compose a new piece with intelligent suggestions

### Claude Desktop Prompt:
```
I'm composing a piece. Please help me:
1. Import my work-in-progress score
2. Analyze what I've written so far
3. Suggest chord progressions for the next section
4. Generate a countermelody
5. Check for voice leading issues
6. Export the complete score
```

### Step-by-Step Commands:
```python
# Step 1: Import WIP
import_score("my_piece", "composition_draft.mid", "file")

# Step 2: Analyze existing material
key_analysis("my_piece")
chord_analysis("my_piece")
pattern_recognition("my_piece", "melodic")

# Step 3: Harmonize next section
harmonize_melody("my_piece", style="classical")
# Creates "my_piece_harmonized"

# Step 4: Add counterpoint
generate_counterpoint("my_piece_harmonized", species=5, voice_position="above")
# Creates "my_piece_complete"

# Step 5: Quality check
voice_leading_analysis("my_piece_complete")

# Step 6: Export final version
export_score("my_piece_complete", "musicxml")
export_score("my_piece_complete", "midi")
```

---

## 🔬 Corpus Analysis Research Workflow

**Goal**: Analyze patterns across multiple pieces for musicology research

### Claude Desktop Prompt:
```
Research Baroque harmonic progressions:
1. Import 10 Bach chorales
2. Analyze key relationships
3. Find common chord progressions
4. Calculate statistical patterns
5. Generate summary report
```

### Step-by-Step Commands:
```python
# Step 1: Batch import
chorales = ["bwv66.6", "bwv267", "bwv269", "bwv270", "bwv271"]
for i, chorale in enumerate(chorales):
    import_score(f"bach_{i}", f"bach/{chorale}", "corpus")

# Step 2: Analyze all keys
results = []
for i in range(len(chorales)):
    results.append(key_analysis(f"bach_{i}"))

# Step 3: Find progressions
progressions = []
for i in range(len(chorales)):
    progressions.append(pattern_recognition(f"bach_{i}", "harmonic"))

# Step 4: Statistical analysis
harmony_results = []
for i in range(len(chorales)):
    harmony_results.append(harmony_analysis(f"bach_{i}"))

# Step 5: View all scores info
list_scores()
```

---

## 🎭 Performance Preparation Workflow

**Goal**: Prepare a score for performance with analysis markings

### Claude Desktop Prompt:
```
Prepare my score for performance:
1. Import the piece I'm performing
2. Analyze the formal structure
3. Mark all important harmonic points
4. Identify technically challenging passages
5. Create a practice version with annotations
6. Export with measure numbers and rehearsal marks
```

### Step-by-Step Commands:
```python
# Step 1: Import performance piece
import_score("performance", "beethoven_sonata.xml", "file")

# Step 2: Structural analysis
score_info("performance")
# Shows form, sections, measures

# Step 3: Harmonic analysis
harmony_analysis("performance")
chord_analysis("performance")
# Mark modulations and key areas

# Step 4: Pattern recognition
pattern_recognition("performance", "melodic")
pattern_recognition("performance", "rhythmic")
# Find difficult passages

# Step 5: Voice leading check
voice_leading_analysis("performance")
# Understand voice movement

# Step 6: Export annotated version
export_score("performance", "musicxml")
export_score("performance", "pdf")
```

---

## 💡 Tips for Effective Workflows

### Chaining Operations
- Always import scores first before analysis
- Save intermediate results with descriptive names
- Export in multiple formats for different uses

### Batch Processing
```python
# Process multiple files
scores = ["piece1.mid", "piece2.mid", "piece3.mid"]
for score in scores:
    import_score(score, score, "file")
    key_analysis(score)
    harmony_analysis(score)
    export_score(score, "musicxml")
```

### Error Recovery
- If a tool times out, try with a simpler score first
- Break large scores into sections
- Use list_scores() to check what's loaded

### Claude Desktop Integration
When using with Claude Desktop, you can ask natural language questions:
- "What key is this piece in?"
- "Generate a Bach-style harmonization"
- "Find all the ii-V-I progressions"
- "Create a countermelody above this tune"

Claude will automatically chain the appropriate tools together!

---

## 🚀 Quick Start Examples

### For Music Students:
```
"Import Bach BWV 66.6 and show me all the chords with Roman numerals"
```

### For Composers:
```
"Harmonize my melody in jazz style with extended chords"
```

### For Teachers:
```
"Generate a first species counterpoint exercise above this cantus firmus"
```

### For Researchers:
```
"Analyze harmonic patterns across all Bach chorales in F major"
```

These workflows can be customized and combined for your specific needs!