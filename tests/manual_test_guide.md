# Manual Testing Guide for Music21 MCP Server

## 🚨 CRITICAL: Core Validation Steps

### 1. Server Startup Test
```bash
# Test STDIO mode (for MCP clients)
python -m music21_mcp.server --stdio

# Test HTTP mode (for debugging)
python -m music21_mcp.server
```

### 2. MCP client Integration
1. Copy configuration to MCP client:
   ```bash
   cp mcp_config.json ~/Library/Application\ Support/Claude/mcp_config.json
   ```

2. Restart MCP client

3. Test in Claude:
   ```
   "Can you analyze this melody: C D E F G A B C"
   "Import the Bach BWV 846 prelude and analyze its harmony"
   "What key is this progression in: Cmaj7 Am7 Dm7 G7"
   ```

### 3. Basic Functionality Tests

#### Test Score Import
```python
# In MCP client, try:
"Import a C major scale as 'test_scale'"
"List all imported scores"
"Analyze the key of 'test_scale'"
```

#### Test Real MIDI Import
```python
# Download a MIDI file, then:
"Import the MIDI file at /path/to/file.mid as 'my_song'"
"Analyze the harmony of 'my_song'"
"Export 'my_song' as MusicXML"
```

#### Test Error Handling
```python
"Analyze the key of 'nonexistent_score'"  # Should fail gracefully
"Import invalid data as 'bad_score'"      # Should report error
```

### 4. Performance Tests

#### Large File Test
- Import a full orchestral score (>100KB MusicXML)
- Verify it completes within reasonable time (<10 seconds)
- Check memory usage doesn't spike excessively

#### Batch Operations
```python
"Import 10 different melodies"
"Analyze all of them"
"List all scores"  # Should show all 10
```

### 5. Real Music Analysis Tests

#### Bach Chorale Test
1. Download Bach BWV 66.6 from music21 corpus
2. Import it
3. Run these analyses:
   - Key detection (should be F# minor)
   - Harmony analysis (should find ~30-50 chords)
   - Voice leading check (should have high smoothness)
   - Bach chorale style check (should pass)

#### Jazz Standard Test
1. Create a jazz lead sheet
2. Import it
3. Run:
   - Jazz harmony analysis
   - Chord substitution detection
   - Extended chord identification

#### Contemporary Music Test
1. Import a 20th century piece (e.g., Schoenberg)
2. Run:
   - Pitch class set analysis
   - Twelve-tone row detection
   - Dissonance analysis

### 6. Integration Validation

#### Cursor IDE Test
1. Add to Cursor settings
2. Open a Python file
3. Ask Cursor to generate code that uses the music21 server

#### API Workflow Test
Simulate a complete workflow:
1. Import score
2. Analyze multiple aspects
3. Modify the score
4. Export in different format
5. Verify round-trip integrity

### 7. Stress Testing

#### Concurrent Requests
- Import 5 scores simultaneously
- Run different analyses on each
- Verify no data corruption

#### Memory Leak Test
- Run 100 import/analyze/delete cycles
- Monitor memory usage
- Should stabilize, not grow indefinitely

## 🔴 Red Flags to Watch For

1. **Server doesn't start in STDIO mode** - Critical MCP issue
2. **MCP client can't discover tools** - Protocol problem
3. **Import succeeds but analysis fails** - Data structure issue
4. **Memory usage grows unbounded** - Memory leak
5. **Errors not reported properly** - Exception handling issue
6. **Round-trip conversion loses data** - Format conversion bug

## ✅ Success Criteria

- [ ] Server starts in both STDIO and HTTP modes
- [ ] MCP client can discover and call all tools
- [ ] Real MIDI/MusicXML files import successfully
- [ ] All analysis tools produce meaningful results
- [ ] Error handling is graceful and informative
- [ ] Performance is acceptable (<5s for most operations)
- [ ] Memory usage is stable
- [ ] Round-trip conversions preserve data

## 📊 Validation Report Template

```
Date: ___________
Tester: _________

Core Functionality:
- Server Startup: [ ] Pass [ ] Fail
- Tool Discovery: [ ] Pass [ ] Fail
- Basic Import/Export: [ ] Pass [ ] Fail

Real Music Tests:
- Bach Chorale: [ ] Pass [ ] Fail
- Jazz Standard: [ ] Pass [ ] Fail
- Contemporary: [ ] Pass [ ] Fail

Integration:
- MCP client: [ ] Pass [ ] Fail
- Cursor IDE: [ ] Pass [ ] Fail

Performance:
- Large Files: [ ] Pass [ ] Fail
- Concurrent Ops: [ ] Pass [ ] Fail

Issues Found:
1. ________________________
2. ________________________
3. ________________________

Overall Status: [ ] Ready [ ] Needs Work
```