"""
Integration tests for MCP server endpoints
"""
import pytest
import asyncio
from unittest.mock import Mock, patch
from music21 import stream, note, chord, key, meter, tempo

# Import server components
import sys
sys.path.insert(0, 'src')
from music21_mcp.server import (
    score_manager, import_score, export_score, analyze_key,
    analyze_scale, analyze_chord_progressions, analyze_rhythm,
    identify_scale, interval_vector, chromatic_analysis,
    secondary_dominants, phrase_structure, comprehensive_analysis,
    batch_analysis, generate_report
)


class TestScoreManagement:
    """Test score import/export endpoints"""
    
    @pytest.mark.asyncio
    async def test_import_score_from_file(self, temp_midi_file):
        """Test importing a score from file"""
        result = await import_score(
            score_id="test_score",
            source=temp_midi_file,
            validate=True
        )
        
        assert result['status'] == 'success'
        assert result['score_id'] == 'test_score'
        assert 'metadata' in result
        assert 'validation' in result
    
    @pytest.mark.asyncio
    async def test_import_score_from_content(self):
        """Test importing from direct content"""
        abc_content = """X:1
T:Test Tune
M:4/4
K:C
CDEF|GABc|"""
        
        result = await import_score(
            score_id="abc_test",
            source={'content': abc_content},
            source_type='abc'
        )
        
        assert result['status'] == 'success'
        assert result['metadata']['note_count'] == 8
    
    @pytest.mark.asyncio
    async def test_export_score_to_midi(self, simple_score):
        """Test exporting score to MIDI"""
        # First import the score
        score_manager.add_score("export_test", simple_score, {})
        
        result = await export_score(
            score_id="export_test",
            format="midi"
        )
        
        assert result['status'] == 'success'
        assert result['format'] == 'midi'
        assert 'result' in result
    
    @pytest.mark.asyncio
    async def test_export_score_invalid_format(self, simple_score):
        """Test export with invalid format"""
        score_manager.add_score("export_test2", simple_score, {})
        
        result = await export_score(
            score_id="export_test2",
            format="invalid_format"
        )
        
        assert result['status'] == 'error'


class TestTheoryAnalysis:
    """Test music theory analysis endpoints"""
    
    @pytest.mark.asyncio
    async def test_analyze_key_endpoint(self, simple_score):
        """Test key analysis endpoint"""
        score_manager.add_score("key_test", simple_score, {})
        
        result = await analyze_key(
            score_id="key_test",
            method="hybrid"
        )
        
        assert result['status'] == 'success'
        assert 'key' in result
        assert 'confidence' in result
        assert 'alternatives' in result
    
    @pytest.mark.asyncio
    async def test_analyze_scale_endpoint(self, simple_score):
        """Test scale analysis endpoint"""
        score_manager.add_score("scale_test", simple_score, {})
        
        result = await analyze_scale(
            score_id="scale_test",
            include_modes=True,
            include_exotic=True
        )
        
        assert result['status'] == 'success'
        assert 'possible_scales' in result
        assert 'best_match' in result
    
    @pytest.mark.asyncio
    async def test_analyze_chord_progressions(self, chord_progression_score):
        """Test chord progression analysis"""
        score_manager.add_score("chord_test", chord_progression_score, {})
        
        result = await analyze_chord_progressions(
            score_id="chord_test",
            analysis_type="roman"
        )
        
        assert result['status'] == 'success'
        assert 'progression' in result
        assert result['chord_count'] == 4


class TestRhythmAnalysis:
    """Test rhythm analysis endpoints"""
    
    @pytest.mark.asyncio
    async def test_analyze_rhythm_endpoint(self, complex_rhythm_score):
        """Test rhythm analysis endpoint"""
        score_manager.add_score("rhythm_test", complex_rhythm_score, {})
        
        result = await analyze_rhythm(
            score_id="rhythm_test",
            include_patterns=True
        )
        
        assert result['status'] == 'success'
        assert 'tempo' in result
        assert 'meter' in result
        assert 'complexity' in result
        assert 'patterns' in result
    
    @pytest.mark.asyncio
    async def test_find_rhythmic_patterns(self, complex_rhythm_score):
        """Test rhythmic pattern finding"""
        # Create score with repeating pattern
        s = stream.Score()
        p = stream.Part()
        
        pattern = [1, 0.5, 0.5]
        for _ in range(5):
            for dur in pattern:
                p.append(note.Note('C4', quarterLength=dur))
        
        s.append(p)
        score_manager.add_score("pattern_test", s, {})
        
        # This test would need the find_rhythmic_patterns endpoint to be imported
        # Currently it's defined in server.py but not exposed in __all__


class TestAdvancedTheory:
    """Test advanced theory endpoints"""
    
    @pytest.mark.asyncio
    async def test_identify_scale(self, simple_score):
        """Test scale identification endpoint"""
        score_manager.add_score("scale_id_test", simple_score, {})
        
        result = await identify_scale(
            score_id="scale_id_test",
            confidence_threshold=0.7
        )
        
        assert result['status'] == 'success'
        assert 'detected_scales' in result
        assert 'best_match' in result
    
    @pytest.mark.asyncio
    async def test_interval_vector(self, chord_progression_score):
        """Test interval vector calculation"""
        score_manager.add_score("iv_test", chord_progression_score, {})
        
        result = await interval_vector(
            score_id="iv_test"
        )
        
        assert result['status'] == 'success'
        assert 'interval_vector' in result
        assert len(result['interval_vector']) == 6
        assert 'consonance_ratio' in result
    
    @pytest.mark.asyncio
    async def test_chromatic_analysis(self):
        """Test chromatic analysis endpoint"""
        # Create score with chromatic notes
        s = stream.Score()
        p = stream.Part()
        p.append(key.KeySignature(0))  # C major
        
        # Add chromatic passing tones
        notes = ['C4', 'C#4', 'D4', 'E4', 'F#4', 'G4']
        for pitch in notes:
            p.append(note.Note(pitch, quarterLength=1))
        
        s.append(p)
        score_manager.add_score("chromatic_test", s, {})
        
        result = await chromatic_analysis(
            score_id="chromatic_test",
            include_voice_leading=True
        )
        
        assert result['status'] == 'success'
        assert result['chromatic_density'] > 0
        assert 'chromatic_functions' in result
    
    @pytest.mark.asyncio
    async def test_secondary_dominants(self):
        """Test secondary dominant detection"""
        # Create score with secondary dominants
        s = stream.Score()
        p = stream.Part()
        p.append(key.KeySignature(0))  # C major
        
        # I - V/V - V - I
        chords = [
            chord.Chord(['C4', 'E4', 'G4']),     # I
            chord.Chord(['D4', 'F#4', 'A4']),    # V/V
            chord.Chord(['G4', 'B4', 'D5']),     # V
            chord.Chord(['C4', 'E4', 'G4'])      # I
        ]
        
        for c in chords:
            c.quarterLength = 2
            p.append(c)
        
        s.append(p)
        score_manager.add_score("secondary_test", s, {})
        
        result = await secondary_dominants(
            score_id="secondary_test"
        )
        
        assert result['status'] == 'success'
        assert result['count'] > 0
        assert 'secondary_dominants' in result
    
    @pytest.mark.asyncio
    async def test_phrase_structure(self, simple_score):
        """Test phrase structure analysis"""
        score_manager.add_score("phrase_test", simple_score, {})
        
        result = await phrase_structure(
            score_id="phrase_test",
            include_motives=False
        )
        
        assert result['status'] == 'success'
        assert 'phrase_type' in result
        assert 'phrase_lengths' in result
        assert 'cadences' in result


class TestIntegrationTools:
    """Test integration and workflow tools"""
    
    @pytest.mark.asyncio
    async def test_comprehensive_analysis(self, simple_score):
        """Test comprehensive analysis"""
        score_manager.add_score("comprehensive_test", simple_score, {})
        
        result = await comprehensive_analysis(
            score_id="comprehensive_test",
            include_advanced=True
        )
        
        assert result['status'] == 'success'
        assert 'analyses' in result
        assert 'key' in result['analyses']
        assert 'scale' in result['analyses']
        assert 'harmony' in result['analyses']
        assert 'rhythm' in result['analyses']
    
    @pytest.mark.asyncio
    async def test_batch_analysis(self, simple_score, chord_progression_score):
        """Test batch analysis"""
        score_manager.add_score("batch1", simple_score, {})
        score_manager.add_score("batch2", chord_progression_score, {})
        
        result = await batch_analysis(
            score_ids=["batch1", "batch2"],
            analysis_types=["key", "rhythm"]
        )
        
        assert result['status'] == 'success'
        assert result['total_scores'] == 2
        assert 'batch1' in result['analyses']
        assert 'batch2' in result['analyses']
        assert 'key' in result['analyses']['batch1']
        assert 'rhythm' in result['analyses']['batch1']
    
    @pytest.mark.asyncio
    async def test_generate_report(self, simple_score):
        """Test report generation"""
        score_manager.add_score("report_test", simple_score, {})
        
        # Test summary report
        result = await generate_report(
            score_id="report_test",
            report_format="summary"
        )
        
        assert result['status'] == 'success'
        assert result['report_type'] == 'summary'
        assert 'summary' in result
        
        # Test educational report
        result = await generate_report(
            score_id="report_test",
            report_format="educational"
        )
        
        assert result['report_type'] == 'educational'
        assert 'explanations' in result


class TestErrorHandling:
    """Test error handling in endpoints"""
    
    @pytest.mark.asyncio
    async def test_analyze_nonexistent_score(self):
        """Test analyzing a score that doesn't exist"""
        result = await analyze_key(score_id="nonexistent")
        
        assert result['status'] == 'error'
        assert 'not found' in result['message']
    
    @pytest.mark.asyncio
    async def test_import_invalid_source(self):
        """Test importing from invalid source"""
        result = await import_score(
            score_id="invalid_test",
            source="not_a_file.xyz"
        )
        
        assert result['status'] == 'error'
    
    @pytest.mark.asyncio
    async def test_batch_analysis_with_missing_scores(self):
        """Test batch analysis with some missing scores"""
        score_manager.add_score("exists", Mock(), {})
        
        result = await batch_analysis(
            score_ids=["exists", "missing"],
            analysis_types=["key"]
        )
        
        # Should still succeed but with errors for missing scores
        assert result['status'] == 'success'
        assert 'error' in result['analyses']['missing']['key']