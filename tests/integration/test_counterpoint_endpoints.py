"""
Integration tests for counterpoint and voice leading MCP endpoints
"""
import pytest
from music21 import stream, note, chord, key, meter, interval

import sys
sys.path.insert(0, 'src')
from music21_mcp.server import (
    score_manager, import_score, check_voice_leading,
    analyze_species_counterpoint, analyze_bach_chorale_style,
    analyze_voice_independence, analyze_fugue
)


class TestCounterpointEndpoints:
    """Test counterpoint and voice leading analysis endpoints"""
    
    @pytest.fixture
    def setup_two_voice_score(self):
        """Create a simple two-voice counterpoint score"""
        s = stream.Score()
        
        # Upper voice
        upper = stream.Part()
        upper.partName = "Upper Voice"
        upper_notes = ['C5', 'D5', 'E5', 'F5', 'G5', 'F5', 'E5', 'D5', 'C5']
        for pitch in upper_notes:
            upper.append(note.Note(pitch, quarterLength=1))
        
        # Lower voice
        lower = stream.Part()
        lower.partName = "Lower Voice"
        lower_notes = ['C4', 'B3', 'A3', 'D4', 'E4', 'F4', 'G4', 'F4', 'E4']
        for pitch in lower_notes:
            lower.append(note.Note(pitch, quarterLength=1))
        
        s.insert(0, upper)
        s.insert(0, lower)
        score_manager.add_score("two_voice", s, {})
        return "two_voice"
    
    @pytest.fixture
    def setup_parallel_fifths_score(self):
        """Create a score with parallel fifths"""
        s = stream.Score()
        
        # Upper voice
        upper = stream.Part()
        upper_notes = ['C5', 'D5', 'E5', 'F5']
        for pitch in upper_notes:
            upper.append(note.Note(pitch, quarterLength=1))
        
        # Lower voice - parallel fifths!
        lower = stream.Part()
        lower_notes = ['F4', 'G4', 'A4', 'B4']
        for pitch in lower_notes:
            lower.append(note.Note(pitch, quarterLength=1))
        
        s.insert(0, upper)
        s.insert(0, lower)
        score_manager.add_score("parallel_fifths", s, {})
        return "parallel_fifths"
    
    @pytest.fixture
    def setup_chorale_score(self):
        """Create a Bach chorale-style score"""
        s = stream.Score()
        
        # Soprano
        soprano = stream.Part()
        soprano.partName = "Soprano"
        # Using chords to represent homophonic texture
        soprano.append(chord.Chord(['E5', 'C5', 'G4', 'C4'], quarterLength=1))
        soprano.append(chord.Chord(['D5', 'B4', 'G4', 'G3'], quarterLength=1))
        soprano.append(chord.Chord(['C5', 'C5', 'E4', 'C4'], quarterLength=2))
        
        s.append(soprano)
        score_manager.add_score("chorale", s, {})
        return "chorale"
    
    @pytest.fixture
    def setup_fugue_score(self):
        """Create a simple fugue exposition"""
        s = stream.Score()
        
        # Subject in first voice
        voice1 = stream.Part()
        voice1.partName = "Voice 1"
        subject = ['C4', 'E4', 'G4', 'F4', 'E4', 'D4', 'C4', 'D4']
        for pitch in subject:
            voice1.append(note.Note(pitch, quarterLength=0.5))
        # Continue with free material
        for pitch in ['E4', 'F4', 'G4', 'A4']:
            voice1.append(note.Note(pitch, quarterLength=1))
        
        # Answer in second voice (at the fifth)
        voice2 = stream.Part()
        voice2.partName = "Voice 2"
        # Rest during subject
        voice2.append(note.Rest(quarterLength=4))
        # Answer at the fifth
        answer = ['G4', 'B4', 'D5', 'C5', 'B4', 'A4', 'G4', 'A4']
        for pitch in answer:
            voice2.append(note.Note(pitch, quarterLength=0.5))
        
        s.insert(0, voice1)
        s.insert(0, voice2)
        score_manager.add_score("fugue", s, {})
        return "fugue"
    
    @pytest.mark.asyncio
    async def test_voice_leading_check_clean(self, setup_two_voice_score):
        """Test voice leading check on clean counterpoint"""
        result = await check_voice_leading(
            score_id=setup_two_voice_score,
            check_parallels=True,
            check_voice_crossing=True
        )
        
        assert result['status'] == 'success'
        assert 'parallel_fifths' in result
        assert 'parallel_octaves' in result
        assert 'voice_crossings' in result
        assert 'voice_overlaps' in result
        assert 'smoothness_score' in result
        
        # Should have good voice leading
        assert result['smoothness_score'] > 0.5
    
    @pytest.mark.asyncio
    async def test_voice_leading_parallel_fifths(self, setup_parallel_fifths_score):
        """Test detection of parallel fifths"""
        result = await check_voice_leading(
            score_id=setup_parallel_fifths_score,
            check_parallels=True
        )
        
        assert result['status'] == 'success'
        # Should detect parallel fifths
        assert result['parallel_fifths'] > 0
    
    @pytest.mark.asyncio
    async def test_species_counterpoint_first(self, setup_two_voice_score):
        """Test first species counterpoint analysis"""
        result = await analyze_species_counterpoint(
            cantus_firmus_id=setup_two_voice_score,
            counterpoint_id=setup_two_voice_score,
            species="first"
        )
        
        assert result['status'] == 'success'
        assert 'species' in result
        assert result['species'] == 'first'
        assert 'errors' in result
        assert 'warnings' in result
        assert 'interval_distribution' in result
        assert 'overall_score' in result
        
        # Check interval distribution
        assert isinstance(result['interval_distribution'], dict)
    
    @pytest.mark.asyncio
    async def test_species_counterpoint_second(self, setup_two_voice_score):
        """Test second species counterpoint analysis"""
        result = await analyze_species_counterpoint(
            cantus_firmus_id=setup_two_voice_score,
            counterpoint_id=setup_two_voice_score,
            species="second"
        )
        
        assert result['status'] == 'success'
        assert result['species'] == 'second'
    
    @pytest.mark.asyncio
    async def test_bach_chorale_style(self, setup_chorale_score):
        """Test Bach chorale style analysis"""
        result = await analyze_bach_chorale_style(score_id=setup_chorale_score)
        
        assert result['status'] == 'success'
        assert 'voice_leading_errors' in result
        assert 'voice_ranges' in result
        assert 'texture_analysis' in result
        assert 'style_conformance_score' in result
        
        # Should identify SATB voices
        assert 'Soprano' in result['voice_ranges'] or len(result['voice_ranges']) > 0
    
    @pytest.mark.asyncio
    async def test_voice_independence(self, setup_two_voice_score):
        """Test voice independence analysis"""
        result = await analyze_voice_independence(score_id=setup_two_voice_score)
        
        assert result['status'] == 'success'
        assert 'voice_count' in result
        assert result['voice_count'] == 2
        assert 'rhythmic_independence' in result
        assert 'parallel_ratio' in result
        assert 'contrary_ratio' in result
        assert 'oblique_ratio' in result
        assert 'similar_ratio' in result
        assert 'overall_independence' in result
        
        # Should have some independence
        assert result['overall_independence'] > 0
    
    @pytest.mark.asyncio
    async def test_fugue_analysis(self, setup_fugue_score):
        """Test fugue analysis"""
        result = await analyze_fugue(
            score_id=setup_fugue_score,
            subject_length=8
        )
        
        assert result['status'] == 'success'
        assert 'subject' in result
        assert 'subject_entries' in result
        assert 'answer_entries' in result
        assert 'strettos' in result
        assert 'episodes' in result
        assert 'fugal_score' in result
        
        # Should detect subject
        assert result['subject']['length'] == 8
        assert result['subject_entries'] > 0
        assert result['answer_entries'] > 0
    
    @pytest.mark.asyncio
    async def test_empty_score_handling(self):
        """Test handling of empty scores"""
        s = stream.Score()
        score_manager.add_score("empty_counterpoint", s, {})
        
        result = await check_voice_leading(score_id="empty_counterpoint")
        
        assert result['status'] == 'success'
        assert result['message'] == "Score has no parts"
    
    @pytest.mark.asyncio
    async def test_single_voice_handling(self):
        """Test handling of single voice scores"""
        s = stream.Score()
        p = stream.Part()
        p.append(note.Note('C4', quarterLength=1))
        s.append(p)
        score_manager.add_score("single_voice", s, {})
        
        result = await analyze_voice_independence(score_id="single_voice")
        
        assert result['status'] == 'success'
        assert result['message'] == "Score has fewer than 2 parts"
        assert result['overall_independence'] == 1.0
    
    @pytest.mark.asyncio
    async def test_invalid_species_handling(self, setup_two_voice_score):
        """Test handling of invalid species type"""
        result = await analyze_species_counterpoint(
            cantus_firmus_id=setup_two_voice_score,
            counterpoint_id=setup_two_voice_score,
            species="invalid_species"
        )
        
        # Should still work with a default or handle gracefully
        assert result['status'] in ['success', 'error']


class TestComplexCounterpointScenarios:
    """Test complex counterpoint scenarios"""
    
    @pytest.mark.asyncio
    async def test_three_part_counterpoint(self):
        """Test three-part counterpoint analysis"""
        s = stream.Score()
        
        # Three independent voices
        for i, starting_pitch in enumerate(['C5', 'G4', 'C4']):
            part = stream.Part()
            part.partName = f"Voice {i+1}"
            
            # Create melodic lines
            if i == 0:  # Top voice
                pitches = ['C5', 'B4', 'C5', 'D5', 'E5', 'D5', 'C5']
            elif i == 1:  # Middle voice
                pitches = ['G4', 'G4', 'A4', 'B4', 'C5', 'B4', 'G4']
            else:  # Bass
                pitches = ['C4', 'D4', 'F4', 'G4', 'C4', 'G3', 'C4']
            
            for pitch in pitches:
                part.append(note.Note(pitch, quarterLength=1))
            
            s.insert(0, part)
        
        score_manager.add_score("three_part", s, {})
        
        result = await analyze_voice_independence(score_id="three_part")
        
        assert result['status'] == 'success'
        assert result['voice_count'] == 3
        assert result['overall_independence'] > 0
    
    @pytest.mark.asyncio
    async def test_fugue_with_stretto(self):
        """Test fugue with stretto detection"""
        s = stream.Score()
        
        # Create overlapping subject entries
        subject = ['C4', 'D4', 'E4', 'F4', 'E4', 'D4', 'C4', 'B3']
        
        # Voice 1
        v1 = stream.Part()
        for pitch in subject:
            v1.append(note.Note(pitch, quarterLength=0.5))
        # Continue
        for pitch in ['C4', 'D4', 'E4', 'F4']:
            v1.append(note.Note(pitch, quarterLength=1))
        
        # Voice 2 - enters early (stretto)
        v2 = stream.Part()
        v2.append(note.Rest(quarterLength=2))  # Short rest
        # Subject in stretto
        for pitch in subject:
            v2.append(note.Note(pitch, quarterLength=0.5))
        
        s.insert(0, v1)
        s.insert(0, v2)
        score_manager.add_score("stretto_fugue", s, {})
        
        result = await analyze_fugue(
            score_id="stretto_fugue",
            subject_length=8
        )
        
        assert result['status'] == 'success'
        # Should detect overlapping entries
        assert result['subject_entries'] >= 2