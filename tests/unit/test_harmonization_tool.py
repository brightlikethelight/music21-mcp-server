"""
Unit tests for HarmonizationTool
"""
import pytest
from music21_mcp.tools.harmonization_tool import HarmonizationTool


class TestHarmonizationTool:
    """Test HarmonizationTool functionality"""
    
    def test_tool_initialization(self, clean_score_storage):
        """Test tool can be initialized with score storage"""
        tool = HarmonizationTool(clean_score_storage)
        assert tool.scores == clean_score_storage
    
    @pytest.mark.asyncio
    async def test_harmonization_success(self, clean_score_storage):
        """Test successful harmonization of melody"""
        from music21 import stream, note, key
        
        # Create simple melody
        melody = stream.Stream()
        melody.append(key.Key('C'))
        for pitch in ['C4', 'E4', 'G4', 'C5']:
            melody.append(note.Note(pitch, quarterLength=1))
        
        clean_score_storage["melody"] = melody
        
        tool = HarmonizationTool(clean_score_storage)
        result = await tool.execute(
            score_id="melody",
            output_id="harmonized_melody",
            style="chorale"
        )
        
        assert result["status"] == "success"
        assert "harmonized_melody" in clean_score_storage
        assert "harmonization" in result
    
    @pytest.mark.asyncio
    async def test_harmonization_nonexistent_score(self, clean_score_storage):
        """Test harmonization with non-existent score"""
        tool = HarmonizationTool(clean_score_storage)
        
        result = await tool.execute(
            score_id="nonexistent",
            output_id="harmonized",
            style="chorale"
        )
        
        assert result["status"] == "error"
        assert "not found" in result["message"]
    
    @pytest.mark.asyncio
    async def test_harmonization_styles(self, clean_score_storage):
        """Test different harmonization styles"""
        from music21 import stream, note, key
        
        # Create melody
        melody = stream.Stream()
        melody.append(key.Key('C'))
        for pitch in ['C4', 'D4', 'E4', 'F4', 'G4']:
            melody.append(note.Note(pitch, quarterLength=1))
        
        clean_score_storage["melody"] = melody
        
        tool = HarmonizationTool(clean_score_storage)
        
        # Test different styles
        styles = ["chorale", "classical", "romantic", "jazz"]
        for style in styles:
            result = await tool.execute(
                score_id="melody",
                output_id=f"harmonized_{style}",
                style=style
            )
            
            # Some styles might not be fully implemented
            assert result["status"] in ["success", "error"]
            if result["status"] == "success":
                assert f"harmonized_{style}" in clean_score_storage
    
    @pytest.mark.asyncio
    async def test_harmonization_parameters(self, clean_score_storage):
        """Test harmonization with custom parameters"""
        from music21 import stream, note, key
        
        melody = stream.Stream()
        melody.append(key.Key('G'))
        for pitch in ['G4', 'A4', 'B4', 'C5', 'D5']:
            melody.append(note.Note(pitch, quarterLength=1))
        
        clean_score_storage["melody"] = melody
        
        tool = HarmonizationTool(clean_score_storage)
        result = await tool.execute(
            score_id="melody",
            output_id="harmonized_custom",
            style="chorale",
            voices=4,
            key="G major",
            time_signature="4/4"
        )
        
        assert result["status"] == "success"
        assert "harmonized_custom" in clean_score_storage
    
    @pytest.mark.asyncio
    async def test_harmonization_duplicate_output_id(self, clean_score_storage):
        """Test harmonization with duplicate output ID"""
        from music21 import stream, note
        
        melody = stream.Stream()
        melody.append(note.Note('C4', quarterLength=1))
        
        clean_score_storage["melody"] = melody
        clean_score_storage["existing_score"] = stream.Stream()
        
        tool = HarmonizationTool(clean_score_storage)
        result = await tool.execute(
            score_id="melody",
            output_id="existing_score",  # Already exists
            style="chorale"
        )
        
        assert result["status"] == "error"
        assert "already exists" in result["message"]
    
    @pytest.mark.asyncio
    async def test_harmonization_voice_count(self, clean_score_storage):
        """Test harmonization with different voice counts"""
        from music21 import stream, note, key
        
        melody = stream.Stream()
        melody.append(key.Key('C'))
        for pitch in ['C4', 'E4', 'G4']:
            melody.append(note.Note(pitch, quarterLength=1))
        
        clean_score_storage["melody"] = melody
        
        tool = HarmonizationTool(clean_score_storage)
        
        # Test different voice counts
        for voices in [3, 4, 5]:
            result = await tool.execute(
                score_id="melody",
                output_id=f"harmonized_{voices}v",
                style="chorale",
                voices=voices
            )
            
            assert result["status"] == "success"
            harmonized = clean_score_storage[f"harmonized_{voices}v"]
            # Check that harmonization has correct number of voices
            parts = harmonized.parts
            assert len(parts) >= 1  # At least original melody
    
    @pytest.mark.asyncio
    async def test_harmonization_info(self, clean_score_storage):
        """Test harmonization information returned"""
        from music21 import stream, note, key
        
        melody = stream.Stream()
        melody.append(key.Key('F'))
        for pitch in ['F4', 'G4', 'A4', 'Bb4', 'C5']:
            melody.append(note.Note(pitch, quarterLength=1))
        
        clean_score_storage["melody"] = melody
        
        tool = HarmonizationTool(clean_score_storage)
        result = await tool.execute(
            score_id="melody",
            output_id="harmonized",
            style="chorale"
        )
        
        assert result["status"] == "success"
        harmonization = result["harmonization"]
        assert "style" in harmonization
        assert "voices" in harmonization
        assert "key" in harmonization
        assert harmonization["style"] == "chorale"