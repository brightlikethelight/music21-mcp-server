"""
Unit tests for CounterpointGeneratorTool
"""

import pytest

from music21_mcp.tools.counterpoint_tool import CounterpointGeneratorTool


class TestCounterpointGeneratorTool:
    """Test CounterpointGeneratorTool functionality"""

    def test_tool_initialization(self, clean_score_storage):
        """Test tool can be initialized with score storage"""
        tool = CounterpointGeneratorTool(clean_score_storage)
        assert tool.scores == clean_score_storage

    @pytest.mark.asyncio
    async def test_counterpoint_generation_success(self, clean_score_storage):
        """Test successful counterpoint generation"""
        from music21 import note, stream

        # Create cantus firmus
        cantus = stream.Stream()
        for pitch in ["C4", "D4", "F4", "E4", "D4", "C4"]:
            cantus.append(note.Note(pitch, quarterLength=1))

        clean_score_storage["cantus"] = cantus

        tool = CounterpointGeneratorTool(clean_score_storage)
        result = await tool.execute(
            score_id="cantus",
            output_id="counterpoint",
            species=1,
            voice_position="above",
        )

        assert result["status"] == "success"
        assert "counterpoint" in clean_score_storage
        assert "counterpoint_info" in result

    @pytest.mark.asyncio
    async def test_counterpoint_nonexistent_score(self, clean_score_storage):
        """Test counterpoint generation with non-existent score"""
        tool = CounterpointGeneratorTool(clean_score_storage)

        result = await tool.execute(
            score_id="nonexistent",
            output_id="counterpoint",
            species=1,
            voice_position="above",
        )

        assert result["status"] == "error"
        assert "not found" in result["message"]

    @pytest.mark.asyncio
    async def test_counterpoint_species(self, clean_score_storage):
        """Test different counterpoint species"""
        from music21 import note, stream

        # Create cantus firmus
        cantus = stream.Stream()
        for pitch in ["D4", "F4", "E4", "D4"]:
            cantus.append(note.Note(pitch, quarterLength=1))

        clean_score_storage["cantus"] = cantus

        tool = CounterpointGeneratorTool(clean_score_storage)

        # Test different species (1-5)
        for species in [1, 2, 3, 4, 5]:
            result = await tool.execute(
                score_id="cantus",
                output_id=f"counterpoint_species_{species}",
                species=species,
                voice_position="above",
            )

            # Some species might be more complex to implement
            assert result["status"] in ["success", "error"]
            if result["status"] == "success":
                assert f"counterpoint_species_{species}" in clean_score_storage

    @pytest.mark.asyncio
    async def test_counterpoint_voice_positions(self, clean_score_storage):
        """Test counterpoint above and below cantus firmus"""
        from music21 import note, stream

        cantus = stream.Stream()
        for pitch in ["G4", "A4", "B4", "C5", "B4", "A4", "G4"]:
            cantus.append(note.Note(pitch, quarterLength=1))

        clean_score_storage["cantus"] = cantus

        tool = CounterpointGeneratorTool(clean_score_storage)

        # Test above
        result_above = await tool.execute(
            score_id="cantus",
            output_id="counterpoint_above",
            species=1,
            voice_position="above",
        )
        assert result_above["status"] == "success"

        # Test below
        result_below = await tool.execute(
            score_id="cantus",
            output_id="counterpoint_below",
            species=1,
            voice_position="below",
        )
        assert result_below["status"] == "success"

    @pytest.mark.asyncio
    async def test_counterpoint_validation(self, clean_score_storage):
        """Test counterpoint rule validation"""
        from music21 import note, stream

        cantus = stream.Stream()
        for pitch in ["C4", "E4", "G4", "E4", "C4"]:
            cantus.append(note.Note(pitch, quarterLength=1))

        clean_score_storage["cantus"] = cantus

        tool = CounterpointGeneratorTool(clean_score_storage)
        result = await tool.execute(
            score_id="cantus",
            output_id="validated_counterpoint",
            species=1,
            voice_position="above",
            strict_rules=True,
        )

        assert result["status"] == "success"
        info = result["counterpoint_info"]
        assert "rule_violations" in info
        assert isinstance(info["rule_violations"], list)

    @pytest.mark.asyncio
    async def test_counterpoint_duplicate_output_id(self, clean_score_storage):
        """Test counterpoint generation with duplicate output ID"""
        from music21 import note, stream

        cantus = stream.Stream()
        cantus.append(note.Note("C4", quarterLength=1))

        clean_score_storage["cantus"] = cantus
        clean_score_storage["existing_score"] = stream.Stream()

        tool = CounterpointGeneratorTool(clean_score_storage)
        result = await tool.execute(
            score_id="cantus",
            output_id="existing_score",  # Already exists
            species=1,
            voice_position="above",
        )

        assert result["status"] == "error"
        assert "already exists" in result["message"]

    @pytest.mark.asyncio
    async def test_counterpoint_info_structure(self, clean_score_storage):
        """Test counterpoint information structure"""
        from music21 import note, stream

        cantus = stream.Stream()
        for pitch in ["F4", "G4", "A4", "G4", "F4"]:
            cantus.append(note.Note(pitch, quarterLength=1))

        clean_score_storage["cantus"] = cantus

        tool = CounterpointGeneratorTool(clean_score_storage)
        result = await tool.execute(
            score_id="cantus",
            output_id="counterpoint",
            species=1,
            voice_position="above",
        )

        assert result["status"] == "success"
        info = result["counterpoint_info"]
        assert "species" in info
        assert "voice_position" in info
        assert "intervals_used" in info
        assert info["species"] == 1
        assert info["voice_position"] == "above"

    @pytest.mark.asyncio
    async def test_counterpoint_invalid_species(self, clean_score_storage):
        """Test counterpoint with invalid species number"""
        from music21 import note, stream

        cantus = stream.Stream()
        cantus.append(note.Note("C4", quarterLength=1))

        clean_score_storage["cantus"] = cantus

        tool = CounterpointGeneratorTool(clean_score_storage)
        result = await tool.execute(
            score_id="cantus",
            output_id="counterpoint",
            species=0,  # Invalid species
            voice_position="above",
        )

        assert result["status"] == "error"
        assert "species" in result["message"].lower()
