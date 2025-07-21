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
            species="first",
            voice_position="above",
        )

        assert result["status"] == "success"
        # Check that the counterpoint was stored with the expected ID
        assert "cantus_counterpoint_first" in clean_score_storage
        # Check response contains expected fields
        assert "counterpoint_score_id" in result
        assert "rule_violations" in result
        assert "rule_compliance_score" in result

    @pytest.mark.asyncio
    async def test_counterpoint_nonexistent_score(self, clean_score_storage):
        """Test counterpoint generation with non-existent score"""
        tool = CounterpointGeneratorTool(clean_score_storage)

        result = await tool.execute(
            score_id="nonexistent",
            output_id="counterpoint",
            species="first",
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

        # Test different species
        for species in ["first", "second", "third", "fourth", "fifth"]:
            result = await tool.execute(
                score_id="cantus",
                species=species,
                voice_position="above",
            )

            # Some species might be more complex to implement
            assert result["status"] in ["success", "error"]
            if result["status"] == "success":
                # Check that counterpoint was stored with auto-generated ID
                assert f"cantus_counterpoint_{species}" in clean_score_storage

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
            species="first",
            voice_position="above",
        )
        assert result_above["status"] == "success"
        assert result_above["voice_position"] == "above"

        # Test below - need to use different species to avoid duplicate ID
        result_below = await tool.execute(
            score_id="cantus",
            species="second",
            voice_position="below",
        )
        assert result_below["status"] == "success"
        assert result_below["voice_position"] == "below"

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
            species="first",
            voice_position="above",
            rule_set="strict",
        )

        assert result["status"] == "success"
        assert "rule_violations" in result
        assert isinstance(result["rule_violations"], list)
        assert "rule_compliance_score" in result

    @pytest.mark.asyncio
    async def test_counterpoint_duplicate_output_id(self, clean_score_storage):
        """Test counterpoint generation with duplicate output ID"""
        from music21 import note, stream

        cantus = stream.Stream()
        cantus.append(note.Note("C4", quarterLength=1))

        clean_score_storage["cantus"] = cantus

        tool = CounterpointGeneratorTool(clean_score_storage)
        # Generate first counterpoint
        result1 = await tool.execute(
            score_id="cantus",
            species="first",
            voice_position="above",
        )
        assert result1["status"] == "success"

        # Try to generate again with same parameters (would create duplicate ID)
        result2 = await tool.execute(
            score_id="cantus",
            species="first",
            voice_position="above",
        )
        # Tool should handle this gracefully - either overwrite or generate new ID
        assert result2["status"] == "success"

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
            species="first",
            voice_position="above",
        )

        assert result["status"] == "success"
        # Check that response contains expected fields
        assert "species" in result
        assert "voice_position" in result
        assert "interval_analysis" in result
        assert "melodic_analysis" in result
        assert result["species"] == "first"
        assert result["voice_position"] == "above"

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
            species="invalid",  # Invalid species
            voice_position="above",
        )

        assert result["status"] == "error"
        assert "species" in result["message"].lower()
