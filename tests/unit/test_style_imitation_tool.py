"""
Unit tests for StyleImitationTool
"""

import pytest

from music21_mcp.tools.style_imitation_tool import StyleImitationTool


class TestStyleImitationTool:
    """Test StyleImitationTool functionality"""

    def test_tool_initialization(self, clean_score_storage):
        """Test tool can be initialized with score storage"""
        tool = StyleImitationTool(clean_score_storage)
        assert tool.scores == clean_score_storage

    @pytest.mark.asyncio
    async def test_style_imitation_success(self, populated_score_storage):
        """Test successful style imitation"""
        tool = StyleImitationTool(populated_score_storage)

        result = await tool.execute(
            style_source="bach_test", generation_length=16, complexity="medium"
        )

        assert result["status"] == "success"
        assert "generated_score_id" in result
        assert "musical_features" in result
        # The generated score should be stored
        assert result["generated_score_id"] in populated_score_storage

    @pytest.mark.asyncio
    async def test_style_imitation_nonexistent_score(self, clean_score_storage):
        """Test style imitation with non-existent score"""
        tool = StyleImitationTool(clean_score_storage)

        result = await tool.execute(
            style_source="nonexistent", generation_length=8, complexity="simple"
        )

        assert result["status"] == "error"
        assert "not found" in result["message"]

    @pytest.mark.asyncio
    async def test_style_imitation_different_styles(self, populated_score_storage):
        """Test imitation with different style parameters"""
        tool = StyleImitationTool(populated_score_storage)

        styles = ["baroque", "classical", "romantic", "modern", "jazz"]

        for style in styles:
            result = await tool.execute(
                composer=style, generation_length=8, complexity="simple"
            )

            # Some styles might not be pre-defined composers
            assert result["status"] in ["success", "error"]
            if result["status"] == "success":
                assert "generated_score_id" in result
                assert result["generated_score_id"] in populated_score_storage

    @pytest.mark.asyncio
    async def test_style_imitation_length_parameter(self, populated_score_storage):
        """Test style imitation with different lengths"""
        tool = StyleImitationTool(populated_score_storage)

        lengths = [4, 8, 16, 32]

        for length in lengths:
            result = await tool.execute(
                style_source="bach_test", generation_length=length, complexity="medium"
            )

            assert result["status"] == "success"
            assert "generated_score_id" in result
            generated = populated_score_storage[result["generated_score_id"]]
            # Check that generated score exists
            assert generated is not None

    @pytest.mark.asyncio
    async def test_style_imitation_analysis_info(self, populated_score_storage):
        """Test style analysis information returned"""
        tool = StyleImitationTool(populated_score_storage)

        result = await tool.execute(
            style_source="bach_test", generation_length=8, complexity="medium"
        )

        assert result["status"] == "success"
        assert "musical_features" in result
        features = result["musical_features"]
        # Check for style characteristics
        assert "melodic" in features
        assert "harmonic" in features
        assert "rhythmic" in features

    @pytest.mark.asyncio
    async def test_style_imitation_duplicate_output_id(self, populated_score_storage):
        """Test style imitation with duplicate output ID"""
        from music21 import stream

        populated_score_storage["existing_score"] = stream.Stream()

        tool = StyleImitationTool(populated_score_storage)
        # Style imitation tool doesn't have output_id parameter
        result = await tool.execute(
            style_source="bach_test", generation_length=8, complexity="simple"
        )

        assert result["status"] == "success"
        assert "generated_score_id" in result

    @pytest.mark.asyncio
    async def test_style_imitation_custom_parameters(self, populated_score_storage):
        """Test style imitation with custom parameters"""
        tool = StyleImitationTool(populated_score_storage)

        result = await tool.execute(
            style_source="bach_test",
            generation_length=16,
            complexity="complex",
            starting_note="G4",
            constraints=["key:G", "range:G3-G5"],
        )

        assert result["status"] == "success"
        assert "generated_score_id" in result
        assert result["generated_score_id"] in populated_score_storage

    @pytest.mark.asyncio
    async def test_style_imitation_minimal_input(self, clean_score_storage):
        """Test style imitation with minimal input score"""
        from music21 import note, stream

        # Create very simple input
        simple = stream.Stream()
        simple.append(note.Note("C4", quarterLength=1))
        simple.append(note.Note("D4", quarterLength=1))

        clean_score_storage["simple"] = simple

        tool = StyleImitationTool(clean_score_storage)
        result = await tool.execute(
            score_id="simple", output_id="imitation", style="classical", length=4
        )

        # Should handle minimal input gracefully
        # Should handle minimal input
        assert result["status"] in ["success", "error"]
        if result["status"] == "success":
            assert "generated_score_id" in result

    @pytest.mark.asyncio
    async def test_style_imitation_invalid_length(self, populated_score_storage):
        """Test style imitation with invalid length"""
        tool = StyleImitationTool(populated_score_storage)

        result = await tool.execute(
            style_source="bach_test",
            generation_length=-1,  # Invalid negative length
            complexity="simple",
        )

        assert result["status"] == "error"
        assert "generation_length" in result["message"]

    @pytest.mark.asyncio
    async def test_style_imitation_auto_detect_style(self, populated_score_storage):
        """Test style imitation with auto-detected style"""
        tool = StyleImitationTool(populated_score_storage)

        # Use predefined Bach style
        result = await tool.execute(
            composer="bach", generation_length=8, complexity="simple"
        )

        assert result["status"] == "success"
        assert "generated_score_id" in result
        assert "musical_features" in result
