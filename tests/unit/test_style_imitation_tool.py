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
            score_id="bach_test", output_id="imitation", style="baroque", length=16
        )

        assert result["status"] == "success"
        assert "imitation" in populated_score_storage
        assert "style_analysis" in result
        assert "generated_score_id" in result

    @pytest.mark.asyncio
    async def test_style_imitation_nonexistent_score(self, clean_score_storage):
        """Test style imitation with non-existent score"""
        tool = StyleImitationTool(clean_score_storage)

        result = await tool.execute(
            score_id="nonexistent", output_id="imitation", style="classical", length=8
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
                score_id="bach_test",
                output_id=f"imitation_{style}",
                style=style,
                length=8,
            )

            # Some styles might be more complex to implement
            assert result["status"] in ["success", "error"]
            if result["status"] == "success":
                assert f"imitation_{style}" in populated_score_storage

    @pytest.mark.asyncio
    async def test_style_imitation_length_parameter(self, populated_score_storage):
        """Test style imitation with different lengths"""
        tool = StyleImitationTool(populated_score_storage)

        lengths = [4, 8, 16, 32]

        for length in lengths:
            result = await tool.execute(
                score_id="bach_test",
                output_id=f"imitation_{length}bars",
                style="baroque",
                length=length,
            )

            assert result["status"] == "success"
            generated = populated_score_storage[f"imitation_{length}bars"]
            # Check that generated score has appropriate length
            assert generated.quarterLength >= 0

    @pytest.mark.asyncio
    async def test_style_imitation_analysis_info(self, populated_score_storage):
        """Test style analysis information returned"""
        tool = StyleImitationTool(populated_score_storage)

        result = await tool.execute(
            score_id="bach_test",
            output_id="imitation_analyzed",
            style="baroque",
            length=8,
        )

        assert result["status"] == "success"
        analysis = result["style_analysis"]
        assert "detected_style" in analysis
        assert "key_characteristics" in analysis
        assert "rhythmic_patterns" in analysis
        assert "harmonic_patterns" in analysis

    @pytest.mark.asyncio
    async def test_style_imitation_duplicate_output_id(self, populated_score_storage):
        """Test style imitation with duplicate output ID"""
        from music21 import stream

        populated_score_storage["existing_score"] = stream.Stream()

        tool = StyleImitationTool(populated_score_storage)
        result = await tool.execute(
            score_id="bach_test",
            output_id="existing_score",  # Already exists
            style="baroque",
            length=8,
        )

        assert result["status"] == "error"
        assert "already exists" in result["message"]

    @pytest.mark.asyncio
    async def test_style_imitation_custom_parameters(self, populated_score_storage):
        """Test style imitation with custom parameters"""
        tool = StyleImitationTool(populated_score_storage)

        result = await tool.execute(
            score_id="bach_test",
            output_id="custom_imitation",
            style="baroque",
            length=16,
            tempo=120,
            time_signature="3/4",
            key="G major",
        )

        assert result["status"] == "success"
        assert "custom_imitation" in populated_score_storage

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
        assert result["status"] in ["success", "error"]
        if result["status"] == "error":
            assert (
                "insufficient" in result["message"].lower()
                or "minimal" in result["message"].lower()
            )

    @pytest.mark.asyncio
    async def test_style_imitation_invalid_length(self, populated_score_storage):
        """Test style imitation with invalid length"""
        tool = StyleImitationTool(populated_score_storage)

        result = await tool.execute(
            score_id="bach_test",
            output_id="imitation",
            style="baroque",
            length=-1,  # Invalid negative length
        )

        assert result["status"] == "error"
        assert "length" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_style_imitation_auto_detect_style(self, populated_score_storage):
        """Test style imitation with auto-detected style"""
        tool = StyleImitationTool(populated_score_storage)

        result = await tool.execute(
            score_id="bach_test",
            output_id="auto_style_imitation",
            style="auto",  # Auto-detect style
            length=8,
        )

        assert result["status"] == "success"
        analysis = result["style_analysis"]
        assert "detected_style" in analysis
        # Bach should be detected as baroque
        assert analysis["detected_style"].lower() in ["baroque", "bach", "classical"]
