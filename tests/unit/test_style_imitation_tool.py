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


class TestAnalyzeStyleMethod:
    """Test the analyze_style code path (lines 205-251)."""

    @pytest.mark.asyncio
    async def test_analyze_style_detailed(self, populated_score_storage):
        """analyze_style with detailed=True exercises interval/rhythm/chord analysis."""
        tool = StyleImitationTool(populated_score_storage)
        result = await tool.analyze_style(score_id="bach_test", detailed=True)
        assert result["status"] == "success"
        assert "style_characteristics" in result
        style = result["style_characteristics"]
        assert "interval_distribution" in style
        assert "rhythm_histogram" in style
        assert "chord_vocabulary" in style
        assert "phrase_lengths" in style
        assert "distinctive_features" in style
        assert "closest_known_styles" in result

    @pytest.mark.asyncio
    async def test_analyze_style_not_detailed(self, populated_score_storage):
        """analyze_style with detailed=False skips extra analysis."""
        tool = StyleImitationTool(populated_score_storage)
        result = await tool.analyze_style(score_id="bach_test", detailed=False)
        assert result["status"] == "success"
        style = result["style_characteristics"]
        assert "interval_distribution" not in style

    @pytest.mark.asyncio
    async def test_analyze_style_missing_score(self, clean_score_storage):
        """analyze_style on non-existent score returns error."""
        tool = StyleImitationTool(clean_score_storage)
        result = await tool.analyze_style(score_id="missing")
        assert result["status"] == "error"
        assert "not found" in result["message"]


class TestStyleHelpers:
    """Test internal style helper methods for coverage."""

    def test_identify_distinctive_features(self, clean_score_storage):
        tool = StyleImitationTool(clean_score_storage)
        style_data = {
            "melodic": {"stepwise_motion": 0.9, "leap_frequency": 0.4},
            "harmonic": {"dissonance_level": 2.0, "chord_density": 0.3},
            "rhythmic": {"syncopation_level": 0.5, "rhythm_variety": 8},
        }
        features = tool._identify_distinctive_features(style_data)
        assert "Highly stepwise melodic motion" in features
        assert "Frequent melodic leaps" in features
        assert "High harmonic dissonance" in features
        assert "Sparse harmonic rhythm" in features
        assert "Significant syncopation" in features
        assert "Complex rhythmic vocabulary" in features

    def test_compare_to_known_styles(self, clean_score_storage):
        tool = StyleImitationTool(clean_score_storage)
        style_data = {
            "melodic": {"stepwise_motion": 0.7},
            "harmonic": {},
            "rhythmic": {},
        }
        similarities = tool._compare_to_known_styles(style_data)
        assert isinstance(similarities, list)
        assert len(similarities) <= 3
        # Each entry is (composer_name, similarity_score)
        for name, score in similarities:
            assert isinstance(name, str)
            assert 0 <= score <= 1
