"""
Unit tests for individual tool classes
Tests each tool in isolation with mocked dependencies
"""

from unittest.mock import AsyncMock, Mock

import pytest
from music21 import chord, note, stream

from music21_mcp.tools import (
    ChordAnalysisTool,
    CounterpointGeneratorTool,
    DeleteScoreTool,
    ExportScoreTool,
    HarmonizationTool,
    HarmonyAnalysisTool,
    ImportScoreTool,
    KeyAnalysisTool,
    ListScoresTool,
    PatternRecognitionTool,
    ScoreInfoTool,
    StyleImitationTool,
    VoiceLeadingAnalysisTool,
)


@pytest.fixture
def mock_score_manager():
    """Mock score manager for testing"""
    # Create a custom class that inherits from dict but has async methods
    class MockScoreManager(dict):
        def __init__(self):
            super().__init__()
            self.add_score = AsyncMock()
            self.get_score = AsyncMock()
            self.list_scores = AsyncMock()
            self.remove_score = AsyncMock()
            # Override get to be a Mock
            self._dict_get = super().get
            self.get = Mock()
        
        def __getitem__(self, key):
            # When tests set mock_score_manager[key] = value
            if key in self:
                return super().__getitem__(key)
            # When tools call score_manager.get(key), use the mock
            return self.get.return_value
    
    return MockScoreManager()


@pytest.fixture
def sample_score():
    """Create a sample score for testing"""
    from music21 import metadata

    s = stream.Score()
    part = stream.Part()

    # Add some measures with notes
    for i in range(4):
        m = stream.Measure(number=i + 1)
        m.append(note.Note("C4", quarterLength=1))
        m.append(note.Note("E4", quarterLength=1))
        m.append(note.Note("G4", quarterLength=1))
        m.append(note.Note("C5", quarterLength=1))
        part.append(m)

    s.append(part)

    # Initialize metadata properly
    s.metadata = metadata.Metadata()
    s.metadata.title = "Test Score"
    s.metadata.composer = "Test Composer"
    return s


class TestImportScoreTool:
    """Test ImportScoreTool functionality"""

    @pytest.mark.asyncio
    async def test_import_corpus_score(self, mock_score_manager):
        tool = ImportScoreTool(mock_score_manager)

        result = await tool.execute(
            score_id="test_score", source="bach/bwv66.6", source_type="corpus"
        )

        assert result["status"] == "success"
        assert result["score_id"] == "test_score"
        # ImportScoreTool uses dict assignment, not add_score method
        assert "test_score" in mock_score_manager

    @pytest.mark.asyncio
    async def test_import_text_score(self, mock_score_manager):
        tool = ImportScoreTool(mock_score_manager)

        result = await tool.execute(
            score_id="test_score",
            source="tinyNotation: 4/4 c4 d4 e4 f4",
            source_type="text",
        )

        assert result["status"] == "success"
        assert result["score_id"] == "test_score"
        # ImportScoreTool uses dict assignment, not add_score method
        assert "test_score" in mock_score_manager

    @pytest.mark.asyncio
    async def test_import_file_not_found(self, mock_score_manager):
        tool = ImportScoreTool(mock_score_manager)

        result = await tool.execute(
            score_id="test_score", source="/nonexistent/file.xml", source_type="file"
        )

        assert result["status"] == "error"
        assert "failed" in result["message"].lower()


class TestListScoresTool:
    """Test ListScoresTool functionality"""

    @pytest.mark.asyncio
    async def test_list_empty_scores(self, mock_score_manager):
        mock_score_manager.list_scores.return_value = []
        tool = ListScoresTool(mock_score_manager)

        result = await tool.execute()

        assert result["count"] == 0
        assert result["scores"] == []

    @pytest.mark.asyncio
    async def test_list_multiple_scores(self, mock_score_manager, sample_score):
        # Add multiple scores to the manager
        mock_score_manager["score1"] = sample_score
        mock_score_manager["score2"] = sample_score
        
        tool = ListScoresTool(mock_score_manager)

        result = await tool.execute()

        assert result["status"] == "success"
        assert result["count"] == 2
        assert len(result["scores"]) == 2
        # Check that both scores are in the result
        score_ids = [s["score_id"] for s in result["scores"]]
        assert "score1" in score_ids
        assert "score2" in score_ids


class TestKeyAnalysisTool:
    """Test KeyAnalysisTool functionality"""

    @pytest.mark.asyncio
    async def test_analyze_key_krumhansl(self, mock_score_manager, sample_score):
        # Add score to manager for existence check
        mock_score_manager["test_score"] = sample_score
        mock_score_manager.get.return_value = sample_score
        tool = KeyAnalysisTool(mock_score_manager)

        result = await tool.execute(score_id="test_score", algorithm="krumhansl")

        assert "key" in result
        assert "confidence" in result
        assert "alternatives" in result
        assert result["algorithm"] == "Krumhansl-Schmuckler"

    @pytest.mark.asyncio
    async def test_analyze_key_score_not_found(self, mock_score_manager):
        # Don't add score - it should not be found
        mock_score_manager.get.return_value = None
        tool = KeyAnalysisTool(mock_score_manager)

        result = await tool.execute(score_id="nonexistent")

        assert result["status"] == "error"
        assert "not found" in result["message"]


class TestChordAnalysisTool:
    """Test ChordAnalysisTool functionality"""

    @pytest.mark.asyncio
    async def test_analyze_chords(self, mock_score_manager, sample_score):
        # Add some chords to the score
        for m in sample_score.parts[0].getElementsByClass("Measure"):
            m.clear()
            c = chord.Chord(["C4", "E4", "G4"])
            m.append(c)

        # Add score to manager for existence check
        mock_score_manager["test_score"] = sample_score
        mock_score_manager.get.return_value = sample_score
        tool = ChordAnalysisTool(mock_score_manager)

        result = await tool.execute(score_id="test_score")

        assert "chord_progression" in result
        assert "total_chords" in result
        assert "summary" in result
        assert result["total_chords"] > 0


class TestScoreInfoTool:
    """Test ScoreInfoTool functionality"""

    @pytest.mark.asyncio
    async def test_get_score_info(self, mock_score_manager, sample_score):
        # Add score to manager for existence check
        mock_score_manager["test_score"] = sample_score
        mock_score_manager.get.return_value = sample_score
        tool = ScoreInfoTool(mock_score_manager)

        result = await tool.execute(score_id="test_score")

        assert result["title"] == "Test Score"
        assert result["composer"] == "Test Composer"
        assert result["num_parts"] == 1
        assert result["num_measures"] == 4
        assert "duration_seconds" in result
        assert "structure" in result


class TestExportScoreTool:
    """Test ExportScoreTool functionality"""

    @pytest.mark.asyncio
    async def test_export_musicxml(self, mock_score_manager, sample_score, tmp_path):
        # Add score to manager for existence check
        mock_score_manager["test_score"] = sample_score
        mock_score_manager.get.return_value = sample_score
        tool = ExportScoreTool(mock_score_manager)

        output_path = str(tmp_path / "test.xml")
        result = await tool.execute(
            score_id="test_score", format="musicxml", output_path=output_path
        )

        assert result["status"] == "success"
        assert result["format"] == "musicxml"
        assert result["file_path"] == output_path

    @pytest.mark.asyncio
    async def test_export_midi(self, mock_score_manager, sample_score, tmp_path):
        # Add score to manager for existence check
        mock_score_manager["test_score"] = sample_score
        mock_score_manager.get.return_value = sample_score
        tool = ExportScoreTool(mock_score_manager)

        output_path = str(tmp_path / "test.mid")
        result = await tool.execute(
            score_id="test_score", format="midi", output_path=output_path
        )

        assert result["status"] == "success"
        assert result["format"] == "midi"


class TestDeleteScoreTool:
    """Test DeleteScoreTool functionality"""

    @pytest.mark.asyncio
    async def test_delete_existing_score(self, mock_score_manager, sample_score):
        # Add a score to delete
        mock_score_manager["test_score"] = sample_score
        tool = DeleteScoreTool(mock_score_manager)

        result = await tool.execute(score_id="test_score")

        assert result["status"] == "success"
        assert "Deleted" in result["message"]
        assert "test_score" not in mock_score_manager

    @pytest.mark.asyncio
    async def test_delete_nonexistent_score(self, mock_score_manager):
        # Don't add any score, so it won't exist
        tool = DeleteScoreTool(mock_score_manager)

        result = await tool.execute(score_id="nonexistent")

        assert result["status"] == "error"
        assert "not found" in result["message"]


class TestHarmonyAnalysisTool:
    """Test HarmonyAnalysisTool functionality"""

    @pytest.mark.asyncio
    async def test_analyze_harmony(self, mock_score_manager, sample_score):
        # Add chords for harmony analysis
        for m in sample_score.parts[0].getElementsByClass("Measure"):
            m.clear()
            m.append(chord.Chord(["C4", "E4", "G4"]))
            m.append(chord.Chord(["G4", "B4", "D5"]))

        # Add score to manager for existence check
        mock_score_manager["test_score"] = sample_score
        mock_score_manager.get.return_value = sample_score
        tool = HarmonyAnalysisTool(mock_score_manager)

        result = await tool.execute(score_id="test_score")

        assert "harmonic_rhythm" in result
        assert "chord_progressions" in result
        assert "roman_numerals" in result
        assert "functional_analysis" in result


class TestVoiceLeadingAnalysisTool:
    """Test VoiceLeadingAnalysisTool functionality"""

    @pytest.mark.asyncio
    async def test_analyze_voice_leading(self, mock_score_manager, sample_score):
        # Add score to manager for existence check
        mock_score_manager["test_score"] = sample_score
        mock_score_manager.get.return_value = sample_score
        tool = VoiceLeadingAnalysisTool(mock_score_manager)

        result = await tool.execute(score_id="test_score")

        assert "parallel_issues" in result
        assert "voice_crossings" in result
        assert "smoothness_analysis" in result
        assert "overall_score" in result


class TestPatternRecognitionTool:
    """Test PatternRecognitionTool functionality"""

    @pytest.mark.asyncio
    async def test_recognize_patterns(self, mock_score_manager, sample_score):
        # Add score to manager for existence check
        mock_score_manager["test_score"] = sample_score
        mock_score_manager.get.return_value = sample_score
        tool = PatternRecognitionTool(mock_score_manager)

        result = await tool.execute(
            score_id="test_score", pattern_types=["melodic", "rhythmic"]
        )

        assert "melodic_patterns" in result
        assert "rhythmic_patterns" in result
        assert "phrase_structure" in result


class TestHarmonizationTool:
    """Test HarmonizationTool functionality"""

    @pytest.mark.asyncio
    async def test_harmonize_melody(self, mock_score_manager, sample_score):
        # Create a simple melody
        melody = stream.Part()
        for pitch_name in ["C4", "D4", "E4", "F4", "G4"]:
            melody.append(note.Note(pitch_name, quarterLength=1))

        # Add score to manager for existence check
        mock_score_manager["melody"] = melody
        mock_score_manager.get.return_value = melody
        tool = HarmonizationTool(mock_score_manager)

        result = await tool.execute(
            score_id="melody",
            output_id="harmonized",
            style="classical",
        )

        assert result["status"] == "success"
        assert result["harmonized_score_id"] == "harmonized"
        assert "harmonization" in result


class TestCounterpointGeneratorTool:
    """Test CounterpointGeneratorTool functionality"""

    @pytest.mark.asyncio
    async def test_generate_counterpoint(self, mock_score_manager):
        # Create cantus firmus
        cantus = stream.Part()
        for pitch_name in ["C4", "D4", "F4", "E4", "D4", "C4"]:
            cantus.append(note.Note(pitch_name, quarterLength=1))

        # Add score to manager for existence check
        mock_score_manager["cantus"] = cantus
        mock_score_manager.get.return_value = cantus
        tool = CounterpointGeneratorTool(mock_score_manager)

        result = await tool.execute(
            score_id="cantus", species="first"
        )

        assert result["status"] == "success"
        assert result["species"] == "first"
        assert "melodic_analysis" in result


class TestStyleImitationTool:
    """Test StyleImitationTool functionality"""

    @pytest.mark.asyncio
    async def test_imitate_style(self, mock_score_manager, sample_score):
        # Add score to manager for existence check
        mock_score_manager["test_score"] = sample_score
        mock_score_manager.get.return_value = sample_score
        tool = StyleImitationTool(mock_score_manager)

        result = await tool.execute(
            style_source="test_score", generation_length=8
        )

        assert result["status"] == "success"
        assert "generated_score_id" in result
        assert "measures_generated" in result
        assert "musical_features" in result


# Integration test for tool interaction
class TestToolIntegration:
    """Test interaction between multiple tools"""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_import_analyze_export_workflow(
        self, mock_score_manager, sample_score, tmp_path
    ):
        # Import
        import_tool = ImportScoreTool(mock_score_manager)
        await import_tool.execute(
            score_id="workflow_test", source="bach/bwv66.6", source_type="corpus"
        )

        # Analyze
        mock_score_manager["workflow_test"] = sample_score
        mock_score_manager.get.return_value = sample_score
        key_tool = KeyAnalysisTool(mock_score_manager)
        key_result = await key_tool.execute(score_id="workflow_test")
        assert "key" in key_result

        # Export
        export_tool = ExportScoreTool(mock_score_manager)
        export_result = await export_tool.execute(
            score_id="workflow_test",
            format="musicxml",
            output_path=str(tmp_path / "workflow.xml"),
        )
        assert export_result["status"] == "success"
