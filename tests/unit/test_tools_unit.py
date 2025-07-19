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
    manager = Mock()
    manager.add_score = AsyncMock()
    manager.get_score = AsyncMock()
    manager.list_scores = AsyncMock()
    manager.remove_score = AsyncMock()
    return manager


@pytest.fixture
def sample_score():
    """Create a sample score for testing"""
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
        assert mock_score_manager.add_score.called

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
        assert mock_score_manager.add_score.called

    @pytest.mark.asyncio
    async def test_import_file_not_found(self, mock_score_manager):
        tool = ImportScoreTool(mock_score_manager)

        result = await tool.execute(
            score_id="test_score", source="/nonexistent/file.xml", source_type="file"
        )

        assert "error" in result
        assert "not found" in result["error"].lower()


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
    async def test_list_multiple_scores(self, mock_score_manager):
        mock_scores = [
            {"id": "score1", "title": "Score 1"},
            {"id": "score2", "title": "Score 2"},
        ]
        mock_score_manager.list_scores.return_value = mock_scores
        tool = ListScoresTool(mock_score_manager)

        result = await tool.execute()

        assert result["count"] == 2
        assert result["scores"] == mock_scores


class TestKeyAnalysisTool:
    """Test KeyAnalysisTool functionality"""

    @pytest.mark.asyncio
    async def test_analyze_key_krumhansl(self, mock_score_manager, sample_score):
        mock_score_manager.get_score.return_value = sample_score
        tool = KeyAnalysisTool(mock_score_manager)

        result = await tool.execute(score_id="test_score", algorithm="krumhansl")

        assert "key" in result
        assert "tonic" in result
        assert "mode" in result
        assert "confidence" in result
        assert result["algorithm"] == "krumhansl"

    @pytest.mark.asyncio
    async def test_analyze_key_score_not_found(self, mock_score_manager):
        mock_score_manager.get_score.return_value = None
        tool = KeyAnalysisTool(mock_score_manager)

        result = await tool.execute(score_id="nonexistent")

        assert "error" in result
        assert "not found" in result["error"]


class TestChordAnalysisTool:
    """Test ChordAnalysisTool functionality"""

    @pytest.mark.asyncio
    async def test_analyze_chords(self, mock_score_manager, sample_score):
        # Add some chords to the score
        for m in sample_score.parts[0].getElementsByClass("Measure"):
            m.clear()
            c = chord.Chord(["C4", "E4", "G4"])
            m.append(c)

        mock_score_manager.get_score.return_value = sample_score
        tool = ChordAnalysisTool(mock_score_manager)

        result = await tool.execute(score_id="test_score")

        assert "chords" in result
        assert "chord_types" in result
        assert "total_chords" in result
        assert result["total_chords"] > 0


class TestScoreInfoTool:
    """Test ScoreInfoTool functionality"""

    @pytest.mark.asyncio
    async def test_get_score_info(self, mock_score_manager, sample_score):
        mock_score_manager.get_score.return_value = sample_score
        tool = ScoreInfoTool(mock_score_manager)

        result = await tool.execute(score_id="test_score")

        assert result["title"] == "Test Score"
        assert result["composer"] == "Test Composer"
        assert result["parts"] == 1
        assert result["measures"] == 4
        assert "duration" in result
        assert "time_signatures" in result


class TestExportScoreTool:
    """Test ExportScoreTool functionality"""

    @pytest.mark.asyncio
    async def test_export_musicxml(self, mock_score_manager, sample_score, tmp_path):
        mock_score_manager.get_score.return_value = sample_score
        tool = ExportScoreTool(mock_score_manager)

        output_path = str(tmp_path / "test.xml")
        result = await tool.execute(
            score_id="test_score", format="musicxml", output_path=output_path
        )

        assert result["status"] == "success"
        assert result["format"] == "musicxml"
        assert result["output_path"] == output_path

    @pytest.mark.asyncio
    async def test_export_midi(self, mock_score_manager, sample_score, tmp_path):
        mock_score_manager.get_score.return_value = sample_score
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
    async def test_delete_existing_score(self, mock_score_manager):
        mock_score_manager.remove_score.return_value = True
        tool = DeleteScoreTool(mock_score_manager)

        result = await tool.execute(score_id="test_score")

        assert result["status"] == "success"
        assert "deleted" in result["message"]

    @pytest.mark.asyncio
    async def test_delete_nonexistent_score(self, mock_score_manager):
        mock_score_manager.remove_score.return_value = False
        tool = DeleteScoreTool(mock_score_manager)

        result = await tool.execute(score_id="nonexistent")

        assert result["status"] == "not_found"


class TestHarmonyAnalysisTool:
    """Test HarmonyAnalysisTool functionality"""

    @pytest.mark.asyncio
    async def test_analyze_harmony(self, mock_score_manager, sample_score):
        # Add chords for harmony analysis
        for m in sample_score.parts[0].getElementsByClass("Measure"):
            m.clear()
            m.append(chord.Chord(["C4", "E4", "G4"]))
            m.append(chord.Chord(["G4", "B4", "D5"]))

        mock_score_manager.get_score.return_value = sample_score
        tool = HarmonyAnalysisTool(mock_score_manager)

        result = await tool.execute(score_id="test_score")

        assert "harmonic_rhythm" in result
        assert "progression_analysis" in result
        assert "cadences" in result
        assert "key_context" in result


class TestVoiceLeadingAnalysisTool:
    """Test VoiceLeadingAnalysisTool functionality"""

    @pytest.mark.asyncio
    async def test_analyze_voice_leading(self, mock_score_manager, sample_score):
        mock_score_manager.get_score.return_value = sample_score
        tool = VoiceLeadingAnalysisTool(mock_score_manager)

        result = await tool.execute(score_id="test_score")

        assert "voice_count" in result
        assert "parallel_intervals" in result
        assert "voice_crossings" in result
        assert "smooth_voice_leading_score" in result


class TestPatternRecognitionTool:
    """Test PatternRecognitionTool functionality"""

    @pytest.mark.asyncio
    async def test_recognize_patterns(self, mock_score_manager, sample_score):
        mock_score_manager.get_score.return_value = sample_score
        tool = PatternRecognitionTool(mock_score_manager)

        result = await tool.execute(
            score_id="test_score", pattern_types=["melodic", "rhythmic"]
        )

        assert "patterns" in result
        assert "summary" in result
        assert isinstance(result["patterns"], dict)


class TestHarmonizationTool:
    """Test HarmonizationTool functionality"""

    @pytest.mark.asyncio
    async def test_harmonize_melody(self, mock_score_manager, sample_score):
        # Create a simple melody
        melody = stream.Part()
        for pitch_name in ["C4", "D4", "E4", "F4", "G4"]:
            melody.append(note.Note(pitch_name, quarterLength=1))

        mock_score_manager.get_score.return_value = melody
        tool = HarmonizationTool(mock_score_manager)

        result = await tool.execute(
            melody_score_id="test_melody",
            output_score_id="harmonized",
            style="bach_chorale",
        )

        assert result["status"] == "success"
        assert result["output_score_id"] == "harmonized"
        assert "parts_created" in result


class TestCounterpointGeneratorTool:
    """Test CounterpointGeneratorTool functionality"""

    @pytest.mark.asyncio
    async def test_generate_counterpoint(self, mock_score_manager):
        # Create cantus firmus
        cantus = stream.Part()
        for pitch_name in ["C4", "D4", "F4", "E4", "D4", "C4"]:
            cantus.append(note.Note(pitch_name, quarterLength=1))

        mock_score_manager.get_score.return_value = cantus
        tool = CounterpointGeneratorTool(mock_score_manager)

        result = await tool.execute(
            cantus_firmus_id="cantus", output_score_id="counterpoint", species=1
        )

        assert result["status"] == "success"
        assert result["species"] == 1
        assert "voices_created" in result


class TestStyleImitationTool:
    """Test StyleImitationTool functionality"""

    @pytest.mark.asyncio
    async def test_imitate_style(self, mock_score_manager, sample_score):
        mock_score_manager.get_score.return_value = sample_score
        tool = StyleImitationTool(mock_score_manager)

        result = await tool.execute(
            source_score_id="test_score", output_score_id="imitation", measures=8
        )

        assert result["status"] == "success"
        assert result["output_score_id"] == "imitation"
        assert result["measures_generated"] == 8
        assert "style_characteristics" in result


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
        mock_score_manager.get_score.return_value = sample_score
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
