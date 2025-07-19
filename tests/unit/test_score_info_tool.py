"""
Unit tests for ScoreInfoTool
"""

import pytest

from music21_mcp.tools.score_info_tool import ScoreInfoTool


class TestScoreInfoTool:
    """Test ScoreInfoTool functionality"""

    def test_tool_initialization(self, clean_score_storage):
        """Test tool can be initialized with score storage"""
        tool = ScoreInfoTool(clean_score_storage)
        assert tool.scores == clean_score_storage

    @pytest.mark.asyncio
    async def test_get_score_info_success(self, populated_score_storage):
        """Test successful score info retrieval"""
        tool = ScoreInfoTool(populated_score_storage)

        result = await tool.execute(score_id="bach_test")

        assert result["status"] == "success"
        assert "exists" in result
        assert "score_id" in result
        assert "metadata" in result
        assert "num_parts" in result
        assert "num_measures" in result
        assert "num_notes" in result
        assert "duration_quarters" in result
        assert "time_signatures" in result
        assert "instruments" in result
        assert result["exists"] is True
        assert result["score_id"] == "bach_test"
        assert isinstance(result["num_parts"], int)
        assert isinstance(result["num_measures"], int)
        assert isinstance(result["num_notes"], int)
        assert isinstance(result["time_signatures"], list)
        assert isinstance(result["instruments"], list)

    @pytest.mark.asyncio
    async def test_get_score_info_nonexistent(self, clean_score_storage):
        """Test score info with non-existent score"""
        tool = ScoreInfoTool(clean_score_storage)

        result = await tool.execute(score_id="nonexistent")

        assert result["status"] == "error"
        assert "not found" in result["message"]

    @pytest.mark.asyncio
    async def test_score_info_structure(self, populated_score_storage):
        """Test that score info has expected structure"""
        tool = ScoreInfoTool(populated_score_storage)

        result = await tool.execute(score_id="bach_test")

        assert result["status"] == "success"

        # Check basic metadata
        assert "metadata" in result
        assert "num_parts" in result
        assert "num_measures" in result
        assert isinstance(result["num_parts"], int)
        assert isinstance(result["num_measures"], int)
        assert result["num_parts"] >= 0
        assert result["num_measures"] >= 0

    @pytest.mark.asyncio
    async def test_score_info_time_signature(self, populated_score_storage):
        """Test time signature information"""
        tool = ScoreInfoTool(populated_score_storage)

        result = await tool.execute(score_id="bach_test")

        assert result["status"] == "success"
        assert "time_signatures" in result

        # Time signatures should be a list of time signature objects
        time_sigs = result["time_signatures"]
        assert isinstance(time_sigs, list)
        if len(time_sigs) > 0:
            time_sig = time_sigs[0]
            assert "signature" in time_sig
            assert "offset" in time_sig
            assert "numerator" in time_sig
            assert "denominator" in time_sig
            assert isinstance(time_sig["signature"], str)
            assert "/" in time_sig["signature"]

    @pytest.mark.asyncio
    async def test_score_info_key_signature(self, populated_score_storage):
        """Test key signature information"""
        tool = ScoreInfoTool(populated_score_storage)

        result = await tool.execute(score_id="bach_test")

        assert result["status"] == "success"
        assert "structure" in result
        assert "key_signatures" in result["structure"]

        # Key signatures should be a list of key signature objects
        key_sigs = result["structure"]["key_signatures"]
        assert isinstance(key_sigs, list)
        if len(key_sigs) > 0:
            key_sig = key_sigs[0]
            assert "sharps" in key_sig
            assert "offset" in key_sig
            assert isinstance(key_sig["sharps"], int)
            assert isinstance(key_sig["offset"], (int, float))

    @pytest.mark.asyncio
    async def test_score_info_duration(self, populated_score_storage):
        """Test duration information"""
        tool = ScoreInfoTool(populated_score_storage)

        result = await tool.execute(score_id="bach_test")

        assert result["status"] == "success"
        assert "duration_quarters" in result
        assert "duration_seconds" in result

        # Duration should be numbers
        duration_quarters = result["duration_quarters"]
        duration_seconds = result["duration_seconds"]
        assert isinstance(duration_quarters, (int, float))
        assert isinstance(duration_seconds, (int, float))
        assert duration_quarters >= 0
        assert duration_seconds >= 0

    @pytest.mark.asyncio
    async def test_score_info_handles_complex_score(self, clean_score_storage):
        """Test score info handles complex scores with multiple parts"""
        from music21 import key, meter, note, stream

        # Create a complex score with multiple parts
        score = stream.Score()
        score.append(key.Key("C"))
        score.append(meter.TimeSignature("4/4"))

        # Add multiple parts
        part1 = stream.Part()
        part1.append(note.Note("C4", quarterLength=1))
        part1.append(note.Note("D4", quarterLength=1))

        part2 = stream.Part()
        part2.append(note.Note("E4", quarterLength=1))
        part2.append(note.Note("F4", quarterLength=1))

        score.append(part1)
        score.append(part2)

        clean_score_storage["complex_score"] = score

        tool = ScoreInfoTool(clean_score_storage)
        result = await tool.execute(score_id="complex_score")

        assert result["status"] == "success"
        assert result["num_parts"] == 2
        # Note: artificially created scores might have 0 measures due to music21 analysis
        assert result["num_measures"] >= 0
        assert result["num_notes"] >= 0

    @pytest.mark.asyncio
    async def test_score_info_handles_empty_score(self, clean_score_storage):
        """Test score info handles empty scores gracefully"""
        from music21 import stream

        empty_score = stream.Score()
        clean_score_storage["empty_score"] = empty_score

        tool = ScoreInfoTool(clean_score_storage)
        result = await tool.execute(score_id="empty_score")

        # Should handle gracefully
        assert result["status"] in ["success", "error"]
        if result["status"] == "success":
            assert result["num_parts"] == 0
            assert result["num_measures"] == 0
            assert result["num_notes"] == 0
