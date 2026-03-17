"""Tests for PythonAdapter async/sync bridge and all wrapper methods."""

import pytest

from music21_mcp.adapters.python_adapter import (
    Music21Analysis,
    PythonAdapter,
    create_music_analyzer,
    create_sync_analyzer,
)


class TestPythonAdapter:
    def test_init_creates_core_service(self):
        adapter = PythonAdapter()
        assert adapter.core_service is not None

    def test_get_available_tools(self):
        adapter = PythonAdapter()
        tools = adapter.get_available_tools()
        assert isinstance(tools, (list, dict))
        assert len(tools) >= 10

    def test_get_score_count(self):
        adapter = PythonAdapter()
        assert adapter.get_score_count() == 0

    def test_get_status(self):
        adapter = PythonAdapter()
        status = adapter.get_status()
        assert status["status"] == "healthy"
        assert status["adapter"] == "PythonAdapter"
        assert "tools_available" in status
        assert "scores_loaded" in status

    @pytest.mark.asyncio
    async def test_list_scores(self):
        adapter = PythonAdapter()
        result = await adapter.list_scores()
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_import_and_analysis_flow(self):
        """Import a score then exercise all analysis methods on it."""
        adapter = PythonAdapter()
        result = await adapter.import_score("pa_bach", "bach/bwv66.6", "corpus")
        assert result["status"] == "success"

        info = await adapter.get_score_info("pa_bach")
        assert isinstance(info, dict)

        key = await adapter.analyze_key("pa_bach")
        assert isinstance(key, dict)

        chords = await adapter.analyze_chords("pa_bach")
        assert isinstance(chords, dict)

        harmony = await adapter.analyze_harmony("pa_bach", "roman")
        assert isinstance(harmony, dict)

        vl = await adapter.analyze_voice_leading("pa_bach")
        assert isinstance(vl, dict)

        patterns = await adapter.recognize_patterns("pa_bach", "melodic")
        assert isinstance(patterns, dict)

        export = await adapter.export_score("pa_bach", "musicxml")
        assert isinstance(export, dict)

        delete = await adapter.delete_score("pa_bach")
        assert isinstance(delete, dict)

    @pytest.mark.asyncio
    async def test_quick_analysis_success(self):
        adapter = PythonAdapter()
        await adapter.import_score("qa_s", "bach/bwv66.6", "corpus")
        result = await adapter.quick_analysis("qa_s")
        assert result["status"] == "success"
        assert "analysis" in result

    @pytest.mark.asyncio
    async def test_quick_analysis_missing_score(self):
        """Service returns error dicts (doesn't raise), so quick_analysis completes."""
        adapter = PythonAdapter()
        result = await adapter.quick_analysis("nonexistent_score")
        # quick_analysis catches exceptions; service returns error dicts for missing scores
        assert result["score_id"] == "nonexistent_score"
        assert result["status"] in ("success", "error")

    @pytest.mark.asyncio
    async def test_batch_import_mixed(self):
        adapter = PythonAdapter()
        scores = [
            {"score_id": "bi_good", "source": "bach/bwv66.6", "source_type": "corpus"},
            {
                "score_id": "bi_bad",
                "source": "nonexistent/fake",
                "source_type": "corpus",
            },
        ]
        result = await adapter.batch_import(scores)
        assert result["status"] == "completed"
        assert result["total_scores"] == 2
        assert result["successful"] >= 1


class TestMusic21AnalysisSync:
    def test_init(self):
        analyzer = Music21Analysis()
        assert analyzer.adapter is not None

    def test_run_async_no_loop(self):
        """_run_async uses asyncio.run when no event loop is running."""
        analyzer = Music21Analysis()

        async def dummy():
            return 42

        result = analyzer._run_async(dummy())
        assert result == 42

    def test_get_status_sync(self):
        analyzer = Music21Analysis()
        status = analyzer.get_status()
        assert status["status"] == "healthy"

    def test_get_available_tools_sync(self):
        analyzer = Music21Analysis()
        tools = analyzer.get_available_tools()
        assert isinstance(tools, (list, dict))

    def test_get_score_count_sync(self):
        analyzer = Music21Analysis()
        assert analyzer.get_score_count() == 0

    def test_list_scores_sync(self):
        analyzer = Music21Analysis()
        result = analyzer.list_scores()
        assert isinstance(result, dict)

    def test_import_and_analyze_sync(self):
        analyzer = Music21Analysis()
        result = analyzer.import_score("sync_test", "bach/bwv66.6", "corpus")
        assert result["status"] == "success"

        key_result = analyzer.analyze_key("sync_test")
        assert isinstance(key_result, dict)

    def test_batch_import_sync(self):
        analyzer = Music21Analysis()
        result = analyzer.batch_import(
            [{"score_id": "batch1", "source": "bach/bwv66.6"}]
        )
        assert result["status"] == "completed"

    def test_quick_analysis_sync_missing(self):
        analyzer = Music21Analysis()
        result = analyzer.quick_analysis("nonexistent")
        assert result["status"] in ("success", "error")


class TestFactoryFunctions:
    def test_create_music_analyzer(self):
        analyzer = create_music_analyzer()
        assert isinstance(analyzer, PythonAdapter)

    def test_create_sync_analyzer(self):
        analyzer = create_sync_analyzer()
        assert isinstance(analyzer, Music21Analysis)
