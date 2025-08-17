#!/usr/bin/env python3
"""
Python Library Adapter
Direct programmatic access to core music analysis service

Provides clean Python API without any protocol overhead.
Ideal for embedding in other Python applications or notebooks.
"""

from typing import Any

from ..services import MusicAnalysisService


class PythonAdapter:
    """
    Python library adapter for music analysis service

    Provides direct programmatic access to core music analysis functionality.
    No protocols, no overhead - just clean Python API.
    """

    def __init__(self):
        """Initialize Python adapter with core service"""
        self.core_service = MusicAnalysisService()

    # === Score Management ===

    async def import_score(
        self, score_id: str, source: str, source_type: str = "corpus"
    ) -> dict[str, Any]:
        """Import a score from various sources

        Args:
            score_id: Unique identifier for the score
            source: Source path or identifier
            source_type: Type of source ("corpus", "file", "url")

        Returns:
            Result dict with status and data
        """
        return await self.core_service.import_score(score_id, source, source_type)

    async def list_scores(self) -> dict[str, Any]:
        """List all imported scores

        Returns:
            Dict containing list of score IDs and metadata
        """
        return await self.core_service.list_scores()

    async def get_score_info(self, score_id: str) -> dict[str, Any]:
        """Get detailed information about a score

        Args:
            score_id: The score identifier

        Returns:
            Dict with comprehensive score information
        """
        return await self.core_service.get_score_info(score_id)

    async def export_score(
        self, score_id: str, format: str = "musicxml"
    ) -> dict[str, Any]:
        """Export a score to various formats

        Args:
            score_id: The score identifier
            format: Export format ("musicxml", "midi", "ly", etc.)

        Returns:
            Dict with export result and file path/content
        """
        return await self.core_service.export_score(score_id, format)

    async def delete_score(self, score_id: str) -> dict[str, Any]:
        """Delete a score from storage

        Args:
            score_id: The score identifier

        Returns:
            Dict confirming deletion
        """
        return await self.core_service.delete_score(score_id)

    # === Analysis Methods ===

    async def analyze_key(self, score_id: str) -> dict[str, Any]:
        """Analyze the key signature of a score

        Args:
            score_id: The score identifier

        Returns:
            Dict with key analysis results
        """
        return await self.core_service.analyze_key(score_id)

    async def analyze_chords(self, score_id: str) -> dict[str, Any]:
        """Analyze chord progressions in a score

        Args:
            score_id: The score identifier

        Returns:
            Dict with chord analysis results
        """
        return await self.core_service.analyze_chords(score_id)

    async def analyze_harmony(
        self, score_id: str, analysis_type: str = "roman"
    ) -> dict[str, Any]:
        """Perform harmony analysis

        Args:
            score_id: The score identifier
            analysis_type: Type of analysis ("roman", "functional", "berklee")

        Returns:
            Dict with harmony analysis results
        """
        return await self.core_service.analyze_harmony(score_id, analysis_type)

    async def analyze_voice_leading(self, score_id: str) -> dict[str, Any]:
        """Analyze voice leading patterns in a score

        Args:
            score_id: The score identifier

        Returns:
            Dict with voice leading analysis results
        """
        return await self.core_service.analyze_voice_leading(score_id)

    async def recognize_patterns(
        self, score_id: str, pattern_type: str = "melodic"
    ) -> dict[str, Any]:
        """Recognize musical patterns in a score

        Args:
            score_id: The score identifier
            pattern_type: Type of patterns to find ("melodic", "rhythmic", "harmonic")

        Returns:
            Dict with pattern recognition results
        """
        return await self.core_service.recognize_patterns(score_id, pattern_type)

    # === Utility Methods ===

    def get_available_tools(self) -> list[str]:
        """Get list of all available analysis tools

        Returns:
            List of tool names
        """
        return self.core_service.get_available_tools()

    def get_score_count(self) -> int:
        """Get number of imported scores

        Returns:
            Number of scores currently loaded
        """
        return self.core_service.get_score_count()

    def get_status(self) -> dict[str, Any]:
        """Get service status information

        Returns:
            Dict with service health and statistics
        """
        return {
            "service": "Music21 Analysis Service",
            "adapter": "PythonAdapter",
            "tools_available": len(self.get_available_tools()),
            "scores_loaded": self.get_score_count(),
            "status": "healthy",
        }

    # === Convenience Methods ===

    async def quick_analysis(self, score_id: str) -> dict[str, Any]:
        """Perform a quick comprehensive analysis of a score

        Args:
            score_id: The score identifier

        Returns:
            Dict with combined analysis results (key, harmony, voice leading)
        """
        results = {}

        try:
            # Get basic info
            results["info"] = await self.get_score_info(score_id)

            # Key analysis
            results["key"] = await self.analyze_key(score_id)

            # Harmony analysis
            results["harmony"] = await self.analyze_harmony(score_id, "roman")

            # Voice leading
            results["voice_leading"] = await self.analyze_voice_leading(score_id)

            return {
                "status": "success",
                "score_id": score_id,
                "analysis": results,
                "message": "Quick analysis completed",
            }

        except Exception as e:
            return {
                "status": "error",
                "score_id": score_id,
                "error": str(e),
                "message": "Quick analysis failed",
            }

    async def batch_import(self, scores: list[dict[str, str]]) -> dict[str, Any]:
        """Import multiple scores in batch

        Args:
            scores: List of dicts with 'score_id', 'source', 'source_type'

        Returns:
            Dict with batch import results
        """
        results = []

        for score_data in scores:
            try:
                result = await self.import_score(
                    score_data["score_id"],
                    score_data["source"],
                    score_data.get("source_type", "corpus"),
                )
                results.append(
                    {
                        "score_id": score_data["score_id"],
                        "status": "success",
                        "result": result,
                    }
                )
            except Exception as e:
                results.append(
                    {
                        "score_id": score_data["score_id"],
                        "status": "error",
                        "error": str(e),
                    }
                )

        successful = len([r for r in results if r["status"] == "success"])

        return {
            "status": "completed",
            "total_scores": len(scores),
            "successful": successful,
            "failed": len(scores) - successful,
            "results": results,
        }


# === Synchronous Wrapper ===
# For easier use in notebooks and scripts that don't want async


class Music21Analysis:
    """
    Synchronous wrapper for PythonAdapter

    Provides blocking interface for easier use in notebooks and simple scripts.
    """

    def __init__(self):
        """Initialize synchronous wrapper"""
        self.adapter = PythonAdapter()

    def _run_async(self, coro):
        """Run async method synchronously"""
        import asyncio

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(coro)

    # === Synchronous API ===

    def import_score(
        self, score_id: str, source: str, source_type: str = "corpus"
    ) -> dict[str, Any]:
        """Import a score (synchronous)"""
        return self._run_async(self.adapter.import_score(score_id, source, source_type))

    def list_scores(self) -> dict[str, Any]:
        """List all imported scores (synchronous)"""
        return self._run_async(self.adapter.list_scores())

    def get_score_info(self, score_id: str) -> dict[str, Any]:
        """Get score information (synchronous)"""
        return self._run_async(self.adapter.get_score_info(score_id))

    def analyze_key(self, score_id: str) -> dict[str, Any]:
        """Analyze key signature (synchronous)"""
        return self._run_async(self.adapter.analyze_key(score_id))

    def analyze_harmony(
        self, score_id: str, analysis_type: str = "roman"
    ) -> dict[str, Any]:
        """Analyze harmony (synchronous)"""
        return self._run_async(self.adapter.analyze_harmony(score_id, analysis_type))

    def analyze_chords(self, score_id: str) -> dict[str, Any]:
        """Analyze chord progressions (synchronous)"""
        return self._run_async(self.adapter.analyze_chords(score_id))

    def analyze_voice_leading(self, score_id: str) -> dict[str, Any]:
        """Analyze voice leading (synchronous)"""
        return self._run_async(self.adapter.analyze_voice_leading(score_id))

    def recognize_patterns(
        self, score_id: str, pattern_type: str = "melodic"
    ) -> dict[str, Any]:
        """Recognize musical patterns (synchronous)"""
        return self._run_async(self.adapter.recognize_patterns(score_id, pattern_type))

    def export_score(
        self, score_id: str, format: str = "musicxml"
    ) -> dict[str, Any]:
        """Export a score to various formats (synchronous)"""
        return self._run_async(self.adapter.export_score(score_id, format))

    def delete_score(self, score_id: str) -> dict[str, Any]:
        """Delete a score from storage (synchronous)"""
        return self._run_async(self.adapter.delete_score(score_id))

    def batch_import(self, scores: list[dict[str, str]]) -> dict[str, Any]:
        """Import multiple scores in batch (synchronous)"""
        return self._run_async(self.adapter.batch_import(scores))

    def quick_analysis(self, score_id: str) -> dict[str, Any]:
        """Quick comprehensive analysis (synchronous)"""
        return self._run_async(self.adapter.quick_analysis(score_id))

    def get_available_tools(self) -> list[str]:
        """Get list of all available analysis tools (synchronous)"""
        return self.adapter.get_available_tools()

    def get_score_count(self) -> int:
        """Get number of imported scores (synchronous)"""
        return self.adapter.get_score_count()

    def get_status(self) -> dict[str, Any]:
        """Get service status (synchronous)"""
        return self.adapter.get_status()


# === Factory Functions ===


def create_music_analyzer() -> PythonAdapter:
    """Create a new music analysis adapter instance"""
    return PythonAdapter()


def create_sync_analyzer() -> Music21Analysis:
    """Create a new synchronous music analysis wrapper"""
    return Music21Analysis()


# === Example Usage ===

if __name__ == "__main__":
    import asyncio

    async def demo():
        """Demonstrate Python adapter usage"""

        # Create adapter
        analyzer = PythonAdapter()

        # Show status
        analyzer.get_status()

        # Import a score
        await analyzer.import_score("demo_chorale", "bach/bwv66.6", "corpus")

        # Quick analysis
        await analyzer.quick_analysis("demo_chorale")

        # List tools
        for tool in analyzer.get_available_tools():
            pass

    # Run demo
    asyncio.run(demo())
