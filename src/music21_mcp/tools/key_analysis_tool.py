"""
Key Analysis Tool - Detect musical key with multiple algorithms
"""

import logging
from typing import Any

from music21 import stream

from .base_tool import BaseTool

logger = logging.getLogger(__name__)


class KeyAnalysisTool(BaseTool):
    """Tool for analyzing musical key with confidence scoring"""

    ALGORITHMS = {
        "krumhansl": "Krumhansl-Schmuckler",
        "aarden": "Aarden-Essen",
        "temperley": "Temperley-Kostka-Payne",
        "bellman": "Bellman-Budge",
    }

    async def execute(self, **kwargs: Any) -> dict[str, Any]:
        """
        Analyze the key of a score using various algorithms

        Args:
            **kwargs: Keyword arguments including:
                score_id: ID of the score to analyze
                algorithm: Which algorithm to use ('all', 'krumhansl', 'aarden', 'temperley', 'bellman')
        """
        # Extract parameters from kwargs
        score_id = kwargs.get("score_id", "")
        algorithm = kwargs.get("algorithm", "all")
        # Validate inputs
        error = self.validate_inputs(**kwargs)
        if error:
            return self.create_error_response(error)

        with self.error_handling(f"Key analysis for '{score_id}'"):
            score = self.get_score(score_id)

            self.report_progress(0.1, "Preparing score for analysis")

            # Handle different score types
            if isinstance(score, stream.Part):
                analysis_score = score
            else:
                # Use first part or flatten
                if hasattr(score, "parts") and len(score.parts) > 0:
                    analysis_score = score.parts[0]
                else:
                    analysis_score = score.flatten()

            if algorithm == "all":
                # Run all algorithms and find consensus
                results = await self._analyze_with_all_algorithms(analysis_score)
                key, confidence, alternatives = self._find_consensus(results)

                return self.create_success_response(
                    key=str(key),
                    confidence=confidence,
                    alternatives=alternatives,
                    algorithm_results=results,
                )
            # Run specific algorithm
            self.report_progress(
                0.3,
                f"Running {self.ALGORITHMS.get(algorithm, algorithm)} algorithm",
            )

            result = await self._analyze_with_algorithm(analysis_score, algorithm)

            if result is None:
                return self.create_error_response(f"Failed to analyze with {algorithm}")

            key = result.tonic.name + (" major" if result.mode == "major" else " minor")

            # Get alternatives
            alternatives = []
            for alt in result.alternateInterpretations[:3]:
                alt_key = alt.tonic.name + (
                    " major" if alt.mode == "major" else " minor"
                )
                alternatives.append(
                    {"key": alt_key, "confidence": alt.correlationCoefficient}
                )

            self.report_progress(1.0, "Analysis complete")

            return self.create_success_response(
                key=key,
                confidence=result.correlationCoefficient,
                alternatives=alternatives,
                algorithm=self.ALGORITHMS.get(algorithm, algorithm),
            )

    def validate_inputs(self, **kwargs: Any) -> str | None:
        """Validate input parameters"""
        score_id = kwargs.get("score_id", "")
        algorithm = kwargs.get("algorithm", "all")

        error = self.check_score_exists(score_id)
        if error:
            return error

        valid_algorithms = ["all"] + list(self.ALGORITHMS.keys())
        if algorithm not in valid_algorithms:
            return f"Invalid algorithm: {algorithm}. Valid options: {', '.join(valid_algorithms)}"

        return None

    async def _analyze_with_all_algorithms(
        self, score: stream.Stream
    ) -> dict[str, Any]:
        """Run all key detection algorithms"""
        results = {}

        algorithms = list(self.ALGORITHMS.keys())
        for i, alg in enumerate(algorithms):
            self.report_progress(
                0.2 + (0.6 * i / len(algorithms)), f"Running {self.ALGORITHMS[alg]}"
            )

            try:
                result = await self._analyze_with_algorithm(score, alg)
                if result:
                    key = result.tonic.name + (
                        " major" if result.mode == "major" else " minor"
                    )
                    results[alg] = {
                        "key": key,
                        "confidence": result.correlationCoefficient,
                    }
            except Exception as e:
                logger.warning(f"Algorithm {alg} failed: {e}")

        return results

    async def _analyze_with_algorithm(self, score: stream.Stream, algorithm: str) -> Any:
        """Run a specific key detection algorithm asynchronously"""
        def _run_analysis():
            try:
                if algorithm == "krumhansl":
                    return score.analyze("key.krumhanslschmuckler")
                if algorithm == "aarden":
                    return score.analyze("key.aardenessen")
                if algorithm == "temperley":
                    return score.analyze("key.temperleykostkapayne")
                if algorithm == "bellman":
                    return score.analyze("key.bellmanbudge")
                return score.analyze("key")
            except Exception as e:
                logger.error(f"Key analysis with {algorithm} failed: {e}")
                return None

        return await self.run_music21_operation(_run_analysis)

    def _find_consensus(self, results: dict[str, Any]) -> tuple:
        """Find consensus key from multiple algorithms"""
        if not results:
            return "C major", 0.0, []

        # Count votes for each key
        key_votes = {}
        for alg, data in results.items():
            key = data["key"]
            confidence = data["confidence"]

            if key not in key_votes:
                key_votes[key] = {"votes": 0, "total_confidence": 0}

            key_votes[key]["votes"] += 1
            key_votes[key]["total_confidence"] += confidence

        # Sort by votes, then by average confidence
        sorted_keys = sorted(
            key_votes.items(),
            key=lambda x: (
                x[1]["votes"],
                x[1]["total_confidence"] / x[1]["votes"] if x[1]["votes"] > 0 else 0
            ),
            reverse=True,
        )

        # Best key
        best_key = sorted_keys[0][0]
        best_data = sorted_keys[0][1]
        avg_confidence = (
            best_data["total_confidence"] / best_data["votes"]
            if best_data["votes"] > 0
            else 0.0
        )

        # Adjust confidence based on consensus
        consensus_factor = best_data["votes"] / len(results) if len(results) > 0 else 0
        final_confidence = avg_confidence * (0.5 + 0.5 * consensus_factor)

        # Alternatives
        alternatives = []
        for key, data in sorted_keys[1:4]:  # Top 3 alternatives
            alternatives.append(
                {
                    "key": key,
                    "confidence": (
                        data["total_confidence"] / data["votes"]
                        if data["votes"] > 0
                        else 0.0
                    ),
                    "votes": data["votes"],
                }
            )

        return best_key, final_confidence, alternatives

