"""
Advanced Harmony Analysis Tool - Comprehensive harmonic analysis
Integrates the harmonic_analyzer module for deep harmonic insights
"""

import logging
from typing import Any, Dict, List, Optional

from music21 import roman, stream

from ..core.harmonic_analyzer import HarmonicAnalyzer
from .base_tool import BaseTool

logger = logging.getLogger(__name__)


class HarmonyAnalysisTool(BaseTool):
    """
    Advanced harmonic analysis tool providing:
    1. Roman numeral analysis with key context
    2. Chord function identification (tonic, dominant, etc.)
    3. Harmonic rhythm analysis
    4. Modulation detection and pivot chord identification
    5. Non-chord tone identification
    """

    def __init__(self, score_manager: Dict[str, Any]):
        super().__init__(score_manager)
        self.analyzer = HarmonicAnalyzer()

    async def execute(self, **kwargs: Any) -> Dict[str, Any]:
        """
        Perform advanced harmonic analysis

        Args:
            **kwargs: Keyword arguments including:
                score_id: ID of the score to analyze
                analysis_type: Type of analysis ('comprehensive', 'functional', 'roman', 'cadences')
                include_voice_leading: Include voice leading analysis
                include_jazz_analysis: Include jazz harmony analysis
                include_modulations: Include modulation detection
        """
        # Extract parameters from kwargs
        score_id = kwargs.get("score_id", "")
        analysis_type = kwargs.get("analysis_type", "comprehensive")
        include_voice_leading = kwargs.get("include_voice_leading", True)
        include_jazz_analysis = kwargs.get("include_jazz_analysis", False)
        include_modulations = kwargs.get("include_modulations", True)
        # Validate inputs
        error = self.validate_inputs(**kwargs)
        if error:
            return self.create_error_response(error)

        with self.error_handling(f"Harmonic analysis for '{score_id}'"):
            score = self.get_score(score_id)

            self.report_progress(0.1, "Preparing score for harmonic analysis")

            # Perform requested analysis type
            if analysis_type == "comprehensive":
                return await self._comprehensive_analysis(
                    score,
                    include_voice_leading,
                    include_jazz_analysis,
                    include_modulations,
                )
            elif analysis_type == "functional":
                return await self._functional_harmony_analysis(score)
            elif analysis_type == "roman":
                return await self._roman_numeral_analysis(score)
            elif analysis_type == "cadences":
                return await self._cadence_analysis(score)
            else:
                return self.create_error_response(
                    f"Unknown analysis type: {analysis_type}"
                )

    def validate_inputs(self, **kwargs: Any) -> Optional[str]:
        """Validate input parameters"""
        score_id = kwargs.get("score_id", "")
        analysis_type = kwargs.get("analysis_type", "comprehensive")

        error = self.check_score_exists(score_id)
        if error:
            return error

        valid_types = ["comprehensive", "functional", "roman", "cadences"]
        if analysis_type not in valid_types:
            return f"Invalid analysis type. Choose from: {', '.join(valid_types)}"

        return None

    async def _comprehensive_analysis(
        self,
        score: stream.Score,
        include_voice_leading: bool,
        include_jazz: bool,
        include_modulations: bool,
    ) -> Dict[str, Any]:
        """Perform comprehensive harmonic analysis"""
        results = {}

        try:
            # Functional harmony analysis
            self.report_progress(0.2, "Analyzing functional harmony")
            functional_result = await self.analyzer.analyze_functional_harmony(score)
            results["functional_harmony"] = self._serialize_functional_analysis(
                functional_result
            )

            # Voice leading analysis
            if include_voice_leading:
                self.report_progress(0.4, "Analyzing voice leading")
                voice_result = await self.analyzer.analyze_voice_leading(score)
                results["voice_leading"] = self._serialize_voice_leading(voice_result)

            # Jazz harmony analysis
            if include_jazz:
                self.report_progress(0.6, "Analyzing jazz harmony")
                jazz_result = await self.analyzer.analyze_jazz_harmony(score)
                results["jazz_harmony"] = self._serialize_jazz_analysis(jazz_result)

            # Modulation detection
            if include_modulations:
                self.report_progress(0.8, "Detecting modulations")
                modulation_result = await self.analyzer.detect_modulations(score)
                results["modulations"] = self._serialize_modulation_analysis(
                    modulation_result
                )

            # Harmonic sequences
            self.report_progress(0.9, "Detecting harmonic sequences")
            sequence_result = await self.analyzer.detect_harmonic_sequences(score)
            results["sequences"] = sequence_result

            self.report_progress(1.0, "Analysis complete")

            # Add summary
            results["summary"] = self._generate_harmonic_summary(results)

            return self.create_success_response(**results)

        except Exception as e:
            logger.error(f"Comprehensive analysis failed: {e}")
            return self.create_error_response(f"Analysis failed: {str(e)}")

    async def _functional_harmony_analysis(self, score: stream.Score) -> Dict[str, Any]:
        """Analyze functional harmony only"""
        self.report_progress(0.3, "Analyzing harmonic functions")

        try:
            result = await self.analyzer.analyze_functional_harmony(score)

            self.report_progress(0.8, "Processing results")

            return self.create_success_response(
                roman_numerals=result.roman_numerals,
                functions=[f.value for f in result.functions],
                cadences=result.cadences,
                phrase_model=result.phrase_model,
                tonal_strength=result.tonal_strength,
                tonic_prolongations=result.tonic_prolongations,
                predominant_chords=result.predominant_chords,
                dominant_preparations=result.dominant_preparations,
                deceptive_resolutions=result.deceptive_resolutions,
            )

        except Exception as e:
            logger.error(f"Functional harmony analysis failed: {e}")
            return self.create_error_response(f"Analysis failed: {str(e)}")

    async def _roman_numeral_analysis(self, score: stream.Score) -> Dict[str, Any]:
        """Perform Roman numeral analysis"""
        self.report_progress(0.2, "Detecting key")

        try:
            # Detect key
            key_obj = score.analyze("key")

            self.report_progress(0.4, "Extracting chords")

            # Get chords
            chords = score.chordify()
            chord_list = list(chords.flatten().getElementsByClass("Chord"))

            self.report_progress(0.6, "Analyzing Roman numerals")

            # Analyze each chord
            roman_numerals = []
            for i, ch in enumerate(chord_list):
                if i % 10 == 0:
                    self.report_progress(
                        0.6 + (0.3 * i / len(chord_list)),
                        f"Processing chord {i+1}/{len(chord_list)}",
                    )

                try:
                    rn = roman.romanNumeralFromChord(ch, key_obj)
                    roman_numerals.append(
                        {
                            "roman": str(rn.romanNumeral),
                            "key": str(rn.key),
                            "scale_degree": rn.scaleDegree,
                            "quality": rn.quality,
                            "inversion": rn.inversion(),
                            "offset": float(ch.offset),
                            "duration": float(ch.duration.quarterLength),
                            "confidence": (
                                rn.commonName != "Chord"
                                if hasattr(rn, "commonName")
                                else True
                            ),
                        }
                    )
                except Exception as e:
                    logger.warning(
                        f"Could not analyze chord at offset {ch.offset}: {e}"
                    )
                    roman_numerals.append(
                        {
                            "roman": "?",
                            "offset": float(ch.offset),
                            "duration": float(ch.duration.quarterLength),
                            "error": str(e),
                        }
                    )

            self.report_progress(1.0, "Analysis complete")

            # Generate progression summary
            progression = [rn["roman"] for rn in roman_numerals if rn["roman"] != "?"]
            common_progressions = self._identify_common_progressions(progression)

            return self.create_success_response(
                key=str(key_obj),
                total_chords=len(chord_list),
                roman_numerals=roman_numerals[:100],  # Limit output
                progression_summary=self._summarize_progression(progression),
                common_progressions=common_progressions,
            )

        except Exception as e:
            logger.error(f"Roman numeral analysis failed: {e}")
            return self.create_error_response(f"Analysis failed: {str(e)}")

    async def _cadence_analysis(self, score: stream.Score) -> Dict[str, Any]:
        """Analyze cadences in the score"""
        self.report_progress(0.3, "Identifying cadences")

        try:
            # Use analyzer's cadence detection
            functional_result = await self.analyzer.analyze_functional_harmony(score)
            cadences = functional_result.cadences

            self.report_progress(0.8, "Categorizing cadences")

            # Categorize cadences
            cadence_types = {
                "authentic": [],
                "plagal": [],
                "half": [],
                "deceptive": [],
                "other": [],
            }

            for cadence in cadences:
                cadence_type = cadence.get("type", "other").lower()
                if "authentic" in cadence_type:
                    cadence_types["authentic"].append(cadence)
                elif "plagal" in cadence_type:
                    cadence_types["plagal"].append(cadence)
                elif "half" in cadence_type:
                    cadence_types["half"].append(cadence)
                elif "deceptive" in cadence_type:
                    cadence_types["deceptive"].append(cadence)
                else:
                    cadence_types["other"].append(cadence)

            self.report_progress(1.0, "Analysis complete")

            # Summary statistics
            total_cadences = len(cadences)
            cadence_density = (
                total_cadences / max(1, score.duration.quarterLength) * 16
            )  # Per 16 quarters

            return self.create_success_response(
                total_cadences=total_cadences,
                cadence_density=round(cadence_density, 2),
                cadence_types={k: len(v) for k, v in cadence_types.items()},
                cadences=cadences[:20],  # Limit output
                authentic_cadences=cadence_types["authentic"][:5],
                deceptive_resolutions=functional_result.deceptive_resolutions[:5],
            )

        except Exception as e:
            logger.error(f"Cadence analysis failed: {e}")
            return self.create_error_response(f"Analysis failed: {str(e)}")

    def _serialize_functional_analysis(self, analysis) -> Dict[str, Any]:
        """Convert functional analysis to JSON-serializable format"""
        return {
            "roman_numerals": analysis.roman_numerals[:50],  # Limit
            "functions": [f.value for f in analysis.functions[:50]],
            "tonal_strength": analysis.tonal_strength,
            "phrase_model": analysis.phrase_model,
            "cadences": analysis.cadences[:10],
            "cadence_count": len(analysis.cadences),
        }

    def _serialize_voice_leading(self, analysis) -> Dict[str, Any]:
        """Convert voice leading analysis to JSON-serializable format"""
        return {
            "errors": [
                {
                    "type": (
                        e["type"].value if hasattr(e["type"], "value") else e["type"]
                    ),
                    "location": e.get("location", "unknown"),
                    "voices": e.get("voices", []),
                }
                for e in analysis.errors[:20]
            ],  # Limit
            "error_count": len(analysis.errors),
            "smoothness_score": analysis.smoothness_score,
            "independence_score": analysis.independence_score,
            "parallel_motions": len(analysis.parallel_motions),
            "voice_crossings": len(analysis.voice_crossings),
        }

    def _serialize_jazz_analysis(self, analysis) -> Dict[str, Any]:
        """Convert jazz analysis to JSON-serializable format"""
        return {
            "chord_symbols": analysis.chord_symbols[:50],
            "extended_chords": analysis.extended_chords[:20],
            "substitutions": [
                {
                    "type": (
                        s["type"].value if hasattr(s["type"], "value") else s["type"]
                    ),
                    "original": s.get("original", ""),
                    "substitute": s.get("substitute", ""),
                    "location": s.get("location", 0),
                }
                for s in analysis.substitutions[:10]
            ],
            "modal_interchanges": analysis.modal_interchanges[:10],
        }

    def _serialize_modulation_analysis(self, analysis) -> Dict[str, Any]:
        """Convert modulation analysis to JSON-serializable format"""
        return {
            "modulations": analysis.modulations[:10],
            "modulation_count": len(analysis.modulations),
            "keys_visited": analysis.keys_visited,
            "tonal_plan": analysis.tonal_plan,
            "stability_score": analysis.stability_score,
        }

    def _identify_common_progressions(
        self, roman_numerals: List[str]
    ) -> List[Dict[str, Any]]:
        """Identify common chord progressions"""
        common_patterns = {
            "ii-V-I": ["ii", "V", "I"],
            "IV-V-I": ["IV", "V", "I"],
            "I-V-vi-IV": ["I", "V", "vi", "IV"],
            "I-vi-IV-V": ["I", "vi", "IV", "V"],
            "I-IV-I": ["I", "IV", "I"],
            "V-I": ["V", "I"],
            "ii-V": ["ii", "V"],
        }

        found_progressions = []

        for name, pattern in common_patterns.items():
            # Search for pattern in progression
            pattern_length = len(pattern)
            for i in range(len(roman_numerals) - pattern_length + 1):
                segment = roman_numerals[i : i + pattern_length]
                # Simple pattern matching (could be improved)
                if all(
                    self._roman_match(segment[j], pattern[j])
                    for j in range(pattern_length)
                ):
                    found_progressions.append(
                        {"name": name, "location": i, "pattern": pattern}
                    )

        return found_progressions[:10]  # Limit results

    def _roman_match(self, actual: str, expected: str) -> bool:
        """Check if Roman numeral matches expected pattern"""
        # Simple matching - could be enhanced
        return expected.lower() in actual.lower()

    def _summarize_progression(self, roman_numerals: List[str]) -> Dict[str, Any]:
        """Generate summary statistics of the progression"""
        if not roman_numerals:
            return {}

        # Count occurrences
        from collections import Counter

        chord_counts = Counter(roman_numerals)

        # Most common chords
        most_common = chord_counts.most_common(10)

        # Tonic/dominant ratio
        tonic_count = sum(
            1 for rn in roman_numerals if "I" in rn and "V" not in rn and "I" not in rn
        )
        dominant_count = sum(1 for rn in roman_numerals if "V" in rn)

        return {
            "unique_chords": len(chord_counts),
            "most_common": [
                {"chord": chord, "count": count} for chord, count in most_common
            ],
            "tonic_percentage": round(100 * tonic_count / len(roman_numerals), 1),
            "dominant_percentage": round(100 * dominant_count / len(roman_numerals), 1),
        }

    def _generate_harmonic_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a summary of all harmonic analyses"""
        summary = {
            "complexity_score": 0,
            "style_indicators": [],
            "key_characteristics": [],
        }

        # Calculate complexity based on various factors
        if "functional_harmony" in results:
            fh = results["functional_harmony"]
            summary["tonal_strength"] = fh.get("tonal_strength", 0)
            summary["cadence_count"] = fh.get("cadence_count", 0)

        if "voice_leading" in results:
            vl = results["voice_leading"]
            summary["voice_leading_quality"] = vl.get("smoothness_score", 0)
            summary["voice_errors"] = vl.get("error_count", 0)

        if "modulations" in results:
            mod = results["modulations"]
            summary["modulation_count"] = mod.get("modulation_count", 0)
            summary["keys_visited"] = mod.get("keys_visited", [])

        # Style indicators
        if results.get("jazz_harmony", {}).get("extended_chords"):
            summary["style_indicators"].append("Jazz harmony detected")

        if summary.get("modulation_count", 0) > 3:
            summary["style_indicators"].append("Highly chromatic")

        return summary
