"""
Voice Leading Analysis Tool - Comprehensive voice leading analysis
Detects errors, suggests improvements, and provides educational explanations
"""

import logging
from typing import Any, Dict, List, Optional

from music21 import interval, stream

from ..core.harmonic_analyzer import HarmonicAnalyzer, VoiceLeadingError
from .base_tool import BaseTool

logger = logging.getLogger(__name__)


class VoiceLeadingAnalysisTool(BaseTool):
    """
    Voice leading analysis tool providing:
    1. Parallel fifths/octaves detection
    2. Voice crossing identification
    3. Resolution checking for tendency tones
    4. Voice leading smoothness analysis
    5. Improvement suggestions with explanations
    """

    def __init__(self, score_manager: Dict[str, Any]):
        super().__init__(score_manager)
        self.analyzer = HarmonicAnalyzer()

        # Educational explanations for voice leading rules
        self.rule_explanations = {
            VoiceLeadingError.PARALLEL_FIFTHS: {
                "rule": "Avoid parallel perfect fifths between voices",
                "reason": "Parallel fifths reduce voice independence and create a hollow sound",
                "fix": "Use contrary or oblique motion, or change one voice by step",
            },
            VoiceLeadingError.PARALLEL_OCTAVES: {
                "rule": "Avoid parallel perfect octaves between voices",
                "reason": "Parallel octaves make two voices sound like one, reducing independence",
                "fix": "Use contrary motion or have one voice move by step while the other leaps",
            },
            VoiceLeadingError.DIRECT_FIFTHS: {
                "rule": "Avoid hidden (direct) fifths in outer voices",
                "reason": "Similar motion to a perfect fifth can sound like parallel fifths",
                "fix": "Approach perfect intervals with contrary or oblique motion",
            },
            VoiceLeadingError.VOICE_CROSSING: {
                "rule": "Avoid voice crossing in traditional part writing",
                "reason": "Voice crossing can confuse the listener about which line is which",
                "fix": "Keep voices in their proper ranges: soprano > alto > tenor > bass",
            },
            VoiceLeadingError.LARGE_LEAP: {
                "rule": "Avoid leaps larger than an octave",
                "reason": "Large leaps are difficult to sing and can disrupt melodic flow",
                "fix": "Use stepwise motion or smaller leaps, or fill in large leaps with passing tones",
            },
            VoiceLeadingError.UNRESOLVED_SEVENTH: {
                "rule": "Resolve seventh of chord down by step",
                "reason": "The seventh creates tension that needs resolution",
                "fix": "Move the seventh down by step to the third of the next chord",
            },
            VoiceLeadingError.UNRESOLVED_LEADING_TONE: {
                "rule": "Resolve leading tone to tonic",
                "reason": "The leading tone has a strong tendency to resolve upward",
                "fix": "Move the leading tone up by semitone to the tonic",
            },
        }

    async def execute(self, **kwargs: Any) -> Dict[str, Any]:
        """
        Analyze voice leading in a score

        Args:
            **kwargs: Keyword arguments including:
                score_id: ID of the score to analyze
                style: Analysis style ('strict' for counterpoint, 'free' for general)
                include_suggestions: Include improvement suggestions
                educational_mode: Include educational explanations
        """
        # Extract parameters from kwargs
        score_id = kwargs.get('score_id', '')
        style = kwargs.get('style', 'strict')
        include_suggestions = kwargs.get('include_suggestions', True)
        educational_mode = kwargs.get('educational_mode', True)
        # Validate inputs
        error = self.validate_inputs(**kwargs)
        if error:
            return self.create_error_response(error)

        with self.error_handling(f"Voice leading analysis for '{score_id}'"):
            score = self.get_score(score_id)

            self.report_progress(0.1, "Analyzing voice leading")

            # Get voice leading analysis from analyzer
            vl_result = await self.analyzer.analyze_voice_leading(score)

            self.report_progress(0.3, "Detecting voice leading errors")

            # Analyze errors by type
            errors_by_type = self._categorize_errors(vl_result.errors)

            self.report_progress(0.5, "Analyzing voice motion")

            # Analyze voice motion patterns
            motion_analysis = self._analyze_voice_motion(score)

            self.report_progress(0.7, "Checking resolutions")

            # Check tendency tone resolutions
            resolution_analysis = self._check_resolutions(score)

            # Generate suggestions if requested
            suggestions = []
            if include_suggestions:
                self.report_progress(0.9, "Generating improvement suggestions")
                suggestions = self._generate_suggestions(
                    errors_by_type, motion_analysis
                )

            self.report_progress(1.0, "Analysis complete")

            # Compile results
            result = self.create_success_response(
                total_errors=len(vl_result.errors),
                errors_by_type=self._format_errors_by_type(
                    errors_by_type, educational_mode
                ),
                voice_motion=motion_analysis,
                resolutions=resolution_analysis,
                smoothness_score=vl_result.smoothness_score,
                independence_score=vl_result.independence_score,
                parallel_motions={
                    "count": len(vl_result.parallel_motions),
                    "details": vl_result.parallel_motions[:10],  # Limit
                },
                voice_crossings={
                    "count": len(vl_result.voice_crossings),
                    "locations": vl_result.voice_crossings[:10],
                },
            )

            if include_suggestions:
                result["improvement_suggestions"] = suggestions

            if style == "strict":
                result["counterpoint_assessment"] = self._assess_counterpoint_quality(
                    result
                )

            return result

    def validate_inputs(self, **kwargs: Any) -> Optional[str]:
        """Validate input parameters"""
        score_id = kwargs.get('score_id', '')
        style = kwargs.get('style', 'strict')
        
        error = self.check_score_exists(score_id)
        if error:
            return error

        if style not in ["strict", "free"]:
            return f"Invalid style: {style}. Choose 'strict' or 'free'"

        return None

    def _categorize_errors(self, errors: List[Dict]) -> Dict[str, List]:
        """Categorize errors by type"""
        categorized = {error_type: [] for error_type in VoiceLeadingError}

        for error in errors:
            error_type = error.get("type")
            if isinstance(error_type, VoiceLeadingError):
                categorized[error_type].append(error)
            elif isinstance(error_type, str):
                # Try to match string to enum
                for enum_type in VoiceLeadingError:
                    if enum_type.value == error_type:
                        categorized[enum_type].append(error)
                        break

        return {k: v for k, v in categorized.items() if v}  # Remove empty categories

    def _analyze_voice_motion(self, score: stream.Score) -> Dict[str, Any]:
        """Analyze types of motion between voices"""
        motion_types = {"parallel": 0, "contrary": 0, "oblique": 0, "similar": 0}

        try:
            parts = score.parts
            if len(parts) < 2:
                return {"error": "Need at least 2 parts for voice motion analysis"}

            # Analyze motion between soprano and bass (outer voices)
            soprano = parts[0].flatten().notes
            bass = parts[-1].flatten().notes

            # Pair up simultaneous notes
            for i in range(min(len(soprano), len(bass)) - 1):
                try:
                    # Get intervals
                    int1 = interval.Interval(noteStart=bass[i], noteEnd=soprano[i])
                    int2 = interval.Interval(
                        noteStart=bass[i + 1], noteEnd=soprano[i + 1]
                    )

                    # Determine motion type
                    sop_motion = soprano[i + 1].pitch.midi - soprano[i].pitch.midi
                    bass_motion = bass[i + 1].pitch.midi - bass[i].pitch.midi

                    if sop_motion == 0 or bass_motion == 0:
                        motion_types["oblique"] += 1
                    elif (sop_motion > 0 and bass_motion < 0) or (
                        sop_motion < 0 and bass_motion > 0
                    ):
                        motion_types["contrary"] += 1
                    elif int1.semitones == int2.semitones:
                        motion_types["parallel"] += 1
                    else:
                        motion_types["similar"] += 1

                except Exception as e:
                    logger.debug(f"Error analyzing motion at position {i}: {e}")
                    continue

            total = sum(motion_types.values())
            if total > 0:
                motion_percentages = {
                    k: round(100 * v / total, 1) for k, v in motion_types.items()
                }
            else:
                motion_percentages = motion_types

            return {
                "motion_types": motion_types,
                "motion_percentages": motion_percentages,
                "predominant_motion": max(motion_types, key=motion_types.get),
            }

        except Exception as e:
            logger.error(f"Voice motion analysis failed: {e}")
            return {"error": str(e)}

    def _check_resolutions(self, score: stream.Score) -> Dict[str, Any]:
        """Check resolution of tendency tones"""
        resolutions = {
            "leading_tones": {"resolved": 0, "unresolved": 0},
            "sevenths": {"resolved": 0, "unresolved": 0},
            "suspensions": {"resolved": 0, "unresolved": 0},
        }

        try:
            # Get key for leading tone identification
            key_obj = score.analyze("key")
            leading_tone_pitch = key_obj.pitchFromDegree(7)

            # Analyze each part
            for part in score.parts:
                notes = list(part.flatten().notes)

                for i in range(len(notes) - 1):
                    current = notes[i]
                    next_note = notes[i + 1]

                    # Check leading tone resolution
                    if (
                        hasattr(current, "pitch")
                        and current.pitch.name == leading_tone_pitch.name
                    ):
                        if next_note.pitch.midi == current.pitch.midi + 1:
                            resolutions["leading_tones"]["resolved"] += 1
                        else:
                            resolutions["leading_tones"]["unresolved"] += 1

                    # Check seventh resolutions (simplified)
                    # Would need chord context for accurate analysis

            # Calculate resolution rates
            resolution_rates = {}
            for category, counts in resolutions.items():
                total = counts["resolved"] + counts["unresolved"]
                if total > 0:
                    resolution_rates[category] = round(
                        100 * counts["resolved"] / total, 1
                    )
                else:
                    resolution_rates[category] = None

            return {
                "resolutions": resolutions,
                "resolution_rates": resolution_rates,
                "key_context": str(key_obj),
            }

        except Exception as e:
            logger.error(f"Resolution checking failed: {e}")
            return {"error": str(e)}

    def _format_errors_by_type(
        self, errors_by_type: Dict, educational: bool
    ) -> List[Dict]:
        """Format errors with optional educational content"""
        formatted = []

        for error_type, errors in errors_by_type.items():
            error_info = {
                "type": error_type.value,
                "count": len(errors),
                "severity": self._get_error_severity(error_type),
                "locations": [
                    self._format_error_location(e) for e in errors[:5]
                ],  # Limit
            }

            if educational and error_type in self.rule_explanations:
                error_info["education"] = self.rule_explanations[error_type]

            formatted.append(error_info)

        # Sort by severity and count
        formatted.sort(key=lambda x: (x["severity"], -x["count"]))

        return formatted

    def _get_error_severity(self, error_type: VoiceLeadingError) -> str:
        """Determine severity of voice leading error"""
        high_severity = [
            VoiceLeadingError.PARALLEL_FIFTHS,
            VoiceLeadingError.PARALLEL_OCTAVES,
        ]
        medium_severity = [
            VoiceLeadingError.DIRECT_FIFTHS,
            VoiceLeadingError.DIRECT_OCTAVES,
            VoiceLeadingError.UNRESOLVED_LEADING_TONE,
            VoiceLeadingError.UNRESOLVED_SEVENTH,
        ]

        if error_type in high_severity:
            return "high"
        elif error_type in medium_severity:
            return "medium"
        else:
            return "low"

    def _format_error_location(self, error: Dict) -> Dict:
        """Format error location for output"""
        return {
            "measure": error.get("measure", "unknown"),
            "beat": error.get("beat", "unknown"),
            "voices": error.get("voices", []),
            "description": error.get("description", ""),
        }

    def _generate_suggestions(
        self, errors_by_type: Dict, motion_analysis: Dict
    ) -> List[Dict]:
        """Generate improvement suggestions based on analysis"""
        suggestions = []

        # Suggestions based on errors
        for error_type, errors in errors_by_type.items():
            if error_type in self.rule_explanations:
                suggestion = {
                    "issue": f"{len(errors)} {error_type.value} found",
                    "suggestion": self.rule_explanations[error_type]["fix"],
                    "priority": self._get_error_severity(error_type),
                }
                suggestions.append(suggestion)

        # Suggestions based on motion analysis
        if motion_analysis.get("motion_percentages", {}):
            percentages = motion_analysis["motion_percentages"]

            if percentages.get("parallel", 0) > 40:
                suggestions.append(
                    {
                        "issue": "Excessive parallel motion between voices",
                        "suggestion": "Increase use of contrary motion for better voice independence",
                        "priority": "medium",
                    }
                )

            if percentages.get("contrary", 0) < 20:
                suggestions.append(
                    {
                        "issue": "Insufficient contrary motion",
                        "suggestion": "Use more contrary motion between outer voices for better counterpoint",
                        "priority": "low",
                    }
                )

        # Sort by priority
        priority_order = {"high": 0, "medium": 1, "low": 2}
        suggestions.sort(key=lambda x: priority_order.get(x["priority"], 3))

        return suggestions[:10]  # Limit to top 10 suggestions

    def _assess_counterpoint_quality(self, analysis_result: Dict) -> Dict[str, Any]:
        """Assess overall counterpoint quality for strict style"""
        assessment = {
            "grade": "",
            "strengths": [],
            "weaknesses": [],
            "overall_comment": "",
        }

        # Calculate score based on various factors
        score = 100

        # Deduct for errors
        total_errors = analysis_result.get("total_errors", 0)
        score -= min(40, total_errors * 5)  # Max 40 point deduction

        # Check voice independence
        independence = analysis_result.get("independence_score", 0.5)
        if independence < 0.7:
            score -= 10
            assessment["weaknesses"].append("Limited voice independence")
        elif independence > 0.85:
            assessment["strengths"].append("Excellent voice independence")

        # Check motion variety
        motion = analysis_result.get("voice_motion", {}).get("motion_percentages", {})
        if motion.get("contrary", 0) > 30:
            assessment["strengths"].append("Good use of contrary motion")
        if motion.get("parallel", 0) > 50:
            score -= 15
            assessment["weaknesses"].append("Excessive parallel motion")

        # Assign grade
        if score >= 90:
            assessment["grade"] = "A"
            assessment["overall_comment"] = "Excellent counterpoint with few issues"
        elif score >= 80:
            assessment["grade"] = "B"
            assessment["overall_comment"] = "Good counterpoint with minor issues"
        elif score >= 70:
            assessment["grade"] = "C"
            assessment["overall_comment"] = "Acceptable counterpoint with some problems"
        elif score >= 60:
            assessment["grade"] = "D"
            assessment["overall_comment"] = (
                "Poor counterpoint needing significant improvement"
            )
        else:
            assessment["grade"] = "F"
            assessment["overall_comment"] = (
                "Counterpoint has major issues requiring revision"
            )

        assessment["numeric_score"] = score

        return assessment
