"""
Harmony Analysis Tool - Simple and reliable harmonic analysis
Uses music21 directly for Roman numeral analysis and chord detection
"""

import logging
from typing import Any

from music21 import chord, key, roman, stream

from .base_tool import BaseTool

logger = logging.getLogger(__name__)


class HarmonyAnalysisTool(BaseTool):
    """
    Harmonic analysis tool providing:
    1. Roman numeral analysis
    2. Chord progression analysis
    3. Basic functional harmony detection
    """

    def __init__(self, score_manager: dict[str, Any]):
        super().__init__(score_manager)

    async def execute(self, **kwargs: Any) -> dict[str, Any]:
        """
        Perform harmonic analysis on a score

        Args:
            score_id: ID of the score to analyze
        """
        score_id = kwargs.get("score_id", "")

        # Validate inputs
        error = self.validate_inputs(**kwargs)
        if error:
            return self.create_error_response(error)

        with self.error_handling(f"Harmony analysis for '{score_id}'"):
            # Get the score
            score = self.get_score(score_id)
            if not score:
                return self.create_error_response(f"Score '{score_id}' not found")

            # Perform harmonic analysis
            chords = self._extract_chords(score)

            if not chords:
                return self.create_success_response(
                    "No chords found in score",
                    roman_numerals=[],
                    chord_progressions=[],
                    functional_analysis={},
                    harmonic_rhythm=[],
                    chord_count=0,
                )

            # Analyze with Roman numerals
            roman_numerals = self._analyze_roman_numerals(chords, score)

            # Basic progression analysis
            progressions = self._analyze_progressions(roman_numerals)

            # Functional analysis
            functional = self._analyze_functional_harmony(roman_numerals)

            # Harmonic rhythm
            rhythm = self._analyze_harmonic_rhythm(chords)

            return self.create_success_response(
                f"Harmony analysis complete: {len(roman_numerals)} chords analyzed",
                roman_numerals=roman_numerals,
                chord_progressions=progressions,
                functional_analysis=functional,
                harmonic_rhythm=rhythm,
                chord_count=len(chords),
            )

    def validate_inputs(self, **kwargs: Any) -> str | None:
        """Validate the inputs for harmony analysis"""
        score_id = kwargs.get("score_id", "")
        if not score_id:
            return "score_id is required"
        return None

    def _extract_chords(self, score: stream.Score) -> list[chord.Chord]:
        """Extract chords from the score"""
        chords = []

        # First try to get existing chord objects
        for element in score.recurse():
            if isinstance(element, chord.Chord):
                chords.append(element)

        # If no chords found, try to chordify the score
        if not chords:
            try:
                # Chordify will combine simultaneous notes into chords
                chordified = score.chordify()
                for element in chordified.recurse():
                    if isinstance(element, chord.Chord):
                        chords.append(element)
            except:
                # If chordify fails, return empty list
                pass

        return chords

    def _analyze_roman_numerals(
        self, chords: list[chord.Chord], score: stream.Score
    ) -> list[dict[str, Any]]:
        """Analyze chords using Roman numeral notation"""
        roman_numerals = []

        # Try to get key from score
        try:
            # Get key from score metadata or analyze it
            key_obj = score.analyze("key")
            if not key_obj:
                # Fallback to C major
                key_obj = key.Key("C")
        except:
            key_obj = key.Key("C")

        for i, chord_obj in enumerate(chords):
            try:
                # Create Roman numeral representation
                rn = roman.romanNumeralFromChord(chord_obj, key_obj)

                roman_numerals.append(
                    {
                        "position": i,
                        "chord": chord_obj.pitchedCommonName,
                        "roman_numeral": str(rn),
                        "key": str(key_obj),
                        "root": str(chord_obj.root()),
                        "quality": chord_obj.quality,
                        "inversion": chord_obj.inversion(),
                        "measure": getattr(chord_obj, "measureNumber", None),
                        "offset": (
                            float(chord_obj.offset)
                            if hasattr(chord_obj, "offset")
                            else 0.0
                        ),
                    }
                )
            except Exception as e:
                logger.warning(f"Could not analyze chord {chord_obj}: {e}")
                # Add basic info even if Roman numeral fails
                roman_numerals.append(
                    {
                        "position": i,
                        "chord": chord_obj.pitchedCommonName,
                        "roman_numeral": "?",
                        "key": str(key_obj),
                        "root": str(chord_obj.root()),
                        "quality": chord_obj.quality,
                        "inversion": chord_obj.inversion(),
                        "measure": getattr(chord_obj, "measureNumber", None),
                        "offset": (
                            float(chord_obj.offset)
                            if hasattr(chord_obj, "offset")
                            else 0.0
                        ),
                    }
                )

        return roman_numerals

    def _analyze_progressions(
        self, roman_numerals: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Analyze chord progressions"""
        progressions: list[dict[str, Any]] = []

        if len(roman_numerals) < 2:
            return progressions

        # Look for common progressions
        rn_sequence = [rn["roman_numeral"] for rn in roman_numerals]

        # Check for common progressions
        common_progressions = [
            (["I", "IV", "V", "I"], "Authentic Cadence"),
            (["I", "vi", "IV", "V"], "vi-IV-V Progression"),
            (["ii", "V", "I"], "ii-V-I"),
            (["I", "V", "vi", "IV"], "I-V-vi-IV"),
            (["vi", "IV", "I", "V"], "vi-IV-I-V"),
        ]

        for prog_pattern, prog_name in common_progressions:
            for i in range(len(rn_sequence) - len(prog_pattern) + 1):
                if rn_sequence[i : i + len(prog_pattern)] == prog_pattern:
                    progressions.append(
                        {
                            "name": prog_name,
                            "start_position": i,
                            "end_position": i + len(prog_pattern) - 1,
                            "chords": prog_pattern,
                        }
                    )

        return progressions

    def _analyze_functional_harmony(
        self, roman_numerals: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Analyze functional harmony"""
        if not roman_numerals:
            return {}

        # Categorize chords by function
        tonic_chords = []
        dominant_chords = []
        subdominant_chords = []

        for rn in roman_numerals:
            rn_str = rn["roman_numeral"]
            if rn_str in ["I", "i", "vi", "VI"]:
                tonic_chords.append(rn)
            elif rn_str in ["V", "v", "vii", "VII"]:
                dominant_chords.append(rn)
            elif rn_str in ["IV", "iv", "ii", "II"]:
                subdominant_chords.append(rn)

        return {
            "tonic_function": len(tonic_chords),
            "dominant_function": len(dominant_chords),
            "subdominant_function": len(subdominant_chords),
            "tonic_chords": tonic_chords,
            "dominant_chords": dominant_chords,
            "subdominant_chords": subdominant_chords,
        }

    def _analyze_harmonic_rhythm(
        self, chords: list[chord.Chord]
    ) -> list[dict[str, Any]]:
        """Analyze harmonic rhythm"""
        rhythm = []

        for i, chord_obj in enumerate(chords):
            duration = (
                float(chord_obj.quarterLength)
                if hasattr(chord_obj, "quarterLength")
                else 1.0
            )
            rhythm.append(
                {
                    "position": i,
                    "duration": duration,
                    "chord": chord_obj.pitchedCommonName,
                }
            )

        return rhythm
