"""
Score Info Tool - Extract comprehensive metadata and information from scores
"""

import logging
from typing import Any

from music21 import instrument, meter, stream, tempo

from .base_tool import BaseTool

logger = logging.getLogger(__name__)


class ScoreInfoTool(BaseTool):
    """Tool for extracting comprehensive score information and metadata"""

    async def execute(self, **kwargs: Any) -> dict[str, Any]:
        """
        Get comprehensive information about a score

        Args:
            **kwargs: Keyword arguments including:
                score_id: ID of the score to analyze
                include_instruments: Include instrument analysis
                include_structure: Include structural analysis
        """
        # Extract parameters from kwargs
        score_id = kwargs.get("score_id", "")
        include_instruments = kwargs.get("include_instruments", True)
        include_structure = kwargs.get("include_structure", True)
        # Validate inputs
        error = self.validate_inputs(**kwargs)
        if error:
            return self.create_error_response(error)

        with self.error_handling(f"Get info for '{score_id}'"):
            score = self.get_score(score_id)

            self.report_progress(0.1, "Extracting basic metadata")

            # Basic info
            info = {"exists": True, "score_id": score_id}

            # Extract metadata
            metadata = self._extract_metadata(score)
            info["metadata"] = metadata

            self.report_progress(0.3, "Analyzing score structure")

            # Structure analysis
            structure = self._analyze_structure(score)
            info.update(structure)

            # Time and tempo analysis
            self.report_progress(0.5, "Analyzing time and tempo")
            time_info = self._analyze_time_and_tempo(score)
            info.update(time_info)

            # Instrument analysis
            if include_instruments:
                self.report_progress(0.7, "Analyzing instrumentation")
                instruments = self._analyze_instruments(score)
                info["instruments"] = instruments

            # Additional structure analysis
            if include_structure:
                self.report_progress(0.9, "Analyzing detailed structure")
                detailed_structure = self._analyze_detailed_structure(score)
                info["structure"] = detailed_structure

            self.report_progress(1.0, "Analysis complete")

            return self.create_success_response(**info)

    def validate_inputs(self, **kwargs: Any) -> str | None:
        """Validate input parameters"""
        score_id = kwargs.get("score_id", "")
        return self.check_score_exists(score_id)

    def _extract_metadata(self, score: stream.Score) -> dict[str, Any]:
        """Extract metadata from score"""
        metadata = {}

        try:
            if hasattr(score, "metadata"):
                md = score.metadata

                # Title and movement
                if md.title:
                    metadata["title"] = md.title
                if md.movementName:
                    metadata["movementName"] = md.movementName
                if md.movementNumber:
                    metadata["movementNumber"] = md.movementNumber

                # Composer and contributors
                if md.composer:
                    metadata["composer"] = md.composer
                if md.lyricist:
                    metadata["lyricist"] = md.lyricist
                if md.arranger:
                    metadata["arranger"] = md.arranger

                # Dates
                if md.date:
                    metadata["date"] = str(md.date)

                # Copyright and source
                if md.copyright:
                    metadata["copyright"] = md.copyright

                # Additional metadata
                for attr in [
                    "alternativeTitle",
                    "popularTitle",
                    "parentTitle",
                    "groupTitle",
                    "localeOfComposition",
                ]:
                    if hasattr(md, attr) and getattr(md, attr):
                        metadata[attr] = getattr(md, attr)

        except Exception as e:
            logger.warning(f"Error extracting metadata: {e}")

        return metadata

    def _analyze_structure(self, score: stream.Score) -> dict[str, Any]:
        """Analyze basic structure of the score"""
        try:
            flat = score.flatten()

            # Count elements
            num_notes = len(list(flat.notes))
            num_measures = len(list(flat.getElementsByClass("Measure")))
            num_parts = len(score.parts) if hasattr(score, "parts") else 1

            # Duration
            duration_quarters = float(score.duration.quarterLength)

            # Estimate duration in seconds (assuming 120 BPM if no tempo marking)
            tempo_marking = self._get_first_tempo(score)
            bpm = tempo_marking.number if tempo_marking else 120
            duration_seconds = (duration_quarters / 4) * (60 / bpm) * 4

            return {
                "num_parts": num_parts,
                "num_measures": num_measures,
                "num_notes": num_notes,
                "duration_quarters": duration_quarters,
                "duration_seconds": round(duration_seconds, 1),
            }

        except Exception as e:
            logger.warning(f"Error analyzing structure: {e}")
            return {
                "num_parts": 0,
                "num_measures": 0,
                "num_notes": 0,
                "duration_quarters": 0,
                "duration_seconds": 0,
            }

    def _analyze_time_and_tempo(self, score: stream.Score) -> dict[str, Any]:
        """Analyze time signatures and tempo markings"""
        result: dict[str, Any] = {}

        try:
            flat = score.flatten()

            # Time signatures
            time_sigs = list(flat.getElementsByClass(meter.TimeSignature))
            if time_sigs:
                result["time_signatures"] = []
                for ts in time_sigs[:5]:  # Limit to first 5
                    result["time_signatures"].append(
                        {
                            "signature": ts.ratioString,
                            "offset": float(ts.offset),
                            "numerator": ts.numerator,
                            "denominator": ts.denominator,
                        }
                    )

            # Tempo markings
            tempo_markings = list(flat.getElementsByClass(tempo.MetronomeMark))
            if tempo_markings:
                result["tempo_markings"] = []
                for tm in tempo_markings[:5]:  # Limit to first 5
                    result["tempo_markings"].append(
                        {
                            "bpm": tm.number,
                            "offset": float(tm.offset),
                            "text": tm.text if hasattr(tm, "text") else None,
                        }
                    )

                # Overall tempo
                result["tempo_bpm"] = tempo_markings[0].number
            else:
                result["tempo_bpm"] = 120  # Default

        except Exception as e:
            logger.warning(f"Error analyzing time/tempo: {e}")
            result["tempo_bpm"] = 120

        return result

    def _analyze_instruments(self, score: stream.Score) -> list[dict[str, Any]]:
        """Analyze instrumentation"""
        instruments_found = []

        try:
            if hasattr(score, "parts"):
                for i, part in enumerate(score.parts):
                    part_info = {
                        "part_number": i + 1,
                        "part_name": (
                            part.partName
                            if hasattr(part, "partName")
                            else f"Part {i + 1}"
                        ),
                    }

                    # Find instrument objects
                    instr_objs = list(part.getElementsByClass(instrument.Instrument))
                    if instr_objs:
                        instr = instr_objs[0]
                        part_info["instrument"] = instr.instrumentName
                        part_info["abbreviation"] = instr.instrumentAbbreviation
                        part_info["midi_program"] = instr.midiProgram

                    # Get range
                    notes = list(part.flatten().notes)
                    if notes:
                        pitches = [n.pitch.midi for n in notes if hasattr(n, "pitch")]
                        if pitches:
                            part_info["lowest_note"] = min(pitches)
                            part_info["highest_note"] = max(pitches)
                            part_info["range"] = max(pitches) - min(pitches)

                    instruments_found.append(part_info)

        except Exception as e:
            logger.warning(f"Error analyzing instruments: {e}")

        return instruments_found

    def _analyze_detailed_structure(self, score: stream.Score) -> dict[str, Any]:
        """Analyze detailed structural elements"""
        structure: dict[str, Any] = {}

        try:
            flat = score.flatten()

            # Key signatures
            key_sigs = list(flat.getElementsByClass("KeySignature"))
            if key_sigs:
                structure["key_signatures"] = []
                for ks in key_sigs[:5]:
                    structure["key_signatures"].append(
                        {"sharps": ks.sharps, "offset": float(ks.offset)}
                    )

            # Rehearsal marks
            rehearsal_marks = list(flat.getElementsByClass("RehearsalMark"))
            if rehearsal_marks:
                structure["rehearsal_marks"] = [str(rm) for rm in rehearsal_marks[:10]]

            # Dynamics
            dynamics = list(flat.getElementsByClass("Dynamic"))
            if dynamics:
                structure["dynamics_count"] = len(dynamics)
                structure["dynamic_range"] = list(
                    {d.value for d in dynamics if hasattr(d, "value")}
                )

            # Measure analysis
            measures = list(flat.getElementsByClass("Measure"))
            if measures:
                structure["first_measure_number"] = measures[0].number
                structure["last_measure_number"] = measures[-1].number

                # Check for pickup measure
                if measures[0].paddingLeft > 0:
                    structure["has_pickup"] = True

        except Exception as e:
            logger.warning(f"Error analyzing detailed structure: {e}")

        return structure

    def _get_first_tempo(self, score: stream.Score) -> tempo.MetronomeMark | None:
        """Get the first tempo marking in the score"""
        try:
            flat = score.flatten()
            tempos = list(flat.getElementsByClass(tempo.MetronomeMark))
            return tempos[0] if tempos else None
        except:
            return None
