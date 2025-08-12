"""
Pattern Recognition Tools - Find melodic and rhythmic patterns
Supports sequence detection, motivic analysis, and fuzzy matching
"""

import logging
from collections import defaultdict
from typing import Any

import numpy as np
from music21 import interval, stream

from .base_tool import BaseTool

logger = logging.getLogger(__name__)


class PatternRecognitionTool(BaseTool):
    """
    Pattern recognition tools for:
    1. Melodic pattern detection (sequences, motifs)
    2. Rhythmic pattern detection
    3. Phrase structure analysis
    4. Contour analysis with similarity scoring
    """

    async def execute(self, **kwargs: Any) -> dict[str, Any]:
        """
        Find patterns in a musical score

        Args:
            **kwargs: Keyword arguments including:
                score_id: ID of the score to analyze
                pattern_type: Type of patterns to find ('melodic', 'rhythmic', 'both')
                min_pattern_length: Minimum length for a pattern
                similarity_threshold: Threshold for fuzzy matching (0-1)
                include_transformations: Include inversions, retrogrades, etc.
        """
        # Extract parameters from kwargs
        score_id = kwargs.get("score_id", "")
        pattern_type = kwargs.get("pattern_type", "both")
        min_pattern_length = kwargs.get("min_pattern_length", 3)
        similarity_threshold = kwargs.get("similarity_threshold", 0.85)
        include_transformations = kwargs.get("include_transformations", True)
        # Validate inputs
        error = self.validate_inputs(**kwargs)
        if error:
            return self.create_error_response(error)

        with self.error_handling(f"Pattern recognition for '{score_id}'"):
            score = self.get_score(score_id)

            results = {}

            if pattern_type in ["melodic", "both"]:
                self.report_progress(0.2, "Finding melodic patterns")
                melodic_results = await self._find_melodic_patterns(
                    score,
                    min_pattern_length,
                    similarity_threshold,
                    include_transformations,
                )
                results["melodic_patterns"] = melodic_results

            if pattern_type in ["rhythmic", "both"]:
                self.report_progress(0.5, "Finding rhythmic patterns")
                rhythmic_results = await self._find_rhythmic_patterns(
                    score, min_pattern_length, similarity_threshold
                )
                results["rhythmic_patterns"] = rhythmic_results

            # Phrase structure analysis
            self.report_progress(0.8, "Analyzing phrase structure")
            phrase_structure = self._analyze_phrase_structure(score)
            results["phrase_structure"] = phrase_structure

            self.report_progress(1.0, "Analysis complete")

            return self.create_success_response(
                message="Pattern recognition complete", **results
            )

    def validate_inputs(self, **kwargs: Any) -> str | None:
        """Validate input parameters"""
        score_id = kwargs.get("score_id", "")
        pattern_type = kwargs.get("pattern_type", "both")
        similarity_threshold = kwargs.get("similarity_threshold", 0.85)

        error = self.check_score_exists(score_id)
        if error:
            return error

        if pattern_type not in ["melodic", "rhythmic", "both"]:
            return f"Invalid pattern_type: {pattern_type}"

        if not 0 <= similarity_threshold <= 1:
            return "similarity_threshold must be between 0 and 1"

        return None

    async def _find_melodic_patterns(
        self,
        score: stream.Score,
        min_length: int,
        threshold: float,
        include_transformations: bool,
    ) -> dict[str, Any]:
        """Find melodic patterns including sequences and motifs"""
        results: dict[str, Any] = {
            "sequences": [],
            "motifs": [],
            "contour_patterns": [],
            "interval_patterns": [],
        }

        try:
            # Extract melodies from all parts
            melodies = []
            for part in score.parts:
                melody = [n for n in part.flatten().notes if hasattr(n, "pitch")]
                if melody:
                    melodies.append(melody)

            # Find sequences (exact transpositions)
            for melody in melodies:
                sequences = self._find_sequences(melody, min_length)
                results["sequences"].extend(sequences)

            # Find motifs (recurring patterns)
            for melody in melodies:
                motifs = self._find_motifs(
                    melody, min_length, threshold, include_transformations
                )
                results["motifs"].extend(motifs)

            # Analyze melodic contours
            for i, melody in enumerate(melodies):
                contour = self._analyze_contour(melody)
                if contour:
                    contour["part"] = i
                    results["contour_patterns"].append(contour)

            # Find interval patterns
            for melody in melodies:
                interval_patterns = self._find_interval_patterns(melody, min_length)
                results["interval_patterns"].extend(interval_patterns)

            # Deduplicate and sort by significance
            results["sequences"] = self._deduplicate_patterns(results["sequences"])[:20]
            results["motifs"] = self._deduplicate_patterns(results["motifs"])[:20]

            # Add summary statistics
            results["summary"] = {
                "total_sequences": len(results["sequences"]),
                "total_motifs": len(results["motifs"]),
                "most_common_intervals": self._get_common_intervals(melodies),
                "melodic_density": self._calculate_melodic_density(melodies, score),
            }

        except Exception as e:
            logger.error(f"Melodic pattern finding failed: {e}")
            results["error"] = str(e)

        return results

    async def _find_rhythmic_patterns(
        self, score: stream.Score, min_length: int, threshold: float
    ) -> dict[str, Any]:
        """Find rhythmic patterns"""
        results: dict[str, Any] = {
            "rhythmic_motifs": [],
            "metric_patterns": [],
            "syncopations": [],
            "cross_rhythms": [],
        }

        try:
            # Extract rhythm from all parts
            rhythms = []
            for part in score.parts:
                rhythm_stream = []
                for element in part.flatten().notesAndRests:
                    rhythm_stream.append(
                        {
                            "duration": element.duration.quarterLength,
                            "offset": element.offset,
                            "is_rest": element.isRest,
                        }
                    )
                if rhythm_stream:
                    rhythms.append(rhythm_stream)

            # Find rhythmic motifs
            for rhythm in rhythms:
                motifs = self._find_rhythmic_motifs(rhythm, min_length, threshold)
                results["rhythmic_motifs"].extend(motifs)

            # Analyze metric patterns
            metric_patterns = self._analyze_metric_patterns(score)
            results["metric_patterns"] = metric_patterns

            # Find syncopations
            for part in score.parts:
                syncopations = self._find_syncopations(part)
                results["syncopations"].extend(syncopations)

            # Detect cross-rhythms
            if len(score.parts) > 1:
                cross_rhythms = self._detect_cross_rhythms(score)
                results["cross_rhythms"] = cross_rhythms

            # Deduplicate
            results["rhythmic_motifs"] = self._deduplicate_patterns(
                results["rhythmic_motifs"]
            )[:15]

            # Add summary
            results["summary"] = {
                "total_motifs": len(results["rhythmic_motifs"]),
                "syncopation_density": len(results["syncopations"])
                / max(1, score.duration.quarterLength),
                "has_cross_rhythms": len(results["cross_rhythms"]) > 0,
            }

        except Exception as e:
            logger.error(f"Rhythmic pattern finding failed: {e}")
            results["error"] = str(e)

        return results

    def _find_sequences(self, melody: list, min_length: int) -> list[dict]:
        """Find melodic sequences (exact transpositions)"""
        sequences: list[dict[str, Any]] = []

        if len(melody) < min_length * 2:
            return sequences

        # Get intervals between consecutive notes
        intervals = []
        for i in range(len(melody) - 1):
            try:
                intv = interval.Interval(noteStart=melody[i], noteEnd=melody[i + 1])
                intervals.append(intv.semitones)
            except (AttributeError, TypeError, ValueError) as e:
                logger.debug(f"Interval calculation failed at position {i}: {e}")
                intervals.append(0)

        # Look for repeated interval patterns
        # Ensure we have at least space for two occurrences of the pattern
        max_pattern_length = min(len(intervals) // 2 + 1, len(intervals))
        for pattern_length in range(min_length, max_pattern_length):
            for start in range(len(intervals) - pattern_length * 2 + 1):
                pattern = intervals[start : start + pattern_length]

                # Look for repetitions
                for next_start in range(
                    start + pattern_length, len(intervals) - pattern_length + 1
                ):
                    if intervals[next_start : next_start + pattern_length] == pattern:
                        # Check if it's a true sequence (transposed)
                        transposition = (
                            melody[next_start].pitch.midi - melody[start].pitch.midi
                        )

                        sequences.append(
                            {
                                "type": (
                                    "real_sequence"
                                    if transposition != 0
                                    else "literal_repeat"
                                ),
                                "start_measure": melody[start].measureNumber,
                                "length": pattern_length,
                                "transposition": transposition,
                                "interval_pattern": pattern,
                                "occurrences": [
                                    {
                                        "position": start,
                                        "measure": melody[start].measureNumber,
                                    },
                                    {
                                        "position": next_start,
                                        "measure": melody[next_start].measureNumber,
                                    },
                                ],
                            }
                        )

        return sequences

    def _find_motifs(
        self,
        melody: list,
        min_length: int,
        threshold: float,
        include_transformations: bool,
    ) -> list[dict]:
        """Find recurring melodic motifs with fuzzy matching"""
        motifs: list[dict[str, Any]] = []

        if len(melody) < min_length:
            return motifs

        # Extract pitch sequences
        pitch_sequences = []
        for i in range(len(melody) - min_length + 1):
            segment = melody[i : i + min_length]
            pitches = [n.pitch.midi for n in segment]
            pitch_sequences.append(
                {"pitches": pitches, "start": i, "measure": melody[i].measureNumber}
            )

        # Find similar patterns
        processed = set()

        for i, seq1 in enumerate(pitch_sequences):
            if i in processed:
                continue

            similar = [seq1]

            for j, seq2 in enumerate(pitch_sequences[i + 1 :], i + 1):
                if j in processed:
                    continue

                # Calculate similarity
                similarity = self._calculate_pitch_similarity(
                    seq1["pitches"], seq2["pitches"], include_transformations
                )

                if similarity >= threshold:
                    similar.append(seq2)
                    processed.add(j)

            if len(similar) >= 2:  # At least 2 occurrences
                motifs.append(
                    {
                        "pitch_contour": self._get_contour(seq1["pitches"]),
                        "length": min_length,
                        "occurrences": len(similar),
                        "instances": [
                            {"measure": s["measure"], "position": s["start"]}
                            for s in similar
                        ],
                        "transformations": (
                            self._identify_transformations(similar)
                            if include_transformations
                            else []
                        ),
                    }
                )
                processed.add(i)

        # Sort by frequency
        motifs.sort(key=lambda x: x.get("occurrences", 0), reverse=True)

        return motifs

    def _calculate_pitch_similarity(
        self, seq1: list[int], seq2: list[int], include_transformations: bool
    ) -> float:
        """Calculate similarity between two pitch sequences"""
        if len(seq1) != len(seq2):
            return 0.0

        # Direct comparison
        if seq1 == seq2:
            return 1.0

        # Transposition check
        interval_diff = seq2[0] - seq1[0]
        transposed = [p + interval_diff for p in seq1]
        if transposed == seq2:
            return 0.95

        if include_transformations:
            # Inversion check
            inverted = [seq1[0] - (p - seq1[0]) for p in seq1]
            if inverted == seq2:
                return 0.9

            # Retrograde check
            if seq1[::-1] == seq2:
                return 0.9

        # Contour similarity
        contour1 = self._get_contour(seq1)
        contour2 = self._get_contour(seq2)

        if contour1 == contour2:
            return 0.85

        # Fuzzy contour matching
        matches = sum(1 for a, b in zip(contour1, contour2, strict=False) if a == b)
        return matches / len(contour1) * 0.8

    def _get_contour(self, pitches: list[int]) -> list[str]:
        """Get melodic contour (up, down, same)"""
        contour = []
        for i in range(len(pitches) - 1):
            if pitches[i + 1] > pitches[i]:
                contour.append("U")
            elif pitches[i + 1] < pitches[i]:
                contour.append("D")
            else:
                contour.append("S")
        return contour

    def _analyze_contour(self, melody: list) -> dict[str, Any]:
        """Analyze overall melodic contour"""
        if len(melody) < 2:
            return {}

        pitches = [n.pitch.midi for n in melody if hasattr(n, "pitch")]
        if not pitches:
            return {}

        # Calculate contour features
        contour_string = "".join(self._get_contour(pitches))

        # Find peak and trough
        peak_idx = pitches.index(max(pitches))
        trough_idx = pitches.index(min(pitches))

        # Classify overall shape
        shape = self._classify_contour_shape(pitches)

        return {
            "shape": shape,
            "peak_position": peak_idx / len(pitches),  # Normalized position
            "trough_position": trough_idx / len(pitches),
            "range": max(pitches) - min(pitches),
            "contour_string": contour_string[:50],  # Limit length
            "ascending_ratio": contour_string.count("U") / max(1, len(contour_string)),
            "descending_ratio": contour_string.count("D") / max(1, len(contour_string)),
        }

    def _classify_contour_shape(self, pitches: list[int]) -> str:
        """Classify overall melodic shape"""
        if not pitches:
            return "unknown"

        # Simple linear regression for trend
        x = np.arange(len(pitches))
        y = np.array(pitches)

        # Calculate slope
        slope = np.polyfit(x, y, 1)[0]

        # Calculate arch characteristics
        mid_point = len(pitches) // 2
        first_half_avg = np.mean(pitches[:mid_point])
        second_half_avg = np.mean(pitches[mid_point:])
        middle_avg = np.mean(pitches[mid_point // 2 : mid_point + mid_point // 2])

        # Classify
        if abs(slope) < 0.1:  # Relatively flat
            return "static"
        if slope > 0.5:
            return "ascending"
        if slope < -0.5:
            return "descending"
        if middle_avg > first_half_avg and middle_avg > second_half_avg:
            return "arch"
        if middle_avg < first_half_avg and middle_avg < second_half_avg:
            return "inverted_arch"
        return "undulating"

    def _find_interval_patterns(self, melody: list, min_length: int) -> list[dict]:
        """Find recurring interval patterns"""
        patterns: list[dict] = []

        if len(melody) < min_length + 1:
            return patterns

        # Get all intervals
        intervals = []
        for i in range(len(melody) - 1):
            try:
                intv = interval.Interval(noteStart=melody[i], noteEnd=melody[i + 1])
                intervals.append(intv.directedName)
            except (AttributeError, TypeError, ValueError) as e:
                logger.debug(f"Interval name calculation failed at position {i}: {e}")
                intervals.append("P1")

        # Find recurring patterns
        pattern_counts: dict[tuple[str, ...], int] = defaultdict(int)
        pattern_positions: dict[tuple[str, ...], list[int]] = defaultdict(list)

        for length in range(min_length, min(len(intervals), 8)):
            for start in range(len(intervals) - length + 1):
                pattern = tuple(intervals[start : start + length])
                pattern_counts[pattern] += 1
                pattern_positions[pattern].append(start)

        # Filter significant patterns
        for pattern, count in pattern_counts.items():
            if count >= 2:
                patterns.append(
                    {
                        "intervals": list(pattern),
                        "length": len(pattern),
                        "occurrences": count,
                        "positions": pattern_positions[pattern][:10],  # Limit
                        "musical_significance": self._assess_interval_pattern_significance(
                            pattern
                        ),
                    }
                )

        # Sort by significance and frequency
        patterns.sort(
            key=lambda x: (x.get("occurrences", 0), x.get("musical_significance", 0)),
            reverse=True,
        )

        return patterns

    def _assess_interval_pattern_significance(self, pattern: tuple[str, ...]) -> float:
        """Assess musical significance of an interval pattern"""
        significance = 0.5

        # Common melodic patterns get higher significance
        common_patterns = [
            ("M2", "M2"),  # Scale passage
            ("M3", "m3"),  # Triad outline
            ("P4", "P5"),  # Fourth-fifth pattern
            ("M2", "m2"),  # Chromatic neighbor
        ]

        pattern_str = str(pattern)
        for common in common_patterns:
            if str(common) in pattern_str:
                significance += 0.2

        # Penalize very large intervals
        for interval_name in pattern:
            if any(x in interval_name for x in ["6", "7", "8", "9"]):
                significance -= 0.1

        return max(0, min(1, significance))

    def _find_rhythmic_motifs(
        self, rhythm_stream: list[dict], min_length: int, threshold: float
    ) -> list[dict]:
        """Find recurring rhythmic patterns"""
        motifs: list[dict[str, Any]] = []

        if len(rhythm_stream) < min_length:
            return motifs

        # Extract duration sequences
        duration_sequences = []
        for i in range(len(rhythm_stream) - min_length + 1):
            segment = rhythm_stream[i : i + min_length]
            durations = [r["duration"] for r in segment]
            duration_sequences.append(
                {"durations": durations, "start": i, "offset": segment[0]["offset"]}
            )

        # Find similar patterns
        processed = set()

        for i, seq1 in enumerate(duration_sequences):
            if i in processed:
                continue

            similar = [seq1]

            for j, seq2 in enumerate(duration_sequences[i + 1 :], i + 1):
                if j in processed:
                    continue

                # Calculate rhythmic similarity
                similarity = self._calculate_rhythmic_similarity(
                    seq1["durations"], seq2["durations"]
                )

                if similarity >= threshold:
                    similar.append(seq2)
                    processed.add(j)

            if len(similar) >= 2:
                motifs.append(
                    {
                        "pattern": seq1["durations"],
                        "length": min_length,
                        "occurrences": len(similar),
                        "instances": [{"offset": s["offset"]} for s in similar],
                        "total_duration": sum(seq1["durations"]),
                    }
                )
                processed.add(i)

        return motifs

    def _calculate_rhythmic_similarity(
        self, rhythm1: list[float], rhythm2: list[float]
    ) -> float:
        """Calculate similarity between two rhythmic patterns"""
        if len(rhythm1) != len(rhythm2):
            return 0.0

        # Exact match
        if rhythm1 == rhythm2:
            return 1.0

        # Proportional match (augmentation/diminution)
        if len(set(rhythm1)) > 0 and len(set(rhythm2)) > 0:
            ratio = rhythm2[0] / rhythm1[0] if rhythm1[0] != 0 else 0
            if ratio > 0:
                scaled = [r * ratio for r in rhythm1]
                if all(
                    abs(a - b) < 0.01 for a, b in zip(scaled, rhythm2, strict=False)
                ):
                    return 0.9

        # Approximate match with tolerance
        differences = [abs(a - b) for a, b in zip(rhythm1, rhythm2, strict=False)]
        avg_difference = sum(differences) / len(differences)

        if avg_difference < 0.125:  # Within 32nd note
            return 0.8 - (avg_difference * 2)

        return 0.0

    def _analyze_metric_patterns(self, score: stream.Score) -> list[dict]:
        """Analyze metric patterns and accentuation"""
        patterns = []

        try:
            # Get time signature(s)
            time_sigs = list(score.flatten().getElementsByClass("TimeSignature"))

            for ts in time_sigs[:5]:  # Limit analysis
                # Analyze metric hierarchy
                hierarchy = {
                    "time_signature": ts.ratioString,
                    "beats_per_measure": ts.numerator,
                    "beat_unit": ts.denominator,
                    "offset": float(ts.offset),
                    "strong_beats": self._get_strong_beats(ts),
                    "metric_type": self._classify_meter(ts),
                }
                patterns.append(hierarchy)

        except Exception as e:
            logger.error(f"Metric pattern analysis failed: {e}")

        return patterns

    def _get_strong_beats(self, time_sig: Any) -> list[int]:
        """Identify strong beats in a time signature"""
        if time_sig.numerator == 4:
            return [1, 3]
        if time_sig.numerator == 3:
            return [1]
        if time_sig.numerator == 6:
            return [1, 4]
        if time_sig.numerator == 2:
            return [1]
        # Default: first beat and middle if even
        strong = [1]
        if time_sig.numerator % 2 == 0:
            strong.append(time_sig.numerator // 2 + 1)
        return strong

    def _classify_meter(self, time_sig: Any) -> str:
        """Classify the type of meter"""
        if time_sig.numerator in [2, 4]:
            return "simple_duple"
        if time_sig.numerator == 3:
            return "simple_triple"
        if time_sig.numerator == 6:
            return "compound_duple"
        if time_sig.numerator == 9:
            return "compound_triple"
        if time_sig.numerator in [5, 7]:
            return "asymmetric"
        return "complex"

    def _find_syncopations(self, part: stream.Part) -> list[dict]:
        """Find syncopated rhythms"""
        syncopations = []

        try:
            notes = list(part.flatten().notes)

            for note in notes:
                # Check if note starts on weak beat but extends past strong beat
                beat_strength = note.beatStrength
                if beat_strength < 0.5:  # Weak beat
                    # Check if it extends past a stronger beat
                    end_beat = note.offset + note.duration.quarterLength
                    next_strong_beat = int(note.offset) + 1

                    if end_beat > next_strong_beat:
                        syncopations.append(
                            {
                                "offset": float(note.offset),
                                "duration": float(note.duration.quarterLength),
                                "pitch": (
                                    str(note.pitch)
                                    if hasattr(note, "pitch")
                                    else "rest"
                                ),
                                "beat_strength": beat_strength,
                                "measure": note.measureNumber,
                            }
                        )

        except Exception as e:
            logger.debug(f"Syncopation detection error: {e}")

        return syncopations[:20]  # Limit output

    def _detect_cross_rhythms(self, score: stream.Score) -> list[dict]:
        """Detect polyrhythms between parts"""
        cross_rhythms: list[dict[str, Any]] = []

        try:
            if len(score.parts) < 2:
                return cross_rhythms

            # Compare rhythm between first two parts
            part1_rhythm = self._extract_rhythm_profile(score.parts[0])
            part2_rhythm = self._extract_rhythm_profile(score.parts[1])

            # Find conflicting patterns
            # This is simplified - real polyrhythm detection would be more complex
            if part1_rhythm and part2_rhythm:
                if part1_rhythm["primary_division"] != part2_rhythm["primary_division"]:
                    cross_rhythms.append(
                        {
                            "type": "polyrhythm",
                            "part1_division": part1_rhythm["primary_division"],
                            "part2_division": part2_rhythm["primary_division"],
                            "complexity": "simple",
                        }
                    )

        except Exception as e:
            logger.debug(f"Cross-rhythm detection error: {e}")

        return cross_rhythms

    def _extract_rhythm_profile(self, part: stream.Part) -> dict[str, Any]:
        """Extract rhythmic profile of a part"""
        try:
            durations = [n.duration.quarterLength for n in part.flatten().notesAndRests]
            if not durations:
                return {}

            # Find most common duration (simplified)
            from collections import Counter

            duration_counts = Counter(durations)
            primary_duration = duration_counts.most_common(1)[0][0]

            # Determine primary division
            if primary_duration == 0.25:
                primary_division = "sixteenth"
            elif primary_duration == 0.5:
                primary_division = "eighth"
            elif primary_duration == 1.0:
                primary_division = "quarter"
            else:
                primary_division = "other"

            return {
                "primary_duration": primary_duration,
                "primary_division": primary_division,
                "variety": len(set(durations)),
            }

        except Exception as e:
            logger.debug(f"Rhythm profile extraction error: {e}")
            return {}

    def _analyze_phrase_structure(self, score: stream.Score) -> dict[str, Any]:
        """Analyze musical phrase structure"""
        structure: dict[str, Any] = {
            "phrases": [],
            "phrase_lengths": [],
            "symmetry": "unknown",
        }

        try:
            # Simple phrase detection based on rests and cadences
            for part in score.parts[:1]:  # Analyze first part
                melody = part.flatten()
                current_phrase_start = 0

                elements = list(melody.notesAndRests)
                for i, element in enumerate(elements):
                    # Phrase ends at significant rests or end
                    if (
                        (element.isRest and element.duration.quarterLength >= 1.0)
                        or i == len(elements) - 1
                    ) and i > current_phrase_start:
                        phrase_length = sum(
                            e.duration.quarterLength
                            for e in elements[current_phrase_start:i]
                        )

                        structure["phrases"].append(
                            {
                                "start_measure": elements[
                                    current_phrase_start
                                ].measureNumber,
                                "end_measure": (
                                    elements[i - 1].measureNumber if i > 0 else 1
                                ),
                                "length": phrase_length,
                                "note_count": sum(
                                    1
                                    for e in elements[current_phrase_start:i]
                                    if e.isNote
                                ),
                            }
                        )
                        structure["phrase_lengths"].append(phrase_length)

                        current_phrase_start = i + 1

            # Analyze symmetry
            if len(structure["phrase_lengths"]) >= 2:
                lengths = structure["phrase_lengths"]
                if len(set(lengths)) == 1:
                    structure["symmetry"] = "perfectly_symmetric"
                elif (
                    len(lengths) == 4
                    and lengths[0] == lengths[2]
                    and lengths[1] == lengths[3]
                ):
                    structure["symmetry"] = "AABB"
                elif (
                    len(lengths) == 4
                    and lengths[0] == lengths[3]
                    and lengths[1] == lengths[2]
                ):
                    structure["symmetry"] = "ABBA"
                else:
                    structure["symmetry"] = "asymmetric"

        except Exception as e:
            logger.error(f"Phrase structure analysis failed: {e}")
            structure["error"] = str(e)

        return structure

    def _deduplicate_patterns(self, patterns: list[dict]) -> list[dict]:
        """Remove duplicate patterns keeping the most significant"""
        seen = set()
        unique = []

        for pattern in patterns:
            # Create a hashable representation
            if "interval_pattern" in pattern:
                key = tuple(pattern["interval_pattern"])
            elif "pattern" in pattern:
                key = tuple(pattern["pattern"])
            elif "intervals" in pattern:
                key = tuple(pattern["intervals"])
            else:
                continue

            if key not in seen:
                seen.add(key)
                unique.append(pattern)

        return unique

    def _get_common_intervals(self, melodies: list[list]) -> list[dict]:
        """Get most common melodic intervals"""
        interval_counts: dict[str, int] = defaultdict(int)

        for melody in melodies:
            for i in range(len(melody) - 1):
                try:
                    intv = interval.Interval(noteStart=melody[i], noteEnd=melody[i + 1])
                    interval_counts[intv.directedName] += 1
                except (AttributeError, TypeError, ValueError) as e:
                    logger.debug(f"Interval counting failed at position {i}: {e}")
                    pass

        # Get top intervals
        sorted_intervals = sorted(
            interval_counts.items(), key=lambda x: x[1], reverse=True
        )

        return [
            {"interval": name, "count": count} for name, count in sorted_intervals[:10]
        ]

    def _calculate_melodic_density(
        self, melodies: list[list], score: stream.Score
    ) -> float:
        """Calculate notes per quarter note"""
        total_notes = sum(len(m) for m in melodies)
        total_duration = score.duration.quarterLength

        if total_duration > 0:
            return float(round(total_notes / total_duration, 2))
        return 0.0

    def _identify_transformations(self, similar_patterns: list[dict]) -> list[str]:
        """Identify transformations between pattern instances"""
        transformations = set()

        if len(similar_patterns) < 2:
            return []

        base_pitches = similar_patterns[0]["pitches"]

        for pattern in similar_patterns[1:]:
            pitches = pattern["pitches"]

            # Check for transposition
            if len(pitches) == len(base_pitches):
                interval_diff = pitches[0] - base_pitches[0]
                if all(
                    p - b == interval_diff
                    for p, b in zip(pitches, base_pitches, strict=False)
                ):
                    transformations.add(f"T{interval_diff}")

                # Check for inversion
                if all(
                    base_pitches[0] - (p - base_pitches[0]) == pitches[i]
                    for i, p in enumerate(base_pitches)
                ):
                    transformations.add("I")

                # Check for retrograde
                if pitches == base_pitches[::-1]:
                    transformations.add("R")

        return list(transformations)
