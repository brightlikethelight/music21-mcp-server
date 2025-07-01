"""
Melodic Pattern Recognition Engine for music21 MCP server
Implements motivic analysis, contour analysis, pattern matching,
and cross-cultural melodic analysis.
"""

from collections import Counter
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from music21 import note, pitch, stream
from scipy import signal


class ContourType(Enum):
    """Types of melodic contour"""

    ASCENDING = "ascending"
    DESCENDING = "descending"
    ARCH = "arch"
    INVERTED_ARCH = "inverted_arch"
    WAVE = "wave"
    STATIC = "static"
    ZIGZAG = "zigzag"
    GAP_FILL = "gap_fill"


class MotiveType(Enum):
    """Types of melodic motives"""

    HEAD = "head_motive"
    TAIL = "tail_motive"
    CORE = "core_motive"
    RHYTHMIC = "rhythmic_motive"
    INTERVALLIC = "intervallic_motive"
    COMPOUND = "compound_motive"


class PatternTransformation(Enum):
    """Types of motivic transformations"""

    EXACT = "exact"
    TRANSPOSITION = "transposition"
    INVERSION = "inversion"
    RETROGRADE = "retrograde"
    RETROGRADE_INVERSION = "retrograde_inversion"
    AUGMENTATION = "augmentation"
    DIMINUTION = "diminution"
    FRAGMENTATION = "fragmentation"
    SEQUENCE = "sequence"


class MelodicStyle(Enum):
    """Cross-cultural melodic styles"""

    WESTERN_CLASSICAL = "western_classical"
    JAZZ = "jazz"
    INDIAN_RAGA = "indian_raga"
    ARABIC_MAQAM = "arabic_maqam"
    PENTATONIC = "pentatonic"
    BLUES = "blues"
    FOLK = "folk"
    ATONAL = "atonal"


@dataclass
class MelodicMotive:
    """Represents a melodic motive"""

    pitches: List[pitch.Pitch]
    intervals: List[int]  # In semitones
    rhythm: List[float]  # Duration ratios
    contour: List[int]  # -1, 0, 1 for down, same, up
    start_measure: int
    end_measure: int
    occurrences: List[Dict[str, Any]] = field(default_factory=list)
    transformations: List[PatternTransformation] = field(default_factory=list)
    importance_score: float = 0.0

    def __hash__(self) -> int:
        return hash(tuple(self.intervals))


@dataclass
class ContourAnalysis:
    """Results of melodic contour analysis"""

    overall_contour: ContourType
    contour_segments: List[Dict[str, Any]] = field(default_factory=list)
    contour_vector: List[int] = field(default_factory=list)
    prime_form: List[int] = field(default_factory=list)
    arch_points: List[Dict[str, Any]] = field(default_factory=list)
    complexity_score: float = 0.0
    smoothness_score: float = 0.0


@dataclass
class MotivicAnalysis:
    """Results of motivic analysis"""

    motives: List[MelodicMotive] = field(default_factory=list)
    primary_motive: Optional[MelodicMotive] = None
    motive_hierarchy: Dict[str, List[MelodicMotive]] = field(default_factory=dict)
    transformations: List[Dict[str, Any]] = field(default_factory=list)
    development_timeline: List[Dict[str, Any]] = field(default_factory=list)
    coherence_score: float = 0.0


@dataclass
class PatternMatch:
    """Represents a pattern match"""

    pattern: List[Any]
    locations: List[int]
    similarity_score: float
    transformation_type: PatternTransformation
    transposition_interval: Optional[int] = None


@dataclass
class CrossCulturalAnalysis:
    """Results of cross-cultural melodic analysis"""

    detected_styles: List[Tuple[MelodicStyle, float]] = field(default_factory=list)
    scale_characteristics: Dict[str, Any] = field(default_factory=dict)
    ornamentations: List[Dict[str, Any]] = field(default_factory=list)
    microtonal_inflections: List[Dict[str, Any]] = field(default_factory=list)
    cultural_markers: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MelodicSimilarity:
    """Results of melodic similarity analysis"""

    similarity_matrix: np.ndarray = None
    similar_phrases: List[Dict[str, Any]] = field(default_factory=list)
    theme_variations: List[Dict[str, Any]] = field(default_factory=list)
    melodic_families: List[List[int]] = field(default_factory=list)


class MelodicAnalyzer:
    """Advanced melodic pattern recognition engine"""

    def __init__(self) -> None:
        self.min_motive_length = 3
        self.similarity_threshold = 0.75

    async def analyze_melodic_contour(
        self, melody_line: stream.Stream, window_size: int = 8
    ) -> ContourAnalysis:
        """
        Analyze melodic contour and shape.

        Args:
            melody_line: Melodic line to analyze
            window_size: Analysis window in notes

        Returns:
            ContourAnalysis object
        """
        result = ContourAnalysis(overall_contour=ContourType.STATIC)

        # Extract pitches
        notes = [n for n in melody_line.flatten().notes if hasattr(n, "pitch")]
        if len(notes) < 2:
            return result

        # Calculate contour vector
        result.contour_vector = []
        for i in range(len(notes) - 1):
            if notes[i].pitch < notes[i + 1].pitch:
                result.contour_vector.append(1)  # Up
            elif notes[i].pitch > notes[i + 1].pitch:
                result.contour_vector.append(-1)  # Down
            else:
                result.contour_vector.append(0)  # Same

        # Determine overall contour type
        result.overall_contour = self._classify_contour(result.contour_vector)

        # Analyze contour segments
        for i in range(0, len(notes) - window_size + 1, window_size // 2):
            segment_notes = notes[i : i + window_size]
            segment_contour = result.contour_vector[i : i + window_size - 1]

            segment_analysis = {
                "start_index": i,
                "end_index": i + window_size,
                "contour_type": self._classify_contour(segment_contour),
                "pitch_range": max(n.pitch.midi for n in segment_notes)
                - min(n.pitch.midi for n in segment_notes),
                "direction_changes": self._count_direction_changes(segment_contour),
            }
            result.contour_segments.append(segment_analysis)

        # Calculate prime form (normalized contour)
        result.prime_form = self._calculate_prime_form(result.contour_vector)

        # Find arch points (local maxima/minima)
        pitch_values = [n.pitch.midi for n in notes]
        result.arch_points = self._find_arch_points(pitch_values, notes)

        # Calculate complexity and smoothness
        result.complexity_score = self._calculate_contour_complexity(
            result.contour_vector
        )
        result.smoothness_score = self._calculate_melodic_smoothness(notes)

        return result

    async def detect_motives(
        self, score: stream.Score, min_length: int = 3, min_occurrences: int = 2
    ) -> MotivicAnalysis:
        """
        Detect and analyze melodic motives.

        Args:
            score: Score to analyze
            min_length: Minimum motive length in notes
            min_occurrences: Minimum occurrences to be considered a motive

        Returns:
            MotivicAnalysis object
        """
        result = MotivicAnalysis()

        # Extract all melodic lines
        melodic_parts = self._extract_melodic_parts(score)

        all_motives = []

        for part in melodic_parts:
            notes = [n for n in part.flatten().notes if hasattr(n, 'pitch')]
            if len(notes) < min_length:
                continue

            # Generate potential motives
            for length in range(min_length, min(len(notes) // 2, 12)):
                for start in range(len(notes) - length + 1):
                    motive_notes = notes[start : start + length]

                    # Create motive representation
                    motive = self._create_motive(motive_notes, start)

                    # Find occurrences with transformations
                    occurrences = self._find_motive_occurrences(
                        motive, notes, start + length
                    )

                    if len(occurrences) >= min_occurrences - 1:  # -1 for original
                        motive.occurrences = occurrences
                        motive.importance_score = self._calculate_motive_importance(
                            motive, len(occurrences) + 1
                        )
                        all_motives.append(motive)

        # Filter and rank motives
        result.motives = self._filter_overlapping_motives(all_motives)
        result.motives.sort(key=lambda m: m.importance_score, reverse=True)

        if result.motives:
            result.primary_motive = result.motives[0]

        # Create motive hierarchy
        result.motive_hierarchy = self._create_motive_hierarchy(result.motives)

        # Analyze motive development
        result.development_timeline = self._analyze_motive_development(
            result.motives, score
        )

        # Calculate coherence score
        result.coherence_score = self._calculate_motivic_coherence(result.motives)

        return result

    async def find_melodic_patterns(
        self,
        melody: stream.Stream,
        pattern_length: int = 4,
        similarity_threshold: float = 0.8,
    ) -> List[PatternMatch]:
        """
        Find recurring melodic patterns with transformations.

        Args:
            melody: Melodic line to analyze
            pattern_length: Length of patterns to search
            similarity_threshold: Minimum similarity score

        Returns:
            List of PatternMatch objects
        """
        patterns: List[Dict[str, Any]] = []
        notes = [n for n in melody.flatten().notes if hasattr(n, 'pitch')]

        if len(notes) < pattern_length * 2:
            return patterns

        # Extract interval patterns
        for i in range(len(notes) - pattern_length):
            pattern_notes = notes[i : i + pattern_length]
            pattern_intervals = self._get_interval_pattern(pattern_notes)

            # Search for matches
            matches = []
            for j in range(i + 1, len(notes) - pattern_length + 1):
                candidate_notes = notes[j : j + pattern_length]

                # Check various transformations
                for transform in PatternTransformation:
                    similarity, transposition = self._compare_patterns(
                        pattern_notes, candidate_notes, transform
                    )

                    if similarity >= similarity_threshold:
                        matches.append(
                            {
                                "location": j,
                                "similarity": similarity,
                                "transformation": transform,
                                "transposition": transposition,
                            }
                        )

            if matches:
                # Group similar matches
                pattern_match = PatternMatch(
                    pattern=pattern_intervals,
                    locations=[i] + [m["location"] for m in matches],
                    similarity_score=np.mean([m["similarity"] for m in matches]),
                    transformation_type=matches[0]["transformation"],
                    transposition_interval=matches[0].get("transposition"),
                )
                patterns.append(pattern_match)

        # Remove duplicate patterns
        patterns = self._deduplicate_patterns(patterns)

        return patterns

    async def analyze_cross_cultural_elements(
        self, melody: stream.Stream, include_microtones: bool = True
    ) -> CrossCulturalAnalysis:
        """
        Analyze melodic characteristics across different musical cultures.

        Args:
            melody: Melodic line to analyze
            include_microtones: Whether to detect microtonal inflections

        Returns:
            CrossCulturalAnalysis object
        """
        result = CrossCulturalAnalysis()

        notes = [n for n in melody.flatten().notes if hasattr(n, 'pitch')]
        if not notes:
            return result

        # Extract pitch classes and intervals
        pitch_classes = [n.pitch.pitchClass for n in notes]
        intervals = self._get_interval_sequence(notes)

        # Detect scale/mode characteristics
        result.scale_characteristics = self._analyze_scale_characteristics(
            pitch_classes, intervals
        )

        # Classify melodic style
        style_scores = {}

        # Western classical characteristics
        style_scores[MelodicStyle.WESTERN_CLASSICAL] = self._score_western_classical(
            notes, intervals, result.scale_characteristics
        )

        # Jazz characteristics
        style_scores[MelodicStyle.JAZZ] = self._score_jazz_style(
            notes, intervals, result.scale_characteristics
        )

        # Indian raga characteristics
        style_scores[MelodicStyle.INDIAN_RAGA] = self._score_raga_style(
            notes, intervals, result.scale_characteristics
        )

        # Arabic maqam characteristics
        style_scores[MelodicStyle.ARABIC_MAQAM] = self._score_maqam_style(
            notes, intervals, result.scale_characteristics
        )

        # Pentatonic characteristics
        style_scores[MelodicStyle.PENTATONIC] = self._score_pentatonic_style(
            pitch_classes
        )

        # Blues characteristics
        style_scores[MelodicStyle.BLUES] = self._score_blues_style(
            notes, intervals, pitch_classes
        )

        # Sort by score
        result.detected_styles = sorted(
            style_scores.items(), key=lambda x: x[1], reverse=True
        )

        # Detect ornamentations
        result.ornamentations = self._detect_ornamentations(notes)

        # Detect microtonal inflections if requested
        if include_microtones:
            result.microtonal_inflections = self._detect_microtones(notes)

        # Extract cultural markers
        result.cultural_markers = self._extract_cultural_markers(
            notes,
            intervals,
            result.detected_styles[0][0] if result.detected_styles else None,
        )

        return result

    async def calculate_melodic_similarity(
        self, melodies: List[stream.Stream], method: str = "interval"
    ) -> MelodicSimilarity:
        """
        Calculate similarity between multiple melodic lines.

        Args:
            melodies: List of melodic lines to compare
            method: Similarity method ('interval', 'contour', 'rhythm', 'combined')

        Returns:
            MelodicSimilarity object
        """
        result = MelodicSimilarity()
        n_melodies = len(melodies)

        if n_melodies < 2:
            return result

        # Initialize similarity matrix
        result.similarity_matrix = np.zeros((n_melodies, n_melodies))
        np.fill_diagonal(result.similarity_matrix, 1.0)

        # Calculate pairwise similarities
        for i in range(n_melodies):
            for j in range(i + 1, n_melodies):
                similarity = self._calculate_pairwise_similarity(
                    melodies[i], melodies[j], method
                )
                result.similarity_matrix[i, j] = similarity
                result.similarity_matrix[j, i] = similarity

        # Find similar phrases
        threshold = 0.7
        for i in range(n_melodies):
            similar_indices = np.where(result.similarity_matrix[i] > threshold)[0]
            similar_indices = similar_indices[similar_indices != i]

            if len(similar_indices) > 0:
                result.similar_phrases.append(
                    {
                        "melody_index": i,
                        "similar_to": similar_indices.tolist(),
                        "similarity_scores": result.similarity_matrix[
                            i, similar_indices
                        ].tolist(),
                    }
                )

        # Detect theme and variations
        result.theme_variations = self._detect_theme_variations(
            melodies, result.similarity_matrix
        )

        # Cluster melodies into families
        result.melodic_families = self._cluster_melodies(result.similarity_matrix)

        return result

    async def analyze_melodic_development(
        self, score: stream.Score, track_dynamics: bool = True
    ) -> Dict[str, Any]:
        """
        Analyze how melodic material develops throughout a piece.

        Args:
            score: Score to analyze
            track_dynamics: Whether to track dynamic development

        Returns:
            Dictionary with development analysis
        """
        development = {
            "motivic_development": [],
            "range_expansion": [],
            "complexity_curve": [],
            "climax_points": [],
            "thematic_return": [],
        }

        # Extract main melodic line
        melody = self._extract_main_melody(score)
        if not melody:
            return development

        # Divide into sections
        measures = list(melody.getElementsByClass(stream.Measure))
        section_size = max(4, len(measures) // 8)  # Analyze in sections

        # Track motivic development
        motives = await self.detect_motives(score)
        if motives.motives:
            development["motivic_development"] = motives.development_timeline

        # Analyze range expansion
        for i in range(0, len(measures), section_size):
            section = measures[i : i + section_size]
            notes = []
            for m in section:
                notes.extend(list(m.notes))

            if notes:
                pitch_range = {
                    "section": i // section_size + 1,
                    "measures": f"{i+1}-{min(i+section_size, len(measures))}",
                    "lowest": min(n.pitch.midi for n in notes),
                    "highest": max(n.pitch.midi for n in notes),
                    "range": max(n.pitch.midi for n in notes)
                    - min(n.pitch.midi for n in notes),
                }
                development["range_expansion"].append(pitch_range)

        # Calculate complexity curve
        window_size = 8
        for i in range(0, len(measures) - window_size + 1, 2):
            window_measures = measures[i : i + window_size]
            window_stream = stream.Stream()
            for m in window_measures:
                window_stream.append(m)

            contour = await self.analyze_melodic_contour(window_stream)
            complexity_point = {
                "measure": i + 1,
                "complexity": contour.complexity_score,
                "smoothness": contour.smoothness_score,
            }
            development["complexity_curve"].append(complexity_point)

        # Find climax points
        all_notes = [n for n in melody.flatten().notes if hasattr(n, 'pitch')]
        if all_notes:
            pitches = [n.pitch.midi for n in all_notes]
            climax_indices = signal.find_peaks(pitches, prominence=5)[0]

            for idx in climax_indices:
                climax = {
                    "note_index": int(idx),
                    "pitch": all_notes[idx].nameWithOctave,
                    "measure": self._get_measure_number(all_notes[idx], measures),
                    "dynamic": (
                        all_notes[idx].volume.velocity if track_dynamics else None
                    ),
                    "context": self._analyze_climax_context(all_notes, idx),
                }
                development["climax_points"].append(climax)

        # Detect thematic returns
        if motives.primary_motive:
            for occurrence in motives.primary_motive.occurrences:
                if occurrence["transformation"] == PatternTransformation.EXACT:
                    development["thematic_return"].append(
                        {
                            "measure": occurrence["measure"],
                            "confidence": occurrence["similarity"],
                        }
                    )

        return development

    # Helper methods
    def _classify_contour(self, contour_vector: List[int]) -> ContourType:
        """Classify overall contour shape"""
        if not contour_vector:
            return ContourType.STATIC

        ups = contour_vector.count(1)
        downs = contour_vector.count(-1)
        total = len(contour_vector)

        # Predominantly ascending or descending
        if ups > total * 0.7:
            return ContourType.ASCENDING
        elif downs > total * 0.7:
            return ContourType.DESCENDING

        # Check for arch patterns
        midpoint = len(contour_vector) // 2
        first_half = contour_vector[:midpoint]
        second_half = contour_vector[midpoint:]

        if (
            sum(first_half) > len(first_half) * 0.3
            and sum(second_half) < -len(second_half) * 0.3
        ):
            return ContourType.ARCH
        elif (
            sum(first_half) < -len(first_half) * 0.3
            and sum(second_half) > len(second_half) * 0.3
        ):
            return ContourType.INVERTED_ARCH

        # Check for wave pattern
        direction_changes = self._count_direction_changes(contour_vector)
        if direction_changes > len(contour_vector) * 0.3:
            return ContourType.WAVE

        # Default
        return ContourType.ZIGZAG

    def _count_direction_changes(self, contour: List[int]) -> int:
        """Count direction changes in contour"""
        changes = 0
        for i in range(len(contour) - 1):
            if contour[i] * contour[i + 1] < 0:  # Sign change
                changes += 1
        return changes

    def _calculate_prime_form(self, contour: List[int]) -> List[int]:
        """Calculate normalized contour representation"""
        if not contour:
            return []

        # Simplify to basic up/down/same
        prime = []
        run_length = 1
        current_direction = contour[0]

        for i in range(1, len(contour)):
            if contour[i] == current_direction:
                run_length += 1
            else:
                prime.append((current_direction, run_length))
                current_direction = contour[i]
                run_length = 1

        prime.append((current_direction, run_length))

        # Convert to simple list
        return [direction for direction, length in prime]

    def _find_arch_points(
        self, pitches: List[int], notes: List[Any]
    ) -> List[Dict[str, Any]]:
        """Find melodic arch points (peaks and valleys)"""
        arch_points = []

        # Find peaks
        peaks, properties = signal.find_peaks(pitches, prominence=2)
        for i, peak_idx in enumerate(peaks):
            arch_points.append(
                {
                    "type": "peak",
                    "index": int(peak_idx),
                    "pitch": notes[peak_idx].nameWithOctave,
                    "prominence": float(properties["prominences"][i]),
                }
            )

        # Find valleys (invert and find peaks)
        valleys, properties = signal.find_peaks([-p for p in pitches], prominence=2)
        for i, valley_idx in enumerate(valleys):
            arch_points.append(
                {
                    "type": "valley",
                    "index": int(valley_idx),
                    "pitch": notes[valley_idx].nameWithOctave,
                    "prominence": float(properties["prominences"][i]),
                }
            )

        # Sort by index
        arch_points.sort(key=lambda x: x["index"])

        return arch_points

    def _calculate_contour_complexity(self, contour: List[int]) -> float:
        """Calculate complexity of melodic contour"""
        if not contour:
            return 0.0

        # Factors: direction changes, variety, unpredictability
        direction_changes = self._count_direction_changes(contour)
        change_ratio = direction_changes / len(contour)

        # Entropy of contour
        counts = Counter(contour)
        total = len(contour)
        entropy = -sum(
            (count / total) * np.log2(count / total)
            for count in counts.values()
            if count > 0
        )

        # Normalize to 0-1
        max_entropy = np.log2(3)  # Three possible values: -1, 0, 1
        normalized_entropy = entropy / max_entropy

        # Combine factors
        complexity = (change_ratio + normalized_entropy) / 2

        return min(complexity, 1.0)

    def _calculate_melodic_smoothness(self, notes: List[Any]) -> float:
        """Calculate melodic smoothness based on interval sizes"""
        if len(notes) < 2:
            return 1.0

        intervals = []
        for i in range(len(notes) - 1):
            semitones = abs(notes[i].pitch.midi - notes[i + 1].pitch.midi)
            intervals.append(semitones)

        # Stepwise motion is smooth
        stepwise = sum(1 for i in intervals if i <= 2)
        smoothness = stepwise / len(intervals)

        # Penalize large leaps
        large_leaps = sum(1 for i in intervals if i > 7)
        penalty = large_leaps / len(intervals) * 0.3

        return max(0, smoothness - penalty)

    def _extract_melodic_parts(self, score: stream.Score) -> List[stream.Part]:
        """Extract parts likely to contain melodic material"""
        melodic_parts = []

        for part in score.parts:
            # Check if part is likely melodic (not just chords)
            notes = [n for n in part.flatten().notes if hasattr(n, 'pitch')]
            if not notes:
                continue

            # Calculate ratio of single notes to chords
            single_notes = sum(1 for n in notes if n.isNote)
            total_notes = len(notes)

            if single_notes / total_notes > 0.7:  # Mostly single notes
                melodic_parts.append(part)

        return melodic_parts

    def _create_motive(self, notes: List[Any], start_index: int) -> MelodicMotive:
        """Create a motive object from notes"""
        pitches = [n.pitch for n in notes]

        # Calculate intervals
        intervals = []
        for i in range(len(notes) - 1):
            intervals.append(notes[i + 1].pitch.midi - notes[i].pitch.midi)

        # Calculate rhythm ratios
        rhythm = []
        if notes[0].duration.quarterLength > 0:
            for n in notes:
                rhythm.append(
                    n.duration.quarterLength / notes[0].duration.quarterLength
                )
        else:
            rhythm = [1.0] * len(notes)

        # Calculate contour
        contour = []
        for i in range(len(notes) - 1):
            if notes[i].pitch < notes[i + 1].pitch:
                contour.append(1)
            elif notes[i].pitch > notes[i + 1].pitch:
                contour.append(-1)
            else:
                contour.append(0)

        # Get measure numbers
        start_measure = (
            notes[0].measureNumber if hasattr(notes[0], "measureNumber") else 0
        )
        end_measure = (
            notes[-1].measureNumber if hasattr(notes[-1], "measureNumber") else 0
        )

        return MelodicMotive(
            pitches=pitches,
            intervals=intervals,
            rhythm=rhythm,
            contour=contour,
            start_measure=start_measure,
            end_measure=end_measure,
        )

    def _find_motive_occurrences(
        self, motive: MelodicMotive, all_notes: List[Any], start_from: int = 0
    ) -> List[Dict[str, Any]]:
        """Find all occurrences of a motive with transformations"""
        occurrences = []
        motive_length = len(motive.pitches)

        for i in range(start_from, len(all_notes) - motive_length + 1):
            candidate_notes = all_notes[i : i + motive_length]

            # Check each transformation type
            for transformation in PatternTransformation:
                similarity = self._calculate_motive_similarity(
                    motive, candidate_notes, transformation
                )

                if similarity >= self.similarity_threshold:
                    occurrences.append(
                        {
                            "start_index": i,
                            "measure": (
                                candidate_notes[0].measureNumber
                                if hasattr(candidate_notes[0], "measureNumber")
                                else 0
                            ),
                            "transformation": transformation,
                            "similarity": similarity,
                        }
                    )
                    break  # Don't count same occurrence multiple times

        return occurrences

    def _calculate_motive_similarity(
        self,
        motive: MelodicMotive,
        candidate_notes: List[Any],
        transformation: PatternTransformation,
    ) -> float:
        """Calculate similarity between motive and candidate"""
        if len(candidate_notes) != len(motive.pitches):
            return 0.0

        # Get candidate intervals
        candidate_intervals = []
        for i in range(len(candidate_notes) - 1):
            candidate_intervals.append(
                candidate_notes[i + 1].pitch.midi - candidate_notes[i].pitch.midi
            )

        # Apply transformation to motive intervals
        transformed_intervals = self._transform_intervals(
            motive.intervals, transformation
        )

        if transformation == PatternTransformation.TRANSPOSITION:
            # Allow any transposition
            if len(motive.intervals) == len(candidate_intervals):
                # Check if interval patterns match
                return 1.0 if motive.intervals == candidate_intervals else 0.0

        # Calculate interval similarity
        interval_similarity = self._compare_interval_sequences(
            transformed_intervals, candidate_intervals
        )

        # Calculate rhythm similarity
        candidate_rhythm = []
        if candidate_notes[0].duration.quarterLength > 0:
            for n in candidate_notes:
                candidate_rhythm.append(
                    n.duration.quarterLength / candidate_notes[0].duration.quarterLength
                )
        else:
            candidate_rhythm = [1.0] * len(candidate_notes)

        rhythm_similarity = self._compare_rhythm_sequences(
            motive.rhythm, candidate_rhythm, transformation
        )

        # Combine similarities
        return interval_similarity * 0.7 + rhythm_similarity * 0.3

    def _transform_intervals(
        self, intervals: List[int], transformation: PatternTransformation
    ) -> List[int]:
        """Apply transformation to interval sequence"""
        if transformation == PatternTransformation.EXACT:
            return intervals
        elif transformation == PatternTransformation.TRANSPOSITION:
            return intervals  # Intervals remain the same
        elif transformation == PatternTransformation.INVERSION:
            return [-i for i in intervals]
        elif transformation == PatternTransformation.RETROGRADE:
            return intervals[::-1]
        elif transformation == PatternTransformation.RETROGRADE_INVERSION:
            return [-i for i in intervals[::-1]]
        else:
            return intervals

    def _compare_interval_sequences(self, seq1: List[int], seq2: List[int]) -> float:
        """Compare two interval sequences"""
        if len(seq1) != len(seq2):
            return 0.0

        if not seq1:
            return 1.0

        # Exact match
        if seq1 == seq2:
            return 1.0

        # Calculate similarity
        matches = sum(1 for a, b in zip(seq1, seq2) if a == b)
        return matches / len(seq1)

    def _compare_rhythm_sequences(
        self,
        rhythm1: List[float],
        rhythm2: List[float],
        transformation: PatternTransformation,
    ) -> float:
        """Compare two rhythm sequences"""
        if len(rhythm1) != len(rhythm2):
            return 0.0

        # Handle augmentation/diminution
        if transformation == PatternTransformation.AUGMENTATION:
            # Check if rhythm2 is proportionally longer
            if rhythm2[0] > 0:
                ratio = rhythm1[0] / rhythm2[0]
                scaled_rhythm2 = [r * ratio for r in rhythm2]
                return self._calculate_rhythm_similarity(rhythm1, scaled_rhythm2)
        elif transformation == PatternTransformation.DIMINUTION:
            # Check if rhythm2 is proportionally shorter
            if rhythm1[0] > 0:
                ratio = rhythm2[0] / rhythm1[0]
                scaled_rhythm1 = [r * ratio for r in rhythm1]
                return self._calculate_rhythm_similarity(scaled_rhythm1, rhythm2)

        return self._calculate_rhythm_similarity(rhythm1, rhythm2)

    def _calculate_rhythm_similarity(
        self, rhythm1: List[float], rhythm2: List[float]
    ) -> float:
        """Calculate similarity between rhythm patterns"""
        if len(rhythm1) != len(rhythm2):
            return 0.0

        # Calculate normalized difference
        differences = []
        for r1, r2 in zip(rhythm1, rhythm2):
            if max(r1, r2) > 0:
                diff = abs(r1 - r2) / max(r1, r2)
                differences.append(diff)

        if not differences:
            return 1.0

        avg_difference = np.mean(differences)
        return max(0, 1 - avg_difference)

    def _calculate_motive_importance(
        self, motive: MelodicMotive, occurrence_count: int
    ) -> float:
        """Calculate importance score for a motive"""
        # Factors: occurrences, length, placement, transformations

        # Occurrence factor
        occurrence_score = min(occurrence_count / 5, 1.0)

        # Length factor (prefer medium length motives)
        length_score = 1.0
        if len(motive.pitches) < 3:
            length_score = 0.5
        elif len(motive.pitches) > 8:
            length_score = 0.7

        # Placement factor (early motives more important)
        placement_score = 1.0 - (motive.start_measure / 100)
        placement_score = max(0.3, placement_score)

        # Transformation variety
        transformation_types = set()
        for occ in motive.occurrences:
            transformation_types.add(occ["transformation"])

        transformation_score = len(transformation_types) / len(PatternTransformation)

        # Combine scores
        importance = (
            occurrence_score * 0.4
            + length_score * 0.2
            + placement_score * 0.2
            + transformation_score * 0.2
        )

        return importance

    def _filter_overlapping_motives(
        self, motives: List[MelodicMotive]
    ) -> List[MelodicMotive]:
        """Remove overlapping motives, keeping the most important"""
        if not motives:
            return []

        # Sort by importance
        motives.sort(key=lambda m: m.importance_score, reverse=True)

        filtered = []
        used_positions = set()

        for motive in motives:
            # Check if this motive overlaps with already selected ones
            motive_positions = set()
            for i in range(len(motive.pitches)):
                motive_positions.add(motive.start_measure + i)

            for occ in motive.occurrences:
                for i in range(len(motive.pitches)):
                    motive_positions.add(occ["start_index"] + i)

            # If minimal overlap, keep the motive
            overlap = len(motive_positions & used_positions)
            if overlap < len(motive_positions) * 0.5:
                filtered.append(motive)
                used_positions.update(motive_positions)

        return filtered

    def _create_motive_hierarchy(
        self, motives: List[MelodicMotive]
    ) -> Dict[str, List[MelodicMotive]]:
        """Organize motives into hierarchical structure"""
        hierarchy = {"primary": [], "secondary": [], "fragmentary": []}

        if not motives:
            return hierarchy

        # Classify by importance
        for motive in motives:
            if motive.importance_score > 0.7:
                hierarchy["primary"].append(motive)
            elif motive.importance_score > 0.4:
                hierarchy["secondary"].append(motive)
            else:
                hierarchy["fragmentary"].append(motive)

        return hierarchy

    def _analyze_motive_development(
        self, motives: List[MelodicMotive], score: stream.Score
    ) -> List[Dict[str, Any]]:
        """Analyze how motives develop over time"""
        development = []

        # Get all motive occurrences with timestamps
        all_occurrences = []
        for motive_idx, motive in enumerate(motives):
            # Add original occurrence
            all_occurrences.append(
                {
                    "motive_index": motive_idx,
                    "measure": motive.start_measure,
                    "transformation": PatternTransformation.EXACT,
                    "is_primary": motive_idx == 0,
                }
            )

            # Add transformed occurrences
            for occ in motive.occurrences:
                all_occurrences.append(
                    {
                        "motive_index": motive_idx,
                        "measure": occ["measure"],
                        "transformation": occ["transformation"],
                        "is_primary": motive_idx == 0,
                    }
                )

        # Sort by measure
        all_occurrences.sort(key=lambda x: x["measure"])

        # Group into development sections
        if all_occurrences:
            total_measures = all_occurrences[-1]["measure"]
            section_size = max(8, total_measures // 4)

            for section_start in range(0, total_measures, section_size):
                section_end = section_start + section_size
                section_occurrences = [
                    occ
                    for occ in all_occurrences
                    if section_start <= occ["measure"] < section_end
                ]

                if section_occurrences:
                    # Analyze section characteristics
                    transformations = Counter(
                        occ["transformation"] for occ in section_occurrences
                    )

                    development.append(
                        {
                            "section": f"mm. {section_start+1}-{section_end}",
                            "occurrence_count": len(section_occurrences),
                            "primary_motive_count": sum(
                                1 for occ in section_occurrences if occ["is_primary"]
                            ),
                            "transformation_types": dict(transformations),
                            "development_intensity": len(section_occurrences)
                            / section_size,
                        }
                    )

        return development

    def _calculate_motivic_coherence(self, motives: List[MelodicMotive]) -> float:
        """Calculate overall motivic coherence of the piece"""
        if not motives:
            return 0.0

        # Factors: motive coverage, relationship between motives, development

        # Calculate coverage (how much of the piece uses motivic material)
        # This is simplified - would need full piece length
        coverage_score = min(len(motives) / 10, 1.0)

        # Calculate inter-motive relationships
        relationship_score = 0.0
        if len(motives) > 1:
            # Check if motives are related (share intervals, etc.)
            relationships = 0
            for i in range(len(motives)):
                for j in range(i + 1, len(motives)):
                    if self._are_motives_related(motives[i], motives[j]):
                        relationships += 1

            possible_relationships = len(motives) * (len(motives) - 1) / 2
            relationship_score = (
                relationships / possible_relationships
                if possible_relationships > 0
                else 0
            )

        # Calculate development score
        development_score = 0.0
        total_transformations = sum(len(m.occurrences) for m in motives)
        if total_transformations > 0:
            unique_transformations = set()
            for m in motives:
                for occ in m.occurrences:
                    unique_transformations.add(occ["transformation"])

            development_score = len(unique_transformations) / len(PatternTransformation)

        # Combine scores
        coherence = (
            coverage_score * 0.3 + relationship_score * 0.3 + development_score * 0.4
        )

        return coherence

    def _are_motives_related(
        self, motive1: MelodicMotive, motive2: MelodicMotive
    ) -> bool:
        """Check if two motives are related"""
        # Check if one is a fragment of the other
        if len(motive1.intervals) > len(motive2.intervals):
            longer, shorter = motive1.intervals, motive2.intervals
        else:
            longer, shorter = motive2.intervals, motive1.intervals

        # Check for subsequence
        for i in range(len(longer) - len(shorter) + 1):
            if longer[i : i + len(shorter)] == shorter:
                return True

        # Check for similar contour
        contour_similarity = SequenceMatcher(
            None, motive1.contour, motive2.contour
        ).ratio()

        return contour_similarity > 0.7

    def _get_interval_pattern(self, notes: List[Any]) -> List[int]:
        """Extract interval pattern from notes"""
        intervals = []
        for i in range(len(notes) - 1):
            intervals.append(notes[i + 1].pitch.midi - notes[i].pitch.midi)
        return intervals

    def _compare_patterns(
        self,
        pattern1_notes: List[Any],
        pattern2_notes: List[Any],
        transformation: PatternTransformation,
    ) -> Tuple[float, Optional[int]]:
        """Compare two patterns with given transformation"""
        if len(pattern1_notes) != len(pattern2_notes):
            return 0.0, None

        pattern1_intervals = self._get_interval_pattern(pattern1_notes)
        pattern2_intervals = self._get_interval_pattern(pattern2_notes)

        if transformation == PatternTransformation.EXACT:
            # Check if pitches match exactly
            if all(
                n1.pitch.midi == n2.pitch.midi
                for n1, n2 in zip(pattern1_notes, pattern2_notes)
            ):
                return 1.0, 0
            else:
                return 0.0, None

        elif transformation == PatternTransformation.TRANSPOSITION:
            # Check if intervals match
            if pattern1_intervals == pattern2_intervals:
                transposition = (
                    pattern2_notes[0].pitch.midi - pattern1_notes[0].pitch.midi
                )
                return 1.0, transposition
            else:
                return 0.0, None

        elif transformation == PatternTransformation.INVERSION:
            inverted = [-i for i in pattern1_intervals]
            if inverted == pattern2_intervals:
                return 1.0, None
            else:
                return 0.0, None

        elif transformation == PatternTransformation.RETROGRADE:
            if pattern1_intervals[::-1] == pattern2_intervals:
                return 1.0, None
            else:
                return 0.0, None

        # Add more transformations as needed
        return 0.0, None

    def _deduplicate_patterns(self, patterns: List[PatternMatch]) -> List[PatternMatch]:
        """Remove duplicate patterns"""
        unique_patterns = []
        seen = set()

        for pattern in patterns:
            # Create a hashable representation
            pattern_key = (tuple(pattern.pattern), pattern.transformation_type)

            if pattern_key not in seen:
                seen.add(pattern_key)
                unique_patterns.append(pattern)

        return unique_patterns

    def _get_interval_sequence(self, notes: List[Any]) -> List[int]:
        """Get sequence of intervals between consecutive notes"""
        intervals = []
        for i in range(len(notes) - 1):
            if notes[i].isNote and notes[i + 1].isNote:
                intervals.append(notes[i + 1].pitch.midi - notes[i].pitch.midi)
        return intervals

    def _analyze_scale_characteristics(
        self, pitch_classes: List[int], intervals: List[int]
    ) -> Dict[str, Any]:
        """Analyze scale and modal characteristics"""
        characteristics = {
            "pitch_class_set": list(set(pitch_classes)),
            "total_pitch_classes": len(set(pitch_classes)),
            "chromatic_notes": [],
            "interval_content": Counter(abs(i) % 12 for i in intervals),
            "most_common_intervals": [],
        }

        # Find most common intervals
        interval_counts = characteristics["interval_content"]
        characteristics["most_common_intervals"] = [
            interval for interval, _ in interval_counts.most_common(3)
        ]

        return characteristics

    def _score_western_classical(
        self, notes: List[Any], intervals: List[int], scale_chars: Dict[str, Any]
    ) -> float:
        """Score Western classical characteristics"""
        score = 0.0

        # Diatonic scale usage
        if scale_chars["total_pitch_classes"] <= 7:
            score += 0.3

        # Stepwise motion preference
        stepwise = sum(1 for i in intervals if abs(i) <= 2)
        if stepwise / len(intervals) > 0.6:
            score += 0.3

        # Regular phrase lengths
        # Would need phrase detection
        score += 0.2

        # Tonal resolution patterns
        # Simplified check for V-I motion
        for i in range(len(intervals) - 1):
            if intervals[i] == -7:  # Down a fifth
                score += 0.05

        return min(score, 1.0)

    def _score_jazz_style(
        self, notes: List[Any], intervals: List[int], scale_chars: Dict[str, Any]
    ) -> float:
        """Score jazz characteristics"""
        score = 0.0

        # Chromatic elements
        chromatic_intervals = sum(1 for i in intervals if abs(i) == 1)
        if chromatic_intervals / len(intervals) > 0.2:
            score += 0.3

        # Extended intervals (beyond octave)
        extended = sum(1 for i in intervals if abs(i) > 12)
        if extended > 0:
            score += 0.2

        # Blue notes (check for b3, b5, b7)
        # Simplified check
        if 3 in scale_chars["interval_content"] or 6 in scale_chars["interval_content"]:
            score += 0.2

        # Syncopation (would need rhythm analysis)
        score += 0.1

        return min(score, 1.0)

    def _score_raga_style(
        self, notes: List[Any], intervals: List[int], scale_chars: Dict[str, Any]
    ) -> float:
        """Score Indian raga characteristics"""
        score = 0.0

        # Specific interval patterns (e.g., augmented seconds)
        if 3 in scale_chars["interval_content"]:  # Minor third
            score += 0.2

        # Melodic phrases returning to tonic
        # Simplified check
        first_pitch = notes[0].pitch.pitchClass
        returns_to_tonic = sum(1 for n in notes if n.pitch.pitchClass == first_pitch)
        if returns_to_tonic / len(notes) > 0.15:
            score += 0.3

        # Ornamentation patterns
        small_intervals = sum(1 for i in intervals if abs(i) == 1)
        if small_intervals / len(intervals) > 0.3:
            score += 0.2

        return min(score, 1.0)

    def _score_maqam_style(
        self, notes: List[Any], intervals: List[int], scale_chars: Dict[str, Any]
    ) -> float:
        """Score Arabic maqam characteristics"""
        score = 0.0

        # Quarter-tone intervals (would need microtonal analysis)
        # Check for augmented seconds
        aug_seconds = sum(1 for i in intervals if abs(i) == 3)
        if aug_seconds > 0:
            score += 0.3

        # Specific interval patterns
        if scale_chars["total_pitch_classes"] in [7, 8]:
            score += 0.2

        # Melodic emphasis patterns
        # Simplified
        score += 0.2

        return min(score, 1.0)

    def _score_pentatonic_style(self, pitch_classes: List[int]) -> float:
        """Score pentatonic characteristics"""
        unique_pcs = set(pitch_classes)

        # Five-note scale
        if len(unique_pcs) == 5:
            # Check for pentatonic intervals (no semitones)
            # This is simplified
            return 0.8
        elif len(unique_pcs) <= 6:
            return 0.4

        return 0.0

    def _score_blues_style(
        self, notes: List[Any], intervals: List[int], pitch_classes: List[int]
    ) -> float:
        """Score blues characteristics"""
        score = 0.0

        # Blue notes (b3, b5, b7)
        # Simplified check for minor thirds and tritones
        if 3 in Counter(abs(i) % 12 for i in intervals):
            score += 0.3
        if 6 in Counter(abs(i) % 12 for i in intervals):
            score += 0.3

        # Bent notes (would need microtonal analysis)
        score += 0.1

        # Call and response patterns (would need phrase analysis)
        score += 0.1

        return min(score, 1.0)

    def _detect_ornamentations(self, notes: List[Any]) -> List[Dict[str, Any]]:
        """Detect melodic ornamentations"""
        ornamentations = []

        for i in range(1, len(notes) - 1):
            # Check for grace notes (very short durations)
            if notes[i].duration.quarterLength < 0.25:
                # Potential ornament
                ornament_type = "grace_note"

                # Check specific patterns
                if (
                    notes[i - 1].pitch == notes[i + 1].pitch
                    and abs(notes[i].pitch.midi - notes[i - 1].pitch.midi) <= 2
                ):
                    ornament_type = (
                        "mordent"
                        if notes[i].pitch > notes[i - 1].pitch
                        else "lower_mordent"
                    )

                ornamentations.append(
                    {
                        "index": i,
                        "type": ornament_type,
                        "pitch": notes[i].nameWithOctave,
                        "duration": notes[i].duration.quarterLength,
                    }
                )

        # Check for trills (rapid alternation)
        for i in range(len(notes) - 3):
            if (
                notes[i].pitch == notes[i + 2].pitch
                and notes[i + 1].pitch == notes[i + 3].pitch
                and abs(notes[i].pitch.midi - notes[i + 1].pitch.midi) <= 2
            ):
                ornamentations.append(
                    {
                        "index": i,
                        "type": "trill",
                        "pitches": [
                            notes[i].nameWithOctave,
                            notes[i + 1].nameWithOctave,
                        ],
                        "duration": sum(
                            n.duration.quarterLength for n in notes[i : i + 4]
                        ),
                    }
                )

        return ornamentations

    def _detect_microtones(self, notes: List[Any]) -> List[Dict[str, Any]]:
        """Detect microtonal inflections"""
        microtones = []

        for i, note in enumerate(notes):
            # Check if pitch has microtonal alteration
            if hasattr(note.pitch, "microtone") and note.pitch.microtone:
                microtones.append(
                    {
                        "index": i,
                        "pitch": note.nameWithOctave,
                        "cents_deviation": note.pitch.microtone.cents,
                        "direction": (
                            "sharp" if note.pitch.microtone.cents > 0 else "flat"
                        ),
                    }
                )

        return microtones

    def _extract_cultural_markers(
        self,
        notes: List[Any],
        intervals: List[int],
        primary_style: Optional[MelodicStyle],
    ) -> Dict[str, Any]:
        """Extract culture-specific markers"""
        markers = {
            "characteristic_intervals": [],
            "cadential_patterns": [],
            "rhythmic_patterns": [],
            "range_and_tessitura": {},
        }

        # Find characteristic intervals
        interval_counts = Counter(abs(i) for i in intervals)
        markers["characteristic_intervals"] = [
            interval for interval, count in interval_counts.most_common(5) if count >= 2
        ]

        # Analyze range and tessitura
        if notes:
            pitches = [n.pitch.midi for n in notes]
            markers["range_and_tessitura"] = {
                "lowest": min(pitches),
                "highest": max(pitches),
                "range": max(pitches) - min(pitches),
                "tessitura_center": np.median(pitches),
            }

        return markers

    def _calculate_pairwise_similarity(
        self, melody1: stream.Stream, melody2: stream.Stream, method: str
    ) -> float:
        """Calculate similarity between two melodies"""
        notes1 = [n for n in melody1.flatten().notes if hasattr(n, 'pitch')]
        notes2 = [n for n in melody2.flatten().notes if hasattr(n, 'pitch')]

        if not notes1 or not notes2:
            return 0.0

        if method == "interval":
            intervals1 = self._get_interval_sequence(notes1)
            intervals2 = self._get_interval_sequence(notes2)

            # Use sequence matching
            matcher = SequenceMatcher(None, intervals1, intervals2)
            return matcher.ratio()

        elif method == "contour":
            contour1 = self._get_contour_vector(notes1)
            contour2 = self._get_contour_vector(notes2)

            matcher = SequenceMatcher(None, contour1, contour2)
            return matcher.ratio()

        elif method == "rhythm":
            rhythm1 = [n.duration.quarterLength for n in notes1]
            rhythm2 = [n.duration.quarterLength for n in notes2]

            # Normalize rhythms
            if sum(rhythm1) > 0:
                rhythm1 = [r / sum(rhythm1) for r in rhythm1]
            if sum(rhythm2) > 0:
                rhythm2 = [r / sum(rhythm2) for r in rhythm2]

            matcher = SequenceMatcher(None, rhythm1, rhythm2)
            return matcher.ratio()

        elif method == "combined":
            # Combine multiple similarity measures
            interval_sim = self._calculate_pairwise_similarity(
                melody1, melody2, "interval"
            )
            contour_sim = self._calculate_pairwise_similarity(
                melody1, melody2, "contour"
            )
            rhythm_sim = self._calculate_pairwise_similarity(melody1, melody2, "rhythm")

            return interval_sim * 0.5 + contour_sim * 0.3 + rhythm_sim * 0.2

        return 0.0

    def _get_contour_vector(self, notes: List[Any]) -> List[int]:
        """Get contour vector from notes"""
        contour = []
        for i in range(len(notes) - 1):
            if notes[i].pitch < notes[i + 1].pitch:
                contour.append(1)
            elif notes[i].pitch > notes[i + 1].pitch:
                contour.append(-1)
            else:
                contour.append(0)
        return contour

    def _detect_theme_variations(
        self, melodies: List[stream.Stream], similarity_matrix: np.ndarray
    ) -> List[Dict[str, Any]]:
        """Detect theme and variation relationships"""
        variations = []
        n_melodies = len(melodies)

        # Find potential themes (melodies with high similarity to others)
        theme_scores = np.mean(similarity_matrix, axis=1)
        potential_themes = np.argsort(theme_scores)[-3:][::-1]  # Top 3

        for theme_idx in potential_themes:
            # Find variations of this theme
            variation_indices = np.where(
                (similarity_matrix[theme_idx] > 0.6)
                & (similarity_matrix[theme_idx] < 0.95)  # Not exact copy
            )[0]

            if len(variation_indices) > 0:
                variations.append(
                    {
                        "theme_index": int(theme_idx),
                        "variation_indices": variation_indices.tolist(),
                        "similarity_scores": similarity_matrix[
                            theme_idx, variation_indices
                        ].tolist(),
                        "variation_types": self._identify_variation_types(
                            melodies[theme_idx],
                            [melodies[i] for i in variation_indices],
                        ),
                    }
                )

        return variations

    def _identify_variation_types(
        self, theme: stream.Stream, variations: List[stream.Stream]
    ) -> List[str]:
        """Identify types of variations applied"""
        variation_types = []

        theme_notes = [n for n in theme.flatten().notes if hasattr(n, 'pitch')]

        for var in variations:
            var_notes = [n for n in var.flatten().notes if hasattr(n, 'pitch')]

            # Check for different variation techniques
            types = []

            # Rhythmic variation
            if len(theme_notes) == len(var_notes):
                theme_rhythms = [n.duration.quarterLength for n in theme_notes]
                var_rhythms = [n.duration.quarterLength for n in var_notes]
                if theme_rhythms != var_rhythms:
                    types.append("rhythmic")

            # Melodic ornamentation
            if len(var_notes) > len(theme_notes):
                types.append("ornamentation")
            elif len(var_notes) < len(theme_notes):
                types.append("reduction")

            # Intervallic variation
            theme_intervals = self._get_interval_sequence(theme_notes)
            var_intervals = self._get_interval_sequence(var_notes)
            if theme_intervals != var_intervals:
                types.append("intervallic")

            variation_types.append(types if types else ["subtle"])

        return variation_types

    def _cluster_melodies(self, similarity_matrix: np.ndarray) -> List[List[int]]:
        """Cluster melodies into families based on similarity"""
        n_melodies = similarity_matrix.shape[0]

        # Simple clustering based on similarity threshold
        threshold = 0.7
        families = []
        assigned = set()

        for i in range(n_melodies):
            if i in assigned:
                continue

            # Start new family
            family = [i]
            assigned.add(i)

            # Find similar melodies
            for j in range(i + 1, n_melodies):
                if j not in assigned and similarity_matrix[i, j] > threshold:
                    family.append(j)
                    assigned.add(j)

            families.append(family)

        return families

    def _extract_main_melody(self, score: stream.Score) -> Optional[stream.Stream]:
        """Extract the main melodic line from a score"""
        # Try to find the top part or most melodic part
        if not score.parts:
            return None

        # Simple heuristic: take the highest part
        highest_part = None
        highest_avg_pitch = -1

        for part in score.parts:
            notes = [n for n in part.flatten().notes if hasattr(n, "pitch")]
            if notes:
                # Check if mostly single notes (not chords)
                single_notes = [n for n in notes if n.isNote]
                if len(single_notes) / len(notes) > 0.7:
                    avg_pitch = np.mean([n.pitch.midi for n in single_notes])
                    if avg_pitch > highest_avg_pitch:
                        highest_avg_pitch = avg_pitch
                        highest_part = part

        return highest_part

    def _get_measure_number(self, note_obj: note.Note, measures: List) -> int:
        """Get measure number for a note"""
        note_offset = note_obj.offset
        for i, measure in enumerate(measures):
            if measure.offset <= note_offset < measure.offset + measure.quarterLength:
                return i + 1
        return 0

    def _analyze_climax_context(
        self, notes: List[Any], climax_index: int
    ) -> Dict[str, Any]:
        """Analyze the context around a melodic climax"""
        context = {"approach": "unknown", "departure": "unknown", "local_range": 0}

        # Analyze approach (5 notes before)
        if climax_index >= 5:
            approach_notes = notes[climax_index - 5 : climax_index]
            approach_contour = self._get_contour_vector(approach_notes)
            if sum(approach_contour) > 2:
                context["approach"] = "ascending"
            elif sum(approach_contour) < -2:
                context["approach"] = "descending"
            else:
                context["approach"] = "mixed"

        # Analyze departure (5 notes after)
        if climax_index < len(notes) - 5:
            departure_notes = notes[climax_index : climax_index + 5]
            departure_contour = self._get_contour_vector(departure_notes)
            if sum(departure_contour) > 2:
                context["departure"] = "ascending"
            elif sum(departure_contour) < -2:
                context["departure"] = "descending"
            else:
                context["departure"] = "mixed"

        # Local range
        start = max(0, climax_index - 8)
        end = min(len(notes), climax_index + 8)
        local_notes = notes[start:end]
        if local_notes:
            local_pitches = [n.pitch.midi for n in local_notes]
            context["local_range"] = max(local_pitches) - min(local_pitches)

        return context
