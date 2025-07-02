"""
Voice Leading and Counterpoint Analysis Engine for music21 MCP server
Implements species counterpoint checking, Bach chorale analysis,
voice independence metrics, and fugue analysis.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from music21 import chord, interval, note, roman, stream, voiceLeading


class CounterpointSpecies(Enum):
    """Types of species counterpoint"""

    FIRST = "first_species"  # Note against note
    SECOND = "second_species"  # Two notes against one
    THIRD = "third_species"  # Four notes against one
    FOURTH = "fourth_species"  # Syncopation
    FIFTH = "fifth_species"  # Florid counterpoint
    FREE = "free_counterpoint"


class VoiceMotion(Enum):
    """Types of voice motion"""

    PARALLEL = "parallel"
    SIMILAR = "similar"
    CONTRARY = "contrary"
    OBLIQUE = "oblique"
    NONE = "none"


class CounterpointError(Enum):
    """Types of counterpoint errors"""

    PARALLEL_PERFECT = "parallel_perfect_interval"
    DIRECT_PERFECT = "direct_perfect_interval"
    DISSONANCE_UNPREPARED = "unprepared_dissonance"
    DISSONANCE_UNRESOLVED = "unresolved_dissonance"
    VOICE_CROSSING = "voice_crossing"
    VOICE_OVERLAP = "voice_overlap"
    LARGE_LEAP = "large_leap_unprepared"
    TRITONE_LEAP = "tritone_leap"
    POOR_CADENCE = "poor_cadence"
    WEAK_BEAT_DISSONANCE = "weak_beat_dissonance"
    REPEATED_NOTE = "excessive_repetition"
    STATIC_VOICE = "static_voice"
    POOR_CLIMAX = "poor_climax_placement"


class FugueElement(Enum):
    """Elements of fugal writing"""

    SUBJECT = "subject"
    ANSWER = "answer"
    COUNTERSUBJECT = "countersubject"
    EPISODE = "episode"
    STRETTO = "stretto"
    AUGMENTATION = "augmentation"
    DIMINUTION = "diminution"
    INVERSION = "inversion"
    RETROGRADE = "retrograde"
    PEDAL = "pedal_point"
    CODETTA = "codetta"
    CODA = "coda"


@dataclass
class SpeciesAnalysis:
    """Results of species counterpoint analysis"""

    species_type: CounterpointSpecies
    errors: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[Dict[str, Any]] = field(default_factory=list)
    voice_leading_score: float = 0.0
    melodic_quality_score: float = 0.0
    harmonic_quality_score: float = 0.0
    overall_score: float = 0.0
    interval_usage: Dict[str, int] = field(default_factory=dict)
    cadences: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ChoraleAnalysis:
    """Results of Bach chorale style analysis"""

    voice_leading_errors: List[Dict[str, Any]] = field(default_factory=list)
    voice_ranges: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    chord_progressions: List[str] = field(default_factory=list)
    non_chord_tones: List[Dict[str, Any]] = field(default_factory=list)
    cadence_types: List[Dict[str, Any]] = field(default_factory=list)
    texture_analysis: Dict[str, Any] = field(default_factory=dict)
    style_conformance_score: float = 0.0


@dataclass
class VoiceIndependence:
    """Results of voice independence analysis"""

    rhythmic_independence: float = 0.0
    melodic_independence: float = 0.0
    contour_independence: float = 0.0
    onset_independence: float = 0.0
    voice_crossings: List[Dict[str, Any]] = field(default_factory=list)
    voice_overlaps: List[Dict[str, Any]] = field(default_factory=list)
    parallel_motion_ratio: float = 0.0
    contrary_motion_ratio: float = 0.0
    overall_independence: float = 0.0


@dataclass
class FugueAnalysis:
    """Results of fugue analysis"""

    subject: Optional[Dict[str, Any]] = None
    answer: Optional[Dict[str, Any]] = None
    countersubjects: List[Dict[str, Any]] = field(default_factory=list)
    episodes: List[Dict[str, Any]] = field(default_factory=list)
    entries: List[Dict[str, Any]] = field(default_factory=list)
    strettos: List[Dict[str, Any]] = field(default_factory=list)
    key_plan: List[Dict[str, Any]] = field(default_factory=list)
    formal_structure: str = ""
    contrapuntal_devices: List[Dict[str, Any]] = field(default_factory=list)


class CounterpointAnalyzer:
    """Advanced counterpoint and voice leading analysis engine"""

    def __init__(self) -> None:
        self.consonant_intervals = {
            1,
            3,
            4,
            5,
            6,
            8,
        }  # Unison, 3rd, 4th, 5th, 6th, octave
        self.perfect_intervals = {1, 5, 8}  # Unison, fifth, octave
        self.dissonant_intervals = {2, 7}  # 2nd, 7th

    async def analyze_species_counterpoint(
        self,
        cantus_firmus: stream.Stream,
        counterpoint_voice: stream.Stream,
        species: CounterpointSpecies = CounterpointSpecies.FIRST,
    ) -> SpeciesAnalysis:
        """
        Analyze species counterpoint exercises.

        Args:
            cantus_firmus: The given melody (cantus firmus)
            counterpoint_voice: The counterpoint line
            species: Type of species counterpoint

        Returns:
            SpeciesAnalysis object
        """
        result = SpeciesAnalysis(species_type=species)

        # Align voices (filter to only Note objects)
        cf_notes = [n for n in cantus_firmus.flatten().notes if hasattr(n, "pitch")]
        cp_notes = [
            n for n in counterpoint_voice.flatten().notes if hasattr(n, "pitch")
        ]

        if not cf_notes or not cp_notes:
            return result

        # Check species-specific requirements
        if species == CounterpointSpecies.FIRST:
            result = await self._analyze_first_species(cf_notes, cp_notes, result)
        elif species == CounterpointSpecies.SECOND:
            result = await self._analyze_second_species(cf_notes, cp_notes, result)
        elif species == CounterpointSpecies.THIRD:
            result = await self._analyze_third_species(cf_notes, cp_notes, result)
        elif species == CounterpointSpecies.FOURTH:
            result = await self._analyze_fourth_species(cf_notes, cp_notes, result)
        elif species == CounterpointSpecies.FIFTH:
            result = await self._analyze_fifth_species(cf_notes, cp_notes, result)
        else:
            result = await self._analyze_free_counterpoint(cf_notes, cp_notes, result)

        # Common checks for all species
        result = self._check_voice_leading(cf_notes, cp_notes, result)
        result = self._check_melodic_quality(cp_notes, result)
        result = self._check_harmonic_intervals(cf_notes, cp_notes, result)
        result = self._check_cadences(cf_notes, cp_notes, result)

        # Calculate overall score
        result.overall_score = self._calculate_counterpoint_score(result)

        return result

    async def analyze_bach_chorale(self, chorale: stream.Score) -> ChoraleAnalysis:
        """
        Analyze Bach chorale style voice leading and harmony.

        Args:
            chorale: Four-part chorale score

        Returns:
            ChoraleAnalysis object
        """
        result = ChoraleAnalysis()

        # Extract parts (SATB)
        parts = list(chorale.parts)
        if len(parts) < 4:
            # Try to extract from chords
            parts = self._extract_voices_from_chords(chorale)

        if len(parts) < 2:
            return result

        # Analyze voice ranges
        result.voice_ranges = self._analyze_voice_ranges(parts)

        # Check voice leading between all pairs
        for i in range(len(parts)):
            for j in range(i + 1, len(parts)):
                errors = self._check_bach_voice_leading(parts[i], parts[j], i, j)
                result.voice_leading_errors.extend(errors)

        # Analyze harmony
        result = await self._analyze_chorale_harmony(chorale, result)

        # Identify non-chord tones
        result.non_chord_tones = self._identify_non_chord_tones(chorale)

        # Analyze cadences
        result.cadence_types = self._analyze_chorale_cadences(chorale)

        # Analyze texture
        result.texture_analysis = self._analyze_chorale_texture(parts)

        # Calculate style conformance
        result.style_conformance_score = self._calculate_bach_style_score(result)

        return result

    async def analyze_voice_independence(
        self, score: stream.Score
    ) -> VoiceIndependence:
        """
        Analyze independence between voices.

        Args:
            score: Multi-voice score

        Returns:
            VoiceIndependence object
        """
        result = VoiceIndependence()

        parts = list(score.parts)
        if len(parts) < 2:
            return result

        # Analyze rhythmic independence
        result.rhythmic_independence = self._calculate_rhythmic_independence(parts)

        # Analyze melodic independence
        result.melodic_independence = self._calculate_melodic_independence(parts)

        # Analyze contour independence
        result.contour_independence = self._calculate_contour_independence(parts)

        # Analyze onset independence
        result.onset_independence = self._calculate_onset_independence(parts)

        # Find voice crossings and overlaps
        for i in range(len(parts)):
            for j in range(i + 1, len(parts)):
                crossings = self._find_voice_crossings(parts[i], parts[j])
                overlaps = self._find_voice_overlaps(parts[i], parts[j])

                for crossing in crossings:
                    crossing["voices"] = f"{i+1}-{j+1}"
                    result.voice_crossings.append(crossing)

                for overlap in overlaps:
                    overlap["voices"] = f"{i+1}-{j+1}"
                    result.voice_overlaps.append(overlap)

        # Calculate motion ratios
        motion_stats = self._calculate_motion_statistics(parts)
        result.parallel_motion_ratio = motion_stats["parallel"]
        result.contrary_motion_ratio = motion_stats["contrary"]

        # Calculate overall independence
        result.overall_independence = np.mean(
            [
                result.rhythmic_independence,
                result.melodic_independence,
                result.contour_independence,
                result.onset_independence,
                1 - result.parallel_motion_ratio,
            ]
        )

        return result

    async def analyze_fugue(
        self, fugue: stream.Score, expected_voices: int = None
    ) -> FugueAnalysis:
        """
        Analyze fugal writing and structure.

        Args:
            fugue: Fugue score
            expected_voices: Number of voices (auto-detected if None)

        Returns:
            FugueAnalysis object
        """
        result = FugueAnalysis()

        # Detect number of voices if not specified
        if expected_voices is None:
            expected_voices = len(fugue.parts)

        # Find the subject
        result.subject = await self._find_fugue_subject(fugue)

        if result.subject:
            # Find the answer
            result.answer = await self._find_fugue_answer(fugue, result.subject)

            # Find all subject/answer entries
            result.entries = self._find_all_entries(
                fugue, result.subject, result.answer
            )

            # Find countersubjects
            result.countersubjects = await self._find_countersubjects(
                fugue, result.subject, result.entries
            )

            # Find episodes
            result.episodes = self._find_episodes(fugue, result.entries)

            # Find strettos
            result.strettos = self._find_strettos(result.entries)

            # Analyze key plan
            result.key_plan = await self._analyze_fugue_key_plan(fugue, result.entries)

            # Find contrapuntal devices
            result.contrapuntal_devices = self._find_contrapuntal_devices(
                fugue, result.subject
            )

            # Determine formal structure
            result.formal_structure = self._determine_fugue_form(result)

        return result

    # Helper methods for species counterpoint
    async def _analyze_first_species(
        self,
        cf_notes: List[Any],
        cp_notes: List[Any],
        result: SpeciesAnalysis,
    ) -> SpeciesAnalysis:
        """Analyze first species (note against note)"""
        if len(cf_notes) != len(cp_notes):
            result.errors.append(
                {
                    "type": "length_mismatch",
                    "message": "First species requires equal number of notes",
                }
            )
            return result

        # Check each interval
        for i in range(len(cf_notes)):
            cf_pitch = cf_notes[i].pitch
            cp_pitch = cp_notes[i].pitch

            # Calculate interval
            interv = interval.Interval(cf_pitch, cp_pitch)
            generic = abs(interv.generic.value)

            # Track interval usage
            interval_name = interv.simpleName
            result.interval_usage[interval_name] = (
                result.interval_usage.get(interval_name, 0) + 1
            )

            # Check for dissonances (only consonances allowed in first species)
            if generic not in self.consonant_intervals:
                result.errors.append(
                    {
                        "type": CounterpointError.DISSONANCE_UNPREPARED,
                        "measure": i + 1,
                        "interval": interval_name,
                        "message": f"Dissonant interval {interval_name} at position {i+1}",
                    }
                )

        return result

    async def _analyze_second_species(
        self,
        cf_notes: List[Any],
        cp_notes: List[Any],
        result: SpeciesAnalysis,
    ) -> SpeciesAnalysis:
        """Analyze second species (two notes against one)"""
        if len(cp_notes) != len(cf_notes) * 2:
            result.warnings.append(
                {
                    "type": "length_mismatch",
                    "message": "Second species typically has 2:1 note ratio",
                }
            )

        # Check strong and weak beats
        cp_index = 0
        for cf_index, cf_note in enumerate(cf_notes):
            # Strong beat
            if cp_index < len(cp_notes):
                interv = interval.Interval(cf_note.pitch, cp_notes[cp_index].pitch)
                if abs(interv.generic.value) not in self.consonant_intervals:
                    result.errors.append(
                        {
                            "type": CounterpointError.DISSONANCE_UNPREPARED,
                            "beat": "strong",
                            "measure": cf_index + 1,
                            "message": "Dissonance on strong beat",
                        }
                    )
                cp_index += 1

            # Weak beat (passing tones allowed)
            if cp_index < len(cp_notes):
                interv = interval.Interval(cf_note.pitch, cp_notes[cp_index].pitch)
                if abs(interv.generic.value) not in self.consonant_intervals:
                    # Check if it's a valid passing tone
                    if not self._is_valid_passing_tone(cp_notes, cp_index):
                        result.warnings.append(
                            {
                                "type": CounterpointError.WEAK_BEAT_DISSONANCE,
                                "beat": "weak",
                                "measure": cf_index + 1,
                                "message": "Non-passing dissonance on weak beat",
                            }
                        )
                cp_index += 1

        return result

    async def _analyze_third_species(
        self,
        cf_notes: List[Any],
        cp_notes: List[Any],
        result: SpeciesAnalysis,
    ) -> SpeciesAnalysis:
        """Analyze third species (four notes against one)"""
        if len(cp_notes) != len(cf_notes) * 4:
            result.warnings.append(
                {
                    "type": "length_mismatch",
                    "message": "Third species typically has 4:1 note ratio",
                }
            )

        # Similar to second species but with more elaborate rules
        # Check first beat consonance, allow passing/neighbor tones
        cp_index = 0
        for cf_index, cf_note in enumerate(cf_notes):
            # First beat must be consonant
            if cp_index < len(cp_notes):
                interv = interval.Interval(cf_note.pitch, cp_notes[cp_index].pitch)
                if abs(interv.generic.value) not in self.consonant_intervals:
                    result.errors.append(
                        {
                            "type": CounterpointError.DISSONANCE_UNPREPARED,
                            "beat": 1,
                            "measure": cf_index + 1,
                        }
                    )

            # Other beats can have passing/neighbor tones
            for beat in range(1, 4):
                cp_index += 1
                if cp_index < len(cp_notes):
                    # Check melodic flow
                    if cp_index >= 2:
                        leap = abs(
                            cp_notes[cp_index].pitch.midi
                            - cp_notes[cp_index - 1].pitch.midi
                        )
                        if leap > 4 and beat != 1:  # Leap larger than M3
                            result.warnings.append(
                                {
                                    "type": "large_leap_weak_beat",
                                    "measure": cf_index + 1,
                                    "beat": beat + 1,
                                }
                            )

        return result

    async def _analyze_fourth_species(
        self,
        cf_notes: List[Any],
        cp_notes: List[Any],
        result: SpeciesAnalysis,
    ) -> SpeciesAnalysis:
        """Analyze fourth species (syncopation)"""
        # Check for proper suspensions
        for i in range(1, len(cp_notes)):
            # Suspension pattern: preparation - suspension - resolution
            if i < len(cf_notes):
                # Check suspension (dissonance on strong beat)
                susp_interval = interval.Interval(cf_notes[i].pitch, cp_notes[i].pitch)

                if abs(susp_interval.generic.value) in self.dissonant_intervals:
                    # Must resolve down by step
                    if i + 1 < len(cp_notes):
                        resolution = cp_notes[i + 1].pitch.midi - cp_notes[i].pitch.midi
                        if resolution != -1 and resolution != -2:
                            result.errors.append(
                                {
                                    "type": CounterpointError.DISSONANCE_UNRESOLVED,
                                    "measure": i + 1,
                                    "message": "Suspension must resolve down by step",
                                }
                            )

        return result

    async def _analyze_fifth_species(
        self,
        cf_notes: List[Any],
        cp_notes: List[Any],
        result: SpeciesAnalysis,
    ) -> SpeciesAnalysis:
        """Analyze fifth species (florid counterpoint)"""
        # Combination of all species - most complex
        # Check for variety of rhythmic values
        rhythms = [n.duration.quarterLength for n in cp_notes]
        rhythm_variety = len(set(rhythms))

        if rhythm_variety < 3:
            result.warnings.append(
                {
                    "type": "insufficient_variety",
                    "message": "Fifth species should use varied rhythm",
                }
            )

        # Check for appropriate use of all techniques
        return result

    async def _analyze_free_counterpoint(
        self,
        cf_notes: List[Any],
        cp_notes: List[Any],
        result: SpeciesAnalysis,
    ) -> SpeciesAnalysis:
        """Analyze free counterpoint"""
        # More lenient rules
        return result

    def _check_voice_leading(
        self,
        cf_notes: List[Any],
        cp_notes: List[Any],
        result: SpeciesAnalysis,
    ) -> SpeciesAnalysis:
        """Check voice leading rules"""
        for i in range(len(min(cf_notes, cp_notes)) - 1):
            if i + 1 < len(cf_notes) and i + 1 < len(cp_notes):
                # Create voice leading quartet
                vl = voiceLeading.VoiceLeadingQuartet(
                    cf_notes[i], cp_notes[i], cf_notes[i + 1], cp_notes[i + 1]
                )

                # Check for parallel perfects
                if vl.parallelFifth() or vl.parallelOctave():
                    result.errors.append(
                        {
                            "type": CounterpointError.PARALLEL_PERFECT,
                            "measure": i + 1,
                            "interval": "fifth" if vl.parallelFifth() else "octave",
                        }
                    )

                # Check for hidden perfects
                if vl.hiddenFifth() or vl.hiddenOctave():
                    result.warnings.append(
                        {
                            "type": CounterpointError.DIRECT_PERFECT,
                            "measure": i + 1,
                            "interval": "fifth" if vl.hiddenFifth() else "octave",
                        }
                    )

        result.voice_leading_score = self._calculate_voice_leading_score(result)
        return result

    def _check_melodic_quality(
        self, melody_notes: List[Any], result: SpeciesAnalysis
    ) -> SpeciesAnalysis:
        """Check melodic quality of counterpoint line"""
        # Check for large leaps
        for i in range(len(melody_notes) - 1):
            leap = abs(melody_notes[i + 1].pitch.midi - melody_notes[i].pitch.midi)

            # Tritone leap
            if leap == 6:
                result.errors.append(
                    {"type": CounterpointError.TRITONE_LEAP, "measure": i + 1}
                )
            # Large leap (> octave)
            elif leap > 12:
                result.errors.append(
                    {
                        "type": CounterpointError.LARGE_LEAP,
                        "measure": i + 1,
                        "interval": leap,
                    }
                )
            # Leap > 6th should be followed by stepwise motion in opposite direction
            elif leap > 9:
                if i + 2 < len(melody_notes):
                    next_motion = (
                        melody_notes[i + 2].pitch.midi - melody_notes[i + 1].pitch.midi
                    )
                    if (leap > 0 and next_motion > 2) or (
                        leap < 0 and next_motion < -2
                    ):
                        result.warnings.append(
                            {"type": "leap_not_recovered", "measure": i + 1}
                        )

        # Check for repeated notes
        repetitions = 0
        for i in range(1, len(melody_notes)):
            if melody_notes[i].pitch == melody_notes[i - 1].pitch:
                repetitions += 1
            else:
                if repetitions > 2:
                    result.warnings.append(
                        {
                            "type": CounterpointError.REPEATED_NOTE,
                            "measure": i - repetitions,
                        }
                    )
                repetitions = 0

        # Check for melodic contour and climax
        pitches = [n.pitch.midi for n in melody_notes]
        climax_index = pitches.index(max(pitches))

        # Climax should not be at beginning or end
        if climax_index == 0 or climax_index == len(pitches) - 1:
            result.warnings.append(
                {
                    "type": CounterpointError.POOR_CLIMAX,
                    "message": "Climax at extremes of melody",
                }
            )

        result.melodic_quality_score = self._calculate_melodic_score(result)
        return result

    def _check_harmonic_intervals(
        self,
        cf_notes: List[Any],
        cp_notes: List[Any],
        result: SpeciesAnalysis,
    ) -> SpeciesAnalysis:
        """Check harmonic interval usage"""
        # Count interval types
        for i in range(min(len(cf_notes), len(cp_notes))):
            interv = interval.Interval(cf_notes[i].pitch, cp_notes[i].pitch)

            # Avoid too many perfect intervals
            if abs(interv.generic.value) in self.perfect_intervals:
                interval_type = f"P{abs(interv.generic.value)}"
                count = result.interval_usage.get(interval_type, 0)
                if count > len(cf_notes) * 0.5:
                    result.warnings.append(
                        {
                            "type": "excessive_perfect_intervals",
                            "message": f"Too many {interval_type} intervals",
                        }
                    )

        result.harmonic_quality_score = self._calculate_harmonic_score(result)
        return result

    def _check_cadences(
        self,
        cf_notes: List[Any],
        cp_notes: List[Any],
        result: SpeciesAnalysis,
    ) -> SpeciesAnalysis:
        """Check cadence formulas"""
        if len(cf_notes) >= 2 and len(cp_notes) >= 2:
            # Check final cadence
            penult_interval = interval.Interval(cf_notes[-2].pitch, cp_notes[-2].pitch)
            final_interval = interval.Interval(cf_notes[-1].pitch, cp_notes[-1].pitch)

            # Common cadence formulas
            if cf_notes[-2].pitch < cf_notes[-1].pitch:  # CF ascending
                # Should have 6-8 or 3-1 in upper voice
                if not (
                    penult_interval.generic.value == 6
                    and final_interval.generic.value in [1, 8]
                ):
                    result.warnings.append(
                        {
                            "type": CounterpointError.POOR_CADENCE,
                            "message": "Non-standard ascending cadence",
                        }
                    )
            else:  # CF descending
                # Should have 3-1 or 10-8
                valid_cadence = False
                if (
                    penult_interval.generic.value == 3
                    and final_interval.generic.value == 1
                ):
                    valid_cadence = True

                if not valid_cadence:
                    result.warnings.append(
                        {
                            "type": CounterpointError.POOR_CADENCE,
                            "message": "Non-standard descending cadence",
                        }
                    )

            result.cadences.append(
                {
                    "type": "final",
                    "formula": f"{penult_interval.simpleName}-{final_interval.simpleName}",
                }
            )

        return result

    def _is_valid_passing_tone(self, notes: List[Any], index: int) -> bool:
        """Check if note is a valid passing tone"""
        if index == 0 or index >= len(notes) - 1:
            return False

        prev_pitch = notes[index - 1].pitch.midi
        curr_pitch = notes[index].pitch.midi
        next_pitch = notes[index + 1].pitch.midi

        # Passing tone: stepwise motion in same direction
        if prev_pitch < curr_pitch < next_pitch:
            return True
        if prev_pitch > curr_pitch > next_pitch:
            return True

        # Neighbor tone: step away and back
        if abs(prev_pitch - curr_pitch) <= 2 and prev_pitch == next_pitch:
            return True

        return False

    def _calculate_counterpoint_score(self, result: SpeciesAnalysis) -> float:
        """Calculate overall counterpoint quality score"""
        # Weighted average of component scores
        scores = [
            result.voice_leading_score * 0.4,
            result.melodic_quality_score * 0.3,
            result.harmonic_quality_score * 0.3,
        ]

        # Deduct for errors and warnings
        error_penalty = len(result.errors) * 0.1
        warning_penalty = len(result.warnings) * 0.05

        total = sum(scores) - error_penalty - warning_penalty
        return max(0, min(1, total))

    def _calculate_voice_leading_score(self, result: SpeciesAnalysis) -> float:
        """Calculate voice leading quality score"""
        if not result.errors:
            return 1.0

        # Count serious errors
        parallel_errors = sum(
            1 for e in result.errors if e["type"] == CounterpointError.PARALLEL_PERFECT
        )

        score = 1.0 - (parallel_errors * 0.2)
        return max(0, score)

    def _calculate_melodic_score(self, result: SpeciesAnalysis) -> float:
        """Calculate melodic quality score"""
        leap_errors = sum(
            1
            for e in result.errors
            if e["type"]
            in [CounterpointError.LARGE_LEAP, CounterpointError.TRITONE_LEAP]
        )

        repetition_warnings = sum(
            1 for w in result.warnings if w["type"] == CounterpointError.REPEATED_NOTE
        )

        score = 1.0 - (leap_errors * 0.15) - (repetition_warnings * 0.05)
        return max(0, score)

    def _calculate_harmonic_score(self, result: SpeciesAnalysis) -> float:
        """Calculate harmonic quality score"""
        # Variety of intervals is good
        interval_variety = len(result.interval_usage)
        variety_score = min(interval_variety / 6, 1.0)

        # Not too many perfect intervals
        perfect_count = sum(
            count
            for interval, count in result.interval_usage.items()
            if "P" in interval
        )
        total_intervals = sum(result.interval_usage.values())

        if total_intervals > 0:
            perfect_ratio = perfect_count / total_intervals
            balance_score = 1.0 - max(0, perfect_ratio - 0.4)
        else:
            balance_score = 0.5

        return (variety_score + balance_score) / 2

    # Helper methods for Bach chorale analysis
    def _extract_voices_from_chords(self, score: stream.Score) -> List[stream.Part]:
        """Extract individual voices from chord notation"""
        voices = [stream.Part() for _ in range(4)]  # SATB

        for element in score.flatten():
            if isinstance(element, chord.Chord):
                pitches = sorted(element.pitches, key=lambda p: p.midi)

                # Distribute to voices (simple approach)
                if len(pitches) >= 4:
                    voices[3].append(
                        note.Note(pitches[0], quarterLength=element.quarterLength)
                    )  # Bass
                    voices[2].append(
                        note.Note(pitches[1], quarterLength=element.quarterLength)
                    )  # Tenor
                    voices[1].append(
                        note.Note(pitches[2], quarterLength=element.quarterLength)
                    )  # Alto
                    voices[0].append(
                        note.Note(pitches[3], quarterLength=element.quarterLength)
                    )  # Soprano

        return voices

    def _analyze_voice_ranges(
        self, parts: List[stream.Part]
    ) -> Dict[str, Dict[str, Any]]:
        """Analyze vocal ranges in chorale"""
        voice_names = ["Soprano", "Alto", "Tenor", "Bass"]
        ranges = {}

        # Standard Bach chorale ranges (in MIDI numbers)
        standard_ranges = {
            "Soprano": (60, 79),  # C4-G5
            "Alto": (55, 74),  # G3-D5
            "Tenor": (48, 69),  # C3-A4
            "Bass": (40, 60),  # E2-C4
        }

        for i, part in enumerate(parts):
            if i < len(voice_names):
                notes = [n for n in part.flatten().notes if hasattr(n, "pitch")]
                if notes:
                    pitches = [n.pitch.midi for n in notes if n.isNote]
                    if pitches:
                        voice_range = {
                            "lowest": min(pitches),
                            "highest": max(pitches),
                            "average": np.mean(pitches),
                            "tessitura": self._calculate_tessitura(pitches),
                        }

                        # Check against standard range
                        if i < len(standard_ranges):
                            std_low, std_high = standard_ranges[voice_names[i]]
                            if voice_range["lowest"] < std_low:
                                voice_range["below_standard"] = (
                                    std_low - voice_range["lowest"]
                                )
                            if voice_range["highest"] > std_high:
                                voice_range["above_standard"] = (
                                    voice_range["highest"] - std_high
                                )

                        ranges[voice_names[i]] = voice_range

        return ranges

    def _calculate_tessitura(self, pitches: List[int]) -> Dict[str, float]:
        """Calculate comfortable singing range"""
        if not pitches:
            return {}

        # Use middle 80% as tessitura
        return {
            "low": np.percentile(pitches, 10),
            "high": np.percentile(pitches, 90),
            "center": np.median(pitches),
        }

    def _check_bach_voice_leading(
        self, part1: stream.Part, part2: stream.Part, voice1_idx: int, voice2_idx: int
    ) -> List[Dict[str, Any]]:
        """Check Bach-style voice leading between two parts"""
        errors = []

        notes1 = [n for n in part1.flatten().notes if hasattr(n, "pitch")]
        notes2 = [n for n in part2.flatten().notes if hasattr(n, "pitch")]

        # Align notes by offset
        aligned = self._align_voices_by_offset(notes1, notes2)

        for i in range(len(aligned) - 1):
            if all(aligned[i]) and all(aligned[i + 1]):
                curr1, curr2 = aligned[i]
                next1, next2 = aligned[i + 1]

                # Create voice leading object
                vl = voiceLeading.VoiceLeadingQuartet(curr1, curr2, next1, next2)

                # Parallel fifths/octaves (forbidden)
                if vl.parallelFifth():
                    errors.append(
                        {
                            "type": "parallel_fifth",
                            "voices": f"{voice1_idx+1}-{voice2_idx+1}",
                            "measure": self._get_measure_number(curr1),
                        }
                    )

                if vl.parallelOctave():
                    errors.append(
                        {
                            "type": "parallel_octave",
                            "voices": f"{voice1_idx+1}-{voice2_idx+1}",
                            "measure": self._get_measure_number(curr1),
                        }
                    )

                # Hidden fifths/octaves (avoided in outer voices)
                if voice1_idx == 0 or voice2_idx == 0:  # Involves soprano
                    if vl.hiddenFifth() or vl.hiddenOctave():
                        errors.append(
                            {
                                "type": "hidden_perfect",
                                "voices": f"{voice1_idx+1}-{voice2_idx+1}",
                                "measure": self._get_measure_number(curr1),
                                "severity": "warning",
                            }
                        )

        return errors

    def _align_voices_by_offset(
        self, notes1: List[Any], notes2: List[Any]
    ) -> List[Tuple[Optional[note.Note], Optional[note.Note]]]:
        """Align two voices by musical offset"""
        aligned = []
        i, j = 0, 0

        while i < len(notes1) or j < len(notes2):
            if i >= len(notes1):
                aligned.append((None, notes2[j]))
                j += 1
            elif j >= len(notes2):
                aligned.append((notes1[i], None))
                i += 1
            else:
                offset1 = notes1[i].offset
                offset2 = notes2[j].offset

                if abs(offset1 - offset2) < 0.01:  # Same offset
                    aligned.append((notes1[i], notes2[j]))
                    i += 1
                    j += 1
                elif offset1 < offset2:
                    aligned.append((notes1[i], None))
                    i += 1
                else:
                    aligned.append((None, notes2[j]))
                    j += 1

        return aligned

    async def _analyze_chorale_harmony(
        self, chorale: stream.Score, result: ChoraleAnalysis
    ) -> ChoraleAnalysis:
        """Analyze harmonic progressions in chorale"""
        # Get chords and analyze
        chords = chorale.chordify()

        for c in chords.flatten().getElementsByClass(chord.Chord):
            # Get Roman numeral
            try:
                rn = roman.romanNumeralFromChord(c, chorale.analyze("key"))
                result.chord_progressions.append(rn.figure)
            except:
                result.chord_progressions.append("?")

        return result

    def _identify_non_chord_tones(self, chorale: stream.Score) -> List[Dict[str, Any]]:
        """Identify non-harmonic tones in chorale"""
        ncts = []

        # This would require detailed harmonic analysis
        # Simplified version

        return ncts

    def _analyze_chorale_cadences(self, chorale: stream.Score) -> List[Dict[str, Any]]:
        """Analyze cadence types in chorale"""
        cadences = []

        # Find phrase endings (fermatas often mark these)
        for element in chorale.flatten():
            if element.expressions:
                for expr in element.expressions:
                    if "fermata" in str(expr).lower():
                        # Analyze cadence at this point
                        cadence_info = {
                            "measure": element.measureNumber,
                            "type": "authentic",  # Would need harmonic analysis
                            "strength": "perfect",
                        }
                        cadences.append(cadence_info)

        return cadences

    def _analyze_chorale_texture(self, parts: List[stream.Part]) -> Dict[str, Any]:
        """Analyze textural characteristics of chorale"""
        texture = {
            "voice_count": len(parts),
            "homophonic_ratio": 0.0,
            "average_voice_spacing": 0.0,
            "crossing_count": 0,
        }

        if len(parts) >= 2:
            # Check for homorhythm
            rhythm_patterns = []
            for part in parts:
                rhythms = [n.duration.quarterLength for n in part.flatten().notes]
                rhythm_patterns.append(rhythms)

            # Compare rhythms
            if rhythm_patterns:
                matching_rhythms = 0
                total_comparisons = 0

                for i in range(len(rhythm_patterns[0])):
                    if all(i < len(rp) for rp in rhythm_patterns):
                        if all(
                            rp[i] == rhythm_patterns[0][i] for rp in rhythm_patterns
                        ):
                            matching_rhythms += 1
                        total_comparisons += 1

                if total_comparisons > 0:
                    texture["homophonic_ratio"] = matching_rhythms / total_comparisons

        return texture

    def _calculate_bach_style_score(self, result: ChoraleAnalysis) -> float:
        """Calculate Bach chorale style conformance score"""
        score = 1.0

        # Penalize voice leading errors
        error_penalty = len(result.voice_leading_errors) * 0.05
        score -= min(error_penalty, 0.5)

        # Reward appropriate texture
        if result.texture_analysis.get("homophonic_ratio", 0) > 0.7:
            score += 0.1

        # Check voice ranges
        for voice, range_info in result.voice_ranges.items():
            if "below_standard" in range_info or "above_standard" in range_info:
                score -= 0.05

        return max(0, min(1, score))

    # Helper methods for voice independence
    def _calculate_rhythmic_independence(self, parts: List[stream.Part]) -> float:
        """Calculate rhythmic independence between voices"""
        if len(parts) < 2:
            return 1.0

        independence_scores = []

        for i in range(len(parts)):
            for j in range(i + 1, len(parts)):
                rhythms1 = [n.duration.quarterLength for n in parts[i].flatten().notes]
                rhythms2 = [n.duration.quarterLength for n in parts[j].flatten().notes]

                if rhythms1 and rhythms2:
                    # Compare rhythm patterns
                    min_len = min(len(rhythms1), len(rhythms2))
                    different_rhythms = sum(
                        1 for k in range(min_len) if rhythms1[k] != rhythms2[k]
                    )

                    independence = different_rhythms / min_len if min_len > 0 else 0
                    independence_scores.append(independence)

        return np.mean(independence_scores) if independence_scores else 0.5

    def _calculate_melodic_independence(self, parts: List[stream.Part]) -> float:
        """Calculate melodic independence between voices"""
        if len(parts) < 2:
            return 1.0

        independence_scores = []

        for i in range(len(parts)):
            for j in range(i + 1, len(parts)):
                intervals1 = self._get_melodic_intervals(parts[i])
                intervals2 = self._get_melodic_intervals(parts[j])

                if intervals1 and intervals2:
                    # Compare interval patterns
                    min_len = min(len(intervals1), len(intervals2))
                    different_intervals = sum(
                        1 for k in range(min_len) if intervals1[k] != intervals2[k]
                    )

                    independence = different_intervals / min_len if min_len > 0 else 0
                    independence_scores.append(independence)

        return np.mean(independence_scores) if independence_scores else 0.5

    def _get_melodic_intervals(self, part: stream.Part) -> List[int]:
        """Get melodic intervals from a part"""
        notes = [n for n in part.flatten().notes if hasattr(n, "pitch")]
        intervals = []

        for i in range(len(notes) - 1):
            if notes[i].isNote and notes[i + 1].isNote:
                interval = notes[i + 1].pitch.midi - notes[i].pitch.midi
                intervals.append(interval)

        return intervals

    def _calculate_contour_independence(self, parts: List[stream.Part]) -> float:
        """Calculate contour independence between voices"""
        if len(parts) < 2:
            return 1.0

        contours = []
        for part in parts:
            notes = [n for n in part.flatten().notes if hasattr(n, "pitch")]
            contour = []
            for i in range(len(notes) - 1):
                if notes[i].isNote and notes[i + 1].isNote:
                    if notes[i].pitch < notes[i + 1].pitch:
                        contour.append(1)
                    elif notes[i].pitch > notes[i + 1].pitch:
                        contour.append(-1)
                    else:
                        contour.append(0)
            contours.append(contour)

        # Compare contours
        independence_scores = []
        for i in range(len(contours)):
            for j in range(i + 1, len(contours)):
                if contours[i] and contours[j]:
                    min_len = min(len(contours[i]), len(contours[j]))
                    different = sum(
                        1 for k in range(min_len) if contours[i][k] != contours[j][k]
                    )

                    independence = different / min_len if min_len > 0 else 0
                    independence_scores.append(independence)

        return np.mean(independence_scores) if independence_scores else 0.5

    def _calculate_onset_independence(self, parts: List[stream.Part]) -> float:
        """Calculate onset time independence"""
        if len(parts) < 2:
            return 1.0

        # Get onset times for each part
        onset_sets = []
        for part in parts:
            onsets = set()
            for n in part.flatten().notes:
                onsets.add(n.offset)
            onset_sets.append(onsets)

        # Calculate independence
        independence_scores = []
        for i in range(len(onset_sets)):
            for j in range(i + 1, len(onset_sets)):
                union = onset_sets[i] | onset_sets[j]
                intersection = onset_sets[i] & onset_sets[j]

                if union:
                    independence = 1 - (len(intersection) / len(union))
                    independence_scores.append(independence)

        return np.mean(independence_scores) if independence_scores else 0.5

    def _find_voice_crossings(
        self, part1: stream.Part, part2: stream.Part
    ) -> List[Dict[str, Any]]:
        """Find voice crossing instances"""
        crossings = []

        notes1 = [n for n in part1.flatten().notes if hasattr(n, "pitch")]
        notes2 = [n for n in part2.flatten().notes if hasattr(n, "pitch")]

        aligned = self._align_voices_by_offset(notes1, notes2)

        for i, (n1, n2) in enumerate(aligned):
            if n1 and n2 and n1.isNote and n2.isNote:
                # Assuming part1 should be higher than part2
                if n1.pitch < n2.pitch:
                    crossings.append(
                        {
                            "measure": self._get_measure_number(n1),
                            "pitches": [n1.nameWithOctave, n2.nameWithOctave],
                            "offset": n1.offset,
                        }
                    )

        return crossings

    def _find_voice_overlaps(
        self, part1: stream.Part, part2: stream.Part
    ) -> List[Dict[str, Any]]:
        """Find voice overlap instances"""
        overlaps = []

        notes1 = [n for n in part1.flatten().notes if hasattr(n, "pitch")]
        notes2 = [n for n in part2.flatten().notes if hasattr(n, "pitch")]

        # Check if voices get too close
        for i in range(min(len(notes1), len(notes2))):
            if notes1[i].isNote and notes2[i].isNote:
                interval = abs(notes1[i].pitch.midi - notes2[i].pitch.midi)
                if interval < 2:  # Less than a whole step
                    overlaps.append(
                        {
                            "measure": self._get_measure_number(notes1[i]),
                            "interval": interval,
                            "pitches": [
                                notes1[i].nameWithOctave,
                                notes2[i].nameWithOctave,
                            ],
                        }
                    )

        return overlaps

    def _calculate_motion_statistics(
        self, parts: List[stream.Part]
    ) -> Dict[str, float]:
        """Calculate motion type statistics between all voice pairs"""
        motion_counts = {
            "parallel": 0,
            "similar": 0,
            "contrary": 0,
            "oblique": 0,
            "total": 0,
        }

        for i in range(len(parts)):
            for j in range(i + 1, len(parts)):
                notes1 = [n for n in parts[i].flatten().notes if hasattr(n, "pitch")]
                notes2 = [n for n in parts[j].flatten().notes if hasattr(n, "pitch")]

                aligned = self._align_voices_by_offset(notes1, notes2)

                for k in range(len(aligned) - 1):
                    if all(aligned[k]) and all(aligned[k + 1]):
                        motion = self._classify_motion(
                            aligned[k][0],
                            aligned[k][1],
                            aligned[k + 1][0],
                            aligned[k + 1][1],
                        )

                        if motion != VoiceMotion.NONE:
                            motion_counts[motion.value] += 1
                            motion_counts["total"] += 1

        # Calculate ratios
        if motion_counts["total"] > 0:
            return {
                motion: count / motion_counts["total"]
                for motion, count in motion_counts.items()
                if motion != "total"
            }
        else:
            return {"parallel": 0, "similar": 0, "contrary": 0, "oblique": 0}

    def _classify_motion(
        self, note1a: note.Note, note1b: note.Note, note2a: note.Note, note2b: note.Note
    ) -> VoiceMotion:
        """Classify motion type between two voice pairs"""
        if not all([note1a, note1b, note2a, note2b]):
            return VoiceMotion.NONE

        # Calculate melodic intervals
        interval1 = note2a.pitch.midi - note1a.pitch.midi
        interval2 = note2b.pitch.midi - note1b.pitch.midi

        # Oblique: one voice stays same
        if interval1 == 0 or interval2 == 0:
            return VoiceMotion.OBLIQUE

        # Same direction
        if (interval1 > 0 and interval2 > 0) or (interval1 < 0 and interval2 < 0):
            # Check if parallel (same interval)
            if interval1 == interval2:
                return VoiceMotion.PARALLEL
            else:
                return VoiceMotion.SIMILAR

        # Opposite directions
        return VoiceMotion.CONTRARY

    # Helper methods for fugue analysis
    async def _find_fugue_subject(
        self, fugue: stream.Score
    ) -> Optional[Dict[str, Any]]:
        """Find the fugue subject"""
        # Usually in the first voice to enter
        for part in fugue.parts:
            notes = [n for n in part.flatten().notes if hasattr(n, "pitch")]

            # Look for initial melodic material
            if len(notes) >= 4:  # Minimum subject length
                # Find first substantial melodic phrase
                subject_notes = []
                for n in notes:
                    if n.isNote:
                        subject_notes.append(n)
                        # Subject usually ends at first significant rest or cadence
                        if len(subject_notes) >= 4 and len(subject_notes) <= 16:
                            # Check for a rest after
                            next_elem = part.getElementAfterElement(n)
                            if next_elem and next_elem.isRest:
                                break

                if subject_notes:
                    return {
                        "notes": subject_notes,
                        "pitches": [n.pitch for n in subject_notes],
                        "intervals": self._get_interval_pattern(subject_notes),
                        "rhythm": [n.duration.quarterLength for n in subject_notes],
                        "length": len(subject_notes),
                        "first_note": subject_notes[0].nameWithOctave,
                        "last_note": subject_notes[-1].nameWithOctave,
                    }

        return None

    def _get_interval_pattern(self, notes: List[Any]) -> List[int]:
        """Get interval pattern from notes"""
        intervals = []
        for i in range(len(notes) - 1):
            if notes[i].isNote and notes[i + 1].isNote:
                intervals.append(notes[i + 1].pitch.midi - notes[i].pitch.midi)
        return intervals

    async def _find_fugue_answer(
        self, fugue: stream.Score, subject: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Find the fugue answer (real or tonal)"""
        subject_intervals = subject["intervals"]

        for part in fugue.parts:
            notes = [n for n in part.flatten().notes if hasattr(n, "pitch")]

            # Look for transposed subject
            for i in range(len(notes) - len(subject_intervals)):
                candidate_notes = notes[i : i + len(subject_intervals) + 1]

                if all(n.isNote for n in candidate_notes):
                    candidate_intervals = self._get_interval_pattern(candidate_notes)

                    # Check for real answer (exact transposition)
                    if candidate_intervals == subject_intervals:
                        transposition = (
                            candidate_notes[0].pitch.midi
                            - subject["notes"][0].pitch.midi
                        )

                        # Typical answer is at fifth (7 semitones)
                        if abs(transposition) in [7, -5]:  # Fifth up or fourth down
                            return {
                                "notes": candidate_notes,
                                "type": "real",
                                "transposition": transposition,
                                "measure": self._get_measure_number(candidate_notes[0]),
                            }

                    # Check for tonal answer (modified intervals)
                    elif self._is_tonal_answer(subject_intervals, candidate_intervals):
                        return {
                            "notes": candidate_notes,
                            "type": "tonal",
                            "modifications": self._get_answer_modifications(
                                subject_intervals, candidate_intervals
                            ),
                            "measure": self._get_measure_number(candidate_notes[0]),
                        }

        return None

    def _is_tonal_answer(
        self, subject_intervals: List[int], candidate_intervals: List[int]
    ) -> bool:
        """Check if candidate is a tonal answer"""
        if len(subject_intervals) != len(candidate_intervals):
            return False

        # Tonal answers typically modify first/last intervals
        differences = [
            abs(s - c) for s, c in zip(subject_intervals, candidate_intervals)
        ]

        # Allow up to 2 semitone differences in up to 2 places
        significant_differences = sum(1 for d in differences if d > 0)
        max_difference = max(differences) if differences else 0

        return significant_differences <= 2 and max_difference <= 2

    def _get_answer_modifications(
        self, subject_intervals: List[int], answer_intervals: List[int]
    ) -> List[Dict[str, Any]]:
        """Get modifications in tonal answer"""
        modifications = []

        for i, (subj, ans) in enumerate(zip(subject_intervals, answer_intervals)):
            if subj != ans:
                modifications.append(
                    {
                        "position": i,
                        "original": subj,
                        "modified": ans,
                        "difference": ans - subj,
                    }
                )

        return modifications

    def _find_all_entries(
        self,
        fugue: stream.Score,
        subject: Dict[str, Any],
        answer: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Find all subject and answer entries"""
        entries = []
        subject_intervals = subject["intervals"]

        for part_idx, part in enumerate(fugue.parts):
            notes = [n for n in part.flatten().notes if hasattr(n, "pitch")]

            for i in range(len(notes) - len(subject_intervals)):
                candidate_notes = notes[i : i + len(subject_intervals) + 1]

                if all(n.isNote for n in candidate_notes):
                    candidate_intervals = self._get_interval_pattern(candidate_notes)

                    # Check for subject or answer
                    if candidate_intervals == subject_intervals:
                        entries.append(
                            {
                                "type": "subject",
                                "voice": part_idx + 1,
                                "measure": self._get_measure_number(candidate_notes[0]),
                                "pitch_level": candidate_notes[0].nameWithOctave,
                            }
                        )
                    elif answer and self._matches_answer(candidate_intervals, answer):
                        entries.append(
                            {
                                "type": "answer",
                                "voice": part_idx + 1,
                                "measure": self._get_measure_number(candidate_notes[0]),
                                "pitch_level": candidate_notes[0].nameWithOctave,
                            }
                        )

        return sorted(entries, key=lambda e: e["measure"])

    def _matches_answer(
        self, candidate_intervals: List[int], answer: Dict[str, Any]
    ) -> bool:
        """Check if intervals match the answer"""
        if "notes" in answer:
            answer_intervals = self._get_interval_pattern(answer["notes"])
            return candidate_intervals == answer_intervals
        return False

    async def _find_countersubjects(
        self,
        fugue: stream.Score,
        subject: Dict[str, Any],
        entries: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Find countersubjects"""
        countersubjects = []

        # Look for consistent material against subject entries
        # This is simplified - real implementation would be more complex

        return countersubjects

    def _find_episodes(
        self, fugue: stream.Score, entries: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Find episodic passages between entries"""
        episodes = []

        # Episodes occur between subject/answer entries
        for i in range(len(entries) - 1):
            start_measure = entries[i]["measure"] + 2  # Approximate
            end_measure = entries[i + 1]["measure"] - 1

            if end_measure > start_measure:
                episodes.append(
                    {
                        "start_measure": start_measure,
                        "end_measure": end_measure,
                        "type": (
                            "sequential"
                            if (end_measure - start_measure) > 4
                            else "link"
                        ),
                    }
                )

        return episodes

    def _find_strettos(self, entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find stretto passages (overlapping entries)"""
        strettos = []

        # Check for entries that start close together
        for i in range(len(entries) - 1):
            if entries[i + 1]["measure"] - entries[i]["measure"] < 3:
                strettos.append(
                    {
                        "voices": [entries[i]["voice"], entries[i + 1]["voice"]],
                        "measure": entries[i]["measure"],
                        "interval": entries[i + 1]["measure"] - entries[i]["measure"],
                    }
                )

        return strettos

    async def _analyze_fugue_key_plan(
        self, fugue: stream.Score, entries: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Analyze key areas in fugue"""
        key_plan = []

        # Simplified - would need proper key analysis
        initial_key = fugue.analyze("key")
        key_plan.append(
            {"measure": 1, "key": str(initial_key), "relationship": "tonic"}
        )

        return key_plan

    def _find_contrapuntal_devices(
        self, fugue: stream.Score, subject: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Find special contrapuntal devices"""
        devices = []

        # Look for inversions, augmentations, etc.
        # This is a simplified implementation

        return devices

    def _determine_fugue_form(self, analysis: FugueAnalysis) -> str:
        """Determine overall fugue form"""
        if not analysis.entries:
            return "incomplete"

        # Simple classification based on entries
        exposition_entries = [e for e in analysis.entries if e["measure"] < 20]

        if len(exposition_entries) >= 3:
            if analysis.strettos:
                return "fugue_with_stretto"
            elif len(analysis.episodes) > 3:
                return "episodic_fugue"
            else:
                return "simple_fugue"

        return "fugal_exposition"

    def _get_measure_number(self, note_obj: note.Note) -> int:
        """Get measure number for a note"""
        if hasattr(note_obj, "measureNumber"):
            return note_obj.measureNumber
        return 0
