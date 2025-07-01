"""
Harmonic Analysis Engine for music21 MCP server
Implements advanced harmonic analysis including functional harmony,
voice leading, jazz harmony, and modulation detection.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from music21 import (chord, harmony, interval, key, note, roman, stream,
                     voiceLeading)


class HarmonicFunction(Enum):
    """Enumeration of harmonic functions"""

    TONIC = "T"
    SUBDOMINANT = "S"
    DOMINANT = "D"
    PREDOMINANT = "PD"
    SECONDARY_DOMINANT = "V/"
    SECONDARY_SUBDOMINANT = "IV/"
    AUGMENTED_SIXTH = "+6"
    NEAPOLITAN = "N"
    BORROWED = "b"
    CHROMATIC_MEDIANT = "CM"
    PASSING = "P"
    NEIGHBOR = "N"


class VoiceLeadingError(Enum):
    """Types of voice leading errors"""

    PARALLEL_FIFTHS = "parallel_fifths"
    PARALLEL_OCTAVES = "parallel_octaves"
    DIRECT_FIFTHS = "direct_fifths"
    DIRECT_OCTAVES = "direct_octaves"
    VOICE_CROSSING = "voice_crossing"
    VOICE_OVERLAP = "voice_overlap"
    LARGE_LEAP = "large_leap"
    AUGMENTED_INTERVAL = "augmented_interval"
    UNRESOLVED_SEVENTH = "unresolved_seventh"
    UNRESOLVED_LEADING_TONE = "unresolved_leading_tone"


class ChordSubstitutionType(Enum):
    """Types of chord substitutions in jazz"""

    TRITONE = "tritone_substitution"
    MINOR_FOR_DOMINANT = "minor_for_dominant"
    DIMINISHED_PASSING = "diminished_passing"
    CHROMATIC_MEDIANT = "chromatic_mediant"
    MODAL_INTERCHANGE = "modal_interchange"
    EXTENDED_DOMINANT = "extended_dominant"
    ALTERED_DOMINANT = "altered_dominant"


@dataclass
class FunctionalHarmonyAnalysis:
    """Results of functional harmony analysis"""

    roman_numerals: List[str] = field(default_factory=list)
    functions: List[HarmonicFunction] = field(default_factory=list)
    tonic_prolongations: List[Dict[str, Any]] = field(default_factory=list)
    predominant_chords: List[Dict[str, Any]] = field(default_factory=list)
    dominant_preparations: List[Dict[str, Any]] = field(default_factory=list)
    deceptive_resolutions: List[Dict[str, Any]] = field(default_factory=list)
    cadences: List[Dict[str, Any]] = field(default_factory=list)
    phrase_model: str = ""
    tonal_strength: float = 0.0


@dataclass
class VoiceLeadingAnalysis:
    """Results of voice leading analysis"""

    voice_movements: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[Dict[str, Any]] = field(default_factory=list)
    parallel_motions: List[Dict[str, Any]] = field(default_factory=list)
    voice_crossings: List[Dict[str, Any]] = field(default_factory=list)
    voice_ranges: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    smoothness_score: float = 0.0
    independence_score: float = 0.0


@dataclass
class JazzHarmonyAnalysis:
    """Results of jazz harmony analysis"""

    chord_symbols: List[str] = field(default_factory=list)
    extended_chords: List[Dict[str, Any]] = field(default_factory=list)
    substitutions: List[Dict[str, Any]] = field(default_factory=list)
    reharmonizations: List[Dict[str, Any]] = field(default_factory=list)
    modal_interchanges: List[Dict[str, Any]] = field(default_factory=list)
    chord_scales: List[Dict[str, Any]] = field(default_factory=list)
    tension_analysis: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ModulationAnalysis:
    """Results of modulation analysis"""

    key_areas: List[Dict[str, Any]] = field(default_factory=list)
    modulations: List[Dict[str, Any]] = field(default_factory=list)
    pivot_chords: List[Dict[str, Any]] = field(default_factory=list)
    common_tone_modulations: List[Dict[str, Any]] = field(default_factory=list)
    chromatic_modulations: List[Dict[str, Any]] = field(default_factory=list)
    enharmonic_modulations: List[Dict[str, Any]] = field(default_factory=list)
    tonicization_regions: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class HarmonicSequence:
    """Represents a harmonic sequence"""

    pattern: List[str]
    intervals: List[int]
    occurrences: List[Dict[str, Any]]
    sequence_type: str
    direction: str  # ascending/descending
    real_or_tonal: str  # real/tonal


class HarmonicAnalyzer:
    """Advanced harmonic analysis engine"""

    def __init__(self):
        self.current_key = None
        self.roman_analyzer = None

    async def analyze_functional_harmony(
        self, score: stream.Score, window_size: int = 4
    ) -> FunctionalHarmonyAnalysis:
        """
        Analyze functional harmony including prolongations and cadences.

        Args:
            score: Music21 score to analyze
            window_size: Analysis window in measures

        Returns:
            FunctionalHarmonyAnalysis object
        """
        result = FunctionalHarmonyAnalysis()

        # Get key and create Roman numeral analysis
        score_key = score.analyze("key")
        self.current_key = score_key

        # Extract chords and analyze
        chords = list(score.flatten().getElementsByClass(chord.Chord))
        measures = score.parts[0].getElementsByClass(stream.Measure)

        # Roman numeral analysis
        for i, ch in enumerate(chords):
            try:
                rn = roman.romanNumeralFromChord(ch, score_key)
                result.roman_numerals.append(rn.figure)
                result.functions.append(self._classify_function(rn))

                # Check for specific progressions
                if i > 0:
                    prev_rn = roman.romanNumeralFromChord(chords[i - 1], score_key)

                    # Tonic prolongation
                    if self._is_tonic_prolongation(prev_rn, rn):
                        result.tonic_prolongations.append(
                            {
                                "start_measure": self._get_measure_number(
                                    chords[i - 1], measures
                                ),
                                "end_measure": self._get_measure_number(ch, measures),
                                "type": self._get_prolongation_type(prev_rn, rn),
                                "chords": [prev_rn.figure, rn.figure],
                            }
                        )

                    # Predominant analysis
                    if self._is_predominant(rn):
                        result.predominant_chords.append(
                            {
                                "measure": self._get_measure_number(ch, measures),
                                "chord": rn.figure,
                                "function": "predominant",
                                "leads_to": self._get_next_function(
                                    i, chords, score_key
                                ),
                            }
                        )

                    # Dominant preparation
                    if self._is_dominant_prep(prev_rn, rn):
                        result.dominant_preparations.append(
                            {
                                "measure": self._get_measure_number(ch, measures),
                                "preparation": prev_rn.figure,
                                "dominant": rn.figure,
                                "strength": self._get_prep_strength(prev_rn, rn),
                            }
                        )

                    # Deceptive resolution
                    if self._is_deceptive_resolution(prev_rn, rn):
                        result.deceptive_resolutions.append(
                            {
                                "measure": self._get_measure_number(ch, measures),
                                "expected": "I" if prev_rn.figure == "V" else "i",
                                "actual": rn.figure,
                                "type": self._get_deception_type(rn),
                            }
                        )

            except Exception:
                # Handle non-diatonic chords
                pass

        # Analyze cadences
        result.cadences = await self._analyze_cadences(score, measures)

        # Determine phrase model
        result.phrase_model = self._determine_phrase_model(result.cadences)

        # Calculate tonal strength
        result.tonal_strength = self._calculate_tonal_strength(result)

        return result

    async def analyze_voice_leading(
        self, score: stream.Score, strict: bool = False
    ) -> VoiceLeadingAnalysis:
        """
        Analyze voice leading including errors and smoothness.

        Args:
            score: Music21 score to analyze
            strict: Whether to apply strict counterpoint rules

        Returns:
            VoiceLeadingAnalysis object
        """
        result = VoiceLeadingAnalysis()

        # Separate into voices
        parts = score.parts
        if len(parts) < 2:
            return result  # Need at least 2 voices

        # Analyze each voice pair
        for i in range(len(parts)):
            for j in range(i + 1, len(parts)):
                voice1_notes = list(parts[i].flatten().notes)
                voice2_notes = list(parts[j].flatten().notes)

                # Align voices by offset
                aligned_pairs = self._align_voices(voice1_notes, voice2_notes)

                for k in range(len(aligned_pairs) - 1):
                    curr_pair = aligned_pairs[k]
                    next_pair = aligned_pairs[k + 1]

                    if all(curr_pair) and all(next_pair):
                        # Create VoiceLeading object
                        vl = voiceLeading.VoiceLeadingQuartet(
                            curr_pair[0], curr_pair[1], next_pair[0], next_pair[1]
                        )

                        # Check for errors
                        errors = self._check_voice_leading_errors(vl, strict)
                        for error in errors:
                            error["voices"] = f"{i+1}-{j+1}"
                            error["measure"] = self._get_offset_measure(
                                curr_pair[0].offset, score
                            )
                            result.errors.append(error)

                        # Analyze motion type
                        motion = self._analyze_motion(vl)
                        if motion["type"] == "parallel":
                            result.parallel_motions.append(
                                {
                                    "voices": f"{i+1}-{j+1}",
                                    "measure": self._get_offset_measure(
                                        curr_pair[0].offset, score
                                    ),
                                    "interval": motion["interval"],
                                    "perfect": motion["perfect"],
                                }
                            )

                        # Check voice crossing
                        if curr_pair[0].pitch > curr_pair[1].pitch:
                            result.voice_crossings.append(
                                {
                                    "voices": f"{i+1}-{j+1}",
                                    "measure": self._get_offset_measure(
                                        curr_pair[0].offset, score
                                    ),
                                    "pitches": [
                                        str(curr_pair[0].pitch),
                                        str(curr_pair[1].pitch),
                                    ],
                                }
                            )

        # Analyze voice ranges
        for i, part in enumerate(parts):
            notes = list(part.flatten().notes)
            if notes:
                result.voice_ranges[f"voice_{i+1}"] = {
                    "lowest": min(n.pitch.midi for n in notes),
                    "highest": max(n.pitch.midi for n in notes),
                    "average": np.mean([n.pitch.midi for n in notes]),
                    "tessitura": self._calculate_tessitura(notes),
                }

        # Calculate overall scores
        result.smoothness_score = self._calculate_smoothness(score)
        result.independence_score = self._calculate_independence(parts)

        return result

    async def analyze_jazz_harmony(
        self, score: stream.Score, include_tensions: bool = True
    ) -> JazzHarmonyAnalysis:
        """
        Analyze jazz harmony including extended chords and substitutions.

        Args:
            score: Music21 score to analyze
            include_tensions: Whether to analyze chord tensions

        Returns:
            JazzHarmonyAnalysis object
        """
        result = JazzHarmonyAnalysis()

        # Get all chords
        chords = list(score.flatten().getElementsByClass(chord.Chord))

        for i, ch in enumerate(chords):
            # Get jazz chord symbol
            symbol = self._get_jazz_chord_symbol(ch)
            result.chord_symbols.append(symbol)

            # Identify extended chords
            if self._is_extended_chord(ch):
                result.extended_chords.append(
                    {
                        "measure": self._get_chord_measure(ch, score),
                        "symbol": symbol,
                        "extensions": self._get_extensions(ch),
                        "alterations": self._get_alterations(ch),
                        "quality": self._get_jazz_quality(ch),
                    }
                )

            # Check for substitutions
            if i > 0:
                prev_ch = chords[i - 1]
                sub = self._detect_substitution(prev_ch, ch, i, chords)
                if sub:
                    result.substitutions.append(sub)

            # Analyze tensions if requested
            if include_tensions:
                tensions = self._analyze_tensions(ch)
                if tensions:
                    result.tension_analysis.append(
                        {
                            "measure": self._get_chord_measure(ch, score),
                            "chord": symbol,
                            "tensions": tensions,
                            "tension_resolution": self._get_tension_resolution(
                                ch, i, chords
                            ),
                        }
                    )

        # Detect modal interchange
        result.modal_interchanges = await self._detect_modal_interchange(score, chords)

        # Suggest chord scales
        for ch in chords:
            scale_options = self._suggest_chord_scales(ch, self.current_key)
            if scale_options:
                result.chord_scales.append(
                    {
                        "measure": self._get_chord_measure(ch, score),
                        "chord": self._get_jazz_chord_symbol(ch),
                        "scales": scale_options,
                    }
                )

        return result

    async def detect_harmonic_sequences(
        self, score: stream.Score, min_pattern_length: int = 2, min_occurrences: int = 2
    ) -> List[HarmonicSequence]:
        """
        Detect harmonic sequences in the score.

        Args:
            score: Music21 score to analyze
            min_pattern_length: Minimum length of pattern
            min_occurrences: Minimum times pattern must occur

        Returns:
            List of HarmonicSequence objects
        """
        sequences = []

        # Get chord progression
        chords = list(score.flatten().getElementsByClass(chord.Chord))
        if not chords:
            return sequences

        # Create Roman numeral progression
        score_key = score.analyze("key")
        roman_progression = []

        for ch in chords:
            try:
                rn = roman.romanNumeralFromChord(ch, score_key)
                roman_progression.append(rn.figure)
            except:
                roman_progression.append("?")

        # Find patterns
        for length in range(min_pattern_length, len(roman_progression) // 2):
            for start in range(len(roman_progression) - length):
                pattern = roman_progression[start : start + length]

                # Skip if pattern contains unknowns
                if "?" in pattern:
                    continue

                # Find occurrences
                occurrences = []
                for i in range(len(roman_progression) - length):
                    if roman_progression[i : i + length] == pattern:
                        occurrences.append(
                            {
                                "start_index": i,
                                "start_measure": self._get_chord_measure(
                                    chords[i], score
                                ),
                            }
                        )

                if len(occurrences) >= min_occurrences:
                    # Determine sequence type
                    seq_type = self._classify_sequence(pattern, occurrences, chords)

                    # Calculate intervallic relationship
                    intervals = []
                    for i in range(1, len(occurrences)):
                        root1 = chords[occurrences[i - 1]["start_index"]].root()
                        root2 = chords[occurrences[i]["start_index"]].root()
                        if root1 and root2:
                            intervals.append(interval.Interval(root1, root2).semitones)

                    sequences.append(
                        HarmonicSequence(
                            pattern=pattern,
                            intervals=intervals,
                            occurrences=occurrences,
                            sequence_type=seq_type,
                            direction=(
                                "descending"
                                if all(i < 0 for i in intervals)
                                else (
                                    "ascending"
                                    if all(i > 0 for i in intervals)
                                    else "mixed"
                                )
                            ),
                            real_or_tonal=(
                                "real" if len(set(intervals)) == 1 else "tonal"
                            ),
                        )
                    )

        # Remove duplicate/overlapping sequences
        sequences = self._filter_sequences(sequences)

        return sequences

    async def analyze_modulations(
        self, score: stream.Score, sensitivity: float = 0.7
    ) -> ModulationAnalysis:
        """
        Detect and analyze modulations.

        Args:
            score: Music21 score to analyze
            sensitivity: Detection sensitivity (0-1)

        Returns:
            ModulationAnalysis object
        """
        result = ModulationAnalysis()

        # Windowed key analysis
        measures = list(score.parts[0].getElementsByClass(stream.Measure))
        window_size = 4

        current_key = score.analyze("key")
        key_regions = []

        for i in range(0, len(measures) - window_size + 1):
            window = stream.Stream()
            for j in range(i, i + window_size):
                window.append(measures[j])

            window_key = window.analyze("key")
            confidence = window_key.correlationCoefficient

            if confidence > sensitivity:
                key_regions.append(
                    {
                        "start_measure": i + 1,
                        "end_measure": i + window_size,
                        "key": str(window_key),
                        "confidence": confidence,
                    }
                )

        # Consolidate regions and detect modulations
        consolidated = self._consolidate_key_regions(key_regions)

        for i in range(1, len(consolidated)):
            if consolidated[i]["key"] != consolidated[i - 1]["key"]:
                mod = {
                    "measure": consolidated[i]["start_measure"],
                    "from_key": consolidated[i - 1]["key"],
                    "to_key": consolidated[i]["key"],
                    "type": self._classify_modulation_type(
                        consolidated[i - 1]["key"], consolidated[i]["key"]
                    ),
                }

                # Find pivot chord if applicable
                pivot = self._find_pivot_chord(
                    score, consolidated[i - 1], consolidated[i]
                )
                if pivot:
                    mod["pivot_chord"] = pivot
                    result.pivot_chords.append(pivot)

                result.modulations.append(mod)

        # Detect specific modulation types
        await self._detect_chromatic_modulations(score, result)
        await self._detect_enharmonic_modulations(score, result)
        await self._detect_common_tone_modulations(score, result)

        # Identify tonicization regions
        result.tonicization_regions = self._find_tonicizations(score)

        result.key_areas = consolidated

        return result

    # Helper methods
    def _classify_function(self, rn: roman.RomanNumeral) -> HarmonicFunction:
        """Classify harmonic function of Roman numeral"""
        degree = rn.scaleDegree
        quality = rn.quality

        if degree == 1:
            return HarmonicFunction.TONIC
        elif degree == 5:
            return HarmonicFunction.DOMINANT
        elif degree == 4:
            return HarmonicFunction.SUBDOMINANT
        elif degree in [2, 4, 6]:
            return HarmonicFunction.PREDOMINANT
        else:
            return HarmonicFunction.TONIC  # Default

    def _is_tonic_prolongation(
        self, prev: roman.RomanNumeral, curr: roman.RomanNumeral
    ) -> bool:
        """Check if progression prolongs tonic"""
        tonic_chords = ["I", "i", "I6", "i6", "I64", "i64"]
        prolonging_chords = ["V", "V6", "V7", "vii°", "vii°6", "IV", "iv"]

        return (prev.figure in tonic_chords and curr.figure in prolonging_chords) or (
            prev.figure in prolonging_chords and curr.figure in tonic_chords
        )

    def _get_prolongation_type(
        self, prev: roman.RomanNumeral, curr: roman.RomanNumeral
    ) -> str:
        """Determine type of tonic prolongation"""
        if "V" in curr.figure:
            return "dominant"
        elif "IV" in curr.figure or "iv" in curr.figure:
            return "subdominant"
        elif "vii" in curr.figure:
            return "leading-tone"
        else:
            return "neighboring"

    def _is_predominant(self, rn: roman.RomanNumeral) -> bool:
        """Check if chord has predominant function"""
        predominants = [
            "ii",
            "II",
            "ii6",
            "II6",
            "ii7",
            "II7",
            "iv",
            "IV",
            "iv6",
            "IV6",
            "N6",
            "It+6",
            "Fr+6",
            "Ger+6",
        ]
        return any(p in rn.figure for p in predominants)

    def _is_dominant_prep(
        self, prev: roman.RomanNumeral, curr: roman.RomanNumeral
    ) -> bool:
        """Check if progression prepares dominant"""
        return self._is_predominant(prev) and "V" in curr.figure

    def _get_prep_strength(
        self, prep: roman.RomanNumeral, dom: roman.RomanNumeral
    ) -> float:
        """Calculate strength of dominant preparation"""
        # Stronger preps get higher scores
        if "ii" in prep.figure and "7" in prep.figure:
            return 0.9  # ii7 is strong prep
        elif "+6" in prep.figure:
            return 0.95  # Aug 6 is very strong
        elif "IV" in prep.figure or "iv" in prep.figure:
            return 0.7  # Subdominant is moderate
        else:
            return 0.5

    def _is_deceptive_resolution(
        self, prev: roman.RomanNumeral, curr: roman.RomanNumeral
    ) -> bool:
        """Check for deceptive resolution"""
        return "V" in prev.figure and curr.scaleDegree not in [1, 8]

    def _get_deception_type(self, rn: roman.RomanNumeral) -> str:
        """Classify type of deceptive resolution"""
        if rn.scaleDegree == 6:
            return "standard"  # V-vi
        elif rn.scaleDegree == 4:
            return "plagal"  # V-IV
        elif "b" in rn.figure:
            return "chromatic"
        else:
            return "other"

    async def _analyze_cadences(
        self, score: stream.Score, measures: List[stream.Measure]
    ) -> List[Dict[str, Any]]:
        """Analyze cadences in the score"""
        cadences = []

        # Simple cadence detection - look for phrase endings
        for i, measure in enumerate(measures):
            chords = list(measure.getElementsByClass(chord.Chord))
            if len(chords) >= 2:
                # Check last two chords of measure
                penult = chords[-2]
                final = chords[-1]

                try:
                    rn_penult = roman.romanNumeralFromChord(penult, self.current_key)
                    rn_final = roman.romanNumeralFromChord(final, self.current_key)

                    cadence_type = self._identify_cadence_type(rn_penult, rn_final)
                    if cadence_type:
                        cadences.append(
                            {
                                "measure": i + 1,
                                "type": cadence_type,
                                "progression": f"{rn_penult.figure}-{rn_final.figure}",
                                "strength": self._get_cadence_strength(cadence_type),
                            }
                        )
                except:
                    pass

        return cadences

    def _identify_cadence_type(
        self, penult: roman.RomanNumeral, final: roman.RomanNumeral
    ) -> Optional[str]:
        """Identify cadence type from two chords"""
        prog = f"{penult.figure}-{final.figure}"

        # Authentic cadences
        if "V" in penult.figure and final.scaleDegree == 1:
            if penult.figure == "V" and final.inversion() == 0:
                return "perfect_authentic"
            else:
                return "imperfect_authentic"

        # Half cadence
        elif "V" in final.figure:
            return "half"

        # Plagal
        elif (
            "IV" in penult.figure or "iv" in penult.figure
        ) and final.scaleDegree == 1:
            return "plagal"

        # Deceptive
        elif "V" in penult.figure and final.scaleDegree == 6:
            return "deceptive"

        return None

    def _get_cadence_strength(self, cadence_type: str) -> float:
        """Get relative strength of cadence"""
        strengths = {
            "perfect_authentic": 1.0,
            "imperfect_authentic": 0.7,
            "half": 0.5,
            "plagal": 0.6,
            "deceptive": 0.4,
        }
        return strengths.get(cadence_type, 0.3)

    def _determine_phrase_model(self, cadences: List[Dict]) -> str:
        """Determine overall phrase model from cadences"""
        if len(cadences) < 2:
            return "incomplete"

        # Look for period structure (HC then PAC)
        if (
            len(cadences) >= 2
            and cadences[0]["type"] == "half"
            and "authentic" in cadences[1]["type"]
        ):
            return "period"

        # Look for sentence structure
        elif len(cadences) >= 3:
            return "sentence"

        else:
            return "phrase_group"

    def _calculate_tonal_strength(self, analysis: FunctionalHarmonyAnalysis) -> float:
        """Calculate overall tonal strength"""
        score = 0.0

        # Strong cadences increase tonal strength
        for cadence in analysis.cadences:
            score += cadence["strength"]

        # Clear functional progressions
        for i in range(len(analysis.functions) - 1):
            if (
                analysis.functions[i] == HarmonicFunction.PREDOMINANT
                and analysis.functions[i + 1] == HarmonicFunction.DOMINANT
            ):
                score += 0.3
            elif (
                analysis.functions[i] == HarmonicFunction.DOMINANT
                and analysis.functions[i + 1] == HarmonicFunction.TONIC
            ):
                score += 0.5

        # Normalize
        return min(score / max(len(analysis.roman_numerals), 1), 1.0)

    def _get_measure_number(self, chord_obj: chord.Chord, measures: List) -> int:
        """Get measure number for a chord"""
        chord_offset = chord_obj.offset
        for i, measure in enumerate(measures):
            if measure.offset <= chord_offset < measure.offset + measure.quarterLength:
                return i + 1
        return 0

    def _get_next_function(
        self, index: int, chords: List[chord.Chord], key: key.Key
    ) -> str:
        """Get the function of the next chord"""
        if index + 1 < len(chords):
            try:
                next_rn = roman.romanNumeralFromChord(chords[index + 1], key)
                return next_rn.figure
            except:
                pass
        return "unknown"

    def _align_voices(
        self, voice1: List[note.Note], voice2: List[note.Note]
    ) -> List[Tuple[Optional[note.Note], Optional[note.Note]]]:
        """Align two voices by offset"""
        aligned = []
        i, j = 0, 0

        while i < len(voice1) or j < len(voice2):
            if i >= len(voice1):
                aligned.append((None, voice2[j]))
                j += 1
            elif j >= len(voice2):
                aligned.append((voice1[i], None))
                i += 1
            elif abs(voice1[i].offset - voice2[j].offset) < 0.1:
                aligned.append((voice1[i], voice2[j]))
                i += 1
                j += 1
            elif voice1[i].offset < voice2[j].offset:
                aligned.append((voice1[i], None))
                i += 1
            else:
                aligned.append((None, voice2[j]))
                j += 1

        return aligned

    def _check_voice_leading_errors(
        self, vl: voiceLeading.VoiceLeadingQuartet, strict: bool
    ) -> List[Dict[str, Any]]:
        """Check for voice leading errors"""
        errors = []

        # Parallel fifths/octaves
        if vl.parallelFifth():
            errors.append(
                {
                    "type": VoiceLeadingError.PARALLEL_FIFTHS,
                    "severity": "high" if strict else "medium",
                }
            )

        if vl.parallelOctave():
            errors.append(
                {"type": VoiceLeadingError.PARALLEL_OCTAVES, "severity": "high"}
            )

        # Hidden fifths/octaves (only in strict mode)
        if strict:
            if vl.hiddenFifth():
                errors.append(
                    {"type": VoiceLeadingError.DIRECT_FIFTHS, "severity": "medium"}
                )

            if vl.hiddenOctave():
                errors.append(
                    {"type": VoiceLeadingError.DIRECT_OCTAVES, "severity": "medium"}
                )

        # Large leaps
        for v in [vl.v1n1, vl.v2n1]:
            if v and vl.v1n2 and abs(v.pitch.midi - vl.v1n2.pitch.midi) > 12:
                errors.append(
                    {
                        "type": VoiceLeadingError.LARGE_LEAP,
                        "severity": "low",
                        "interval": abs(v.pitch.midi - vl.v1n2.pitch.midi),
                    }
                )

        return errors

    def _analyze_motion(self, vl: voiceLeading.VoiceLeadingQuartet) -> Dict[str, Any]:
        """Analyze motion type between voices"""
        motion_type = None

        if vl.parallelMotion():
            motion_type = "parallel"
        elif vl.contraryMotion():
            motion_type = "contrary"
        elif vl.obliqueMotion():
            motion_type = "oblique"
        elif vl.similarMotion():
            motion_type = "similar"

        # Get interval if parallel
        interval_obj = None
        perfect = False
        if motion_type == "parallel" and vl.v1n1 and vl.v2n1:
            interval_obj = interval.Interval(vl.v1n1, vl.v2n1)
            perfect = interval_obj.isPerfectConsonance()

        return {
            "type": motion_type,
            "interval": str(interval_obj) if interval_obj else None,
            "perfect": perfect,
        }

    def _get_offset_measure(self, offset: float, score: stream.Score) -> int:
        """Get measure number from offset"""
        for measure in score.parts[0].getElementsByClass(stream.Measure):
            if measure.offset <= offset < measure.offset + measure.quarterLength:
                return measure.number
        return 0

    def _calculate_tessitura(self, notes: List[note.Note]) -> Dict[str, float]:
        """Calculate tessitura (comfortable range) of voice"""
        if not notes:
            return {}

        pitches = [n.pitch.midi for n in notes]

        # Use 80th percentile range as tessitura
        return {
            "low": np.percentile(pitches, 10),
            "high": np.percentile(pitches, 90),
            "center": np.median(pitches),
        }

    def _calculate_smoothness(self, score: stream.Score) -> float:
        """Calculate voice leading smoothness score"""
        total_movement = 0
        note_count = 0

        for part in score.parts:
            notes = list(part.flatten().notes)
            for i in range(len(notes) - 1):
                if notes[i].isNote and notes[i + 1].isNote:
                    # Smaller intervals = smoother
                    semitones = abs(notes[i].pitch.midi - notes[i + 1].pitch.midi)
                    total_movement += min(semitones, 12)  # Cap at octave
                    note_count += 1

        if note_count == 0:
            return 0.0

        avg_movement = total_movement / note_count
        # Convert to 0-1 scale (2 = stepwise average)
        return max(0, 1 - (avg_movement - 2) / 10)

    def _calculate_independence(self, parts: List[stream.Part]) -> float:
        """Calculate voice independence score"""
        if len(parts) < 2:
            return 1.0

        independence_scores = []

        for i in range(len(parts)):
            for j in range(i + 1, len(parts)):
                # Compare rhythm patterns
                rhythm1 = [n.duration.quarterLength for n in parts[i].flatten().notes]
                rhythm2 = [n.duration.quarterLength for n in parts[j].flatten().notes]

                # Simple rhythm similarity check
                if rhythm1 and rhythm2:
                    min_len = min(len(rhythm1), len(rhythm2))
                    same_rhythm = sum(
                        1 for k in range(min_len) if rhythm1[k] == rhythm2[k]
                    )
                    rhythm_independence = 1 - (same_rhythm / min_len)
                    independence_scores.append(rhythm_independence)

        return np.mean(independence_scores) if independence_scores else 0.5

    def _get_chord_measure(self, chord: chord.Chord, score: stream.Score) -> int:
        """Get measure number for chord"""
        return self._get_offset_measure(chord.offset, score)

    def _get_jazz_chord_symbol(self, ch: chord.Chord) -> str:
        """Convert chord to jazz symbol"""
        try:
            # Try to get harmony symbol
            harmony_obj = harmony.chordSymbolFromChord(ch)
            return harmony_obj.figure
        except:
            # Fallback to basic symbol
            return ch.pitchedCommonName

    def _is_extended_chord(self, ch: chord.Chord) -> bool:
        """Check if chord has extensions beyond 7th"""
        chord_tones = ch.normalOrder
        if len(chord_tones) > 4:
            return True

        # Check for 9th, 11th, 13th
        root = ch.root()
        if root:
            for p in ch.pitches:
                interval_from_root = interval.Interval(root, p).semitones % 12
                if interval_from_root in [2, 5, 9]:  # 9th, 11th, 13th
                    return True

        return False

    def _get_extensions(self, ch: chord.Chord) -> List[str]:
        """Get chord extensions"""
        extensions = []
        root = ch.root()

        if not root:
            return extensions

        for p in ch.pitches:
            semitones = interval.Interval(root, p).semitones % 12
            if semitones == 2:
                extensions.append("9")
            elif semitones == 5:
                extensions.append("11")
            elif semitones == 9:
                extensions.append("13")

        return extensions

    def _get_alterations(self, ch: chord.Chord) -> List[str]:
        """Get chord alterations"""
        alterations = []

        # This would need more sophisticated analysis
        # For now, return empty list
        return alterations

    def _get_jazz_quality(self, ch: chord.Chord) -> str:
        """Get jazz chord quality"""
        if ch.isMajorTriad():
            return "major"
        elif ch.isMinorTriad():
            return "minor"
        elif ch.isDominantSeventh():
            return "dominant"
        elif ch.isDiminishedSeventh():
            return "diminished"
        else:
            return "other"

    def _detect_substitution(
        self,
        prev: chord.Chord,
        curr: chord.Chord,
        index: int,
        all_chords: List[chord.Chord],
    ) -> Optional[Dict[str, Any]]:
        """Detect chord substitution"""
        # Tritone substitution
        if prev.isDominantSeventh() and curr.isDominantSeventh():
            root_interval = interval.Interval(prev.root(), curr.root())
            if root_interval.semitones == 6:  # Tritone
                return {
                    "measure": self._get_chord_measure(curr, None),
                    "type": ChordSubstitutionType.TRITONE,
                    "original": self._get_jazz_chord_symbol(prev),
                    "substitute": self._get_jazz_chord_symbol(curr),
                    "target": self._get_expected_resolution(prev),
                }

        return None

    def _get_expected_resolution(self, dom_chord: chord.Chord) -> str:
        """Get expected resolution of dominant chord"""
        root = dom_chord.root()
        if root:
            # Dominant resolves down a fifth
            resolution_root = root.transpose("-P5")
            return f"{resolution_root.name}maj7"
        return "unknown"

    def _analyze_tensions(self, ch: chord.Chord) -> List[str]:
        """Analyze chord tensions"""
        tensions = []

        # Get extensions
        extensions = self._get_extensions(ch)

        # Determine if tensions are altered
        for ext in extensions:
            tensions.append(f"natural {ext}")

        return tensions

    def _get_tension_resolution(
        self, ch: chord.Chord, index: int, all_chords: List[chord.Chord]
    ) -> Optional[str]:
        """Analyze how tensions resolve"""
        if index + 1 < len(all_chords):
            next_chord = all_chords[index + 1]
            # Simplified - would need voice leading analysis
            return "resolved"
        return None

    async def _detect_modal_interchange(
        self, score: stream.Score, chords: List[chord.Chord]
    ) -> List[Dict[str, Any]]:
        """Detect modal interchange (borrowed chords)"""
        interchanges = []

        if not self.current_key:
            self.current_key = score.analyze("key")

        parallel_key = self.current_key.parallel()

        for i, ch in enumerate(chords):
            # Check if chord belongs to parallel mode
            try:
                # Try in current key
                rn_current = roman.romanNumeralFromChord(ch, self.current_key)

                # Try in parallel key
                rn_parallel = roman.romanNumeralFromChord(ch, parallel_key)

                # If chord fits better in parallel key, it's borrowed
                if (
                    rn_parallel.pitches == ch.pitches
                    and rn_current.pitches != ch.pitches
                ):
                    interchanges.append(
                        {
                            "measure": self._get_chord_measure(ch, score),
                            "chord": self._get_jazz_chord_symbol(ch),
                            "borrowed_from": str(parallel_key),
                            "function_in_parallel": rn_parallel.figure,
                            "modal_color": (
                                "darker"
                                if self.current_key.mode == "major"
                                else "brighter"
                            ),
                        }
                    )
            except:
                pass

        return interchanges

    def _suggest_chord_scales(
        self, ch: chord.Chord, key_context: Optional[key.Key]
    ) -> List[Dict[str, Any]]:
        """Suggest appropriate scales for improvisation"""
        suggestions = []

        # Simplified version - would need comprehensive mapping
        if ch.isMajorTriad():
            suggestions.append({"scale": "Ionian", "priority": "primary"})
            suggestions.append({"scale": "Lydian", "priority": "alternative"})
        elif ch.isMinorTriad():
            suggestions.append({"scale": "Dorian", "priority": "primary"})
            suggestions.append({"scale": "Aeolian", "priority": "alternative"})
        elif ch.isDominantSeventh():
            suggestions.append({"scale": "Mixolydian", "priority": "primary"})
            suggestions.append({"scale": "Altered", "priority": "alternative"})

        return suggestions

    def _classify_sequence(
        self, pattern: List[str], occurrences: List[Dict], chords: List[chord.Chord]
    ) -> str:
        """Classify type of harmonic sequence"""
        # Check for common sequence types
        if all("V" in p or "I" in p for p in pattern):
            return "circle_of_fifths"
        elif any("ii" in p for p in pattern) and any("V" in p for p in pattern):
            return "ii_V_sequence"
        else:
            return "sequential_progression"

    def _filter_sequences(
        self, sequences: List[HarmonicSequence]
    ) -> List[HarmonicSequence]:
        """Remove overlapping sequences, keeping the most significant"""
        if not sequences:
            return sequences

        # Sort by number of occurrences and pattern length
        sequences.sort(key=lambda s: (len(s.occurrences), len(s.pattern)), reverse=True)

        filtered = []
        used_positions = set()

        for seq in sequences:
            # Check if this sequence overlaps with already selected ones
            positions = set()
            for occ in seq.occurrences:
                for i in range(len(seq.pattern)):
                    positions.add(occ["start_index"] + i)

            if not positions & used_positions:
                filtered.append(seq)
                used_positions.update(positions)

        return filtered

    def _consolidate_key_regions(
        self, regions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Consolidate adjacent regions with same key"""
        if not regions:
            return regions

        consolidated = [regions[0]]

        for region in regions[1:]:
            if (
                region["key"] == consolidated[-1]["key"]
                and region["start_measure"] <= consolidated[-1]["end_measure"] + 1
            ):
                # Extend previous region
                consolidated[-1]["end_measure"] = region["end_measure"]
                consolidated[-1]["confidence"] = max(
                    consolidated[-1]["confidence"], region["confidence"]
                )
            else:
                consolidated.append(region)

        return consolidated

    def _classify_modulation_type(self, from_key: str, to_key: str) -> str:
        """Classify the type of modulation"""
        from_key_obj = key.Key(from_key)
        to_key_obj = key.Key(to_key)

        # Calculate relationship
        interval_between = interval.Interval(from_key_obj.tonic, to_key_obj.tonic)

        # Common modulations
        if interval_between.semitones == 7:  # Up a fifth
            return "dominant"
        elif interval_between.semitones == 5:  # Up a fourth
            return "subdominant"
        elif interval_between.semitones == 0 and from_key_obj.mode != to_key_obj.mode:
            return "parallel"
        elif abs(interval_between.semitones) == 2:
            return "step"
        elif abs(interval_between.semitones) in [3, 4]:
            return "mediant"
        else:
            return "distant"

    def _find_pivot_chord(
        self,
        score: stream.Score,
        from_region: Dict[str, Any],
        to_region: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Find pivot chord between two key areas"""
        # Get chords around modulation point
        transition_measure = to_region["start_measure"] - 1

        # This would need to examine specific chords
        # For now, return None
        return None

    async def _detect_chromatic_modulations(
        self, score: stream.Score, result: ModulationAnalysis
    ) -> None:
        """Detect chromatic modulations"""
        # Simplified - would need chromatic voice leading analysis
        pass

    async def _detect_enharmonic_modulations(
        self, score: stream.Score, result: ModulationAnalysis
    ) -> None:
        """Detect enharmonic modulations"""
        # Look for diminished 7th and augmented 6th chords
        # that could be reinterpreted enharmonically
        pass

    async def _detect_common_tone_modulations(
        self, score: stream.Score, result: ModulationAnalysis
    ) -> None:
        """Detect common tone modulations"""
        # Look for sustained pitches across key changes
        pass

    def _find_tonicizations(self, score: stream.Score) -> List[Dict[str, Any]]:
        """Find brief tonicizations"""
        tonicizations = []

        # Look for secondary dominants and their resolutions
        # This would overlap with secondary dominant detection

        return tonicizations
