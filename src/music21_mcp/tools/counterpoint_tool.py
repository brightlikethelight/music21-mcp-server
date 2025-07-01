"""
Counterpoint Generator Tool - Create species counterpoint
Follows Fux rules with options for strict or relaxed application
"""

import logging
import random
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from music21 import interval
from music21 import key as m21_key
from music21 import note, pitch, stream

from .base_tool import BaseTool

logger = logging.getLogger(__name__)


class Species(Enum):
    """Types of species counterpoint"""

    FIRST = "1:1"  # Note against note
    SECOND = "2:1"  # Two notes against one
    THIRD = "3:1"  # Three notes against one
    FOURTH = "4:1"  # Four notes against one
    FIFTH = "florid"  # Mixed rhythms


class CounterpointRule(Enum):
    """Counterpoint rules that can be enforced or relaxed"""

    NO_PARALLEL_FIFTHS = "no_parallel_fifths"
    NO_PARALLEL_OCTAVES = "no_parallel_octaves"
    NO_DIRECT_FIFTHS_OCTAVES = "no_direct_fifths_octaves"
    CONSONANT_DOWNBEATS = "consonant_downbeats"
    PROPER_CADENCE = "proper_cadence"
    NO_LARGE_LEAPS = "no_large_leaps"
    LEAP_RECOVERY = "leap_recovery"
    NO_REPEATED_NOTES = "no_repeated_notes"
    CLIMAX_ONCE = "climax_once"
    MOSTLY_STEPWISE = "mostly_stepwise"
    NO_CROSS_VOICES = "no_cross_voices"
    STAY_IN_MODE = "stay_in_mode"


class CounterpointGeneratorTool(BaseTool):
    """
    Counterpoint generator providing:
    1. All five species of counterpoint
    2. Strict Fux rules or relaxed modern rules
    3. Multiple voice support (2-4 parts)
    4. Above or below cantus firmus
    5. Rule violation checking with explanations
    """

    def __init__(self, score_manager: Dict[str, Any]):
        super().__init__(score_manager)

        # Define consonant and dissonant intervals
        self.consonances = {
            0: "P1",  # Unison
            3: "m3",  # Minor third
            4: "M3",  # Major third
            5: "P4",  # Perfect fourth (contextual)
            7: "P5",  # Perfect fifth
            8: "m6",  # Minor sixth
            9: "M6",  # Major sixth
            12: "P8",  # Octave
            15: "m10",  # Minor tenth
            16: "M10",  # Major tenth
        }

        self.perfect_consonances = {0, 7, 12}  # Unison, fifth, octave
        self.imperfect_consonances = {3, 4, 8, 9, 15, 16}  # Thirds and sixths

        # Species-specific rules
        self.species_rules = {
            Species.FIRST: {
                "rhythm_ratio": 1,
                "allow_dissonance": False,
                "passing_tones": False,
            },
            Species.SECOND: {
                "rhythm_ratio": 2,
                "allow_dissonance": True,  # On weak beats
                "passing_tones": True,
            },
            Species.THIRD: {
                "rhythm_ratio": 3,
                "allow_dissonance": True,
                "passing_tones": True,
                "neighbor_notes": True,
            },
            Species.FOURTH: {
                "rhythm_ratio": 4,
                "allow_dissonance": True,
                "suspensions": True,
                "passing_tones": True,
            },
            Species.FIFTH: {
                "rhythm_ratio": "mixed",
                "allow_dissonance": True,
                "all_embellishments": True,
            },
        }

    async def execute(
        self,
        score_id: str,
        species: str = "first",
        voice_position: str = "above",
        rule_set: str = "strict",
        custom_rules: Optional[List[str]] = None,
        mode: str = "major",
    ) -> Dict[str, Any]:
        """
        Generate counterpoint for a given cantus firmus

        Args:
            score_id: ID of the cantus firmus score
            species: Species type ('first', 'second', 'third', 'fourth', 'fifth')
            voice_position: Position relative to CF ('above', 'below')
            rule_set: Rule strictness ('strict', 'relaxed', 'custom')
            custom_rules: List of specific rules to enforce if rule_set is 'custom'
            mode: Modal context ('major', 'minor', 'dorian', 'phrygian', etc.)
        """
        # Validate inputs
        error = self.validate_inputs(
            score_id=score_id,
            species=species,
            voice_position=voice_position,
            rule_set=rule_set,
        )
        if error:
            return self.create_error_response(error)

        with self.error_handling(f"Counterpoint generation for '{score_id}'"):
            cf_score = self.get_score(score_id)

            self.report_progress(0.1, "Extracting cantus firmus")

            # Extract cantus firmus
            cantus_firmus = self._extract_cantus_firmus(cf_score)
            if not cantus_firmus:
                return self.create_error_response("No valid cantus firmus found")

            # Determine key/mode
            cf_key = self._determine_key(cantus_firmus, mode)

            # Convert species string to enum
            species_enum = self._get_species_enum(species)

            self.report_progress(0.3, f"Generating {species} species counterpoint")

            # Generate counterpoint based on species
            if species_enum == Species.FIRST:
                counterpoint = await self._generate_first_species(
                    cantus_firmus, cf_key, voice_position, rule_set, custom_rules
                )
            elif species_enum == Species.SECOND:
                counterpoint = await self._generate_second_species(
                    cantus_firmus, cf_key, voice_position, rule_set, custom_rules
                )
            elif species_enum == Species.THIRD:
                counterpoint = await self._generate_third_species(
                    cantus_firmus, cf_key, voice_position, rule_set, custom_rules
                )
            elif species_enum == Species.FOURTH:
                counterpoint = await self._generate_fourth_species(
                    cantus_firmus, cf_key, voice_position, rule_set, custom_rules
                )
            elif species_enum == Species.FIFTH:
                counterpoint = await self._generate_fifth_species(
                    cantus_firmus, cf_key, voice_position, rule_set, custom_rules
                )
            else:
                return self.create_error_response(f"Unknown species: {species}")

            self.report_progress(0.7, "Checking counterpoint rules")

            # Check rules and generate report
            rule_report = self._check_counterpoint_rules(
                cantus_firmus,
                counterpoint["notes"],
                species_enum,
                rule_set,
                custom_rules,
            )

            self.report_progress(0.9, "Creating score")

            # Create final score
            final_score = self._create_counterpoint_score(
                cantus_firmus, counterpoint["notes"], voice_position, species_enum
            )

            # Store the result
            result_id = f"{score_id}_counterpoint_{species}"
            self.score_manager[result_id] = final_score

            self.report_progress(1.0, "Counterpoint complete")

            return self.create_success_response(
                counterpoint_score_id=result_id,
                species=species,
                voice_position=voice_position,
                rule_set=rule_set,
                mode=str(cf_key),
                rule_violations=rule_report["violations"],
                rule_compliance_score=rule_report["compliance_score"],
                interval_analysis=counterpoint["interval_analysis"],
                melodic_analysis=counterpoint["melodic_analysis"],
                explanations=self._generate_explanations(species_enum, rule_report),
            )

    def validate_inputs(
        self, score_id: str, species: str, voice_position: str, rule_set: str, **kwargs
    ) -> Optional[str]:
        """Validate input parameters"""
        error = self.check_score_exists(score_id)
        if error:
            return error

        valid_species = ["first", "second", "third", "fourth", "fifth"]
        if species not in valid_species:
            return (
                f"Invalid species: {species}. Choose from: {', '.join(valid_species)}"
            )

        if voice_position not in ["above", "below"]:
            return "voice_position must be 'above' or 'below'"

        if rule_set not in ["strict", "relaxed", "custom"]:
            return "rule_set must be 'strict', 'relaxed', or 'custom'"

        return None

    def _extract_cantus_firmus(self, score: stream.Score) -> List[note.Note]:
        """Extract the cantus firmus melody"""
        # Get the first part or flattened notes
        if hasattr(score, "parts") and len(score.parts) > 0:
            cf_part = score.parts[0]
        else:
            cf_part = score

        # Extract notes
        cf_notes = []
        for element in cf_part.flatten():
            if isinstance(element, note.Note):
                cf_notes.append(element)

        return cf_notes

    def _determine_key(self, cantus_firmus: List[note.Note], mode: str) -> m21_key.Key:
        """Determine the key/mode of the cantus firmus"""
        try:
            # Create stream for analysis
            cf_stream = stream.Stream(cantus_firmus)

            if mode in ["major", "minor"]:
                # Standard key detection
                detected_key = cf_stream.analyze("key")
                if mode == "minor" and detected_key.mode == "major":
                    detected_key = detected_key.relative
                elif mode == "major" and detected_key.mode == "minor":
                    detected_key = detected_key.relative
                return detected_key
            else:
                # Modal detection (simplified)
                # Get the final note as likely tonic
                final_note = cantus_firmus[-1].pitch

                if mode == "dorian":
                    return m21_key.Key(final_note.name, "minor")
                elif mode == "phrygian":
                    return m21_key.Key(final_note.name, "minor")
                elif mode == "lydian":
                    return m21_key.Key(final_note.name, "major")
                elif mode == "mixolydian":
                    return m21_key.Key(final_note.name, "major")
                else:
                    return m21_key.Key(final_note.name, "major")

        except Exception as e:
            logger.error(f"Key determination failed: {e}")
            # Default to C major
            return m21_key.Key("C", "major")

    def _get_species_enum(self, species: str) -> Species:
        """Convert species string to enum"""
        mapping = {
            "first": Species.FIRST,
            "second": Species.SECOND,
            "third": Species.THIRD,
            "fourth": Species.FOURTH,
            "fifth": Species.FIFTH,
        }
        return mapping.get(species, Species.FIRST)

    async def _generate_first_species(
        self,
        cantus_firmus: List[note.Note],
        cf_key: m21_key.Key,
        voice_position: str,
        rule_set: str,
        custom_rules: Optional[List[str]],
    ) -> Dict[str, Any]:
        """Generate first species (1:1) counterpoint"""
        counterpoint_notes = []
        interval_analysis = []

        # Determine starting interval
        if voice_position == "above":
            start_intervals = [0, 7, 12]  # Unison, fifth, octave
        else:
            start_intervals = [-12, -7, 0]  # Octave, fifth, unison below

        # Generate note for each CF note
        for i, cf_note in enumerate(cantus_firmus):
            self.report_progress(
                0.3 + (0.4 * i / len(cantus_firmus)),
                f"Generating note {i+1}/{len(cantus_firmus)}",
            )

            if i == 0:
                # First note - perfect consonance
                interval_choice = random.choice(start_intervals)
                cp_pitch = pitch.Pitch(midi=cf_note.pitch.midi + interval_choice)
            elif i == len(cantus_firmus) - 1:
                # Last note - octave or unison
                if voice_position == "above":
                    interval_choice = random.choice([0, 12])
                else:
                    interval_choice = random.choice([-12, 0])
                cp_pitch = pitch.Pitch(midi=cf_note.pitch.midi + interval_choice)
            else:
                # Middle notes - choose best consonance
                cp_pitch = self._choose_first_species_note(
                    cf_note,
                    cf_key,
                    voice_position,
                    counterpoint_notes,
                    cantus_firmus,
                    i,
                    rule_set,
                    custom_rules,
                )

            # Create note with same duration as CF
            cp_note = note.Note(cp_pitch, quarterLength=cf_note.duration.quarterLength)
            counterpoint_notes.append(cp_note)

            # Analyze interval
            intv = interval.Interval(noteStart=cf_note, noteEnd=cp_note)
            interval_analysis.append(
                {
                    "interval": intv.name,
                    "semitones": intv.semitones,
                    "consonance": abs(intv.semitones % 12) in self.consonances,
                }
            )

        # Melodic analysis
        melodic_analysis = self._analyze_melodic_line(counterpoint_notes)

        return {
            "notes": counterpoint_notes,
            "interval_analysis": interval_analysis,
            "melodic_analysis": melodic_analysis,
        }

    def _choose_first_species_note(
        self,
        cf_note: note.Note,
        cf_key: m21_key.Key,
        voice_position: str,
        previous_notes: List[note.Note],
        cantus_firmus: List[note.Note],
        cf_index: int,
        rule_set: str,
        custom_rules: Optional[List[str]],
    ) -> pitch.Pitch:
        """Choose appropriate note for first species counterpoint"""
        # Get available pitches in key
        available_pitches = self._get_scale_pitches(
            cf_key, cf_note.pitch, voice_position
        )

        # Filter by consonance
        consonant_pitches = []
        for p in available_pitches:
            interval_semitones = abs(p.midi - cf_note.pitch.midi) % 12
            if interval_semitones in self.consonances:
                consonant_pitches.append(p)

        if not consonant_pitches:
            consonant_pitches = available_pitches

        # Score each option
        best_pitch = None
        best_score = -1000

        for candidate in consonant_pitches:
            score = 0

            # Prefer imperfect consonances
            interval_semitones = abs(candidate.midi - cf_note.pitch.midi) % 12
            if interval_semitones in self.imperfect_consonances:
                score += 20

            # Check voice leading if not first note
            if previous_notes:
                prev_note = previous_notes[-1]
                motion = abs(candidate.midi - prev_note.pitch.midi)

                # Prefer stepwise motion
                if motion <= 2:
                    score += 15
                elif motion <= 4:
                    score += 10
                elif motion > 7:
                    score -= 10  # Penalize large leaps

                # Check for parallel fifths/octaves
                if cf_index > 0:
                    prev_cf = cantus_firmus[cf_index - 1]
                    prev_interval = (prev_note.pitch.midi - prev_cf.pitch.midi) % 12
                    curr_interval = (candidate.midi - cf_note.pitch.midi) % 12

                    if prev_interval == curr_interval and prev_interval in [0, 7]:
                        if rule_set == "strict":
                            score -= 100  # Heavily penalize parallels
                        else:
                            score -= 30

            # Avoid repetition
            if previous_notes and candidate.midi == previous_notes[-1].pitch.midi:
                score -= 5

            # Update best
            if score > best_score:
                best_score = score
                best_pitch = candidate

        return best_pitch or consonant_pitches[0]

    def _get_scale_pitches(
        self, key: m21_key.Key, reference_pitch: pitch.Pitch, voice_position: str
    ) -> List[pitch.Pitch]:
        """Get available pitches from the scale"""
        scale = key.getScale()

        # Determine range based on voice position
        if voice_position == "above":
            min_midi = reference_pitch.midi
            max_midi = reference_pitch.midi + 15  # About a tenth
        else:
            min_midi = reference_pitch.midi - 15
            max_midi = reference_pitch.midi

        available = []

        # Get scale pitches in range
        for midi in range(min_midi, max_midi + 1):
            p = pitch.Pitch(midi=midi)
            if scale.isScaleDegree(p):
                available.append(p)

        return available

    async def _generate_second_species(
        self,
        cantus_firmus: List[note.Note],
        cf_key: m21_key.Key,
        voice_position: str,
        rule_set: str,
        custom_rules: Optional[List[str]],
    ) -> Dict[str, Any]:
        """Generate second species (2:1) counterpoint"""
        counterpoint_notes = []
        interval_analysis = []

        for i, cf_note in enumerate(cantus_firmus):
            # Generate two notes for each CF note
            for beat in range(2):
                is_strong_beat = beat == 0

                if i == 0 and beat == 0:
                    # First note
                    cp_pitch = self._choose_opening_pitch(cf_note, voice_position)
                elif i == len(cantus_firmus) - 1 and beat == 0:
                    # Penultimate note - leading tone
                    cp_pitch = self._choose_penultimate_pitch(
                        cf_note, cf_key, voice_position
                    )
                elif i == len(cantus_firmus) - 1 and beat == 1:
                    # Final note
                    cp_pitch = self._choose_final_pitch(cf_note, voice_position)
                else:
                    # Middle notes
                    cp_pitch = self._choose_second_species_note(
                        cf_note,
                        cf_key,
                        voice_position,
                        counterpoint_notes,
                        is_strong_beat,
                        rule_set,
                        custom_rules,
                    )

                # Create note with half duration
                cp_note = note.Note(
                    cp_pitch, quarterLength=cf_note.duration.quarterLength / 2
                )
                counterpoint_notes.append(cp_note)

                # Analyze interval
                intv = interval.Interval(noteStart=cf_note, noteEnd=cp_note)
                interval_analysis.append(
                    {
                        "interval": intv.name,
                        "semitones": intv.semitones,
                        "beat": "strong" if is_strong_beat else "weak",
                        "consonance": abs(intv.semitones % 12) in self.consonances,
                    }
                )

        melodic_analysis = self._analyze_melodic_line(counterpoint_notes)

        return {
            "notes": counterpoint_notes,
            "interval_analysis": interval_analysis,
            "melodic_analysis": melodic_analysis,
        }

    def _choose_second_species_note(
        self,
        cf_note: note.Note,
        cf_key: m21_key.Key,
        voice_position: str,
        previous_notes: List[note.Note],
        is_strong_beat: bool,
        rule_set: str,
        custom_rules: Optional[List[str]],
    ) -> pitch.Pitch:
        """Choose note for second species with passing tones allowed"""
        available_pitches = self._get_scale_pitches(
            cf_key, cf_note.pitch, voice_position
        )

        if is_strong_beat:
            # Strong beat - must be consonant
            candidates = []
            for p in available_pitches:
                interval_semitones = abs(p.midi - cf_note.pitch.midi) % 12
                if interval_semitones in self.consonances:
                    candidates.append(p)
        else:
            # Weak beat - can be dissonant if passing
            candidates = available_pitches

        # Choose based on melodic considerations
        if previous_notes:
            prev_pitch = previous_notes[-1].pitch

            # Prefer stepwise motion
            stepwise_candidates = [
                p for p in candidates if abs(p.midi - prev_pitch.midi) <= 2
            ]

            if stepwise_candidates:
                return random.choice(stepwise_candidates)

        return random.choice(candidates) if candidates else available_pitches[0]

    async def _generate_third_species(
        self,
        cantus_firmus: List[note.Note],
        cf_key: m21_key.Key,
        voice_position: str,
        rule_set: str,
        custom_rules: Optional[List[str]],
    ) -> Dict[str, Any]:
        """Generate third species (3:1) counterpoint"""
        counterpoint_notes = []
        interval_analysis = []

        for i, cf_note in enumerate(cantus_firmus):
            # Generate three notes for each CF note
            for beat in range(3):
                # Similar logic to second species but with 3 notes
                # First beat consonant, others can use passing/neighbor tones
                if i == 0 and beat == 0:
                    cp_pitch = self._choose_opening_pitch(cf_note, voice_position)
                elif i == len(cantus_firmus) - 1:
                    # Special handling for final measure
                    if beat == 0:
                        cp_pitch = self._choose_penultimate_pitch(
                            cf_note, cf_key, voice_position
                        )
                    else:
                        cp_pitch = self._choose_final_pitch(cf_note, voice_position)
                        if beat == 2:
                            break  # End early
                else:
                    is_strong = beat == 0
                    cp_pitch = self._choose_third_species_note(
                        cf_note,
                        cf_key,
                        voice_position,
                        counterpoint_notes,
                        beat,
                        is_strong,
                        rule_set,
                        custom_rules,
                    )

                cp_note = note.Note(
                    cp_pitch, quarterLength=cf_note.duration.quarterLength / 3
                )
                counterpoint_notes.append(cp_note)

        melodic_analysis = self._analyze_melodic_line(counterpoint_notes)

        return {
            "notes": counterpoint_notes,
            "interval_analysis": interval_analysis,
            "melodic_analysis": melodic_analysis,
        }

    def _choose_third_species_note(
        self,
        cf_note: note.Note,
        cf_key: m21_key.Key,
        voice_position: str,
        previous_notes: List[note.Note],
        beat: int,
        is_strong: bool,
        rule_set: str,
        custom_rules: Optional[List[str]],
    ) -> pitch.Pitch:
        """Choose note for third species with more embellishments"""
        available_pitches = self._get_scale_pitches(
            cf_key, cf_note.pitch, voice_position
        )

        if is_strong:
            # First beat - consonant
            candidates = [
                p
                for p in available_pitches
                if abs(p.midi - cf_note.pitch.midi) % 12 in self.consonances
            ]
        else:
            # Can use passing tones, neighbor notes
            candidates = available_pitches

            if previous_notes and beat == 1:
                # Possibly use neighbor note
                prev_pitch = previous_notes[-1].pitch
                neighbors = [
                    p for p in candidates if abs(p.midi - prev_pitch.midi) == 1
                ]
                if neighbors and random.random() < 0.3:
                    return random.choice(neighbors)

        # Default to stepwise motion when possible
        if previous_notes:
            prev_pitch = previous_notes[-1].pitch
            stepwise = [p for p in candidates if abs(p.midi - prev_pitch.midi) <= 2]
            if stepwise:
                return random.choice(stepwise)

        return random.choice(candidates) if candidates else available_pitches[0]

    async def _generate_fourth_species(
        self,
        cantus_firmus: List[note.Note],
        cf_key: m21_key.Key,
        voice_position: str,
        rule_set: str,
        custom_rules: Optional[List[str]],
    ) -> Dict[str, Any]:
        """Generate fourth species (syncopated) counterpoint"""
        counterpoint_notes = []
        interval_analysis = []

        # Fourth species uses suspensions
        for i, cf_note in enumerate(cantus_firmus):
            if i == 0:
                # First measure - start with rest then consonance
                rest = note.Rest(quarterLength=cf_note.duration.quarterLength / 2)
                counterpoint_notes.append(rest)

                cp_pitch = self._choose_opening_pitch(cf_note, voice_position)
                cp_note = note.Note(
                    cp_pitch, quarterLength=cf_note.duration.quarterLength / 2
                )
                counterpoint_notes.append(cp_note)
            elif i == len(cantus_firmus) - 1:
                # Final measure - resolve to tonic
                cp_pitch = self._choose_final_pitch(cf_note, voice_position)
                cp_note = note.Note(
                    cp_pitch, quarterLength=cf_note.duration.quarterLength
                )
                counterpoint_notes.append(cp_note)
            else:
                # Create suspension
                suspension_notes = self._create_suspension(
                    cf_note,
                    cantus_firmus[i - 1] if i > 0 else None,
                    cf_key,
                    voice_position,
                    counterpoint_notes,
                    rule_set,
                )
                counterpoint_notes.extend(suspension_notes)

        melodic_analysis = self._analyze_melodic_line(counterpoint_notes)

        return {
            "notes": counterpoint_notes,
            "interval_analysis": interval_analysis,
            "melodic_analysis": melodic_analysis,
        }

    def _create_suspension(
        self,
        cf_note: note.Note,
        prev_cf: Optional[note.Note],
        cf_key: m21_key.Key,
        voice_position: str,
        previous_notes: List[note.Note],
        rule_set: str,
    ) -> List[note.Note]:
        """Create a suspension figure"""
        notes = []

        # Preparation (consonant)
        if previous_notes and not isinstance(previous_notes[-1], note.Rest):
            prep_pitch = previous_notes[-1].pitch
        else:
            prep_pitch = self._choose_consonant_pitch(cf_note, cf_key, voice_position)

        # Suspension (becomes dissonant)
        suspension_note = note.Note(
            prep_pitch, quarterLength=cf_note.duration.quarterLength / 2
        )
        suspension_note.tie = note.Tie("start")
        notes.append(suspension_note)

        # Resolution (step down to consonance)
        resolution_pitch = pitch.Pitch(midi=prep_pitch.midi - 1)  # Step down

        # Ensure resolution is consonant
        while (
            abs(resolution_pitch.midi - cf_note.pitch.midi) % 12 not in self.consonances
        ):
            resolution_pitch.midi -= 1
            if resolution_pitch.midi < cf_note.pitch.midi - 12:
                # Fallback to consonant interval
                resolution_pitch = self._choose_consonant_pitch(
                    cf_note, cf_key, voice_position
                )
                break

        resolution_note = note.Note(
            resolution_pitch, quarterLength=cf_note.duration.quarterLength / 2
        )
        notes.append(resolution_note)

        return notes

    def _choose_consonant_pitch(
        self, cf_note: note.Note, cf_key: m21_key.Key, voice_position: str
    ) -> pitch.Pitch:
        """Choose a consonant pitch above or below CF note"""
        available = self._get_scale_pitches(cf_key, cf_note.pitch, voice_position)
        consonant = [
            p
            for p in available
            if abs(p.midi - cf_note.pitch.midi) % 12 in self.consonances
        ]

        if consonant:
            # Prefer imperfect consonances
            imperfect = [
                p
                for p in consonant
                if abs(p.midi - cf_note.pitch.midi) % 12 in self.imperfect_consonances
            ]
            if imperfect:
                return random.choice(imperfect)
            return random.choice(consonant)

        return available[0] if available else cf_note.pitch

    async def _generate_fifth_species(
        self,
        cantus_firmus: List[note.Note],
        cf_key: m21_key.Key,
        voice_position: str,
        rule_set: str,
        custom_rules: Optional[List[str]],
    ) -> Dict[str, Any]:
        """Generate fifth species (florid) counterpoint"""
        counterpoint_notes = []
        interval_analysis = []

        # Fifth species mixes all previous species
        for i, cf_note in enumerate(cantus_firmus):
            # Randomly choose species type for this measure
            if i == 0:
                species_choice = Species.FIRST  # Start simple
            elif i == len(cantus_firmus) - 1:
                species_choice = Species.FIRST  # End simple
            else:
                species_choice = random.choice(
                    [
                        Species.FIRST,
                        Species.SECOND,
                        Species.SECOND,  # Weight towards 2:1
                        Species.THIRD,
                        Species.FOURTH,
                    ]
                )

            # Generate notes based on chosen species
            if species_choice == Species.FIRST:
                cp_pitch = self._choose_first_species_note(
                    cf_note,
                    cf_key,
                    voice_position,
                    counterpoint_notes,
                    cantus_firmus,
                    i,
                    rule_set,
                    custom_rules,
                )
                cp_note = note.Note(
                    cp_pitch, quarterLength=cf_note.duration.quarterLength
                )
                counterpoint_notes.append(cp_note)

            elif species_choice == Species.SECOND:
                for beat in range(2):
                    is_strong = beat == 0
                    cp_pitch = self._choose_second_species_note(
                        cf_note,
                        cf_key,
                        voice_position,
                        counterpoint_notes,
                        is_strong,
                        rule_set,
                        custom_rules,
                    )
                    cp_note = note.Note(
                        cp_pitch, quarterLength=cf_note.duration.quarterLength / 2
                    )
                    counterpoint_notes.append(cp_note)

            # Add other species as needed...

        melodic_analysis = self._analyze_melodic_line(counterpoint_notes)

        return {
            "notes": counterpoint_notes,
            "interval_analysis": interval_analysis,
            "melodic_analysis": melodic_analysis,
        }

    def _choose_opening_pitch(
        self, cf_note: note.Note, voice_position: str
    ) -> pitch.Pitch:
        """Choose opening pitch (perfect consonance)"""
        if voice_position == "above":
            options = [0, 7, 12]  # Unison, fifth, octave
        else:
            options = [-12, -7, 0]

        interval_choice = random.choice(options)
        return pitch.Pitch(midi=cf_note.pitch.midi + interval_choice)

    def _choose_penultimate_pitch(
        self, cf_note: note.Note, cf_key: m21_key.Key, voice_position: str
    ) -> pitch.Pitch:
        """Choose penultimate pitch (usually leading tone)"""
        if cf_key.mode == "major":
            # In major, use leading tone
            leading_tone = cf_key.pitchFromDegree(7)

            # Adjust octave for voice position
            while voice_position == "above" and leading_tone.midi < cf_note.pitch.midi:
                leading_tone.midi += 12
            while voice_position == "below" and leading_tone.midi > cf_note.pitch.midi:
                leading_tone.midi -= 12

            return leading_tone
        else:
            # In minor, could use natural or raised 7th
            return self._choose_consonant_pitch(cf_note, cf_key, voice_position)

    def _choose_final_pitch(
        self, cf_note: note.Note, voice_position: str
    ) -> pitch.Pitch:
        """Choose final pitch (octave or unison)"""
        if voice_position == "above":
            interval_choice = random.choice([0, 12])  # Unison or octave
        else:
            interval_choice = random.choice([-12, 0])

        return pitch.Pitch(midi=cf_note.pitch.midi + interval_choice)

    def _analyze_melodic_line(self, notes: List[note.Note]) -> Dict[str, Any]:
        """Analyze the melodic characteristics of the counterpoint"""
        analysis = {
            "total_notes": len(notes),
            "range": 0,
            "stepwise_motion_percentage": 0,
            "leap_count": 0,
            "direction_changes": 0,
            "repeated_notes": 0,
            "climax_position": 0,
        }

        if len(notes) < 2:
            return analysis

        # Filter out rests
        pitched_notes = [n for n in notes if isinstance(n, note.Note)]

        if len(pitched_notes) < 2:
            return analysis

        # Range
        pitches = [n.pitch.midi for n in pitched_notes]
        analysis["range"] = max(pitches) - min(pitches)

        # Motion analysis
        stepwise = 0
        leaps = 0
        direction_changes = 0
        prev_direction = 0

        for i in range(len(pitched_notes) - 1):
            interval_size = abs(
                pitched_notes[i + 1].pitch.midi - pitched_notes[i].pitch.midi
            )

            if interval_size == 0:
                analysis["repeated_notes"] += 1
            elif interval_size <= 2:
                stepwise += 1
            elif interval_size > 4:
                leaps += 1

            # Direction changes
            current_direction = (
                pitched_notes[i + 1].pitch.midi - pitched_notes[i].pitch.midi
            )
            if current_direction != 0:
                if prev_direction != 0 and (current_direction > 0) != (
                    prev_direction > 0
                ):
                    direction_changes += 1
                prev_direction = current_direction

        analysis["stepwise_motion_percentage"] = (
            stepwise / (len(pitched_notes) - 1)
        ) * 100
        analysis["leap_count"] = leaps
        analysis["direction_changes"] = direction_changes
        analysis["climax_position"] = pitches.index(max(pitches)) / len(pitches)

        return analysis

    def _check_counterpoint_rules(
        self,
        cantus_firmus: List[note.Note],
        counterpoint: List[note.Note],
        species: Species,
        rule_set: str,
        custom_rules: Optional[List[str]],
    ) -> Dict[str, Any]:
        """Check counterpoint against rules"""
        violations = []
        total_checks = 0
        passed_checks = 0

        # Determine which rules to check
        if rule_set == "strict":
            rules_to_check = list(CounterpointRule)
        elif rule_set == "relaxed":
            rules_to_check = [
                CounterpointRule.NO_PARALLEL_FIFTHS,
                CounterpointRule.NO_PARALLEL_OCTAVES,
                CounterpointRule.PROPER_CADENCE,
                CounterpointRule.STAY_IN_MODE,
            ]
        else:  # custom
            rules_to_check = []
            if custom_rules:
                for rule_name in custom_rules:
                    try:
                        rules_to_check.append(CounterpointRule(rule_name))
                    except:
                        pass

        # Check each rule
        for rule in rules_to_check:
            total_checks += 1

            if rule == CounterpointRule.NO_PARALLEL_FIFTHS:
                parallel_fifths = self._check_parallel_intervals(
                    cantus_firmus, counterpoint, 7
                )
                if parallel_fifths:
                    violations.append(
                        {
                            "rule": rule.value,
                            "locations": parallel_fifths,
                            "severity": "high",
                        }
                    )
                else:
                    passed_checks += 1

            elif rule == CounterpointRule.NO_PARALLEL_OCTAVES:
                parallel_octaves = self._check_parallel_intervals(
                    cantus_firmus, counterpoint, 0
                )
                if parallel_octaves:
                    violations.append(
                        {
                            "rule": rule.value,
                            "locations": parallel_octaves,
                            "severity": "high",
                        }
                    )
                else:
                    passed_checks += 1

            elif rule == CounterpointRule.PROPER_CADENCE:
                if self._check_cadence(cantus_firmus, counterpoint):
                    passed_checks += 1
                else:
                    violations.append(
                        {
                            "rule": rule.value,
                            "locations": ["final cadence"],
                            "severity": "medium",
                        }
                    )

            # Add other rule checks...

        compliance_score = (
            (passed_checks / total_checks * 100) if total_checks > 0 else 100
        )

        return {
            "violations": violations,
            "compliance_score": compliance_score,
            "total_rules_checked": total_checks,
            "rules_passed": passed_checks,
        }

    def _check_parallel_intervals(
        self,
        cantus_firmus: List[note.Note],
        counterpoint: List[note.Note],
        interval_size: int,
    ) -> List[int]:
        """Check for parallel fifths or octaves"""
        locations = []

        # Align CF and CP notes
        cf_notes = [n for n in cantus_firmus if isinstance(n, note.Note)]
        cp_notes = [n for n in counterpoint if isinstance(n, note.Note)]

        # For species other than first, we need to align properly
        # This is simplified - real implementation would be more complex

        for i in range(min(len(cf_notes), len(cp_notes)) - 1):
            # Current interval
            curr_interval = (cp_notes[i].pitch.midi - cf_notes[i].pitch.midi) % 12

            # Next interval
            if i + 1 < min(len(cf_notes), len(cp_notes)):
                next_interval = (
                    cp_notes[i + 1].pitch.midi - cf_notes[i + 1].pitch.midi
                ) % 12

                # Check for parallels
                if curr_interval == interval_size and next_interval == interval_size:
                    # Check that both voices moved in same direction
                    cf_motion = cf_notes[i + 1].pitch.midi - cf_notes[i].pitch.midi
                    cp_motion = cp_notes[i + 1].pitch.midi - cp_notes[i].pitch.midi

                    if (
                        cf_motion != 0
                        and cp_motion != 0
                        and (cf_motion > 0) == (cp_motion > 0)
                    ):
                        locations.append(i)

        return locations

    def _check_cadence(
        self, cantus_firmus: List[note.Note], counterpoint: List[note.Note]
    ) -> bool:
        """Check if cadence is proper"""
        if not cantus_firmus or not counterpoint:
            return False

        # Get final CF and CP notes
        final_cf = cantus_firmus[-1]
        cp_notes = [n for n in counterpoint if isinstance(n, note.Note)]

        if not cp_notes:
            return False

        final_cp = cp_notes[-1]

        # Check for octave or unison at cadence
        interval = (final_cp.pitch.midi - final_cf.pitch.midi) % 12

        return interval in [0, 12]  # Unison or octave

    def _create_counterpoint_score(
        self,
        cantus_firmus: List[note.Note],
        counterpoint: List[note.Note],
        voice_position: str,
        species: Species,
    ) -> stream.Score:
        """Create a score with CF and counterpoint"""
        score = stream.Score()

        # Create parts
        cf_part = stream.Part()
        cp_part = stream.Part()

        cf_part.partName = "Cantus Firmus"
        cp_part.partName = f"{species.value} Counterpoint"

        # Add notes to parts
        for n in cantus_firmus:
            cf_part.append(n)

        for n in counterpoint:
            cp_part.append(n)

        # Add parts to score in correct order
        if voice_position == "above":
            score.insert(0, cp_part)
            score.insert(0, cf_part)
        else:
            score.insert(0, cf_part)
            score.insert(0, cp_part)

        return score

    def _generate_explanations(
        self, species: Species, rule_report: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """Generate educational explanations"""
        explanations = []

        # Explain species characteristics
        species_explanations = {
            Species.FIRST: "First species uses note-against-note counterpoint with only consonances",
            Species.SECOND: "Second species uses 2:1 rhythm, allowing passing tones on weak beats",
            Species.THIRD: "Third species uses 3:1 rhythm with more melodic freedom",
            Species.FOURTH: "Fourth species uses syncopation and suspensions for expressive dissonance",
            Species.FIFTH: "Fifth species (florid) combines all previous species for maximum variety",
        }

        explanations.append(
            {
                "topic": "Species Type",
                "explanation": species_explanations.get(species, "Unknown species"),
            }
        )

        # Explain rule compliance
        if rule_report["compliance_score"] >= 90:
            explanations.append(
                {
                    "topic": "Rule Compliance",
                    "explanation": "Excellent adherence to counterpoint rules with minimal violations",
                }
            )
        elif rule_report["compliance_score"] >= 70:
            explanations.append(
                {
                    "topic": "Rule Compliance",
                    "explanation": "Good counterpoint with some minor rule violations that don't affect musicality",
                }
            )
        else:
            explanations.append(
                {
                    "topic": "Rule Compliance",
                    "explanation": "Several rule violations present - review marked locations for improvement",
                }
            )

        # Explain specific violations
        for violation in rule_report["violations"][:3]:  # First 3 violations
            if violation["rule"] == "no_parallel_fifths":
                explanations.append(
                    {
                        "topic": "Parallel Fifths",
                        "explanation": "Parallel perfect fifths reduce voice independence. Try contrary or oblique motion instead.",
                    }
                )
            elif violation["rule"] == "no_parallel_octaves":
                explanations.append(
                    {
                        "topic": "Parallel Octaves",
                        "explanation": "Parallel octaves make voices sound like one. Use different intervals to maintain independence.",
                    }
                )

        return explanations
