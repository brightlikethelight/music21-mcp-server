"""
Style Imitation Tool - Analyze and generate music in specific composer styles
Uses machine learning-inspired techniques for style analysis and generation
"""

import logging
import random
from collections import Counter, defaultdict
from typing import Any

import numpy as np
from music21 import chord, interval, key, note, pitch, stream

from .base_tool import BaseTool

logger = logging.getLogger(__name__)


class StyleImitationTool(BaseTool):
    """
    Style imitation tool providing:
    1. Style analysis from example pieces
    2. Markov chain-based generation
    3. Rule extraction and application
    4. Multi-parametric style modeling
    5. Hybrid generation with constraints
    """

    def __init__(self, score_manager: dict[str, Any]):
        super().__init__(score_manager)

        # Pre-defined style characteristics
        self.style_profiles = {
            "bach": {
                "melodic": {
                    "stepwise_preference": 0.7,
                    "leap_recovery": True,
                    "sequence_use": 0.8,
                    "chromatic_passing": 0.3,
                },
                "harmonic": {
                    "progression_strictness": 0.9,
                    "voice_independence": 0.95,
                    "cadence_types": ["PAC", "IAC", "HC"],
                    "modulation_frequency": 0.3,
                },
                "rhythmic": {
                    "motor_rhythm": True,
                    "syncopation": 0.2,
                    "consistent_subdivision": True,
                },
                "texture": "polyphonic",
            },
            "mozart": {
                "melodic": {
                    "stepwise_preference": 0.6,
                    "leap_recovery": True,
                    "alberti_bass": True,
                    "ornamental_figures": 0.4,
                },
                "harmonic": {
                    "progression_strictness": 0.8,
                    "chromatic_harmony": 0.2,
                    "cadence_types": ["PAC", "HC", "DC"],
                    "tonicization": 0.4,
                },
                "rhythmic": {
                    "periodic_phrasing": True,
                    "rhythmic_clarity": True,
                    "grace_notes": 0.3,
                },
                "texture": "homophonic",
            },
            "chopin": {
                "melodic": {
                    "lyrical_lines": True,
                    "chromatic_inflection": 0.6,
                    "wide_leaps": 0.3,
                    "rubato_implied": True,
                },
                "harmonic": {
                    "chromatic_harmony": 0.7,
                    "extended_chords": 0.6,
                    "pedal_points": 0.4,
                    "modal_mixture": 0.5,
                },
                "rhythmic": {
                    "flexible_tempo": True,
                    "polyrhythm": 0.3,
                    "ornamental_rhythm": 0.5,
                },
                "texture": "melody_dominated_homophony",
            },
            "debussy": {
                "melodic": {
                    "pentatonic_scales": 0.4,
                    "whole_tone": 0.3,
                    "modal_scales": 0.5,
                    "atmospheric": True,
                },
                "harmonic": {
                    "parallel_motion": 0.6,
                    "non_functional": 0.7,
                    "extended_tertian": 0.8,
                    "quartal_harmony": 0.3,
                },
                "rhythmic": {
                    "fluid_rhythm": True,
                    "cross_rhythm": 0.4,
                    "metric_ambiguity": 0.5,
                },
                "texture": "layered",
            },
        }

        self.transition_matrices: dict[str, dict[Any, Any]] = {}
        self.rhythm_patterns: dict[str, Any] = {}
        self.harmonic_progressions: dict[str, Any] = {}

    async def execute(self, **kwargs: Any) -> dict[str, Any]:
        """
        Generate music in a specific style

        Args:
            **kwargs: Keyword arguments including:
                style_source: Score ID to analyze for style
                composer: Pre-defined composer style to use
                generation_length: Number of measures to generate
                starting_note: Starting pitch (e.g., 'C4')
                constraints: List of constraints (e.g., ['key:C', 'range:C3-C6'])
                complexity: Generation complexity ('simple', 'medium', 'complex')
        """
        # Extract parameters from kwargs
        style_source = kwargs.get("style_source")
        composer = kwargs.get("composer")
        generation_length = kwargs.get("generation_length", 16)
        starting_note = kwargs.get("starting_note")
        constraints = kwargs.get("constraints")
        complexity = kwargs.get("complexity", "medium")
        # Validate inputs
        error = self.validate_inputs(**kwargs)
        if error:
            return self.create_error_response(error)

        with self.error_handling("Style imitation"):
            # Learn style characteristics
            if style_source:
                self.report_progress(0.1, f"Analyzing style from '{style_source}'")
                source_score = self.get_score(style_source)
                style_data = await self._analyze_style(source_score)
            elif composer:
                self.report_progress(0.1, f"Loading {composer} style profile")
                style_data = self._load_composer_style(composer)
            else:
                return self.create_error_response(
                    "Must provide either style_source or composer"
                )

            self.report_progress(0.3, "Building generative models")

            # Build transition matrices
            if style_source:
                self._build_transition_matrices(source_score, style_data)
            elif composer and isinstance(composer, str):
                self._load_preset_transitions(composer)

            self.report_progress(0.5, "Generating new composition")

            # Generate based on style
            generated_score = await self._generate_style_based_music(
                style_data, generation_length, starting_note, constraints, complexity
            )

            self.report_progress(0.8, "Applying style refinements")

            # Refine with style-specific rules
            refined_score = self._apply_style_refinements(
                generated_score, style_data, composer
            )

            # Analyze the generated piece
            self.report_progress(0.9, "Analyzing generated music")
            generation_analysis = await self._analyze_generation(
                refined_score, style_data
            )

            # Store result
            result_id = f"generated_{composer or 'custom'}_style"
            self.score_manager[result_id] = refined_score

            self.report_progress(1.0, "Style imitation complete")

            return self.create_success_response(
                generated_score_id=result_id,
                style_source=style_source or composer,
                measures_generated=generation_length,
                complexity=complexity,
                style_adherence=generation_analysis["style_adherence"],
                musical_features=generation_analysis["features"],
                transition_types_used=generation_analysis["transitions"],
                harmonic_language=generation_analysis["harmony"],
            )

    async def analyze_style(self, **kwargs: Any) -> dict[str, Any]:
        """
        Analyze the style characteristics of a score

        Args:
            **kwargs: Keyword arguments including:
                score_id: ID of score to analyze
                detailed: Whether to include detailed statistics
        """
        # Extract parameters from kwargs
        score_id = kwargs.get("score_id", "")
        detailed = kwargs.get("detailed", True)
        error = self.check_score_exists(score_id)
        if error:
            return self.create_error_response(error)

        with self.error_handling(f"Style analysis of '{score_id}'"):
            score = self.get_score(score_id)

            self.report_progress(0.2, "Analyzing melodic characteristics")
            style_data = await self._analyze_style(score)

            if detailed:
                self.report_progress(0.5, "Computing detailed statistics")

                # Additional detailed analysis
                style_data["interval_distribution"] = (
                    self._compute_interval_distribution(score)
                )
                style_data["rhythm_histogram"] = self._compute_rhythm_histogram(score)
                style_data["chord_vocabulary"] = self._extract_chord_vocabulary(score)
                style_data["phrase_lengths"] = self._analyze_phrase_structure(score)

                self.report_progress(0.8, "Identifying style markers")
                style_data["distinctive_features"] = (
                    self._identify_distinctive_features(style_data)
                )

            # Compare to known styles
            self.report_progress(0.9, "Comparing to known styles")
            style_similarities = self._compare_to_known_styles(style_data)

            return self.create_success_response(
                style_characteristics=style_data,
                closest_known_styles=style_similarities,
                detailed_analysis=detailed,
            )

    def validate_inputs(self, **kwargs: Any) -> str | None:
        """Validate input parameters"""
        if "style_source" in kwargs and kwargs["style_source"]:
            error = self.check_score_exists(kwargs["style_source"])
            if error:
                return error

        if "composer" in kwargs and kwargs["composer"]:
            if kwargs["composer"] not in self.style_profiles:
                return f"Unknown composer: {kwargs['composer']}. Available: {list(self.style_profiles.keys())}"

        if "generation_length" in kwargs:
            if not 1 <= kwargs["generation_length"] <= 64:
                return "generation_length must be between 1 and 64 measures"

        if "complexity" in kwargs:
            if kwargs["complexity"] not in ["simple", "medium", "complex"]:
                return "complexity must be 'simple', 'medium', or 'complex'"

        if not kwargs.get("style_source") and not kwargs.get("composer"):
            return "Must provide either style_source or composer"

        return None

    async def _analyze_style(self, score: stream.Score) -> dict[str, Any]:
        """Analyze style characteristics of a score"""
        style_data: dict[str, dict[str, Any]] = {
            "melodic": {},
            "harmonic": {},
            "rhythmic": {},
            "textural": {},
            "formal": {},
        }

        try:
            # Melodic analysis
            all_notes = list(score.flatten().notes)
            if all_notes:
                intervals = []
                for i in range(len(all_notes) - 1):
                    current_note = all_notes[i]
                    next_note = all_notes[i + 1]
                    if isinstance(current_note, note.Note) and isinstance(
                        next_note, note.Note
                    ):
                        try:
                            intv = interval.Interval(
                                noteStart=current_note, noteEnd=next_note
                            )
                            intervals.append(intv.semitones)
                        except (AttributeError, TypeError, ValueError) as e:
                            logger.debug(f"Style interval calculation failed: {e}")
                            continue

                # Calculate melodic statistics
                if intervals:
                    style_data["melodic"]["avg_interval"] = np.mean(np.abs(intervals))
                    style_data["melodic"]["stepwise_motion"] = sum(
                        1 for i in intervals if abs(i) <= 2
                    ) / len(intervals)
                    style_data["melodic"]["leap_frequency"] = sum(
                        1 for i in intervals if abs(i) > 4
                    ) / len(intervals)
                    style_data["melodic"]["largest_leap"] = max(
                        abs(i) for i in intervals
                    )
                    style_data["melodic"]["contour_changes"] = (
                        self._count_contour_changes([int(i) for i in intervals])
                    )

            # Harmonic analysis
            chords = list(score.flatten().getElementsByClass(chord.Chord))
            if chords:
                style_data["harmonic"]["chord_density"] = (
                    len(chords) / score.duration.quarterLength
                )
                style_data["harmonic"]["unique_chords"] = len(
                    {ch.pitchedCommonName for ch in chords}
                )
                style_data["harmonic"]["dissonance_level"] = (
                    self._calculate_dissonance_level(chords)
                )

                # Progression tendencies
                progressions = self._extract_progression_patterns(chords)
                style_data["harmonic"]["common_progressions"] = progressions

            # Rhythmic analysis
            durations = [
                n.duration.quarterLength for n in all_notes if isinstance(n, note.Note)
            ]
            if durations:
                style_data["rhythmic"]["avg_duration"] = np.mean(durations)
                style_data["rhythmic"]["rhythm_variety"] = len(set(durations))
                note_list = [n for n in all_notes if isinstance(n, note.Note)]
                style_data["rhythmic"]["syncopation_level"] = (
                    self._calculate_syncopation(note_list)
                )
                style_data["rhythmic"]["common_patterns"] = (
                    self._extract_rhythm_patterns(note_list)
                )

            # Textural analysis
            parts = score.parts
            if len(parts) > 0:
                style_data["textural"]["voice_count"] = len(parts)
                parts_list = list(parts)
                style_data["textural"]["density_profile"] = (
                    self._analyze_texture_density(parts_list)
                )
                style_data["textural"]["voice_independence"] = (
                    self._calculate_voice_independence(parts_list)
                )

            # Formal analysis
            style_data["formal"]["total_measures"] = sum(
                1 for _ in score.parts[0].getElementsByClass("Measure")
            )
            style_data["formal"]["phrase_structure"] = self._detect_phrase_structure(
                score
            )

        except Exception as e:
            logger.error(f"Style analysis error: {e}")

        return style_data

    def _count_contour_changes(self, intervals: list[int]) -> int:
        """Count melodic contour direction changes"""
        changes = 0
        for i in range(len(intervals) - 1):
            if intervals[i] * intervals[i + 1] < 0:  # Different signs
                changes += 1
        return changes

    def _calculate_dissonance_level(self, chords: list[chord.Chord]) -> float:
        """Calculate overall dissonance level"""
        if not chords:
            return 0.0

        dissonance_sum = 0
        for ch in chords:
            # Simple dissonance calculation based on intervals
            pitches = [p.midi for p in ch.pitches]
            for i in range(len(pitches)):
                for j in range(i + 1, len(pitches)):
                    interval_class = abs(pitches[i] - pitches[j]) % 12
                    # Dissonant intervals: m2, M2, tritone, m7, M7
                    if interval_class in [1, 2, 6, 10, 11]:
                        dissonance_sum += 1

        return dissonance_sum / len(chords)

    def _extract_progression_patterns(
        self, chords: list[chord.Chord]
    ) -> list[tuple[tuple[str, str], int]]:
        """Extract common chord progression patterns"""
        progressions = []

        for i in range(len(chords) - 1):
            prog = (chords[i].pitchedCommonName, chords[i + 1].pitchedCommonName)
            progressions.append(prog)

        # Count frequencies
        prog_counter = Counter(progressions)
        return prog_counter.most_common(5)

    def _calculate_syncopation(self, notes: list[note.Note]) -> float:
        """Calculate syncopation level"""
        syncopated = 0
        total = 0

        for n in notes:
            if isinstance(n, note.Note):
                # Check if note starts on weak beat
                if n.beat and n.beat % 1 != 0:
                    syncopated += 1
                total += 1

        return syncopated / max(total, 1)

    def _extract_rhythm_patterns(
        self, notes: list[note.Note]
    ) -> list[tuple[tuple[float, ...], int]]:
        """Extract common rhythm patterns"""
        patterns = []
        window_size = 4

        for i in range(len(notes) - window_size + 1):
            pattern = tuple(
                n.duration.quarterLength
                for n in notes[i : i + window_size]
                if isinstance(n, note.Note)
            )
            if len(pattern) == window_size:
                patterns.append(pattern)

        pattern_counter = Counter(patterns)
        return pattern_counter.most_common(5)

    def _analyze_texture_density(self, parts: list[stream.Part]) -> dict[str, float]:
        """Analyze texture density across time"""
        return {"average": 0.7, "variation": 0.2}  # Simplified

    def _calculate_voice_independence(self, parts: list[stream.Part]) -> float:
        """Calculate independence between voices"""
        if len(parts) < 2:
            return 1.0

        # Simplified - check rhythm independence
        independence_scores = []

        for i in range(len(parts) - 1):
            for j in range(i + 1, len(parts)):
                part1_rhythm = [
                    n.duration.quarterLength for n in parts[i].flatten().notes
                ]
                part2_rhythm = [
                    n.duration.quarterLength for n in parts[j].flatten().notes
                ]

                # Compare rhythms
                if part1_rhythm and part2_rhythm:
                    min_len = min(len(part1_rhythm), len(part2_rhythm))
                    matching = sum(
                        1 for k in range(min_len) if part1_rhythm[k] == part2_rhythm[k]
                    )
                    independence = 1 - (matching / min_len)
                    independence_scores.append(independence)

        return float(np.mean(independence_scores) if independence_scores else 0.5)

    def _detect_phrase_structure(self, score: stream.Score) -> list[int]:
        """Detect phrase boundaries"""
        # Simplified - look for rests or long notes
        phrase_lengths = []
        current_phrase = 0

        for element in score.flatten():
            if isinstance(element, note.Note):
                current_phrase += element.duration.quarterLength
            elif isinstance(element, note.Rest) and element.duration.quarterLength >= 1:
                if current_phrase > 0:
                    phrase_lengths.append(current_phrase)
                    current_phrase = 0

        if current_phrase > 0:
            phrase_lengths.append(current_phrase)

        return phrase_lengths

    def _load_composer_style(self, composer: str) -> dict[str, Any]:
        """Load pre-defined composer style profile"""
        profile = self.style_profiles.get(composer, {})

        # Convert to analysis format
        melodic_data = profile.get("melodic", {}) if isinstance(profile, dict) else {}
        harmonic_data = profile.get("harmonic", {}) if isinstance(profile, dict) else {}
        rhythmic_data = profile.get("rhythmic", {}) if isinstance(profile, dict) else {}
        texture_type = (
            profile.get("texture", "homophonic")
            if isinstance(profile, dict)
            else "homophonic"
        )

        style_data: dict[str, Any] = {
            "melodic": melodic_data,
            "harmonic": harmonic_data,
            "rhythmic": rhythmic_data,
            "textural": {"texture_type": texture_type},
            "formal": {},
        }

        return style_data

    def _build_transition_matrices(
        self, score: stream.Score, style_data: dict[str, Any]
    ) -> None:
        """Build Markov transition matrices from score"""
        # Pitch transitions
        pitch_transitions: dict[int, dict[int, float]] = defaultdict(
            lambda: defaultdict(int)
        )
        notes = [n for n in score.flatten().notes if isinstance(n, note.Note)]

        for i in range(len(notes) - 1):
            current = notes[i].pitch.midi
            next_pitch = notes[i + 1].pitch.midi
            pitch_transitions[current][next_pitch] += 1

        # Normalize to probabilities
        for current in pitch_transitions:
            total = sum(pitch_transitions[current].values())
            if total > 0:
                for next_pitch in pitch_transitions[current]:
                    pitch_transitions[current][next_pitch] = (
                        float(pitch_transitions[current][next_pitch]) / total
                    )

        self.transition_matrices["pitch"] = dict(pitch_transitions)

        # Rhythm transitions
        rhythm_transitions: dict[float, dict[float, float]] = defaultdict(
            lambda: defaultdict(int)
        )
        for i in range(len(notes) - 1):
            current = notes[i].duration.quarterLength
            next_dur = notes[i + 1].duration.quarterLength
            rhythm_transitions[current][next_dur] += 1

        # Normalize
        for current in rhythm_transitions:
            total = sum(rhythm_transitions[current].values())
            if total > 0:
                for next_dur in rhythm_transitions[current]:
                    rhythm_transitions[current][next_dur] = (
                        float(rhythm_transitions[current][next_dur]) / total
                    )

        self.transition_matrices["rhythm"] = dict(rhythm_transitions)

    def _load_preset_transitions(self, composer: str) -> None:
        """Load preset transition matrices for known composers"""
        # Simplified preset transitions
        if composer == "bach":
            self.transition_matrices["pitch"] = {
                # Simplified - would be much more complex in reality
                60: {62: 0.3, 64: 0.2, 59: 0.2, 60: 0.1, 65: 0.2},  # C
                62: {64: 0.3, 60: 0.2, 65: 0.2, 62: 0.1, 67: 0.2},  # D
                # ... etc
            }
        elif composer == "mozart":
            self.transition_matrices["pitch"] = {
                60: {62: 0.25, 64: 0.25, 67: 0.25, 60: 0.25},  # More balanced
                # ... etc
            }
        # ... other composers

    async def _generate_style_based_music(
        self,
        style_data: dict[str, Any],
        generation_length: int,
        starting_note: str | None,
        constraints: list[str] | None,
        complexity: str,
    ) -> stream.Score:
        """Generate music based on learned style"""
        generated_score = stream.Score()
        part = stream.Part()

        # Parse constraints
        range_constraint = None
        key_constraint = None
        if constraints:
            for constraint in constraints:
                if constraint.startswith("key:"):
                    # Implement key constraint
                    key_str = constraint.split(":")[1]
                    try:
                        key_constraint = key.Key(key_str)
                    except (ValueError, AttributeError) as e:
                        logger.warning(f"Invalid key constraint '{key_str}': {e}")
                elif constraint.startswith("range:"):
                    range_constraint = constraint.split(":")[1]

        # Determine starting pitch
        if starting_note:
            current_pitch = pitch.Pitch(starting_note)
        else:
            current_pitch = pitch.Pitch("C4")

        # Generate notes
        total_duration: float = 0.0
        target_duration = generation_length * 4  # Assume 4/4 time

        while total_duration < target_duration:
            # Choose next pitch
            if (
                "pitch" in self.transition_matrices
                and current_pitch.midi in self.transition_matrices["pitch"]
            ):
                # Use Markov chain
                transitions = self.transition_matrices["pitch"][current_pitch.midi]
                next_pitches = list(transitions.keys())
                probabilities = list(transitions.values())

                if next_pitches:
                    next_midi = np.random.choice(next_pitches, p=probabilities)
                    current_pitch = pitch.Pitch(midi=next_midi)
                else:
                    # Random step
                    current_pitch = self._generate_stylistic_pitch(
                        current_pitch, style_data
                    )
            else:
                # Use style-based generation
                current_pitch = self._generate_stylistic_pitch(
                    current_pitch, style_data
                )

            # Apply key constraint
            if key_constraint:
                # Snap pitch to nearest scale degree in the key
                scale_pitches = key_constraint.pitches
                pitch_classes = [p.pitchClass for p in scale_pitches]
                
                # Find nearest pitch in key
                current_pc = current_pitch.pitchClass
                if current_pc not in pitch_classes:
                    # Find closest pitch class in key
                    distances = [(abs(current_pc - pc), pc) for pc in pitch_classes]
                    distances.extend([(abs(current_pc - pc - 12), pc) for pc in pitch_classes])
                    distances.extend([(abs(current_pc - pc + 12), pc) for pc in pitch_classes])
                    closest_pc = min(distances)[1]
                    
                    # Adjust to nearest octave
                    octave = current_pitch.octave
                    current_pitch = pitch.Pitch()
                    current_pitch.pitchClass = closest_pc
                    current_pitch.octave = octave
            
            # Apply range constraint
            if range_constraint:
                min_pitch, max_pitch = range_constraint.split("-")
                min_midi = pitch.Pitch(min_pitch).midi
                max_midi = pitch.Pitch(max_pitch).midi
                current_pitch.midi = max(min_midi, min(max_midi, current_pitch.midi))

            # Choose duration
            duration = self._generate_stylistic_duration(style_data, complexity)

            # Create note
            n = note.Note(current_pitch, quarterLength=duration)
            part.append(n)

            total_duration = total_duration + float(duration)

        generated_score.insert(0, part)

        # Add additional voices for complex styles
        if (
            complexity == "complex"
            and style_data.get("textural", {}).get("voice_count", 1) > 1
        ):
            # Add bass line
            bass_part = self._generate_bass_line(part, style_data)
            generated_score.insert(0, bass_part)

        return generated_score

    def _generate_stylistic_pitch(
        self, current: pitch.Pitch, style_data: dict[str, Any]
    ) -> pitch.Pitch:
        """Generate next pitch based on style characteristics"""
        stepwise_pref = style_data.get("melodic", {}).get("stepwise_motion", 0.7)

        if random.random() < stepwise_pref:
            # Stepwise motion
            step = random.choice([-2, -1, 1, 2])
        else:
            # Leap
            leap_size = style_data.get("melodic", {}).get("avg_interval", 3)
            step = random.choice([-1, 1]) * random.randint(3, int(leap_size * 2))

        return pitch.Pitch(midi=current.midi + step)

    def _generate_stylistic_duration(
        self, style_data: dict[str, Any], complexity: str
    ) -> float:
        """Generate rhythm based on style"""
        if "rhythm" in self.transition_matrices:
            # Use learned transitions
            durations = list(self.transition_matrices["rhythm"].keys())
            if durations:
                return float(random.choice(durations))

        # Use style-based generation with average duration from style data
        avg_dur = style_data.get("rhythmic", {}).get("avg_duration", 1.0)
        
        # Adjust duration choices based on average duration from style
        if complexity == "simple":
            # Use durations close to average
            if avg_dur < 0.5:
                return random.choice([0.25, 0.5])
            elif avg_dur > 1.5:
                return random.choice([1.0, 2.0])
            else:
                return random.choice([0.5, 1.0])
        
        if complexity == "medium":
            # More variety around average
            if avg_dur < 0.5:
                return random.choice([0.125, 0.25, 0.5, 0.75])
            elif avg_dur > 1.5:
                return random.choice([0.5, 1.0, 1.5, 2.0])
            else:
                return random.choice([0.25, 0.5, 1.0, 1.5])
        
        # Complex - wide variety centered on average
        if avg_dur < 0.5:
            base_durs = [0.0625, 0.125, 0.25, 0.375, 0.5, 0.75, 1.0]
            weights = [0.05, 0.15, 0.3, 0.2, 0.15, 0.1, 0.05]
        elif avg_dur > 1.5:
            base_durs = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]
            weights = [0.05, 0.1, 0.15, 0.3, 0.2, 0.15, 0.05]
        else:
            base_durs = [0.125, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
            weights = [0.1, 0.2, 0.3, 0.1, 0.2, 0.05, 0.05]
        
        return float(np.random.choice(base_durs, p=weights))

    def _generate_bass_line(
        self, melody_part: stream.Part, style_data: dict[str, Any]
    ) -> stream.Part:
        """Generate a bass line to accompany melody"""
        bass_part = stream.Part()
        bass_part.partName = "Bass"

        # Simple bass generation - root notes on strong beats
        for element in melody_part:
            if isinstance(element, note.Note):
                # Create bass note a fifth or octave below
                bass_pitch = pitch.Pitch(midi=element.pitch.midi - 12)
                bass_note = note.Note(
                    bass_pitch, quarterLength=element.duration.quarterLength
                )
                bass_part.append(bass_note)

        return bass_part

    def _apply_style_refinements(
        self, score: stream.Score, style_data: dict[str, Any], composer: str | None
    ) -> stream.Score:
        """Apply style-specific refinements to generated music"""
        import copy

        refined_score = copy.deepcopy(score)

        if composer == "bach":
            # Add passing tones
            refined_score = self._add_bach_ornaments(refined_score)
        elif composer == "mozart":
            # Add Alberti bass patterns
            refined_score = self._add_mozart_accompaniment(refined_score)
        elif composer == "chopin":
            # Add rubato markings and pedal
            refined_score = self._add_chopin_expression(refined_score)
        elif composer == "debussy":
            # Add whole-tone passages
            refined_score = self._add_debussy_colors(refined_score)

        return score

    def _add_bach_ornaments(self, score: stream.Score) -> stream.Score:
        """Add Bach-style ornaments"""
        # Simplified - add some passing tones
        for part in score.parts:
            notes = list(part.flatten().notes)
            for i in range(len(notes) - 1):
                current_note = notes[i]
                next_note = notes[i + 1]
                if isinstance(current_note, note.Note) and isinstance(
                    next_note, note.Note
                ):
                    interval_size = abs(current_note.pitch.midi - next_note.pitch.midi)
                    if interval_size == 4:  # Major third
                        # Could add passing tone
                        pass
        return score

    def _add_mozart_accompaniment(self, score: stream.Score) -> stream.Score:
        """Add Mozart-style accompaniment patterns"""
        # Simplified implementation
        return score

    def _add_chopin_expression(self, score: stream.Score) -> stream.Score:
        """Add Chopin-style expression"""
        # Add tempo markings, dynamics
        return score

    def _add_debussy_colors(self, score: stream.Score) -> stream.Score:
        """Add Debussy-style harmonic colors"""
        # Add pedal markings, parallel motion
        return score

    async def _analyze_generation(
        self, score: stream.Score, style_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Analyze how well the generation matches the style"""
        # Analyze the generated score
        generated_style = await self._analyze_style(score)

        # Compare to target style
        adherence_scores = {}

        # Melodic adherence
        if "melodic" in style_data and "melodic" in generated_style:
            melodic_score = 0
            comparisons = 0

            for key in ["stepwise_motion", "leap_frequency", "avg_interval"]:
                if key in style_data["melodic"] and key in generated_style["melodic"]:
                    target = style_data["melodic"][key]
                    actual = generated_style["melodic"][key]
                    # Calculate similarity (simple difference)
                    similarity = 1 - min(abs(target - actual) / max(target, 0.1), 1)
                    melodic_score += similarity
                    comparisons += 1

            if comparisons > 0:
                adherence_scores["melodic"] = melodic_score / comparisons

        # Overall adherence
        overall_adherence = (
            np.mean(list(adherence_scores.values())) if adherence_scores else 0.5
        )

        return {
            "style_adherence": overall_adherence,
            "features": generated_style,
            "transitions": len(self.transition_matrices.get("pitch", {})),
            "harmony": generated_style.get("harmonic", {}),
        }

    def _compute_interval_distribution(self, score: stream.Score) -> dict[str, float]:
        """Compute distribution of melodic intervals"""
        intervals: dict[str, int] = defaultdict(int)
        total = 0

        notes = [n for n in score.flatten().notes if isinstance(n, note.Note)]
        for i in range(len(notes) - 1):
            intv = interval.Interval(notes[i], notes[i + 1])
            intervals[intv.name] += 1
            total += 1

        # Normalize
        distribution = {}
        if total > 0:
            for intv_name, count in intervals.items():
                distribution[intv_name] = count / total

        return distribution

    def _compute_rhythm_histogram(self, score: stream.Score) -> dict[float, float]:
        """Compute histogram of rhythm values"""
        durations: dict[float, int] = defaultdict(int)
        total = 0

        for n in score.flatten().notesAndRests:
            dur = n.duration.quarterLength
            durations[dur] += 1
            total += 1

        # Normalize
        histogram = {}
        if total > 0:
            for dur, count in durations.items():
                histogram[dur] = count / total

        return histogram

    def _extract_chord_vocabulary(self, score: stream.Score) -> list[str]:
        """Extract unique chord types used"""
        chord_types = set()

        for ch in score.flatten().getElementsByClass(chord.Chord):
            chord_types.add(ch.pitchedCommonName)

        return sorted(chord_types)

    def _analyze_phrase_structure(self, score: stream.Score) -> list[int]:
        """Analyze phrase lengths"""
        return self._detect_phrase_structure(score)

    def _identify_distinctive_features(self, style_data: dict[str, Any]) -> list[str]:
        """Identify distinctive style features"""
        features = []

        # Check melodic features
        melodic = style_data.get("melodic", {})
        if melodic.get("stepwise_motion", 0) > 0.8:
            features.append("Highly stepwise melodic motion")
        if melodic.get("leap_frequency", 0) > 0.3:
            features.append("Frequent melodic leaps")

        # Check harmonic features
        harmonic = style_data.get("harmonic", {})
        if harmonic.get("dissonance_level", 0) > 1.5:
            features.append("High harmonic dissonance")
        if harmonic.get("chord_density", 0) < 0.5:
            features.append("Sparse harmonic rhythm")

        # Check rhythmic features
        rhythmic = style_data.get("rhythmic", {})
        if rhythmic.get("syncopation_level", 0) > 0.3:
            features.append("Significant syncopation")
        if rhythmic.get("rhythm_variety", 0) > 6:
            features.append("Complex rhythmic vocabulary")

        return features

    def _compare_to_known_styles(
        self, style_data: dict[str, Any]
    ) -> list[tuple[str, float]]:
        """Compare analyzed style to known composer styles"""
        similarities = []

        for composer, profile in self.style_profiles.items():
            similarity = 0
            comparisons = 0

            # Compare melodic characteristics
            if (
                "melodic" in style_data
                and isinstance(profile, dict)
                and "melodic" in profile
            ):
                melodic_profile = profile["melodic"]
                if (
                    "stepwise_motion" in style_data["melodic"]
                    and isinstance(melodic_profile, dict)
                    and "stepwise_preference" in melodic_profile
                ):
                    diff = abs(
                        style_data["melodic"]["stepwise_motion"]
                        - melodic_profile["stepwise_preference"]
                    )
                    similarity += 1 - diff
                    comparisons += 1

            # Average similarity
            if comparisons > 0:
                similarities.append((composer, similarity / comparisons))

        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:3]  # Top 3 matches
