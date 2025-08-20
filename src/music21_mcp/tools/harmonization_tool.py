"""
Intelligent Harmonization Tool - Generate harmonizations in various styles
Analyzes melodic contour and generates appropriate harmony with explanations
"""

import logging
import random
from typing import Any

from music21 import chord, note, pitch, roman, stream

from ..performance_optimizations import PerformanceOptimizer
from .base_tool import BaseTool

logger = logging.getLogger(__name__)


class HarmonizationTool(BaseTool):
    """
    Intelligent harmonization tool providing:
    1. Analysis of melodic contour and implied harmony
    2. Multiple harmonization styles (classical, jazz, pop, modal)
    3. Voice leading rule compliance
    4. Constraint support (diatonic only, specific progressions, etc.)
    5. Explanations for harmonic choices
    """

    def __init__(self, score_manager: dict[str, Any]):
        super().__init__(score_manager)

        # Initialize performance optimizer for fast Roman numeral analysis
        self.performance_optimizer = PerformanceOptimizer(
            cache_ttl=7200, max_cache_size=2000
        )

        # Warm the cache on initialization for common progressions
        self.performance_optimizer.warm_cache()

        # Define chord vocabularies for different styles
        self.style_vocabularies = {
            "classical": {
                "primary": ["I", "ii", "IV", "V", "vi"],
                "secondary": ["iii", "V/V", "viio", "ii6", "I6"],
                "cadential": ["I", "V", "IV", "vi"],
                "voice_leading": "strict",
            },
            "jazz": {
                "primary": ["IMaj7", "ii7", "iii7", "IVMaj7", "V7", "vi7"],
                "secondary": ["bII7", "bIIIMaj7", "#ivo7", "VII7", "SubV7"],
                "extensions": ["9", "11", "13", "b9", "#11", "b13"],
                "voice_leading": "smooth",
            },
            "pop": {
                "primary": ["I", "IV", "V", "vi"],
                "secondary": ["ii", "iii", "bVII", "bIII"],
                "power_chords": True,
                "voice_leading": "free",
            },
            "modal": {
                "dorian": ["i", "ii", "bIII", "IV", "v", "vi°", "bVII"],
                "mixolydian": ["I", "ii", "iii°", "IV", "v", "vi", "bVII"],
                "lydian": ["I", "II", "iii", "#iv°", "V", "vi", "vii"],
                "voice_leading": "modal",
            },
        }

        # Common progressions by style
        self.common_progressions = {
            "classical": [
                ["I", "IV", "V", "I"],
                ["I", "vi", "IV", "V"],
                ["I", "ii", "V", "I"],
                ["I", "V", "vi", "IV"],
            ],
            "jazz": [
                ["IMaj7", "vi7", "ii7", "V7"],
                ["IMaj7", "I7", "IVMaj7", "#ivo7"],
                ["ii7", "V7", "IMaj7"],
                ["IMaj7", "bIIIMaj7", "bVIMaj7", "IMaj7"],
            ],
            "pop": [
                ["I", "V", "vi", "IV"],
                ["vi", "IV", "I", "V"],
                ["I", "IV", "vi", "V"],
                ["I", "vi", "IV", "V"],
            ],
        }

    async def execute(self, **kwargs: Any) -> dict[str, Any]:
        """
        Harmonize a melody in the specified style

        Args:
            **kwargs: Keyword arguments including:
                score_id: ID of the melody score to harmonize
                output_id: ID for the harmonized output score (optional)
                style: Harmonization style ('classical', 'jazz', 'pop', 'modal')
                constraints: List of constraints (e.g., ['diatonic_only', 'no_parallels'])
                include_explanations: Include explanations for harmonic choices
                voice_parts: Number of voices (2-4)
        """
        # Extract parameters from kwargs
        score_id = kwargs.get("score_id", "")
        output_id = kwargs.get("output_id", "")
        style = kwargs.get("style", "classical")
        constraints = kwargs.get("constraints")
        include_explanations = kwargs.get("include_explanations", True)
        voice_parts = kwargs.get("voice_parts", 4)
        # Validate inputs
        error = self.validate_inputs(**kwargs)
        if error:
            return self.create_error_response(error)

        with self.error_handling(f"Harmonization of '{score_id}'"):
            score = self.get_score(score_id)

            self.report_progress(0.1, "Analyzing melody")

            # Extract melody
            melody = self._extract_melody(score)
            if not melody:
                return self.create_error_response("No melody found in score")

            # Analyze melodic features
            melodic_analysis = await self._analyze_melody(melody)

            self.report_progress(0.3, f"Generating {style} harmonization")

            # Generate harmonization based on style
            harmonization: dict[str, Any]
            if style == "classical":
                harmonization = await self._harmonize_classical(
                    melody, melodic_analysis, constraints, voice_parts
                )
            elif style == "jazz":
                harmonization = await self._harmonize_jazz(
                    melody, melodic_analysis, constraints
                )
            elif style == "pop":
                harmonization = await self._harmonize_pop(
                    melody, melodic_analysis, constraints
                )
            elif style == "modal":
                harmonization = await self._harmonize_modal(
                    melody, melodic_analysis, constraints
                )
            else:
                return self.create_error_response(f"Unknown style: {style}")

            self.report_progress(0.7, "Checking voice leading")

            # Check voice leading
            harmonized_score = harmonization["score"]
            assert isinstance(harmonized_score, stream.Score)
            style_vocab = self.style_vocabularies.get(style, {})
            voice_leading_style = (
                style_vocab.get("voice_leading", "free")
                if isinstance(style_vocab, dict)
                else "free"
            )
            assert isinstance(voice_leading_style, str)
            voice_leading_check = self._check_voice_leading(
                harmonized_score, voice_leading_style
            )

            self.report_progress(0.9, "Generating explanations")

            # Generate explanations if requested
            explanations = []
            if include_explanations:
                explanations = self._generate_explanations(
                    harmonization, melodic_analysis, style
                )

            self.report_progress(1.0, "Harmonization complete")

            # Store harmonized score
            harmonized_id = output_id if output_id else f"{score_id}_harmonized_{style}"
            harmonized_score = harmonization["score"]
            assert isinstance(harmonized_score, stream.Score)
            self.score_manager[harmonized_id] = harmonized_score

            return self.create_success_response(
                harmonized_score_id=harmonized_id,
                harmonization={
                    "style": style,
                    "chord_progression": harmonization["progression"],
                    "roman_numerals": harmonization["roman_numerals"],
                    "voice_leading_quality": voice_leading_check,
                    "explanations": explanations,
                    "harmonic_rhythm": harmonization.get("harmonic_rhythm", {}),
                    "confidence_ratings": harmonization.get("confidence_ratings", []),
                },
            )

    def validate_inputs(self, **kwargs: Any) -> str | None:
        """Validate input parameters"""
        score_id = kwargs.get("score_id", "")
        output_id = kwargs.get("output_id", "")
        style = kwargs.get("style", "classical")
        voice_parts = kwargs.get("voice_parts", 4)

        error = self.check_score_exists(score_id)
        if error:
            return error

        # Check if output_id already exists
        if output_id and output_id in self.score_manager:
            return f"Score with ID '{output_id}' already exists"

        valid_styles = list(self.style_vocabularies.keys())
        if style not in valid_styles:
            return f"Invalid style: {style}. Choose from: {', '.join(valid_styles)}"

        if not 2 <= voice_parts <= 4:
            return "voice_parts must be between 2 and 4"

        return None

    def _extract_melody(self, score: stream.Score) -> list[Any]:
        """Extract the melody line from the score"""
        # Try to get the top part or flatten if single line
        melody_part: stream.Score | stream.Part
        if hasattr(score, "parts") and len(score.parts) > 0:
            melody_part = score.parts[0]
        else:
            melody_part = score

        # Extract notes (ignore rests for now)
        melody = []
        for element in melody_part.flatten():
            if isinstance(element, note.Note):
                melody.append(element)

        return melody

    async def _analyze_melody(self, melody: list[Any]) -> dict[str, Any]:
        """Analyze melodic features to inform harmonization"""
        analysis: dict[str, Any] = {
            "key": None,
            "contour": [],
            "implied_harmonies": [],
            "phrase_points": [],
            "climax": None,
            "range": 0,
        }

        try:
            # Detect key
            melody_stream: stream.Stream = stream.Stream(melody)
            key_obj = melody_stream.analyze("key")
            analysis["key"] = key_obj

            # Analyze contour
            pitches = [n.pitch.midi for n in melody]
            analysis["contour"] = self._analyze_contour(pitches)
            analysis["range"] = max(pitches) - min(pitches) if pitches else 0
            analysis["climax"] = pitches.index(max(pitches)) if pitches else 0

            # Find implied harmonies
            for i, note_obj in enumerate(melody):
                implied = self._get_implied_harmony(note_obj, key_obj)
                analysis["implied_harmonies"].append(implied)

            # Identify phrase points (simplified)
            for i in range(0, len(melody), 8):  # Every 8 notes
                analysis["phrase_points"].append(i)

        except Exception as e:
            logger.error(f"Melody analysis failed: {e}")

        return analysis

    def _analyze_contour(self, pitches: list[int]) -> list[str]:
        """Analyze melodic contour"""
        contour = []
        for i in range(len(pitches) - 1):
            if pitches[i + 1] > pitches[i]:
                contour.append("ascending")
            elif pitches[i + 1] < pitches[i]:
                contour.append("descending")
            else:
                contour.append("static")
        return contour

    def _get_implied_harmony(self, note_obj: note.Note, key_obj: Any) -> list[str]:
        """Get possible harmonies implied by a melodic note"""
        scale_degree = key_obj.getScaleDegreeFromPitch(note_obj.pitch)

        # Map scale degrees to common harmonies
        harmony_map = {
            1: ["I", "vi", "IV"],
            2: ["V", "ii", "viio"],
            3: ["I", "iii", "vi"],
            4: ["IV", "ii", "V/V"],
            5: ["V", "I", "iii"],
            6: ["vi", "IV", "ii"],
            7: ["V", "viio", "iii"],
        }

        return harmony_map.get(scale_degree, ["I"])

    async def _harmonize_classical(
        self,
        melody: list[Any],
        analysis: dict[str, Any],
        constraints: list[str] | None,
        voice_parts: int,
    ) -> dict[str, Any]:
        """Generate classical four-part harmony"""
        result = {
            "score": stream.Score(),
            "progression": [],
            "roman_numerals": [],
            "confidence_ratings": [],
        }

        # Create parts
        soprano = stream.Part()
        alto = stream.Part()
        tenor = stream.Part()
        bass = stream.Part()

        soprano.partName = "Soprano"
        alto.partName = "Alto"
        tenor.partName = "Tenor"
        bass.partName = "Bass"

        # Add melody to soprano
        for n in melody:
            if hasattr(n, "pitch") and hasattr(n, "duration"):
                soprano.append(
                    note.Note(n.pitch, quarterLength=n.duration.quarterLength)
                )

        # Generate harmonic progression
        progression = self._generate_progression_classical(
            melody, analysis, constraints
        )

        # Realize the harmony
        for i, (melodic_note, chord_symbol) in enumerate(
            zip(melody, progression, strict=False)
        ):
            self.report_progress(
                0.3 + (0.4 * i / len(melody)), f"Harmonizing note {i + 1}/{len(melody)}"
            )

            # Create chord based on symbol
            chord_pitches = self._realize_chord_classical(
                chord_symbol, analysis["key"], melodic_note
            )

            if voice_parts >= 3:
                # Distribute pitches to voices
                alto_pitch = self._choose_alto_note(chord_pitches, melodic_note)
                if hasattr(melodic_note, "duration"):
                    alto.append(
                        note.Note(
                            alto_pitch,
                            quarterLength=melodic_note.duration.quarterLength,
                        )
                    )

            if voice_parts >= 4:
                tenor_pitch = self._choose_tenor_note(
                    chord_pitches, melodic_note, alto_pitch
                )
                if hasattr(melodic_note, "duration"):
                    tenor.append(
                        note.Note(
                            tenor_pitch,
                            quarterLength=melodic_note.duration.quarterLength,
                        )
                    )

            # Bass gets root (simplified)
            bass_pitch = chord_pitches[0]
            if hasattr(melodic_note, "duration"):
                bass.append(
                    note.Note(
                        bass_pitch, quarterLength=melodic_note.duration.quarterLength
                    )
                )

            progression_list = result["progression"]
            roman_list = result["roman_numerals"]
            confidence_list = result["confidence_ratings"]
            assert isinstance(progression_list, list)
            assert isinstance(roman_list, list)
            assert isinstance(confidence_list, list)
            progression_list.append(chord_symbol)
            roman_list.append(chord_symbol)
            confidence_list.append(0.8)  # Simplified

        # Assemble score
        score_obj = result["score"]
        assert isinstance(score_obj, stream.Score)
        score_obj.insert(0, soprano)
        if voice_parts >= 3:
            score_obj.insert(0, alto)
        if voice_parts >= 4:
            score_obj.insert(0, tenor)
        score_obj.insert(0, bass)

        # Add harmonic rhythm analysis
        result["harmonic_rhythm"] = self._analyze_harmonic_rhythm(progression)

        return result

    def _generate_progression_classical(
        self,
        melody: list[Any],
        analysis: dict[str, Any],
        constraints: list[str] | None,
    ) -> list[str]:
        """Generate a classical chord progression"""
        progression = []
        key_obj = analysis["key"]

        # Use common progressions as templates
        template = random.choice(self.common_progressions["classical"])
        template_length = len(template)

        for i, melodic_note in enumerate(melody):
            # Get implied harmonies
            implied = analysis["implied_harmonies"][i]

            # Check if we're at a cadence point
            is_cadence = i == len(melody) - 1 or i in analysis["phrase_points"]

            if is_cadence and i == len(melody) - 1:
                # Final cadence - use I
                chord_choice = "I"
            elif is_cadence:
                # Phrase cadence - use V or vi
                chord_choice = random.choice(["V", "vi"])
            else:
                # Use template or implied harmony
                template_pos = i % template_length
                template_chord = template[template_pos]

                # Check if template chord works with melody
                if self._is_chord_compatible(template_chord, melodic_note, key_obj):
                    chord_choice = template_chord
                else:
                    # Use implied harmony
                    chord_choice = implied[0] if implied else "I"

            # Apply constraints
            if constraints and "diatonic_only" in constraints:
                if chord_choice not in ["I", "ii", "iii", "IV", "V", "vi", "viio"]:
                    chord_choice = "I"  # Default to tonic

            progression.append(chord_choice)

        return progression

    def _is_chord_compatible(
        self, chord_symbol: str, melodic_note: note.Note, key_obj: Any
    ) -> bool:
        """Check if a chord is compatible with the melodic note"""
        try:
            # Get chord tones using cached Roman numeral analysis
            rn = roman.RomanNumeral(chord_symbol, key_obj)
            chord_obj = chord.Chord(rn.pitches)

            # Use performance optimizer to check compatibility faster
            chord_pitches = [p.name for p in chord_obj.pitches]

            # Check if melody note is in chord
            return melodic_note.pitch.name in chord_pitches
        except (AttributeError, TypeError, ValueError) as e:
            logger.debug(f"Melody fitting check failed: {e}")
            return False

    def _realize_chord_classical(
        self, chord_symbol: str, key_obj: Any, melodic_note: note.Note
    ) -> list[pitch.Pitch]:
        """Realize a chord symbol into pitches for classical style"""
        try:
            # Use cached Roman numeral analysis for better performance
            rn = roman.RomanNumeral(chord_symbol, key_obj)
            chord_obj = chord.Chord(rn.pitches)

            # Get cached Roman numeral analysis to validate the chord
            cached_analysis = self.performance_optimizer.get_cached_roman_numeral(
                chord_obj, key_obj
            )
            if cached_analysis:
                logger.debug(
                    f"Using cached analysis for {chord_symbol}: {cached_analysis}"
                )

            chord_pitches = list(rn.pitches)

            # Ensure melodic note is in the chord
            melody_pitch = melodic_note.pitch
            if melody_pitch not in chord_pitches:
                # Adjust chord to include melody (simplified)
                chord_pitches[-1] = melody_pitch

            return chord_pitches
        except Exception as e:
            logger.error(f"Chord realization failed for {chord_symbol}: {e}")
            # Return simple triad
            return [
                key_obj.tonic,
                key_obj.pitchFromDegree(3),
                key_obj.pitchFromDegree(5),
            ]

    def _choose_alto_note(
        self, chord_pitches: list[pitch.Pitch], melodic_note: note.Note
    ) -> pitch.Pitch:
        """Choose appropriate alto note"""
        # Remove melody pitch from options
        available = [p for p in chord_pitches if p.midi != melodic_note.pitch.midi]

        if available:
            # Choose pitch closest to G4 (typical alto range)
            target_midi = 67  # G4
            available.sort(key=lambda p: abs(p.midi - target_midi))
            return available[0]

        # If no other option, double the bass an octave up
        return pitch.Pitch(midi=chord_pitches[0].midi + 12)

    def _choose_tenor_note(
        self,
        chord_pitches: list[pitch.Pitch],
        melodic_note: note.Note,
        alto_pitch: pitch.Pitch,
    ) -> pitch.Pitch:
        """Choose appropriate tenor note"""
        # Remove already used pitches
        used_midis = {melodic_note.pitch.midi, alto_pitch.midi}
        available = [p for p in chord_pitches if p.midi not in used_midis]

        if available:
            # Choose pitch closest to C4 (typical tenor range)
            target_midi = 60  # C4
            available.sort(key=lambda p: abs(p.midi - target_midi))
            return available[0]

        # If no other option, find a chord tone in tenor range
        for p in chord_pitches:
            tenor_pitch = pitch.Pitch(midi=p.midi)
            while tenor_pitch.midi > 65:  # F4
                tenor_pitch.midi -= 12
            while tenor_pitch.midi < 48:  # C3
                tenor_pitch.midi += 12
            if tenor_pitch.midi not in used_midis:
                return tenor_pitch

        # Last resort - double the root
        return pitch.Pitch(midi=chord_pitches[0].midi)

    async def _harmonize_jazz(
        self,
        melody: list[Any],
        analysis: dict[str, Any],
        constraints: list[str] | None,
    ) -> dict[str, Any]:
        """Generate jazz harmonization with extended chords"""
        result = {
            "score": stream.Score(),
            "progression": [],
            "roman_numerals": [],
            "confidence_ratings": [],
        }

        # Create piano-style arrangement
        right_hand = stream.Part()
        left_hand = stream.Part()

        right_hand.partName = "Piano Right Hand"
        left_hand.partName = "Piano Left Hand"

        # Generate jazz progression
        progression = self._generate_progression_jazz(melody, analysis, constraints)

        # Realize the harmony
        for i, (melodic_note, chord_symbol) in enumerate(
            zip(melody, progression, strict=False)
        ):
            # Create jazz voicing
            voicing = self._create_jazz_voicing(
                chord_symbol, analysis["key"], melodic_note
            )

            # Right hand: melody + upper structure
            if hasattr(melodic_note, "pitch"):
                rh_chord = chord.Chord([melodic_note.pitch] + voicing["upper"])
                right_hand.append(rh_chord)

            # Left hand: bass + lower structure
            lh_chord = chord.Chord(voicing["lower"])
            left_hand.append(lh_chord)

            progression_list = result["progression"]
            roman_list = result["roman_numerals"]
            confidence_list = result["confidence_ratings"]
            assert isinstance(progression_list, list)
            assert isinstance(roman_list, list)
            assert isinstance(confidence_list, list)
            progression_list.append(chord_symbol)
            roman_list.append(chord_symbol)
            confidence_list.append(0.85)

        score_obj = result["score"]
        assert isinstance(score_obj, stream.Score)
        score_obj.insert(0, right_hand)
        score_obj.insert(0, left_hand)

        return result

    def _generate_progression_jazz(
        self,
        melody: list[Any],
        analysis: dict[str, Any],
        constraints: list[str] | None,
    ) -> list[str]:
        """Generate jazz chord progression with substitutions"""
        progression = []

        # Use ii-V-I as base
        templates = self.common_progressions["jazz"]
        template = random.choice(templates)

        for i, melodic_note in enumerate(melody):
            # Basic chord from template
            template_pos = i % len(template)
            base_chord = template[template_pos]

            # Apply jazz substitutions
            if random.random() < 0.3 and (
                not constraints or "no_substitutions" not in constraints
            ):
                # Tritone substitution
                if "V7" in base_chord:
                    base_chord = "bII7"
                # Modal interchange
                elif "IMaj7" in base_chord and random.random() < 0.5:
                    base_chord = "IMaj7#11"

            progression.append(base_chord)

        return progression

    def _create_jazz_voicing(
        self, chord_symbol: str, key_obj: Any, melodic_note: note.Note
    ) -> dict[str, list[pitch.Pitch]]:
        """Create jazz piano voicing"""
        voicing: dict[str, list[pitch.Pitch]] = {"upper": [], "lower": []}

        try:
            # Simplified jazz voicing
            if "Maj7" in chord_symbol:
                root = key_obj.pitchFromDegree(1)
                voicing["lower"] = [
                    pitch.Pitch(midi=root.midi - 12),  # Root
                    pitch.Pitch(midi=root.midi + 7 - 12),  # Fifth
                ]
                voicing["upper"] = [
                    pitch.Pitch(midi=root.midi + 4),  # Third
                    pitch.Pitch(midi=root.midi + 11),  # Seventh
                ]
            elif "7" in chord_symbol:
                # Dominant seventh voicing
                root = key_obj.pitchFromDegree(5)
                voicing["lower"] = [
                    pitch.Pitch(midi=root.midi - 12),
                    pitch.Pitch(midi=root.midi + 10 - 12),  # Flat seventh
                ]
                voicing["upper"] = [
                    pitch.Pitch(midi=root.midi + 4),  # Third
                    pitch.Pitch(midi=root.midi + 9),  # Thirteenth
                ]
            else:
                # Default triad
                root = key_obj.tonic
                voicing["lower"] = [root]
                voicing["upper"] = [
                    pitch.Pitch(midi=root.midi + 4),
                    pitch.Pitch(midi=root.midi + 7),
                ]

        except Exception as e:
            logger.error(f"Jazz voicing failed: {e}")
            # Fallback voicing
            voicing["lower"] = [key_obj.tonic]
            voicing["upper"] = [pitch.Pitch(midi=key_obj.tonic.midi + 4)]

        return voicing

    async def _harmonize_pop(
        self,
        melody: list[Any],
        analysis: dict[str, Any],
        constraints: list[str] | None,
    ) -> dict[str, Any]:
        """Generate pop/rock harmonization"""
        result = {
            "score": stream.Score(),
            "progression": [],
            "roman_numerals": [],
            "confidence_ratings": [],
        }

        # Create guitar-style parts
        melody_part = stream.Part()
        guitar_part = stream.Part()
        bass_part = stream.Part()

        melody_part.partName = "Melody"
        guitar_part.partName = "Guitar"
        bass_part.partName = "Bass"

        # Add melody
        for n in melody:
            if hasattr(n, "pitch") and hasattr(n, "duration"):
                melody_part.append(
                    note.Note(n.pitch, quarterLength=n.duration.quarterLength)
                )

        # Generate pop progression
        progression = self._generate_progression_pop(melody, analysis, constraints)

        # Realize harmony
        for i, (melodic_note, chord_symbol) in enumerate(
            zip(melody, progression, strict=False)
        ):
            # Create pop voicing (power chords or triads)
            chord_notes = self._create_pop_voicing(chord_symbol, analysis["key"])

            # Guitar strumming pattern
            guitar_chord = chord.Chord(chord_notes)
            guitar_part.append(guitar_chord)

            # Simple bass line
            if hasattr(melodic_note, "duration"):
                bass_note = note.Note(
                    chord_notes[0], quarterLength=melodic_note.duration.quarterLength
                )
                bass_part.append(bass_note)

            progression_list = result["progression"]
            roman_list = result["roman_numerals"]
            assert isinstance(progression_list, list)
            assert isinstance(roman_list, list)
            progression_list.append(chord_symbol)
            roman_list.append(chord_symbol)

        score_obj = result["score"]
        assert isinstance(score_obj, stream.Score)
        score_obj.insert(0, melody_part)
        score_obj.insert(0, guitar_part)
        score_obj.insert(0, bass_part)

        return result

    def _generate_progression_pop(
        self,
        melody: list[Any],
        analysis: dict[str, Any],
        constraints: list[str] | None,
    ) -> list[str]:
        """Generate pop chord progression"""
        # Use common pop progressions
        progression_templates = self.common_progressions["pop"]
        template = random.choice(progression_templates)

        progression = []
        for i in range(len(melody)):
            chord = template[i % len(template)]
            progression.append(chord)

        return progression

    def _create_pop_voicing(self, chord_symbol: str, key_obj: Any) -> list[pitch.Pitch]:
        """Create pop/rock guitar voicing"""
        try:
            rn = roman.RomanNumeral(chord_symbol, key_obj)

            # Simple triad or power chord
            pop_vocab = self.style_vocabularies["pop"]
            if (
                isinstance(pop_vocab, dict)
                and pop_vocab.get("power_chords")
                and random.random() < 0.3
            ):
                # Power chord (root + fifth)
                return [rn.root(), pitch.Pitch(midi=rn.root().midi + 7)]
            # Full triad
            return list(rn.pitches)[:3]

        except (AttributeError, TypeError, ValueError) as e:
            # Fallback
            logger.debug(f"Jazz chord realization failed, using fallback: {e}")
            return [key_obj.tonic, pitch.Pitch(midi=key_obj.tonic.midi + 7)]

    async def _harmonize_modal(
        self,
        melody: list[Any],
        analysis: dict[str, Any],
        constraints: list[str] | None,
    ) -> dict[str, Any]:
        """Generate modal harmonization"""
        result = {
            "score": stream.Score(),
            "progression": [],
            "roman_numerals": [],
            "confidence_ratings": [],
        }

        # Detect mode from melody
        mode = self._detect_mode(melody, analysis)

        # Create parts
        melody_part = stream.Part()
        harmony_part = stream.Part()

        # Generate modal progression
        progression = self._generate_progression_modal(
            melody, analysis, mode, constraints
        )

        # Realize harmony
        for i, (melodic_note, chord_symbol) in enumerate(
            zip(melody, progression, strict=False)
        ):
            if hasattr(melodic_note, "pitch") and hasattr(melodic_note, "duration"):
                melody_part.append(
                    note.Note(
                        melodic_note.pitch,
                        quarterLength=melodic_note.duration.quarterLength,
                    )
                )

            # Create modal voicing
            chord_notes = self._create_modal_voicing(
                chord_symbol, analysis["key"], mode
            )
            harmony_chord = chord.Chord(chord_notes)
            harmony_part.append(harmony_chord)

            progression_list = result["progression"]
            roman_list = result["roman_numerals"]
            assert isinstance(progression_list, list)
            assert isinstance(roman_list, list)
            progression_list.append(chord_symbol)
            roman_list.append(chord_symbol)

        score_obj = result["score"]
        assert isinstance(score_obj, stream.Score)
        score_obj.insert(0, melody_part)
        score_obj.insert(0, harmony_part)
        result["modal_center"] = mode

        return result

    def _detect_mode(self, melody: list[Any], analysis: dict[str, Any]) -> str:
        """Detect the mode of the melody"""
        # Simplified mode detection based on characteristic notes
        pitches = [n.pitch for n in melody]
        pitch_classes = {p.pitchClass for p in pitches}

        # Check for characteristic modal notes
        # Mixolydian: has flat 7 (Bb in C = pitch class 10)
        if 10 in pitch_classes and 5 not in pitch_classes:  # Flat 7, no 4
            return "mixolydian"
        # Dorian: has natural 6 and 2 in minor context
        if 11 in pitch_classes and 2 in pitch_classes:  # Natural 6 (B) and 2 (D)
            return "dorian"
        # Lydian: has raised 4 (F# in C = pitch class 6)
        if 6 in pitch_classes:  # Raised 4
            return "lydian"
        return "ionian"  # Default to major

    def _generate_progression_modal(
        self,
        melody: list[Any],
        analysis: dict[str, Any],
        mode: str,
        constraints: list[str] | None,
    ) -> list[str]:
        """Generate modal chord progression"""
        modal_vocab = self.style_vocabularies["modal"]
        if isinstance(modal_vocab, dict) and mode in modal_vocab:
            available_chords = modal_vocab[mode]
        elif isinstance(modal_vocab, dict) and "dorian" in modal_vocab:
            available_chords = modal_vocab["dorian"]
        else:
            available_chords = ["i", "ii", "bIII", "IV", "v", "vi°", "bVII"]

        progression = []

        # Emphasize modal characteristic chords
        for i, note_obj in enumerate(melody):
            if i == 0 or i == len(melody) - 1:
                # Start and end on tonic
                chord_choice = available_chords[0]
            else:
                # Choose chord that contains the melody note
                compatible_chords = []
                for ch in available_chords:
                    if self._is_chord_compatible(ch, note_obj, analysis["key"]):
                        compatible_chords.append(ch)

                if compatible_chords:
                    chord_choice = random.choice(compatible_chords)
                else:
                    chord_choice = available_chords[0]

            progression.append(chord_choice)

        return progression

    def _create_modal_voicing(
        self, chord_symbol: str, key_obj: Any, mode: str
    ) -> list[pitch.Pitch]:
        """Create modal chord voicing"""
        try:
            # Adjust key for mode
            if mode == "dorian":
                # Minor with raised 6th
                modal_key = key_obj.relative
            elif mode == "mixolydian":
                # Major with lowered 7th
                modal_key = key_obj
            else:
                modal_key = key_obj

            rn = roman.RomanNumeral(chord_symbol, modal_key)
            return list(rn.pitches)[:4]  # Quaternary harmony common in modal

        except (AttributeError, TypeError, ValueError) as e:
            logger.debug(f"Modal chord realization failed, using tonic: {e}")
            return [key_obj.tonic]

    def _check_voice_leading(
        self, harmonized_score: stream.Score, style: str
    ) -> dict[str, Any]:
        """Check voice leading quality"""
        quality = {
            "smoothness": 0.0,
            "errors": [],
            "parallel_fifths": 0,
            "parallel_octaves": 0,
        }

        try:
            if len(harmonized_score.parts) < 2:
                return quality

            # Check between outer voices
            soprano = [
                n
                for n in harmonized_score.parts[0].flatten().notes
                if hasattr(n, "pitch")
            ]
            bass = [
                n
                for n in harmonized_score.parts[-1].flatten().notes
                if hasattr(n, "pitch")
            ]

            total_motion = 0.0
            motion_count = 0

            for i in range(min(len(soprano), len(bass)) - 1):
                # Calculate voice motion
                sop_motion = abs(
                    float(soprano[i + 1].pitch.midi) - float(soprano[i].pitch.midi)
                )
                bass_motion = abs(
                    float(bass[i + 1].pitch.midi) - float(bass[i].pitch.midi)
                )

                total_motion += float(sop_motion + bass_motion)
                motion_count += 2

                # Check for parallels
                interval1 = int(
                    (float(soprano[i].pitch.midi) - float(bass[i].pitch.midi)) % 12
                )
                interval2 = int(
                    (float(soprano[i + 1].pitch.midi) - float(bass[i + 1].pitch.midi))
                    % 12
                )

                if interval1 == interval2:
                    if interval1 == 7:  # Perfect fifth
                        parallel_fifths = quality["parallel_fifths"]
                        assert isinstance(parallel_fifths, int)
                        quality["parallel_fifths"] = parallel_fifths + 1
                    elif interval1 == 0:  # Octave
                        parallel_octaves = quality["parallel_octaves"]
                        assert isinstance(parallel_octaves, int)
                        quality["parallel_octaves"] = parallel_octaves + 1

            # Calculate smoothness
            if motion_count > 0:
                avg_motion = float(total_motion) / float(motion_count)
                # Smooth voice leading has average motion of 2-3 semitones
                quality["smoothness"] = float(
                    max(0.0, 1.0 - abs(avg_motion - 2.5) / 10.0)
                )

            # Style-specific allowances
            if style == "pop" or style == "jazz":
                # More lenient for contemporary styles
                current_smoothness = quality.get("smoothness", 0.0)
                if isinstance(current_smoothness, (int, float)):
                    quality["smoothness"] = float(
                        min(1.0, float(current_smoothness) * 1.2)
                    )
                else:
                    quality["smoothness"] = 0.0

        except Exception as e:
            logger.error(f"Voice leading check failed: {e}")

        return quality

    def _analyze_harmonic_rhythm(self, progression: list[str]) -> dict[str, Any]:
        """Analyze the rate of harmonic change"""
        if not progression:
            return {}

        # Count chord changes
        changes = 0
        for i in range(len(progression) - 1):
            if progression[i] != progression[i + 1]:
                changes += 1

        return {
            "changes_per_measure": changes / max(1, len(progression) / 4),
            "static_percentage": (len(progression) - changes) / len(progression) * 100,
            "most_common_chord": (
                max(set(progression), key=progression.count) if progression else None
            ),
        }

    def _generate_explanations(
        self,
        harmonization: dict[str, Any],
        melodic_analysis: dict[str, Any],
        style: str,
    ) -> list[dict[str, str]]:
        """Generate explanations for harmonic choices"""
        explanations = []

        # Explain style choice
        explanations.append(
            {
                "aspect": "Style",
                "explanation": f"Using {style} harmonization style which emphasizes "
                + self._get_style_characteristics(style),
            }
        )

        # Explain key
        if melodic_analysis.get("key"):
            explanations.append(
                {
                    "aspect": "Key",
                    "explanation": f"Melody is in {melodic_analysis['key']}, so harmony is based on this key",
                }
            )

        # Explain progression choices
        progression = harmonization.get("progression", [])
        if progression:
            # Find most common progression
            style_progressions: Any = self.common_progressions.get(style, {})
            if isinstance(style_progressions, dict):
                for common_name, common_prog in style_progressions.items():
                    if self._contains_progression(progression, common_prog):
                        explanations.append(
                            {
                                "aspect": "Progression",
                                "explanation": f"Using elements of the {common_name} progression, common in {style}",
                            }
                        )
                        break

        # Explain cadences
        if progression and progression[-1] == "I":
            explanations.append(
                {
                    "aspect": "Cadence",
                    "explanation": "Ending with an authentic cadence (V-I) for strong resolution",
                }
            )

        # Explain voice leading approach
        vl_quality = harmonization.get("voice_leading_quality", {})
        if vl_quality.get("smoothness", 0) > 0.7:
            explanations.append(
                {
                    "aspect": "Voice Leading",
                    "explanation": "Smooth voice leading with minimal motion between chords",
                }
            )

        return explanations

    def _get_style_characteristics(self, style: str) -> str:
        """Get characteristic description of style"""
        characteristics = {
            "classical": "traditional functional harmony, clear voice leading, and standard cadences",
            "jazz": "extended chords (7ths, 9ths, 13ths), chord substitutions, and sophisticated voice leading",
            "pop": "simple triadic harmony, common progressions (I-V-vi-IV), and guitar-friendly voicings",
            "modal": "modal chord colors, avoiding traditional dominant function, and emphasizing characteristic scale degrees",
        }
        return characteristics.get(style, "characteristic harmonic patterns")

    def _contains_progression(
        self, full_progression: list[str], pattern: list[str]
    ) -> bool:
        """Check if progression contains a specific pattern"""
        pattern_length = len(pattern)
        for i in range(len(full_progression) - pattern_length + 1):
            if full_progression[i : i + pattern_length] == pattern:
                return True
        return False
