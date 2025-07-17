"""
Configuration for Musical Accuracy Validation Tests

This module contains test data and expected results for validating
musical analysis accuracy.
"""

from typing import Dict, List, Tuple, Optional


class MusicalTestData:
    """Test data for musical accuracy validation"""
    
    # Known key signatures for Bach chorales
    BACH_CHORALE_KEYS = {
        "bwv66.6": "f# minor",
        "bwv84.5": "a minor", 
        "bwv269": "c major",
        "bwv277": "eb major",
        "bwv86.6": "a major",
        "bwv153.1": "a minor",
        "bwv267": "g major",
        "bwv281": "f major",
    }
    
    # Common chord progressions in different styles
    CHORD_PROGRESSIONS = {
        "classical_cadences": {
            "authentic": ["V", "I"],
            "plagal": ["IV", "I"],
            "deceptive": ["V", "vi"],
            "half": ["I", "V"],
        },
        "jazz_progressions": {
            "ii_v_i": ["ii7", "V7", "Imaj7"],
            "rhythm_changes": ["Imaj7", "VI7", "ii7", "V7"],
            "blues": ["I7", "I7", "I7", "I7", "IV7", "IV7", "I7", "I7", "V7", "IV7", "I7", "V7"],
            "minor_ii_v_i": ["iiø7", "V7alt", "i7"],
        },
        "pop_progressions": {
            "four_chord": ["I", "V", "vi", "IV"],
            "axis": ["I", "V", "vi", "iii", "IV", "I", "IV", "V"],
            "doo_wop": ["I", "vi", "IV", "V"],
        }
    }
    
    # Voice leading rules for validation
    VOICE_LEADING_RULES = {
        "forbidden_parallels": {
            "parallel_fifths": {"interval": 7, "forbidden": True},
            "parallel_octaves": {"interval": 12, "forbidden": True},
            "parallel_unisons": {"interval": 0, "forbidden": True},
        },
        "preferred_motion": {
            "contrary": {"score": 1.0},
            "oblique": {"score": 0.8},
            "similar": {"score": 0.6},
            "parallel": {"score": 0.4},
        },
        "voice_ranges": {
            "soprano": {"min": "C4", "max": "G5"},
            "alto": {"min": "G3", "max": "D5"},
            "tenor": {"min": "C3", "max": "G4"},
            "bass": {"min": "E2", "max": "D4"},
        }
    }
    
    # Counterpoint species rules
    COUNTERPOINT_RULES = {
        "first_species": {
            "consonances": ["P1", "P5", "P8", "m3", "M3", "m6", "M6"],
            "perfect_consonances": ["P1", "P5", "P8"],
            "imperfect_consonances": ["m3", "M3", "m6", "M6"],
            "note_ratio": "1:1",
            "first_interval": ["P1", "P5", "P8"],
            "last_interval": ["P1", "P8"],
        },
        "second_species": {
            "note_ratio": "2:1",
            "strong_beat": "consonant",
            "weak_beat": "consonant_or_passing",
            "dissonances_allowed": ["passing_tone"],
        },
        "third_species": {
            "note_ratio": "4:1",
            "first_beat": "consonant",
            "dissonances_allowed": ["passing_tone", "neighbor_tone"],
        },
        "fourth_species": {
            "note_ratio": "1:1",
            "technique": "syncopation",
            "dissonances_allowed": ["suspension"],
            "resolution": "down_by_step",
        },
        "fifth_species": {
            "note_ratio": "mixed",
            "combines": ["all_species"],
            "additional_allowed": ["eighth_notes", "ornaments"],
        }
    }
    
    # Style characteristics for validation
    STYLE_CHARACTERISTICS = {
        "baroque": {
            "texture": "polyphonic",
            "harmonic_rhythm": "regular",
            "ornamentation": ["trill", "mordent", "turn"],
            "forms": ["fugue", "invention", "suite", "concerto"],
            "typical_rhythms": ["dotted", "continuous_16ths"],
        },
        "classical": {
            "texture": "homophonic",
            "harmonic_rhythm": "clear",
            "dynamics": "terraced",
            "forms": ["sonata", "rondo", "theme_and_variations"],
            "phrasing": "periodic",
        },
        "romantic": {
            "texture": "varied",
            "harmony": "chromatic",
            "dynamics": "expressive",
            "rubato": True,
            "extended_harmony": True,
        },
        "impressionist": {
            "harmony": ["extended", "modal", "whole_tone"],
            "texture": "coloristic",
            "rhythm": "fluid",
            "chords": ["9th", "11th", "13th", "added_note"],
        },
        "jazz": {
            "harmony": "extended",
            "rhythm": "swing",
            "improvisation": True,
            "chords": ["7th", "9th", "11th", "13th", "altered"],
            "scales": ["blues", "bebop", "modal"],
        }
    }
    
    # Expected analysis results for specific pieces
    EXPECTED_RESULTS = {
        "bach_inventions": {
            "texture": "two_voice_polyphony",
            "imitation": True,
            "modulation": True,
            "motivic_development": True,
        },
        "mozart_sonatas": {
            "form": "sonata_allegro",
            "key_areas": ["tonic", "dominant", "tonic"],
            "themes": ["first_theme", "second_theme"],
            "clear_cadences": True,
        },
        "chopin_nocturnes": {
            "texture": "melody_with_accompaniment",
            "ornamentation": "extensive",
            "rubato": True,
            "chromatic_harmony": True,
        }
    }
    
    @classmethod
    def get_expected_key(cls, piece_id: str) -> Optional[str]:
        """Get expected key for a known piece"""
        return cls.BACH_CHORALE_KEYS.get(piece_id)
    
    @classmethod
    def get_progression_template(cls, style: str, progression_type: str) -> Optional[List[str]]:
        """Get a chord progression template"""
        style_progressions = cls.CHORD_PROGRESSIONS.get(style, {})
        return style_progressions.get(progression_type)
    
    @classmethod
    def validate_voice_leading(cls, interval: int, motion_type: str) -> Dict[str, any]:
        """Validate voice leading based on rules"""
        result = {
            "valid": True,
            "issues": [],
            "score": 1.0
        }
        
        # Check for forbidden parallels
        for parallel_type, rule in cls.VOICE_LEADING_RULES["forbidden_parallels"].items():
            if interval == rule["interval"] and motion_type == "parallel":
                result["valid"] = False
                result["issues"].append(f"Forbidden {parallel_type}")
                result["score"] = 0.0
                
        # Score based on motion type
        if motion_type in cls.VOICE_LEADING_RULES["preferred_motion"]:
            result["score"] = cls.VOICE_LEADING_RULES["preferred_motion"][motion_type]["score"]
            
        return result
    
    @classmethod
    def get_species_rules(cls, species: int) -> Optional[Dict[str, any]]:
        """Get counterpoint rules for a specific species"""
        species_key = f"{['first', 'second', 'third', 'fourth', 'fifth'][species-1]}_species"
        return cls.COUNTERPOINT_RULES.get(species_key)
    
    @classmethod
    def get_style_characteristics(cls, style: str) -> Optional[Dict[str, any]]:
        """Get expected characteristics for a musical style"""
        return cls.STYLE_CHARACTERISTICS.get(style.lower())