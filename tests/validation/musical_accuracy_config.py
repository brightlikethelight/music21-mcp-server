"""
Configuration for Musical Accuracy Tests

This module defines the expected accuracy thresholds and test configurations
for different types of musical analysis.
"""

from typing import Dict, List, Any

# Accuracy thresholds for different analysis types
ACCURACY_THRESHOLDS = {
    "key_detection": {
        "bach_chorales": 0.85,  # 85% accuracy on Bach chorales
        "classical_sonatas": 0.90,  # 90% accuracy on Classical sonatas
        "romantic_pieces": 0.80,  # 80% accuracy on Romantic pieces (more chromatic)
        "modal_music": 0.75,  # 75% accuracy on modal music
        "atonal_music": 0.60,  # 60% accuracy on atonal music (harder to define)
    },
    "chord_progression": {
        "common_progressions": 0.95,  # 95% accuracy on ii-V-I, IV-V-I, etc.
        "jazz_progressions": 0.85,  # 85% accuracy on jazz chord progressions
        "chromatic_progressions": 0.80,  # 80% accuracy on chromatic progressions
    },
    "pattern_recognition": {
        "melodic_sequences": 0.90,  # 90% accuracy on melodic sequences
        "rhythmic_patterns": 0.85,  # 85% accuracy on rhythmic patterns
        "fugue_subjects": 0.80,  # 80% accuracy on fugue subject detection
    },
    "voice_leading": {
        "parallel_detection": 0.95,  # 95% accuracy on parallel fifth/octave detection
        "smoothness_score": 0.85,  # 85% accuracy on voice leading smoothness
        "independence_score": 0.80,  # 80% accuracy on voice independence
    },
    "harmony_analysis": {
        "roman_numerals": 0.85,  # 85% accuracy on Roman numeral analysis
        "cadence_detection": 0.90,  # 90% accuracy on cadence detection
        "modulation_detection": 0.80,  # 80% accuracy on modulation detection
    },
    "style_analysis": {
        "period_detection": 0.85,  # 85% accuracy on musical period detection
        "composer_style": 0.75,  # 75% accuracy on composer style recognition
    }
}

# Test corpus selections for different musical periods
TEST_CORPUS = {
    "baroque": {
        "composers": ["bach", "handel", "vivaldi"],
        "forms": ["fugue", "chorale", "suite", "concerto"],
        "key_characteristics": ["functional harmony", "counterpoint", "sequences"],
    },
    "classical": {
        "composers": ["mozart", "haydn", "beethoven"],
        "forms": ["sonata", "symphony", "string quartet"],
        "key_characteristics": ["periodic phrasing", "alberti bass", "clear cadences"],
    },
    "romantic": {
        "composers": ["chopin", "schumann", "brahms"],
        "forms": ["character piece", "art song", "symphonic poem"],
        "key_characteristics": ["chromatic harmony", "rubato", "extended forms"],
    },
    "impressionist": {
        "composers": ["debussy", "ravel"],
        "forms": ["prelude", "tone poem"],
        "key_characteristics": ["extended harmony", "whole tone scales", "parallel motion"],
    },
    "modern": {
        "composers": ["schoenberg", "webern", "berg"],
        "forms": ["twelve-tone", "atonal"],
        "key_characteristics": ["atonality", "serialism", "set theory"],
    }
}

# Expected analysis results for specific pieces
KNOWN_ANALYSES = {
    "bach/bwv66.6": {
        "key": "f# minor",
        "time_signature": "4/4",
        "final_cadence": "authentic",
        "texture": "homophonic",
        "voice_count": 4,
    },
    "mozart/k545/movement1": {
        "key": "C major",
        "time_signature": "4/4",
        "form": "sonata",
        "texture": "homophonic",
        "characteristic_patterns": ["alberti_bass", "scales"],
    },
    "beethoven/opus27no2/movement1": {
        "key": "c# minor",
        "time_signature": "4/4",
        "texture": "homophonic",
        "characteristic_patterns": ["triplets", "sustained_melody"],
    },
}

# Voice leading rules for different style periods
VOICE_LEADING_RULES = {
    "renaissance": {
        "forbidden_parallels": ["P5", "P8"],
        "forbidden_motion": ["aug2", "aug4"],
        "preferred_motion": ["contrary", "oblique"],
        "max_leap": 8,  # Maximum leap in semitones
    },
    "baroque": {
        "forbidden_parallels": ["P5", "P8"],
        "forbidden_motion": ["aug2"],
        "preferred_motion": ["contrary", "similar"],
        "max_leap": 12,
    },
    "classical": {
        "forbidden_parallels": ["P5", "P8"],
        "forbidden_motion": [],
        "preferred_motion": ["smooth"],
        "max_leap": 12,
    },
    "romantic": {
        "forbidden_parallels": [],  # More flexible
        "forbidden_motion": [],
        "preferred_motion": ["expressive"],
        "max_leap": 15,
    },
}

# Harmonic progression templates
PROGRESSION_TEMPLATES = {
    "common": {
        "ii-V-I": ["ii", "V", "I"],
        "IV-V-I": ["IV", "V", "I"],
        "I-vi-IV-V": ["I", "vi", "IV", "V"],
        "I-V-vi-IV": ["I", "V", "vi", "IV"],
        "vi-ii-V-I": ["vi", "ii", "V", "I"],
    },
    "jazz": {
        "ii7-V7-I": ["ii7", "V7", "Imaj7"],
        "I-VI7-ii7-V7": ["Imaj7", "VI7", "ii7", "V7"],
        "iii7-VI7-ii7-V7": ["iii7", "VI7", "ii7", "V7"],
    },
    "classical": {
        "I-IV-V-I": ["I", "IV", "V", "I"],
        "I-ii6-V-I": ["I", "ii6", "V", "I"],
        "I-IV-I6/4-V-I": ["I", "IV", "I64", "V", "I"],
    },
}

# Pattern recognition settings
PATTERN_SETTINGS = {
    "melodic": {
        "min_length": 3,
        "max_length": 16,
        "min_occurrences": 2,
        "transposition_threshold": 0.8,
    },
    "rhythmic": {
        "min_length": 2,
        "max_length": 8,
        "min_occurrences": 3,
        "variation_threshold": 0.1,
    },
    "harmonic": {
        "min_length": 2,
        "max_length": 8,
        "min_occurrences": 2,
        "similarity_threshold": 0.7,
    },
}

# Cadence patterns
CADENCE_PATTERNS = {
    "authentic": {
        "perfect": ["V", "I"],
        "imperfect": ["V", "I"],  # With different voicing
    },
    "plagal": {
        "standard": ["IV", "I"],
    },
    "deceptive": {
        "standard": ["V", "vi"],
        "chromatic": ["V", "bVI"],
    },
    "half": {
        "standard": ["I", "V"],
        "phrygian": ["iv6", "V"],
    },
}

def get_expected_accuracy(analysis_type: str, subtype: str) -> float:
    """Get the expected accuracy threshold for a given analysis type"""
    return ACCURACY_THRESHOLDS.get(analysis_type, {}).get(subtype, 0.8)

def get_corpus_for_period(period: str) -> Dict[str, Any]:
    """Get test corpus information for a musical period"""
    return TEST_CORPUS.get(period, {})

def get_known_analysis(piece: str) -> Dict[str, Any]:
    """Get known analysis results for a specific piece"""
    return KNOWN_ANALYSES.get(piece, {})

def get_voice_leading_rules(period: str) -> Dict[str, Any]:
    """Get voice leading rules for a style period"""
    return VOICE_LEADING_RULES.get(period, VOICE_LEADING_RULES["classical"])

def get_progression_template(type: str, name: str) -> List[str]:
    """Get a chord progression template"""
    return PROGRESSION_TEMPLATES.get(type, {}).get(name, [])