"""
Music theory analysis module with comprehensive theoretical operations
"""
from typing import Dict, List, Any, Optional, Union, Tuple
from music21 import (
    stream, note, chord, key, pitch, interval, scale, roman,
    analysis, harmony, voiceLeading, meter
)
import numpy as np
from collections import Counter, defaultdict
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class KeyDetectionMethod(Enum):
    """Available key detection algorithms"""
    KRUMHANSL = "krumhansl"
    AARDEN = "aarden"
    BELLMAN = "bellman"
    TEMPERLEY = "temperley"
    SIMPLE = "simple"
    HYBRID = "hybrid"


class ScaleDegree(Enum):
    """Scale degrees with functional names"""
    TONIC = 1
    SUPERTONIC = 2
    MEDIANT = 3
    SUBDOMINANT = 4
    DOMINANT = 5
    SUBMEDIANT = 6
    LEADING_TONE = 7
    SUBTONIC = 7  # For natural minor


@dataclass
class KeyAnalysisResult:
    """Comprehensive key analysis result"""
    key: key.Key
    confidence: float
    method: str
    alternatives: List[Tuple[key.Key, float]]
    local_keys: List[Dict[str, Any]]
    modulations: List[Dict[str, Any]]
    evidence: Dict[str, Any]


@dataclass
class IntervalAnalysis:
    """Detailed interval analysis result"""
    interval: interval.Interval
    quality: str
    size: int
    semitones: int
    cents: float
    consonance: str
    enharmonic_equivalent: Optional[str]
    compound: bool
    inverted: Optional[interval.Interval]


class TheoryAnalyzer:
    """Comprehensive music theory analysis"""
    
    # Consonance classifications
    CONSONANCE_MAP = {
        0: "perfect",      # Unison
        3: "imperfect",    # Minor third
        4: "imperfect",    # Major third
        5: "perfect",      # Perfect fourth (contextual)
        7: "perfect",      # Perfect fifth
        8: "imperfect",    # Minor sixth
        9: "imperfect",    # Major sixth
        12: "perfect",     # Octave
    }
    
    # Functional harmony mappings
    FUNCTION_MAP = {
        1: "tonic",
        2: "supertonic",
        3: "mediant",
        4: "subdominant",
        5: "dominant",
        6: "submediant",
        7: "leading-tone"
    }
    
    def __init__(self):
        self.cache = {}
    
    async def analyze_key(
        self,
        score: stream.Score,
        method: KeyDetectionMethod = KeyDetectionMethod.HYBRID,
        window_size: Optional[int] = None,
        confidence_threshold: float = 0.5
    ) -> KeyAnalysisResult:
        """
        Perform comprehensive key analysis
        
        Args:
            score: The score to analyze
            method: Key detection method to use
            window_size: Window size for local analysis (measures)
            confidence_threshold: Minimum confidence for key detection
            
        Returns:
            Comprehensive key analysis result
        """
        # Global key analysis
        global_key, confidence, alternatives = await self._detect_global_key(
            score, method
        )
        
        # Local key analysis
        local_keys = []
        modulations = []
        
        if window_size:
            local_keys = await self._analyze_local_keys(
                score, window_size, method
            )
            modulations = self._detect_modulations(
                local_keys, confidence_threshold
            )
        
        # Gather evidence for key
        evidence = await self._gather_key_evidence(score, global_key)
        
        return KeyAnalysisResult(
            key=global_key,
            confidence=confidence,
            method=method.value,
            alternatives=alternatives,
            local_keys=local_keys,
            modulations=modulations,
            evidence=evidence
        )
    
    async def _detect_global_key(
        self,
        score: stream.Score,
        method: KeyDetectionMethod
    ) -> Tuple[key.Key, float, List[Tuple[key.Key, float]]]:
        """Detect global key using specified method with improved polyphonic handling"""
        
        # Special handling for multi-voice music (e.g., Bach chorales)
        if len(score.parts) > 1:
            # Analyze soprano and bass separately (most important for key)
            results_by_part = {}
            
            # Analyze soprano (usually most melodic)
            if len(score.parts) > 0:
                soprano = score.parts[0]
                try:
                    sop_key = soprano.analyze(f'key.{method.value}' if method != KeyDetectionMethod.SIMPLE else 'key')
                    sop_conf = getattr(sop_key, 'correlationCoefficient', 0.5)
                    results_by_part['soprano'] = (sop_key, sop_conf * 1.2)  # Weight soprano higher
                except:
                    pass
            
            # Analyze bass (harmonic foundation)
            if len(score.parts) > 1:
                bass = score.parts[-1]
                try:
                    bass_key = bass.analyze(f'key.{method.value}' if method != KeyDetectionMethod.SIMPLE else 'key')
                    bass_conf = getattr(bass_key, 'correlationCoefficient', 0.5)
                    results_by_part['bass'] = (bass_key, bass_conf * 1.1)  # Weight bass high
                except:
                    pass
            
            # Analyze full score
            try:
                full_key = score.analyze(f'key.{method.value}' if method != KeyDetectionMethod.SIMPLE else 'key')
                full_conf = getattr(full_key, 'correlationCoefficient', 0.5)
                results_by_part['full'] = (full_key, full_conf)
            except:
                pass
            
            # Combine results intelligently
            key_votes = {}
            for part_name, (k, conf) in results_by_part.items():
                if k not in key_votes:
                    key_votes[k] = 0
                key_votes[k] += conf
            
            # Find best key
            if key_votes:
                best_key = max(key_votes.items(), key=lambda x: x[1])
                # Boost confidence if multiple parts agree
                agreement_bonus = len([k for k, v in results_by_part.items() if v[0] == best_key[0]]) / len(results_by_part)
                final_confidence = min(0.95, (best_key[1] / len(results_by_part)) * agreement_bonus * 1.3)
                
                # Get alternatives
                sorted_keys = sorted(key_votes.items(), key=lambda x: x[1], reverse=True)
                alternatives = [(k, v/sum(key_votes.values())) for k, v in sorted_keys[1:4]]
                
                return best_key[0], final_confidence, alternatives
        
        # For single-voice music or when multi-voice analysis fails
        if method == KeyDetectionMethod.HYBRID:
            # Combine multiple methods
            results = {}
            
            for m in [KeyDetectionMethod.KRUMHANSL, KeyDetectionMethod.AARDEN,
                     KeyDetectionMethod.TEMPERLEY]:
                try:
                    k = score.analyze(f'key.{m.value}')
                    if k:
                        conf = getattr(k, 'correlationCoefficient', 0.5)
                        # Boost confidence for clear tonal music
                        if conf > 0.7:
                            conf = min(0.95, conf * 1.2)
                        results[k] = results.get(k, 0) + conf
                except:
                    continue
            
            # Sort by combined confidence
            sorted_results = sorted(
                results.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            if sorted_results:
                best_key = sorted_results[0][0]
                # Better confidence calculation
                total_weight = sum(v for k, v in results.items())
                confidence = sorted_results[0][1] / total_weight if total_weight > 0 else 0.5
                alternatives = [(k, v/total_weight) for k, v in sorted_results[1:4]]
                return best_key, confidence, alternatives
        
        else:
            # Single method
            if method == KeyDetectionMethod.SIMPLE:
                k = score.analyze('key')
            else:
                k = score.analyze(f'key.{method.value}')
            
            confidence = getattr(k, 'correlationCoefficient', 0.5)
            # Boost confidence for strong correlations
            if confidence > 0.7:
                confidence = min(0.95, confidence * 1.15)
            
            # Get alternatives by analyzing with different methods
            alternatives = []
            for alt_method in ['krumhansl', 'aarden', 'temperley']:
                if alt_method != method.value:
                    try:
                        alt_key = score.analyze(f'key.{alt_method}')
                        if alt_key and alt_key != k:
                            alt_conf = getattr(alt_key, 'correlationCoefficient', 0.3)
                            alternatives.append((alt_key, alt_conf))
                    except:
                        continue
            
            return k, confidence, alternatives
    
    async def _analyze_local_keys(
        self,
        score: stream.Score,
        window_size: int,
        method: KeyDetectionMethod
    ) -> List[Dict[str, Any]]:
        """Analyze keys in local windows"""
        local_keys = []
        measures = list(score.getElementsByClass(stream.Measure))
        
        for i in range(0, len(measures), window_size // 2):  # Overlap windows
            window_start = i
            window_end = min(i + window_size, len(measures))
            
            # Create window stream
            window = stream.Stream()
            for j in range(window_start, window_end):
                window.append(measures[j])
            
            # Analyze window
            try:
                if method == KeyDetectionMethod.SIMPLE:
                    local_key = window.analyze('key')
                else:
                    local_key = window.analyze(f'key.{method.value}')
                
                local_keys.append({
                    'measure_start': window_start + 1,
                    'measure_end': window_end,
                    'key': local_key,
                    'confidence': getattr(local_key, 'correlationCoefficient', 0.5),
                    'pitch_content': self._analyze_pitch_content(window)
                })
            except:
                continue
        
        return local_keys
    
    def _detect_modulations(
        self,
        local_keys: List[Dict[str, Any]],
        threshold: float
    ) -> List[Dict[str, Any]]:
        """Detect modulations from local key analysis"""
        modulations = []
        
        for i in range(1, len(local_keys)):
            prev_key = local_keys[i-1]['key']
            curr_key = local_keys[i]['key']
            
            if prev_key != curr_key and local_keys[i]['confidence'] > threshold:
                # Analyze modulation type
                mod_type = self._classify_modulation(prev_key, curr_key)
                
                modulations.append({
                    'measure': local_keys[i]['measure_start'],
                    'from_key': str(prev_key),
                    'to_key': str(curr_key),
                    'type': mod_type,
                    'confidence': local_keys[i]['confidence'],
                    'pivot_likelihood': self._calculate_pivot_likelihood(
                        prev_key, curr_key
                    )
                })
        
        return modulations
    
    def _classify_modulation(self, from_key: key.Key, to_key: key.Key) -> str:
        """Classify the type of modulation"""
        # Calculate interval between tonics
        tonic_interval = interval.Interval(from_key.tonic, to_key.tonic)
        semitones = tonic_interval.semitones % 12
        
        # Check key relationships
        if to_key == from_key.dominant():
            return "dominant"
        elif to_key == from_key.subdominant():
            return "subdominant"
        elif to_key == from_key.relative:
            return "relative"
        elif to_key == from_key.parallel:
            return "parallel"
        elif semitones == 1:
            return "chromatic"
        elif semitones == 2:
            return "step-wise"
        else:
            return "distant"
    
    def _calculate_pivot_likelihood(self, from_key: key.Key, to_key: key.Key) -> float:
        """Calculate likelihood of pivot chord modulation"""
        # Find common chords between keys
        from_scale = from_key.pitches
        to_scale = to_key.pitches
        
        common_pitches = set(p.name for p in from_scale) & set(p.name for p in to_scale)
        
        # More common pitches = higher pivot likelihood
        return len(common_pitches) / 7.0
    
    async def _gather_key_evidence(
        self,
        score: stream.Score,
        detected_key: key.Key
    ) -> Dict[str, Any]:
        """Gather evidence supporting the key detection"""
        evidence = {
            'scale_degrees': {},
            'leading_tones': 0,
            'cadences': [],
            'first_last_notes': {},
            'accidentals': {}
        }
        
        # Analyze scale degree frequency
        scale_pitches = detected_key.pitches
        pitch_counts = Counter()
        
        for n in score.flatten().notes:
            if isinstance(n, note.Note):
                pitch_counts[n.pitch.pitchClass] += 1
        
        # Map to scale degrees
        for pc, count in pitch_counts.items():
            for i, scale_pitch in enumerate(scale_pitches):
                if scale_pitch.pitchClass == pc:
                    degree = i + 1
                    evidence['scale_degrees'][degree] = count
                    break
        
        # Count leading tone resolutions
        notes = list(score.flatten().notes)
        leading_tone_pc = scale_pitches[6].pitchClass  # 7th degree
        tonic_pc = scale_pitches[0].pitchClass
        
        for i in range(len(notes) - 1):
            if isinstance(notes[i], note.Note) and isinstance(notes[i+1], note.Note):
                if (notes[i].pitch.pitchClass == leading_tone_pc and
                    notes[i+1].pitch.pitchClass == tonic_pc):
                    evidence['leading_tones'] += 1
        
        # Analyze first and last notes
        if notes:
            if isinstance(notes[0], note.Note):
                evidence['first_last_notes']['first'] = notes[0].pitch.name
            if isinstance(notes[-1], note.Note):
                evidence['first_last_notes']['last'] = notes[-1].pitch.name
        
        # Detect cadences
        evidence['cadences'] = await self._detect_cadences(score, detected_key)
        
        return evidence
    
    def _analyze_pitch_content(self, segment: stream.Stream) -> Dict[str, Any]:
        """Analyze pitch content of a stream segment"""
        pitches = []
        for n in segment.flatten().notes:
            if isinstance(n, note.Note):
                pitches.append(n.pitch.pitchClass)
            elif isinstance(n, chord.Chord):
                pitches.extend([p.pitchClass for p in n.pitches])
        
        if not pitches:
            return {}
        
        pitch_counts = Counter(pitches)
        total_pitches = sum(pitch_counts.values())
        
        return {
            'pitch_classes': dict(pitch_counts),
            'pitch_class_distribution': {
                pc: count / total_pitches
                for pc, count in pitch_counts.items()
            },
            'chromatic_coverage': len(pitch_counts) / 12.0,
            'most_common': pitch_counts.most_common(3)
        }
    
    async def analyze_scale(
        self,
        pitches: Union[List[pitch.Pitch], stream.Stream],
        include_modes: bool = True,
        include_exotic: bool = False
    ) -> Dict[str, Any]:
        """
        Analyze scale patterns in pitch content
        
        Args:
            pitches: List of pitches or stream to analyze
            include_modes: Include modal scale detection
            include_exotic: Include exotic scales
            
        Returns:
            Scale analysis results
        """
        # Extract pitches if stream
        if isinstance(pitches, stream.Stream):
            pitch_list = []
            for n in pitches.flatten().notes:
                if isinstance(n, note.Note):
                    pitch_list.append(n.pitch)
                elif isinstance(n, chord.Chord):
                    pitch_list.extend(n.pitches)
            pitches = pitch_list
        
        if not pitches:
            return {"error": "No pitches to analyze"}
        
        # Get unique pitch classes
        pitch_classes = sorted(set(p.pitchClass for p in pitches))
        
        results = {
            'pitch_classes': pitch_classes,
            'pitch_class_count': len(pitch_classes),
            'possible_scales': [],
            'best_match': None,
            'confidence': 0.0
        }
        
        # Standard scales to check
        scales_to_check = [
            ('major', scale.MajorScale),
            ('natural_minor', scale.NaturalMinorScale),
            ('harmonic_minor', scale.HarmonicMinorScale),
            ('melodic_minor', scale.MelodicMinorScale),
        ]
        
        if include_modes:
            scales_to_check.extend([
                ('dorian', scale.DorianScale),
                ('phrygian', scale.PhrygianScale),
                ('lydian', scale.LydianScale),
                ('mixolydian', scale.MixolydianScale),
                ('aeolian', scale.AeolianScale),
                ('locrian', scale.LocrianScale),
            ])
        
        if include_exotic:
            scales_to_check.extend([
                ('whole_tone', scale.WholeToneScale),
                ('chromatic', scale.ChromaticScale),
                ('pentatonic_major', scale.MajorPentatonicScale),
                ('pentatonic_minor', scale.MinorPentatonicScale),
                ('blues', scale.BluesScale),
                ('harmonic_major', scale.HarmonicMajorScale),
                ('hungarian_minor', scale.HungarianMinorScale),
                ('arabic', scale.ArabicScale),
            ])
        
        # Check each scale type
        for scale_name, scale_class in scales_to_check:
            for tonic_pc in range(12):
                tonic = pitch.Pitch(tonic_pc)
                try:
                    test_scale = scale_class(tonic)
                    scale_pcs = [p.pitchClass for p in test_scale.pitches]
                    
                    # Calculate match percentage
                    matches = sum(1 for pc in pitch_classes if pc in scale_pcs)
                    match_percentage = matches / len(pitch_classes)
                    
                    # Calculate how many scale tones are present
                    coverage = sum(1 for pc in scale_pcs if pc in pitch_classes) / len(scale_pcs)
                    
                    # Combined score
                    score = (match_percentage + coverage) / 2
                    
                    if score > 0.7:  # Threshold for considering a match
                        results['possible_scales'].append({
                            'scale': f"{tonic.name} {scale_name}",
                            'tonic': tonic.name,
                            'type': scale_name,
                            'match_score': score,
                            'matched_pitches': matches,
                            'scale_coverage': coverage
                        })
                except:
                    continue
        
        # Sort by match score
        results['possible_scales'].sort(key=lambda x: x['match_score'], reverse=True)
        
        if results['possible_scales']:
            results['best_match'] = results['possible_scales'][0]
            results['confidence'] = results['best_match']['match_score']
        
        return results
    
    async def analyze_intervals(
        self,
        pitch1: Union[pitch.Pitch, note.Note],
        pitch2: Union[pitch.Pitch, note.Note],
        detailed: bool = True
    ) -> IntervalAnalysis:
        """
        Analyze interval between two pitches
        
        Args:
            pitch1: First pitch
            pitch2: Second pitch
            detailed: Include detailed analysis
            
        Returns:
            Comprehensive interval analysis
        """
        # Extract pitches if notes
        if isinstance(pitch1, note.Note):
            pitch1 = pitch1.pitch
        if isinstance(pitch2, note.Note):
            pitch2 = pitch2.pitch
        
        # Create interval
        intv = interval.Interval(pitch1, pitch2)
        
        # Basic properties
        analysis = IntervalAnalysis(
            interval=intv,
            quality=intv.niceName.split()[0] if ' ' in intv.niceName else 'Perfect',
            size=intv.generic.value,
            semitones=intv.semitones,
            cents=intv.cents,
            consonance=self._classify_consonance(intv.semitones % 12),
            enharmonic_equivalent=None,
            compound=intv.semitones > 12,
            inverted=intv.complement if detailed else None
        )
        
        # Find enharmonic equivalent
        if detailed:
            analysis.enharmonic_equivalent = self._find_enharmonic_interval(intv)
        
        return analysis
    
    def _classify_consonance(self, semitones: int) -> str:
        """Classify interval consonance"""
        semitones = semitones % 12
        
        if semitones in self.CONSONANCE_MAP:
            return self.CONSONANCE_MAP[semitones]
        elif semitones in [1, 2, 10, 11]:
            return "dissonant"
        else:
            return "mild_dissonance"
    
    def _find_enharmonic_interval(self, intv: interval.Interval) -> Optional[str]:
        """Find enharmonic equivalent of interval"""
        # Common enharmonic equivalents
        enharmonics = {
            'Augmented Second': 'Minor Third',
            'Diminished Fourth': 'Major Third',
            'Augmented Fourth': 'Diminished Fifth',
            'Augmented Fifth': 'Minor Sixth',
            'Diminished Seventh': 'Major Sixth',
            'Augmented Sixth': 'Minor Seventh'
        }
        
        nice_name = intv.niceName
        return enharmonics.get(nice_name)
    
    async def analyze_chord_quality(
        self,
        chord_obj: chord.Chord,
        include_extensions: bool = True
    ) -> Dict[str, Any]:
        """
        Analyze chord quality and characteristics
        
        Args:
            chord_obj: Chord to analyze
            include_extensions: Include analysis of extensions
            
        Returns:
            Chord quality analysis
        """
        analysis = {
            'root': chord_obj.root().name if chord_obj.root() else None,
            'bass': chord_obj.bass().name if chord_obj.bass() else None,
            'quality': chord_obj.quality,
            'common_name': chord_obj.commonName,
            'pitch_classes': [p.name for p in chord_obj.pitches],
            'intervals_from_bass': [],
            'chord_type': self._classify_chord_type(chord_obj),
            'inversion': chord_obj.inversion(),
            'is_consonant': chord_obj.isConsonant(),
        }
        
        # Analyze intervals from bass
        bass = chord_obj.bass()
        if bass:
            for p in chord_obj.pitches:
                if p != bass:
                    intv = interval.Interval(bass, p)
                    analysis['intervals_from_bass'].append({
                        'pitch': p.name,
                        'interval': intv.niceName,
                        'semitones': intv.semitones
                    })
        
        # Triadic analysis
        if chord_obj.isTriad():
            analysis['triad_type'] = 'major' if chord_obj.isMajorTriad() else \
                                   'minor' if chord_obj.isMinorTriad() else \
                                   'diminished' if chord_obj.isDiminishedTriad() else \
                                   'augmented'
        
        # Seventh chord analysis
        if chord_obj.isSeventh():
            analysis['seventh_type'] = self._classify_seventh_chord(chord_obj)
        
        # Extended harmony
        if include_extensions:
            analysis['extensions'] = self._analyze_extensions(chord_obj)
            analysis['tensions'] = self._identify_tensions(chord_obj)
        
        # Jazz notation
        analysis['jazz_symbol'] = self._get_jazz_symbol(chord_obj)
        
        return analysis
    
    def _classify_chord_type(self, chord_obj: chord.Chord) -> str:
        """Classify general chord type"""
        if chord_obj.isTriad():
            return "triad"
        elif chord_obj.isSeventh():
            return "seventh"
        elif len(chord_obj.pitches) == 2:
            return "dyad"
        elif len(chord_obj.pitches) > 4:
            return "extended"
        else:
            return "other"
    
    def _classify_seventh_chord(self, chord_obj: chord.Chord) -> str:
        """Classify type of seventh chord"""
        if chord_obj.isDominantSeventh():
            return "dominant7"
        elif chord_obj.isMajorSeventh():
            return "major7"
        elif chord_obj.isMinorSeventh():
            return "minor7"
        elif chord_obj.isDiminishedSeventh():
            return "diminished7"
        elif chord_obj.isHalfDiminishedSeventh():
            return "half-diminished7"
        else:
            return "other7"
    
    def _analyze_extensions(self, chord_obj: chord.Chord) -> List[str]:
        """Identify chord extensions beyond the 7th"""
        extensions = []
        if not chord_obj.root():
            return extensions
        
        root = chord_obj.root()
        
        for p in chord_obj.pitches:
            intv = interval.Interval(root, p)
            semitones = intv.semitones % 12
            
            # Common extensions
            if semitones == 2:
                extensions.append("9th")
            elif semitones == 5:
                extensions.append("11th")
            elif semitones == 9:
                extensions.append("13th")
            elif semitones == 1:
                extensions.append("b9")
            elif semitones == 3:
                extensions.append("#9")
            elif semitones == 6:
                extensions.append("#11")
            elif semitones == 8:
                extensions.append("b13")
        
        return list(set(extensions))
    
    def _identify_tensions(self, chord_obj: chord.Chord) -> List[str]:
        """Identify specific chord tensions"""
        tensions = []
        
        # Look for specific interval patterns
        intervals = []
        pitches = sorted(chord_obj.pitches, key=lambda p: p.ps)
        
        for i in range(len(pitches)):
            for j in range(i + 1, len(pitches)):
                intv = interval.Interval(pitches[i], pitches[j])
                intervals.append(intv.semitones % 12)
        
        # Identify tensions based on intervals
        if 6 in intervals:  # Tritone
            tensions.append("tritone")
        if 1 in intervals:  # Minor second
            tensions.append("minor_second")
        if 11 in intervals:  # Major seventh
            tensions.append("major_seventh")
        
        return tensions
    
    def _get_jazz_symbol(self, chord_obj: chord.Chord) -> str:
        """Generate jazz chord symbol"""
        if not chord_obj.root():
            return "N.C."  # No chord
        
        root = chord_obj.root().name
        
        # Basic triads
        if chord_obj.isMajorTriad():
            symbol = root
        elif chord_obj.isMinorTriad():
            symbol = root + "m"
        elif chord_obj.isDiminishedTriad():
            symbol = root + "°"
        elif chord_obj.isAugmentedTriad():
            symbol = root + "+"
        
        # Seventh chords
        elif chord_obj.isDominantSeventh():
            symbol = root + "7"
        elif chord_obj.isMajorSeventh():
            symbol = root + "maj7"
        elif chord_obj.isMinorSeventh():
            symbol = root + "m7"
        elif chord_obj.isDiminishedSeventh():
            symbol = root + "°7"
        elif chord_obj.isHalfDiminishedSeventh():
            symbol = root + "ø7"
        
        else:
            # Try to construct from intervals
            symbol = root
            if chord_obj.isMinorTriad():
                symbol += "m"
            
            # Add extensions
            extensions = self._analyze_extensions(chord_obj)
            if extensions:
                # Find highest extension
                ext_numbers = []
                for ext in extensions:
                    if ext[0].isdigit():
                        ext_numbers.append(int(ext.rstrip('th')))
                
                if ext_numbers:
                    symbol += str(max(ext_numbers))
        
        # Add bass note if different from root
        if chord_obj.bass() != chord_obj.root():
            symbol += "/" + chord_obj.bass().name
        
        return symbol
    
    async def _detect_cadences(
        self,
        score: stream.Score,
        key_context: key.Key
    ) -> List[Dict[str, Any]]:
        """Detect cadences in the score"""
        cadences = []
        
        # Get all chords
        chords = list(score.flatten().getElementsByClass(chord.Chord))
        
        for i in range(1, len(chords)):
            prev_chord = chords[i-1]
            curr_chord = chords[i]
            
            # Try to get Roman numerals
            try:
                prev_rn = roman.romanNumeralFromChord(prev_chord, key_context)
                curr_rn = roman.romanNumeralFromChord(curr_chord, key_context)
                
                cadence_type = self._identify_cadence_type(
                    prev_rn.romanNumeral,
                    curr_rn.romanNumeral
                )
                
                if cadence_type:
                    cadences.append({
                        'type': cadence_type,
                        'measure': curr_chord.measureNumber,
                        'progression': f"{prev_rn.romanNumeral} - {curr_rn.romanNumeral}",
                        'chords': [str(prev_chord), str(curr_chord)]
                    })
            except:
                continue
        
        return cadences
    
    def _identify_cadence_type(self, prev_rn: str, curr_rn: str) -> Optional[str]:
        """Identify cadence type from Roman numerals"""
        # Normalize Roman numerals
        prev = prev_rn.upper().replace('7', '')
        curr = curr_rn.upper().replace('7', '')
        
        # Perfect authentic cadence
        if prev == 'V' and curr == 'I':
            return "perfect_authentic"
        
        # Imperfect authentic cadence
        elif prev in ['V', 'VII'] and curr == 'I':
            return "imperfect_authentic"
        
        # Plagal cadence
        elif prev == 'IV' and curr == 'I':
            return "plagal"
        
        # Deceptive cadence
        elif prev == 'V' and curr == 'VI':
            return "deceptive"
        
        # Half cadence
        elif curr == 'V':
            return "half"
        
        return None
    
    async def find_parallel_keys(
        self,
        reference_key: Union[key.Key, str]
    ) -> Dict[str, Any]:
        """Find parallel and related keys"""
        if isinstance(reference_key, str):
            reference_key = key.Key(reference_key)
        
        return {
            'reference': str(reference_key),
            'parallel': str(reference_key.parallel),
            'relative': str(reference_key.relative),
            'dominant': str(reference_key.getDominant()),
            'subdominant': str(reference_key.getSubdominant()),
            'enharmonic': self._find_enharmonic_key(reference_key),
            'closely_related': self._find_closely_related_keys(reference_key)
        }
    
    def _find_enharmonic_key(self, k: key.Key) -> Optional[str]:
        """Find enharmonic equivalent of key"""
        tonic = k.tonic
        
        # Get enharmonic pitch
        if tonic.name in ['C#', 'D#', 'F#', 'G#', 'A#']:
            enharm_tonic = tonic.getEnharmonic()
            return key.Key(enharm_tonic.name, k.mode).name
        elif tonic.name in ['Db', 'Eb', 'Gb', 'Ab', 'Bb']:
            enharm_tonic = tonic.getEnharmonic()
            return key.Key(enharm_tonic.name, k.mode).name
        
        return None
    
    def _find_closely_related_keys(self, k: key.Key) -> List[str]:
        """Find all closely related keys (differ by one accidental)"""
        related = []
        
        # Keys that differ by one sharp/flat
        current_sharps = k.sharps
        
        # One more sharp/flat
        for sharps in [current_sharps - 1, current_sharps + 1]:
            try:
                # Major key with this many sharps
                major_key = key.KeySignature(sharps).asKey('major')
                related.append(str(major_key))
                
                # Minor key with this many sharps
                minor_key = key.KeySignature(sharps).asKey('minor')
                related.append(str(minor_key))
            except:
                continue
        
        # Add relative, parallel, dominant, subdominant
        for related_key in [k.relative, k.parallel, k.getDominant(), k.getSubdominant()]:
            if str(related_key) not in related:
                related.append(str(related_key))
        
        return related