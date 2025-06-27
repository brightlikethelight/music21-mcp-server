"""
Advanced rhythm analysis module with tempo detection, meter analysis, and pattern recognition
"""
from typing import Dict, List, Any, Optional, Union, Tuple, Set
from music21 import (
    stream, note, chord, tempo, meter, rhythm, duration,
    tie, articulations, expressions, dynamics
)
import numpy as np
from collections import Counter, defaultdict
import logging
from dataclasses import dataclass
from enum import Enum
from scipy import signal
from scipy.stats import entropy
import asyncio

logger = logging.getLogger(__name__)


class RhythmicComplexity(Enum):
    """Levels of rhythmic complexity"""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    VERY_COMPLEX = "very_complex"


class BeatStrength(Enum):
    """Beat strength classifications"""
    STRONG = "strong"
    MEDIUM = "medium"
    WEAK = "weak"
    SYNCOPATED = "syncopated"


@dataclass
class TempoAnalysis:
    """Comprehensive tempo analysis result"""
    primary_tempo: float
    tempo_markings: List[Dict[str, Any]]
    tempo_changes: List[Dict[str, Any]]
    tempo_stability: float
    suggested_tempo_range: Tuple[float, float]
    tempo_character: str
    rubato_likelihood: float
    average_tempo: float
    tempo_variance: float


@dataclass
class MeterAnalysis:
    """Meter and time signature analysis result"""
    primary_meter: meter.TimeSignature
    meter_changes: List[Dict[str, Any]]
    metric_complexity: str
    beat_hierarchy: Dict[str, Any]
    is_compound: bool
    is_asymmetric: bool
    is_mixed_meter: bool
    meter_stability: float
    hypermeter: Optional[Dict[str, Any]]


@dataclass
class RhythmicPattern:
    """Identified rhythmic pattern"""
    pattern: List[float]  # Duration values
    occurrences: int
    locations: List[int]  # Measure numbers
    confidence: float
    pattern_type: str
    is_ostinato: bool


@dataclass
class RhythmAnalysisResult:
    """Complete rhythm analysis result"""
    tempo: TempoAnalysis
    meter: MeterAnalysis
    patterns: List[RhythmicPattern]
    complexity: RhythmicComplexity
    syncopation_level: float
    groove_analysis: Dict[str, Any]
    polyrhythms: List[Dict[str, Any]]
    rhythm_histogram: Dict[str, int]
    beat_strength_profile: List[Dict[str, Any]]


class RhythmAnalyzer:
    """Comprehensive rhythm analysis engine"""
    
    # Tempo character mappings
    TEMPO_CHARACTERS = {
        (0, 40): "Grave",
        (40, 60): "Largo",
        (60, 66): "Larghetto",
        (66, 76): "Adagio",
        (76, 108): "Andante",
        (108, 120): "Moderato",
        (120, 168): "Allegro",
        (168, 200): "Presto",
        (200, 300): "Prestissimo"
    }
    
    # Common rhythmic patterns
    PATTERN_LIBRARY = {
        'march': [1.0, 1.0, 1.0, 1.0],
        'waltz': [1.0, 0.5, 0.5],
        'swing': [0.67, 0.33, 0.67, 0.33],
        'tresillo': [0.75, 0.75, 0.5],
        'cinquillo': [0.5, 0.25, 0.25, 0.5, 0.5],
        'habanera': [0.75, 0.25, 0.5, 0.5],
        'clave_3_2': [0.5, 0.5, 0.5, 0.75, 0.75],
        'clave_2_3': [0.75, 0.75, 0.5, 0.5, 0.5],
        'samba': [0.75, 0.25, 0.5, 0.5],
        'bossa': [0.75, 0.25, 0.75, 0.25]
    }
    
    def __init__(self):
        self.cache = {}
    
    async def analyze_rhythm(
        self,
        score: stream.Score,
        include_patterns: bool = True,
        pattern_min_length: int = 2,
        pattern_min_occurrences: int = 3
    ) -> RhythmAnalysisResult:
        """
        Perform comprehensive rhythm analysis
        
        Args:
            score: The score to analyze
            include_patterns: Whether to search for rhythmic patterns
            pattern_min_length: Minimum pattern length to consider
            pattern_min_occurrences: Minimum pattern occurrences
            
        Returns:
            Complete rhythm analysis
        """
        # Analyze tempo
        tempo_analysis = await self._analyze_tempo(score)
        
        # Analyze meter
        meter_analysis = await self._analyze_meter(score)
        
        # Extract rhythm patterns
        patterns = []
        if include_patterns:
            patterns = await self._extract_rhythmic_patterns(
                score, pattern_min_length, pattern_min_occurrences
            )
        
        # Calculate complexity
        complexity = await self._calculate_rhythmic_complexity(score)
        
        # Analyze syncopation
        syncopation_level = await self._analyze_syncopation(score, meter_analysis)
        
        # Groove analysis
        groove_analysis = await self._analyze_groove(score)
        
        # Detect polyrhythms
        polyrhythms = await self._detect_polyrhythms(score)
        
        # Create rhythm histogram
        rhythm_histogram = await self._create_rhythm_histogram(score)
        
        # Analyze beat strength
        beat_strength_profile = await self._analyze_beat_strength(score, meter_analysis)
        
        return RhythmAnalysisResult(
            tempo=tempo_analysis,
            meter=meter_analysis,
            patterns=patterns,
            complexity=complexity,
            syncopation_level=syncopation_level,
            groove_analysis=groove_analysis,
            polyrhythms=polyrhythms,
            rhythm_histogram=rhythm_histogram,
            beat_strength_profile=beat_strength_profile
        )
    
    async def _analyze_tempo(self, score: stream.Score) -> TempoAnalysis:
        """Analyze tempo characteristics"""
        # Extract tempo markings
        tempo_markings = []
        for element in score.flatten():
            if isinstance(element, tempo.TempoIndication):
                tempo_markings.append({
                    'measure': element.measureNumber,
                    'offset': float(element.offset),
                    'tempo': element.number if hasattr(element, 'number') else None,
                    'text': element.text if hasattr(element, 'text') else str(element),
                    'type': type(element).__name__
                })
        
        # Detect tempo from note durations if no markings
        if not tempo_markings:
            detected_tempo = await self._detect_tempo_from_durations(score)
            if detected_tempo:
                tempo_markings.append({
                    'measure': 1,
                    'offset': 0.0,
                    'tempo': detected_tempo,
                    'text': f'Detected: {detected_tempo} BPM',
                    'type': 'Detected'
                })
        
        # Analyze tempo changes
        tempo_changes = []
        for i in range(1, len(tempo_markings)):
            prev = tempo_markings[i-1]
            curr = tempo_markings[i]
            if prev['tempo'] and curr['tempo']:
                change_ratio = curr['tempo'] / prev['tempo']
                tempo_changes.append({
                    'measure': curr['measure'],
                    'from_tempo': prev['tempo'],
                    'to_tempo': curr['tempo'],
                    'ratio': change_ratio,
                    'type': self._classify_tempo_change(change_ratio)
                })
        
        # Calculate tempo statistics
        tempos = [t['tempo'] for t in tempo_markings if t['tempo']]
        
        if tempos:
            primary_tempo = tempos[0]  # First marked tempo
            average_tempo = np.mean(tempos)
            tempo_variance = np.var(tempos) if len(tempos) > 1 else 0.0
            tempo_stability = 1.0 - (tempo_variance / (average_tempo ** 2)) if average_tempo > 0 else 1.0
            
            # Suggest tempo range based on variance
            if tempo_variance > 0:
                std_dev = np.sqrt(tempo_variance)
                suggested_range = (
                    max(30, average_tempo - std_dev),
                    min(300, average_tempo + std_dev)
                )
            else:
                suggested_range = (primary_tempo * 0.9, primary_tempo * 1.1)
            
            # Detect rubato likelihood
            rubato_likelihood = await self._detect_rubato_likelihood(score, tempo_stability)
            
            # Get tempo character
            tempo_character = self._get_tempo_character(primary_tempo)
        else:
            # No tempo information available
            primary_tempo = 120.0  # Default
            average_tempo = 120.0
            tempo_variance = 0.0
            tempo_stability = 1.0
            suggested_range = (108.0, 132.0)
            rubato_likelihood = 0.0
            tempo_character = "Moderato"
        
        return TempoAnalysis(
            primary_tempo=primary_tempo,
            tempo_markings=tempo_markings,
            tempo_changes=tempo_changes,
            tempo_stability=tempo_stability,
            suggested_tempo_range=suggested_range,
            tempo_character=tempo_character,
            rubato_likelihood=rubato_likelihood,
            average_tempo=average_tempo,
            tempo_variance=tempo_variance
        )
    
    async def _detect_tempo_from_durations(self, score: stream.Score) -> Optional[float]:
        """Detect tempo from note durations using autocorrelation"""
        # Get all note onsets
        onsets = []
        for n in score.flatten().notes:
            if isinstance(n, (note.Note, chord.Chord)):
                onsets.append(float(n.offset))
        
        if len(onsets) < 10:
            return None
        
        # Calculate inter-onset intervals
        onsets = sorted(onsets)
        iois = np.diff(onsets)
        
        if len(iois) < 5:
            return None
        
        # Find most common IOI (likely the beat)
        # Quantize to common beat divisions
        quantized_iois = np.round(iois * 4) / 4  # Quantize to 16th notes
        ioi_counts = Counter(quantized_iois)
        
        # Get most common non-zero IOI
        common_iois = [ioi for ioi, count in ioi_counts.most_common() if ioi > 0.1]
        
        if common_iois:
            beat_duration = common_iois[0]
            # Convert to BPM (assuming quarter note beat)
            tempo = 60.0 / beat_duration
            
            # Sanity check
            if 30 <= tempo <= 300:
                return tempo
        
        return None
    
    def _classify_tempo_change(self, ratio: float) -> str:
        """Classify type of tempo change"""
        if 0.98 <= ratio <= 1.02:
            return "stable"
        elif 1.9 <= ratio <= 2.1:
            return "doppio movimento"
        elif 0.45 <= ratio <= 0.55:
            return "half tempo"
        elif ratio > 1.2:
            return "accelerando"
        elif ratio < 0.8:
            return "ritardando"
        else:
            return "gradual"
    
    async def _detect_rubato_likelihood(self, score: stream.Score, tempo_stability: float) -> float:
        """Detect likelihood of rubato/expressive timing"""
        # Factors that suggest rubato
        factors = []
        
        # Low tempo stability suggests rubato
        factors.append(1.0 - tempo_stability)
        
        # Check for expressive markings
        expressive_marks = 0
        total_marks = 0
        
        for element in score.flatten():
            if isinstance(element, expressions.Expression):
                total_marks += 1
                if any(term in str(element).lower() for term in 
                      ['rubato', 'espressivo', 'freely', 'ad lib', 'piacere']):
                    expressive_marks += 1
        
        if total_marks > 0:
            factors.append(expressive_marks / total_marks)
        
        # Check for fermatas
        fermatas = score.flatten().getElementsByClass(expressions.Fermata)
        if fermatas:
            factors.append(min(1.0, len(fermatas) / 20.0))  # Normalize
        
        # Check for tempo variations
        tempo_indications = score.flatten().getElementsByClass(tempo.TempoIndication)
        if len(tempo_indications) > 1:
            factors.append(min(1.0, len(tempo_indications) / 10.0))
        
        # Average all factors
        return np.mean(factors) if factors else 0.0
    
    def _get_tempo_character(self, bpm: float) -> str:
        """Get tempo character description"""
        for (low, high), character in self.TEMPO_CHARACTERS.items():
            if low <= bpm < high:
                return character
        return "Unknown"
    
    async def _analyze_meter(self, score: stream.Score) -> MeterAnalysis:
        """Analyze meter and time signatures"""
        # Extract time signatures
        time_sigs = []
        for ts in score.flatten().getElementsByClass(meter.TimeSignature):
            time_sigs.append({
                'measure': ts.measureNumber,
                'offset': float(ts.offset),
                'signature': ts,
                'numerator': ts.numerator,
                'denominator': ts.denominator,
                'beat_count': ts.beatCount,
                'is_compound': ts.numerator % 3 == 0 and ts.numerator > 3
            })
        
        if not time_sigs:
            # Try to infer from measure lengths
            inferred = self._infer_time_signature(score)
            if inferred:
                time_sigs.append({
                    'measure': 1,
                    'offset': 0.0,
                    'signature': inferred,
                    'numerator': inferred.numerator,
                    'denominator': inferred.denominator,
                    'beat_count': inferred.beatCount,
                    'is_compound': inferred.numerator % 3 == 0 and inferred.numerator > 3
                })
        
        # Determine primary meter
        if time_sigs:
            primary_meter = time_sigs[0]['signature']
        else:
            primary_meter = meter.TimeSignature('4/4')  # Default
        
        # Analyze meter changes
        meter_changes = []
        for i in range(1, len(time_sigs)):
            prev = time_sigs[i-1]
            curr = time_sigs[i]
            
            meter_changes.append({
                'measure': curr['measure'],
                'from': f"{prev['numerator']}/{prev['denominator']}",
                'to': f"{curr['numerator']}/{curr['denominator']}",
                'type': self._classify_meter_change(prev['signature'], curr['signature'])
            })
        
        # Analyze metric properties
        is_compound = primary_meter.numerator % 3 == 0 and primary_meter.numerator > 3
        is_asymmetric = primary_meter.numerator in [5, 7, 11, 13]
        is_mixed_meter = len(set(f"{ts['numerator']}/{ts['denominator']}" for ts in time_sigs)) > 1
        
        # Calculate meter stability
        if len(time_sigs) <= 1:
            meter_stability = 1.0
        else:
            # More changes = less stability
            changes_per_measure = len(meter_changes) / len(score.getElementsByClass(stream.Measure))
            meter_stability = max(0.0, 1.0 - changes_per_measure)
        
        # Analyze beat hierarchy
        beat_hierarchy = self._analyze_beat_hierarchy(primary_meter)
        
        # Detect hypermeter
        hypermeter = await self._detect_hypermeter(score, primary_meter)
        
        # Classify metric complexity
        if is_mixed_meter:
            metric_complexity = "mixed"
        elif is_asymmetric:
            metric_complexity = "asymmetric"
        elif is_compound:
            metric_complexity = "compound"
        else:
            metric_complexity = "simple"
        
        return MeterAnalysis(
            primary_meter=primary_meter,
            meter_changes=meter_changes,
            metric_complexity=metric_complexity,
            beat_hierarchy=beat_hierarchy,
            is_compound=is_compound,
            is_asymmetric=is_asymmetric,
            is_mixed_meter=is_mixed_meter,
            meter_stability=meter_stability,
            hypermeter=hypermeter
        )
    
    def _infer_time_signature(self, score: stream.Score) -> Optional[meter.TimeSignature]:
        """Infer time signature from measure content"""
        measures = score.getElementsByClass(stream.Measure)
        if not measures:
            return None
        
        # Sample first few complete measures
        durations = []
        for m in measures[:10]:
            if m.duration.quarterLength > 0:
                durations.append(m.duration.quarterLength)
        
        if not durations:
            return None
        
        # Most common duration
        common_duration = Counter(durations).most_common(1)[0][0]
        
        # Try to match to common time signatures
        common_sigs = {
            4.0: '4/4',
            3.0: '3/4',
            2.0: '2/4',
            6.0: '6/8',
            1.0: '1/4',
            5.0: '5/4',
            7.0: '7/8'
        }
        
        if common_duration in common_sigs:
            return meter.TimeSignature(common_sigs[common_duration])
        
        # Try to construct from duration
        if common_duration == int(common_duration):
            return meter.TimeSignature(f'{int(common_duration)}/4')
        
        return None
    
    def _classify_meter_change(self, prev: meter.TimeSignature, curr: meter.TimeSignature) -> str:
        """Classify type of meter change"""
        prev_beats = prev.numerator / prev.denominator
        curr_beats = curr.numerator / curr.denominator
        
        if prev_beats == curr_beats:
            return "equivalent"
        elif curr.numerator == prev.numerator:
            return "subdivision_change"
        elif curr.denominator == prev.denominator:
            return "beat_count_change"
        elif (prev.numerator % 3 == 0) != (curr.numerator % 3 == 0):
            return "simple_compound_shift"
        else:
            return "metric_modulation"
    
    def _analyze_beat_hierarchy(self, ts: meter.TimeSignature) -> Dict[str, Any]:
        """Analyze the beat hierarchy of a time signature"""
        hierarchy = {
            'primary_beat': 1.0,
            'beat_pattern': [],
            'accent_pattern': []
        }
        
        # Get beat pattern
        beat_pattern = ts.beatSequence
        if beat_pattern:
            hierarchy['beat_pattern'] = [
                {'beat': i+1, 'duration': beat.duration.quarterLength, 'weight': beat.weight}
                for i, beat in enumerate(beat_pattern)
            ]
        
        # Determine accent pattern based on meter
        if ts.numerator == 4:
            hierarchy['accent_pattern'] = ['strong', 'weak', 'medium', 'weak']
        elif ts.numerator == 3:
            hierarchy['accent_pattern'] = ['strong', 'weak', 'weak']
        elif ts.numerator == 6 and ts.denominator == 8:
            hierarchy['accent_pattern'] = ['strong', 'weak', 'weak', 'medium', 'weak', 'weak']
        elif ts.numerator == 5:
            hierarchy['accent_pattern'] = ['strong', 'weak', 'medium', 'weak', 'weak']
        elif ts.numerator == 7:
            hierarchy['accent_pattern'] = ['strong', 'weak', 'weak', 'medium', 'weak', 'weak', 'weak']
        else:
            # Generic pattern
            pattern = ['strong']
            for i in range(1, ts.numerator):
                if i == ts.numerator // 2:
                    pattern.append('medium')
                else:
                    pattern.append('weak')
            hierarchy['accent_pattern'] = pattern
        
        return hierarchy
    
    async def _detect_hypermeter(self, score: stream.Score, primary_meter: meter.TimeSignature) -> Optional[Dict[str, Any]]:
        """Detect hypermetric patterns"""
        measures = list(score.getElementsByClass(stream.Measure))
        if len(measures) < 8:
            return None
        
        # Look for patterns in measure groupings
        # Check for phrase markings
        phrase_lengths = []
        current_phrase_start = 0
        
        for i, measure in enumerate(measures):
            # Check for phrase endings (fermatas, long notes, rests)
            has_phrase_end = False
            
            for element in measure:
                if isinstance(element, expressions.Fermata):
                    has_phrase_end = True
                    break
                elif isinstance(element, note.Rest) and element.duration.quarterLength >= measure.barDuration.quarterLength * 0.5:
                    has_phrase_end = True
                    break
            
            if has_phrase_end and i > current_phrase_start:
                phrase_lengths.append(i - current_phrase_start + 1)
                current_phrase_start = i + 1
        
        if not phrase_lengths:
            # Try to detect through harmonic rhythm or other means
            # Simplified: look for regular groupings
            if len(measures) >= 16:
                if len(measures) % 4 == 0:
                    return {
                        'level': 4,
                        'pattern': 'four-bar phrases',
                        'confidence': 0.7
                    }
                elif len(measures) % 8 == 0:
                    return {
                        'level': 8,
                        'pattern': 'eight-bar phrases',
                        'confidence': 0.6
                    }
        
        # Analyze phrase lengths
        if phrase_lengths:
            most_common = Counter(phrase_lengths).most_common(1)[0]
            if most_common[1] >= 2:  # At least 2 occurrences
                return {
                    'level': most_common[0],
                    'pattern': f'{most_common[0]}-bar phrases',
                    'confidence': most_common[1] / len(phrase_lengths)
                }
        
        return None
    
    async def _extract_rhythmic_patterns(
        self,
        score: stream.Score,
        min_length: int,
        min_occurrences: int
    ) -> List[RhythmicPattern]:
        """Extract recurring rhythmic patterns"""
        patterns = []
        
        # Extract rhythm sequences from each part
        for part in score.parts:
            rhythm_sequence = []
            current_measure_rhythms = []
            current_measure_num = None
            
            for element in part.flatten():
                if isinstance(element, (note.Note, note.Rest, chord.Chord)):
                    if element.measureNumber != current_measure_num:
                        if current_measure_rhythms:
                            rhythm_sequence.append(current_measure_rhythms)
                        current_measure_rhythms = []
                        current_measure_num = element.measureNumber
                    
                    # Add duration to current measure
                    current_measure_rhythms.append(float(element.duration.quarterLength))
            
            # Add last measure
            if current_measure_rhythms:
                rhythm_sequence.append(current_measure_rhythms)
            
            # Find patterns in rhythm sequence
            part_patterns = self._find_rhythm_patterns(
                rhythm_sequence, min_length, min_occurrences
            )
            
            patterns.extend(part_patterns)
        
        # Deduplicate and sort by frequency
        unique_patterns = self._deduplicate_patterns(patterns)
        unique_patterns.sort(key=lambda p: p.occurrences, reverse=True)
        
        return unique_patterns[:10]  # Return top 10 patterns
    
    def _find_rhythm_patterns(
        self,
        rhythm_sequence: List[List[float]],
        min_length: int,
        min_occurrences: int
    ) -> List[RhythmicPattern]:
        """Find repeating patterns in rhythm sequence"""
        patterns = []
        
        # Convert to flat sequence for pattern matching
        flat_sequence = []
        measure_starts = [0]
        
        for measure_rhythms in rhythm_sequence:
            flat_sequence.extend(measure_rhythms)
            measure_starts.append(len(flat_sequence))
        
        # Try different pattern lengths
        for length in range(min_length, min(len(flat_sequence) // 2, 16)):
            # Slide window through sequence
            for start in range(len(flat_sequence) - length + 1):
                pattern = flat_sequence[start:start + length]
                
                # Count occurrences
                occurrences = []
                for i in range(len(flat_sequence) - length + 1):
                    if self._rhythms_match(flat_sequence[i:i + length], pattern):
                        # Find which measure this occurrence is in
                        measure_num = next(
                            j for j, ms in enumerate(measure_starts[:-1])
                            if ms <= i < measure_starts[j + 1]
                        ) + 1
                        occurrences.append(measure_num)
                
                if len(occurrences) >= min_occurrences:
                    # Classify pattern type
                    pattern_type = self._classify_rhythm_pattern(pattern)
                    
                    # Check if it's an ostinato (continuous repetition)
                    is_ostinato = self._is_ostinato(occurrences)
                    
                    patterns.append(RhythmicPattern(
                        pattern=pattern,
                        occurrences=len(occurrences),
                        locations=list(set(occurrences)),
                        confidence=len(occurrences) / (len(flat_sequence) / length),
                        pattern_type=pattern_type,
                        is_ostinato=is_ostinato
                    ))
        
        return patterns
    
    def _rhythms_match(self, seq1: List[float], seq2: List[float], tolerance: float = 0.01) -> bool:
        """Check if two rhythm sequences match"""
        if len(seq1) != len(seq2):
            return False
        
        for r1, r2 in zip(seq1, seq2):
            if abs(r1 - r2) > tolerance:
                return False
        
        return True
    
    def _classify_rhythm_pattern(self, pattern: List[float]) -> str:
        """Classify a rhythm pattern"""
        # Check against known patterns
        for name, known_pattern in self.PATTERN_LIBRARY.items():
            if self._rhythms_match(pattern[:len(known_pattern)], known_pattern, tolerance=0.1):
                return name
        
        # Classify by characteristics
        total_duration = sum(pattern)
        unique_durations = len(set(pattern))
        
        if unique_durations == 1:
            return "uniform"
        elif unique_durations == 2:
            if max(pattern) / min(pattern) >= 2:
                return "long-short"
            else:
                return "binary"
        elif any(d < 0.25 for d in pattern):
            return "rapid"
        elif any(d >= 2.0 for d in pattern):
            return "sustained"
        else:
            return "mixed"
    
    def _is_ostinato(self, occurrences: List[int]) -> bool:
        """Check if pattern occurrences suggest an ostinato"""
        if len(occurrences) < 4:
            return False
        
        # Check for consecutive occurrences
        sorted_occ = sorted(set(occurrences))
        consecutive_count = 1
        max_consecutive = 1
        
        for i in range(1, len(sorted_occ)):
            if sorted_occ[i] == sorted_occ[i-1] + 1:
                consecutive_count += 1
                max_consecutive = max(max_consecutive, consecutive_count)
            else:
                consecutive_count = 1
        
        return max_consecutive >= 4
    
    def _deduplicate_patterns(self, patterns: List[RhythmicPattern]) -> List[RhythmicPattern]:
        """Remove duplicate patterns"""
        unique = []
        seen = set()
        
        for pattern in patterns:
            # Create pattern signature
            sig = tuple(round(d * 100) for d in pattern.pattern)
            
            if sig not in seen:
                seen.add(sig)
                unique.append(pattern)
            else:
                # Merge with existing pattern
                for up in unique:
                    up_sig = tuple(round(d * 100) for d in up.pattern)
                    if up_sig == sig:
                        up.occurrences += pattern.occurrences
                        up.locations.extend(pattern.locations)
                        up.locations = list(set(up.locations))
                        break
        
        return unique
    
    async def _calculate_rhythmic_complexity(self, score: stream.Score) -> RhythmicComplexity:
        """Calculate overall rhythmic complexity"""
        factors = []
        
        # Factor 1: Duration variety
        durations = []
        for n in score.flatten().notesAndRests:
            durations.append(n.duration.quarterLength)
        
        if durations:
            unique_durations = len(set(durations))
            duration_variety = unique_durations / len(durations)
            factors.append(duration_variety)
        
        # Factor 2: Rhythmic entropy
        if durations:
            duration_counts = Counter(durations)
            total = sum(duration_counts.values())
            probabilities = [count/total for count in duration_counts.values()]
            rhythm_entropy = entropy(probabilities)
            normalized_entropy = rhythm_entropy / np.log(len(duration_counts)) if len(duration_counts) > 1 else 0
            factors.append(normalized_entropy)
        
        # Factor 3: Syncopation presence
        syncopation = await self._quick_syncopation_check(score)
        factors.append(syncopation)
        
        # Factor 4: Tuplet usage
        tuplet_ratio = self._calculate_tuplet_ratio(score)
        factors.append(tuplet_ratio)
        
        # Factor 5: Tempo changes
        tempo_changes = len(score.flatten().getElementsByClass(tempo.TempoIndication))
        tempo_factor = min(1.0, tempo_changes / 5.0)
        factors.append(tempo_factor)
        
        # Calculate overall complexity
        if factors:
            complexity_score = np.mean(factors)
            
            if complexity_score < 0.25:
                return RhythmicComplexity.SIMPLE
            elif complexity_score < 0.5:
                return RhythmicComplexity.MODERATE
            elif complexity_score < 0.75:
                return RhythmicComplexity.COMPLEX
            else:
                return RhythmicComplexity.VERY_COMPLEX
        
        return RhythmicComplexity.SIMPLE
    
    async def _quick_syncopation_check(self, score: stream.Score) -> float:
        """Quick check for syncopation presence"""
        syncopated_notes = 0
        total_notes = 0
        
        for n in score.flatten().notes:
            if isinstance(n, (note.Note, chord.Chord)):
                total_notes += 1
                
                # Check if note starts on weak beat
                beat_strength = n.beatStrength
                if beat_strength is not None and beat_strength < 0.5:
                    # Check if it's accented or longer than surrounding notes
                    if any(isinstance(a, articulations.Accent) for a in n.articulations):
                        syncopated_notes += 1
                    elif n.duration.quarterLength > 1.0:
                        syncopated_notes += 1
        
        return syncopated_notes / total_notes if total_notes > 0 else 0.0
    
    def _calculate_tuplet_ratio(self, score: stream.Score) -> float:
        """Calculate ratio of tuplet notes to total notes"""
        tuplet_notes = 0
        total_notes = 0
        
        for n in score.flatten().notes:
            if isinstance(n, (note.Note, chord.Chord)):
                total_notes += 1
                if n.duration.tuplets:
                    tuplet_notes += 1
        
        return tuplet_notes / total_notes if total_notes > 0 else 0.0
    
    async def _analyze_syncopation(
        self,
        score: stream.Score,
        meter_analysis: MeterAnalysis
    ) -> float:
        """Detailed syncopation analysis"""
        syncopation_events = []
        
        for part in score.parts:
            for n in part.flatten().notes:
                if isinstance(n, (note.Note, chord.Chord)):
                    # Check various syncopation types
                    
                    # 1. Off-beat attacks
                    if n.beat and n.beat % 1 != 0:  # Not on a beat
                        syncopation_events.append({
                            'type': 'off_beat',
                            'strength': 1.0 - (n.beatStrength or 0.5),
                            'duration': n.duration.quarterLength
                        })
                    
                    # 2. Tied notes across strong beats
                    if n.tie and n.tie.type == 'start':
                        next_note = n.next('Note')
                        if next_note and next_note.beat == 1:  # Ties across barline
                            syncopation_events.append({
                                'type': 'tied_across_bar',
                                'strength': 1.0,
                                'duration': n.duration.quarterLength + next_note.duration.quarterLength
                            })
                    
                    # 3. Long notes on weak beats
                    if n.beatStrength and n.beatStrength < 0.5 and n.duration.quarterLength > 1.0:
                        syncopation_events.append({
                            'type': 'long_weak_beat',
                            'strength': 0.7,
                            'duration': n.duration.quarterLength
                        })
        
        # Calculate syncopation level
        if not syncopation_events:
            return 0.0
        
        # Weight by strength and normalize by total duration
        total_duration = score.duration.quarterLength
        weighted_syncopation = sum(e['strength'] * e['duration'] for e in syncopation_events)
        
        return min(1.0, weighted_syncopation / total_duration)
    
    async def _analyze_groove(self, score: stream.Score) -> Dict[str, Any]:
        """Analyze groove and swing characteristics"""
        groove_analysis = {
            'swing_ratio': 0.0,
            'groove_consistency': 0.0,
            'microtiming_variance': 0.0,
            'groove_type': 'straight',
            'drive': 0.0,
            'pocket': 0.0
        }
        
        # Analyze swing ratio (only for binary meters)
        primary_ts = score.getTimeSignatures()[0] if score.getTimeSignatures() else None
        if primary_ts and primary_ts.denominator in [4, 8]:
            swing_ratio = await self._calculate_swing_ratio(score)
            groove_analysis['swing_ratio'] = swing_ratio
            
            if swing_ratio > 1.3:
                groove_analysis['groove_type'] = 'swing'
            elif swing_ratio > 1.1:
                groove_analysis['groove_type'] = 'shuffle'
        
        # Analyze groove consistency
        groove_analysis['groove_consistency'] = await self._calculate_groove_consistency(score)
        
        # Analyze drive (forward motion)
        groove_analysis['drive'] = await self._calculate_rhythmic_drive(score)
        
        # Analyze pocket (how "in the groove" the rhythm is)
        groove_analysis['pocket'] = await self._calculate_pocket(score)
        
        return groove_analysis
    
    async def _calculate_swing_ratio(self, score: stream.Score) -> float:
        """Calculate swing ratio for eighth note pairs"""
        eighth_pairs = []
        
        for part in score.parts:
            notes = list(part.flatten().notes)
            
            for i in range(len(notes) - 1):
                n1 = notes[i]
                n2 = notes[i + 1]
                
                # Look for consecutive eighth notes
                if (n1.duration.quarterLength == 0.5 and 
                    n2.duration.quarterLength == 0.5 and
                    n2.offset == n1.offset + 0.5):
                    
                    # In swing, first eighth is longer
                    # This is a simplification - real swing would need performance data
                    eighth_pairs.append(1.0)  # Straight eighths for now
        
        if not eighth_pairs:
            return 1.0
        
        # In real implementation, would analyze actual timing
        # For now, return straight timing
        return 1.0
    
    async def _calculate_groove_consistency(self, score: stream.Score) -> float:
        """Calculate how consistent the groove is"""
        # Analyze beat placement consistency
        beat_placements = []
        
        for n in score.flatten().notes:
            if isinstance(n, (note.Note, chord.Chord)):
                # Get beat position
                beat_pos = n.beat if n.beat else 0
                beat_placements.append(beat_pos % 1)  # Fractional part
        
        if not beat_placements:
            return 0.0
        
        # Group by similar positions
        position_counts = Counter(round(bp * 4) / 4 for bp in beat_placements)
        
        # Consistency is higher when notes fall on regular positions
        total_notes = sum(position_counts.values())
        regularity = sum(count**2 for count in position_counts.values()) / (total_notes**2)
        
        return regularity
    
    async def _calculate_rhythmic_drive(self, score: stream.Score) -> float:
        """Calculate rhythmic drive/forward motion"""
        drive_factors = []
        
        # Factor 1: Note density
        total_duration = score.duration.quarterLength
        note_count = len(score.flatten().notes)
        density = note_count / total_duration if total_duration > 0 else 0
        drive_factors.append(min(1.0, density / 4.0))  # Normalize
        
        # Factor 2: Continuous motion (few long rests)
        rests = score.flatten().getElementsByClass(note.Rest)
        long_rests = [r for r in rests if r.duration.quarterLength >= 1.0]
        rest_factor = 1.0 - (len(long_rests) / (note_count + len(rests))) if (note_count + len(rests)) > 0 else 1.0
        drive_factors.append(rest_factor)
        
        # Factor 3: Accelerating patterns
        tempo_indications = score.flatten().getElementsByClass(tempo.TempoIndication)
        accel_count = sum(1 for t in tempo_indications if 'accel' in str(t).lower())
        if accel_count > 0:
            drive_factors.append(min(1.0, accel_count / 3.0))
        
        return np.mean(drive_factors) if drive_factors else 0.5
    
    async def _calculate_pocket(self, score: stream.Score) -> float:
        """Calculate how well the rhythm sits in the pocket"""
        # This is a simplified metric
        # Real "pocket" would require performance analysis
        
        pocket_factors = []
        
        # Factor 1: Beat strength alignment
        strong_beat_notes = 0
        total_notes = 0
        
        for n in score.flatten().notes:
            if isinstance(n, (note.Note, chord.Chord)):
                total_notes += 1
                if n.beatStrength and n.beatStrength >= 0.5:
                    strong_beat_notes += 1
        
        if total_notes > 0:
            pocket_factors.append(strong_beat_notes / total_notes)
        
        # Factor 2: Rhythmic repetition (creates groove)
        patterns = await self._extract_rhythmic_patterns(score, 2, 2)
        if patterns:
            # More repetitive patterns = better pocket
            repetition_factor = min(1.0, len(patterns) / 5.0)
            pocket_factors.append(repetition_factor)
        
        return np.mean(pocket_factors) if pocket_factors else 0.5
    
    async def _detect_polyrhythms(self, score: stream.Score) -> List[Dict[str, Any]]:
        """Detect polyrhythmic patterns between parts"""
        polyrhythms = []
        
        if len(score.parts) < 2:
            return polyrhythms
        
        # Compare rhythmic patterns between parts
        part_patterns = []
        
        for part in score.parts[:4]:  # Limit to first 4 parts for efficiency
            # Extract characteristic rhythm
            rhythms = []
            for n in part.flatten().notes[:100]:  # Sample first 100 notes
                if isinstance(n, (note.Note, chord.Chord)):
                    rhythms.append(n.duration.quarterLength)
            
            if rhythms:
                part_patterns.append({
                    'part': part,
                    'rhythms': rhythms,
                    'cycle_length': self._find_rhythm_cycle(rhythms)
                })
        
        # Compare cycles between parts
        for i in range(len(part_patterns)):
            for j in range(i + 1, len(part_patterns)):
                cycle1 = part_patterns[i]['cycle_length']
                cycle2 = part_patterns[j]['cycle_length']
                
                if cycle1 and cycle2 and cycle1 != cycle2:
                    # Check if they form a polyrhythm
                    ratio = max(cycle1, cycle2) / min(cycle1, cycle2)
                    
                    # Common polyrhythms
                    common_ratios = {
                        1.5: "3:2 (hemiola)",
                        1.33: "4:3",
                        1.25: "5:4",
                        1.67: "5:3",
                        2.0: "2:1",
                        1.75: "7:4"
                    }
                    
                    for target_ratio, name in common_ratios.items():
                        if abs(ratio - target_ratio) < 0.1:
                            polyrhythms.append({
                                'type': name,
                                'parts': [i, j],
                                'cycle_lengths': [cycle1, cycle2],
                                'confidence': 1.0 - abs(ratio - target_ratio)
                            })
                            break
        
        return polyrhythms
    
    def _find_rhythm_cycle(self, rhythms: List[float]) -> Optional[float]:
        """Find repeating cycle in rhythm sequence"""
        if len(rhythms) < 4:
            return None
        
        # Try different cycle lengths
        for cycle_len in range(2, min(len(rhythms) // 2, 8)):
            total_duration = sum(rhythms[:cycle_len])
            
            # Check if this pattern repeats
            matches = 0
            for start in range(0, len(rhythms) - cycle_len, cycle_len):
                if abs(sum(rhythms[start:start + cycle_len]) - total_duration) < 0.1:
                    matches += 1
            
            if matches >= 2:
                return total_duration
        
        return None
    
    async def _create_rhythm_histogram(self, score: stream.Score) -> Dict[str, int]:
        """Create histogram of rhythm values"""
        histogram = Counter()
        
        for n in score.flatten().notesAndRests:
            # Convert duration to standard note value
            dur = n.duration.quarterLength
            
            # Map to standard durations
            standard_durations = {
                4.0: "whole",
                3.0: "dotted half",
                2.0: "half",
                1.5: "dotted quarter",
                1.0: "quarter",
                0.75: "dotted eighth",
                0.5: "eighth",
                0.375: "dotted sixteenth",
                0.25: "sixteenth",
                0.125: "thirty-second"
            }
            
            # Find closest standard duration
            closest = min(standard_durations.keys(), key=lambda x: abs(x - dur))
            if abs(closest - dur) < 0.05:  # Tolerance
                histogram[standard_durations[closest]] += 1
            else:
                histogram[f"other ({dur})"] += 1
        
        return dict(histogram)
    
    async def _analyze_beat_strength(
        self,
        score: stream.Score,
        meter_analysis: MeterAnalysis
    ) -> List[Dict[str, Any]]:
        """Analyze beat strength profile"""
        beat_strength_profile = []
        
        # Sample measures throughout the piece
        measures = list(score.getElementsByClass(stream.Measure))
        sample_indices = np.linspace(0, len(measures) - 1, min(10, len(measures)), dtype=int)
        
        for idx in sample_indices:
            measure = measures[idx]
            measure_profile = {
                'measure': idx + 1,
                'beats': []
            }
            
            # Analyze each beat in the measure
            for beat_num in range(1, int(meter_analysis.primary_meter.numerator) + 1):
                beat_notes = []
                
                for n in measure.notes:
                    if isinstance(n, (note.Note, chord.Chord)):
                        if int(n.beat) == beat_num:
                            beat_notes.append(n)
                
                # Calculate beat characteristics
                if beat_notes:
                    avg_velocity = np.mean([n.volume.velocity for n in beat_notes if n.volume.velocity])
                    has_accent = any(isinstance(a, articulations.Accent) for n in beat_notes for a in n.articulations)
                    total_duration = sum(n.duration.quarterLength for n in beat_notes)
                    
                    strength = BeatStrength.STRONG if beat_num == 1 else \
                              BeatStrength.MEDIUM if beat_num == (meter_analysis.primary_meter.numerator + 1) // 2 else \
                              BeatStrength.WEAK
                    
                    if has_accent and strength != BeatStrength.STRONG:
                        strength = BeatStrength.SYNCOPATED
                    
                    measure_profile['beats'].append({
                        'beat': beat_num,
                        'strength': strength.value,
                        'velocity': avg_velocity if avg_velocity else 64,
                        'density': len(beat_notes),
                        'duration': total_duration
                    })
            
            beat_strength_profile.append(measure_profile)
        
        return beat_strength_profile
    
    async def analyze_tempo_stability(
        self,
        score: stream.Score,
        window_size: int = 4
    ) -> Dict[str, Any]:
        """Analyze tempo stability over time"""
        # Extract inter-onset intervals
        onset_times = []
        
        for n in score.flatten().notes:
            if isinstance(n, (note.Note, chord.Chord)):
                onset_times.append(float(n.offset))
        
        if len(onset_times) < 10:
            return {
                'stability': 1.0,
                'variance': 0.0,
                'trend': 'stable'
            }
        
        # Calculate IOIs
        onset_times = sorted(onset_times)
        iois = np.diff(onset_times)
        
        # Analyze in windows
        stability_scores = []
        
        for i in range(0, len(iois) - window_size + 1):
            window = iois[i:i + window_size]
            if len(window) > 1:
                # Calculate coefficient of variation
                cv = np.std(window) / np.mean(window) if np.mean(window) > 0 else 0
                stability_scores.append(1.0 - min(cv, 1.0))
        
        # Overall stability
        overall_stability = np.mean(stability_scores) if stability_scores else 1.0
        
        # Detect tempo trend
        if len(iois) > 10:
            # Linear regression on IOIs
            x = np.arange(len(iois))
            slope, _ = np.polyfit(x, iois, 1)
            
            if slope > 0.01:
                trend = "slowing"
            elif slope < -0.01:
                trend = "accelerating"
            else:
                trend = "stable"
        else:
            trend = "stable"
        
        return {
            'stability': overall_stability,
            'variance': np.var(iois) if len(iois) > 1 else 0.0,
            'trend': trend,
            'mean_ioi': np.mean(iois) if iois.size > 0 else 0.0
        }