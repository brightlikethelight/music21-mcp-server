"""
Advanced music theory analysis module with specialized harmonic and structural analysis
"""
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from music21 import chord, expressions, interval, key, note, pitch, roman, stream

logger = logging.getLogger(__name__)


class ChromaticFunction(Enum):
    """Types of chromatic functions"""
    PASSING = "passing"
    NEIGHBOR = "neighbor"
    APPOGIATURA = "appogiatura"
    ESCAPE = "escape"
    SUSPENSION = "suspension"
    ANTICIPATION = "anticipation"
    SECONDARY_LEADING = "secondary_leading"
    MODAL_MIXTURE = "modal_mixture"
    CHROMATIC_MEDIANT = "chromatic_mediant"


class PhraseType(Enum):
    """Musical phrase structures"""
    PERIOD = "period"
    SENTENCE = "sentence"
    HYBRID = "hybrid"
    COMPOUND_PERIOD = "compound_period"
    COMPOUND_SENTENCE = "compound_sentence"
    PHRASE_GROUP = "phrase_group"
    DOUBLE_PERIOD = "double_period"
    ASYMMETRIC = "asymmetric"


@dataclass
class ScaleDegreeAnalysis:
    """Analysis of melodic motion by scale degrees"""
    scale_degree_histogram: Dict[int, int]
    tendency_tone_resolutions: Dict[str, int]
    melodic_contour: List[int]
    step_leap_ratio: float
    chromatic_percentage: float
    most_common_progressions: List[Tuple[str, int]]


@dataclass
class IntervalVectorAnalysis:
    """Interval class vector analysis"""
    interval_vector: List[int]  # 6-element vector
    total_intervals: int
    consonance_ratio: float
    tritone_count: int
    common_intervals: List[Tuple[str, int]]
    z_relation: Optional[str]  # Z-related set if applicable


@dataclass
class ChromaticAnalysis:
    """Analysis of chromatic elements"""
    chromatic_notes: List[Dict[str, Any]]
    chromatic_density: float
    chromatic_functions: Dict[ChromaticFunction, int]
    chromatic_voice_leading: List[Dict[str, Any]]
    modal_mixture_chords: List[Dict[str, Any]]


@dataclass
class AdvancedHarmonicAnalysis:
    """Advanced harmonic analysis results"""
    secondary_dominants: List[Dict[str, Any]]
    augmented_sixths: List[Dict[str, Any]]
    neapolitan_chords: List[Dict[str, Any]]
    borrowed_chords: List[Dict[str, Any]]
    chromatic_mediants: List[Dict[str, Any]]
    extended_tonality_events: List[Dict[str, Any]]


@dataclass
class PhraseStructure:
    """Musical phrase structure analysis"""
    phrase_type: PhraseType
    phrase_lengths: List[int]
    cadences: List[Dict[str, Any]]
    motivic_analysis: Dict[str, Any]
    hypermetric_structure: List[int]
    elisions: List[int]
    phrase_rhythm: Dict[str, Any]


class AdvancedTheoryAnalyzer:
    """Advanced music theory analysis engine"""
    
    # Tendency tone resolutions
    TENDENCY_TONES = {
        7: [1],      # Leading tone to tonic
        4: [3],      # Subdominant to mediant
        6: [5],      # Submediant can resolve to dominant
        2: [1, 3],   # Supertonic to tonic or mediant
    }
    
    # Common secondary dominant relationships
    SECONDARY_DOMINANTS = {
        'V/V': {'target': 5, 'quality': 'major'},
        'V/ii': {'target': 2, 'quality': 'major'},
        'V/iii': {'target': 3, 'quality': 'major'},
        'V/IV': {'target': 4, 'quality': 'major'},
        'V/vi': {'target': 6, 'quality': 'major'},
        'V7/V': {'target': 5, 'quality': 'dominant7'},
        'V7/ii': {'target': 2, 'quality': 'dominant7'},
        'V7/iii': {'target': 3, 'quality': 'dominant7'},
        'V7/IV': {'target': 4, 'quality': 'dominant7'},
        'V7/vi': {'target': 6, 'quality': 'dominant7'},
    }
    
    def __init__(self):
        self.cache = {}
    
    async def analyze_scale_degrees(
        self,
        score: stream.Score,
        key_context: Optional[key.Key] = None
    ) -> ScaleDegreeAnalysis:
        """
        Analyze melodic motion by scale degrees
        
        Args:
            score: Score to analyze
            key_context: Key context (will detect if not provided)
            
        Returns:
            Comprehensive scale degree analysis
        """
        # Get or detect key
        if not key_context:
            key_context = score.analyze('key')
        
        scale_degrees = []
        chromatic_count = 0
        tendency_resolutions = defaultdict(int)
        
        # Extract all melodic notes
        melody_notes = []
        for n in score.flatten().notes:
            if isinstance(n, note.Note):
                melody_notes.append(n)
        
        # Analyze each note
        scale_pitches = key_context.pitches
        for i, n in enumerate(melody_notes):
            # Find scale degree
            degree = None
            for d, sp in enumerate(scale_pitches):
                if n.pitch.pitchClass == sp.pitchClass:
                    degree = d + 1
                    break
            
            if degree:
                scale_degrees.append(degree)
                
                # Check tendency tone resolution
                if i < len(melody_notes) - 1 and degree in self.TENDENCY_TONES:
                    next_note = melody_notes[i + 1]
                    next_degree = None
                    for d, sp in enumerate(scale_pitches):
                        if next_note.pitch.pitchClass == sp.pitchClass:
                            next_degree = d + 1
                            break
                    
                    if next_degree in self.TENDENCY_TONES[degree]:
                        tendency_resolutions[f"{degree}->{next_degree}"] += 1
            else:
                # Chromatic note
                chromatic_count += 1
                scale_degrees.append(0)  # 0 for chromatic
        
        # Calculate scale degree histogram
        degree_histogram = Counter(d for d in scale_degrees if d > 0)
        
        # Analyze melodic contour (scale degree changes)
        contour = []
        for i in range(1, len(scale_degrees)):
            if scale_degrees[i] > 0 and scale_degrees[i-1] > 0:
                contour.append(scale_degrees[i] - scale_degrees[i-1])
        
        # Calculate step vs leap ratio
        steps = sum(1 for c in contour if abs(c) <= 2)
        leaps = sum(1 for c in contour if abs(c) > 2)
        step_leap_ratio = steps / (steps + leaps) if (steps + leaps) > 0 else 0
        
        # Find most common scale degree progressions
        progressions = []
        for i in range(1, len(scale_degrees)):
            if scale_degrees[i] > 0 and scale_degrees[i-1] > 0:
                progressions.append(f"{scale_degrees[i-1]}->{scale_degrees[i]}")
        
        progression_counts = Counter(progressions)
        
        return ScaleDegreeAnalysis(
            scale_degree_histogram=dict(degree_histogram),
            tendency_tone_resolutions=dict(tendency_resolutions),
            melodic_contour=contour,
            step_leap_ratio=step_leap_ratio,
            chromatic_percentage=chromatic_count / len(melody_notes) if melody_notes else 0,
            most_common_progressions=progression_counts.most_common(10)
        )
    
    async def calculate_interval_vector(
        self,
        segment: Union[stream.Stream, List[pitch.Pitch]],
        include_z_relation: bool = True
    ) -> IntervalVectorAnalysis:
        """
        Calculate interval class vector for a segment
        
        Args:
            segment: Stream or list of pitches to analyze
            include_z_relation: Check for Z-related sets
            
        Returns:
            Interval vector analysis
        """
        # Extract pitches
        if isinstance(segment, stream.Stream):
            pitches = []
            for element in segment.flatten():
                if isinstance(element, note.Note):
                    pitches.append(element.pitch)
                elif isinstance(element, chord.Chord):
                    pitches.extend(element.pitches)
        else:
            pitches = segment
        
        if len(pitches) < 2:
            return IntervalVectorAnalysis(
                interval_vector=[0, 0, 0, 0, 0, 0],
                total_intervals=0,
                consonance_ratio=0,
                tritone_count=0,
                common_intervals=[],
                z_relation=None
            )
        
        # Calculate all intervals
        intervals = []
        interval_classes = [0] * 6  # IC1 through IC6
        
        for i in range(len(pitches)):
            for j in range(i + 1, len(pitches)):
                intv = interval.Interval(pitches[i], pitches[j])
                intervals.append(intv)
                
                # Get interval class (0-6)
                ic = intv.semitones % 12
                if ic > 6:
                    ic = 12 - ic
                
                if ic > 0 and ic <= 6:
                    interval_classes[ic - 1] += 1
        
        # Calculate consonance ratio
        consonant_intervals = interval_classes[2] + interval_classes[4]  # IC3 and IC5
        total_intervals = sum(interval_classes)
        consonance_ratio = consonant_intervals / total_intervals if total_intervals > 0 else 0
        
        # Count specific intervals
        interval_names = []
        for intv in intervals:
            interval_names.append(intv.niceName)
        
        common_intervals = Counter(interval_names).most_common(5)
        
        # Check for Z-relation (simplified - would need full pc set theory implementation)
        z_relation = None
        if include_z_relation and len(pitches) == 6:
            # Common Z-related hexachords
            z_pairs = {
                (1, 4, 3, 4, 3, 2): "Z17",
                (3, 0, 3, 6, 3, 0): "Z18",
                # Add more Z-relations as needed
            }
            
            vector_tuple = tuple(interval_classes)
            z_relation = z_pairs.get(vector_tuple)
        
        return IntervalVectorAnalysis(
            interval_vector=interval_classes,
            total_intervals=total_intervals,
            consonance_ratio=consonance_ratio,
            tritone_count=interval_classes[5],  # IC6 is tritone
            common_intervals=common_intervals,
            z_relation=z_relation
        )
    
    async def analyze_chromatic_elements(
        self,
        score: stream.Score,
        key_context: Optional[key.Key] = None
    ) -> ChromaticAnalysis:
        """
        Analyze chromatic passages and their functions
        
        Args:
            score: Score to analyze
            key_context: Key context
            
        Returns:
            Chromatic analysis results
        """
        if not key_context:
            key_context = score.analyze('key')
        
        chromatic_notes = []
        chromatic_functions = defaultdict(int)
        voice_leading_events = []
        
        # Get diatonic pitch classes
        diatonic_pcs = set(p.pitchClass for p in key_context.pitches)
        
        # Analyze each part for chromatic notes
        for part in score.parts:
            notes = list(part.flatten().notes)
            
            for i, n in enumerate(notes):
                if isinstance(n, note.Note):
                    if n.pitch.pitchClass not in diatonic_pcs:
                        # Chromatic note found
                        chromatic_info = {
                            'pitch': n.pitch.name,
                            'measure': n.measureNumber,
                            'beat': n.beat,
                            'duration': n.duration.quarterLength,
                            'function': None
                        }
                        
                        # Determine function
                        function = await self._classify_chromatic_function(
                            n, i, notes, key_context
                        )
                        chromatic_info['function'] = function
                        chromatic_functions[function] += 1
                        
                        chromatic_notes.append(chromatic_info)
                        
                        # Analyze voice leading
                        if i > 0 and i < len(notes) - 1:
                            prev_note = notes[i - 1]
                            next_note = notes[i + 1]
                            
                            if isinstance(prev_note, note.Note) and isinstance(next_note, note.Note):
                                voice_leading_events.append({
                                    'chromatic_pitch': n.pitch.name,
                                    'approach': interval.Interval(prev_note.pitch, n.pitch).niceName,
                                    'resolution': interval.Interval(n.pitch, next_note.pitch).niceName,
                                    'type': self._classify_voice_leading(prev_note, n, next_note)
                                })
        
        # Find modal mixture chords
        modal_mixture = await self._find_modal_mixture(score, key_context)
        
        # Calculate chromatic density
        total_notes = len(score.flatten().notes)
        chromatic_density = len(chromatic_notes) / total_notes if total_notes > 0 else 0
        
        return ChromaticAnalysis(
            chromatic_notes=chromatic_notes,
            chromatic_density=chromatic_density,
            chromatic_functions=dict(chromatic_functions),
            chromatic_voice_leading=voice_leading_events,
            modal_mixture_chords=modal_mixture
        )
    
    async def _classify_chromatic_function(
        self,
        chromatic_note: note.Note,
        index: int,
        notes: List[Union[note.Note, chord.Chord]],
        key_context: key.Key
    ) -> ChromaticFunction:
        """Classify the function of a chromatic note"""
        # Get previous and next notes
        prev_note = notes[index - 1] if index > 0 else None
        next_note = notes[index + 1] if index < len(notes) - 1 else None
        
        if not (prev_note and next_note):
            return ChromaticFunction.PASSING
        
        if not (isinstance(prev_note, note.Note) and isinstance(next_note, note.Note)):
            return ChromaticFunction.PASSING
        
        # Calculate intervals
        approach_interval = interval.Interval(prev_note.pitch, chromatic_note.pitch).semitones
        departure_interval = interval.Interval(chromatic_note.pitch, next_note.pitch).semitones
        
        # Passing tone: stepwise motion in same direction
        if abs(approach_interval) <= 2 and abs(departure_interval) <= 2:
            if (approach_interval > 0 and departure_interval > 0) or \
               (approach_interval < 0 and departure_interval < 0):
                return ChromaticFunction.PASSING
        
        # Neighbor tone: step away and back
        if abs(approach_interval) <= 2 and departure_interval == -approach_interval:
            return ChromaticFunction.NEIGHBOR
        
        # Appoggiatura: leap to dissonance, step resolution
        if abs(approach_interval) > 2 and abs(departure_interval) <= 2:
            return ChromaticFunction.APPOGIATURA
        
        # Escape tone: step to dissonance, leap away
        if abs(approach_interval) <= 2 and abs(departure_interval) > 2:
            return ChromaticFunction.ESCAPE
        
        # Check for secondary leading tone
        target_pitch = next_note.pitch
        if chromatic_note.pitch.midi == target_pitch.midi - 1:
            # Half step below target - could be secondary leading tone
            for degree in range(1, 8):
                degree_pitch = key_context.pitches[degree - 1]
                if target_pitch.pitchClass == degree_pitch.pitchClass and degree != 1:
                    return ChromaticFunction.SECONDARY_LEADING
        
        # Check for modal mixture (requires chord context)
        parallel_key = key_context.parallel
        if chromatic_note.pitch.pitchClass in [p.pitchClass for p in parallel_key.pitches]:
            return ChromaticFunction.MODAL_MIXTURE
        
        # Default to passing if unclear
        return ChromaticFunction.PASSING
    
    def _classify_voice_leading(
        self,
        prev_note: note.Note,
        chromatic_note: note.Note,
        next_note: note.Note
    ) -> str:
        """Classify voice leading pattern"""
        approach = interval.Interval(prev_note.pitch, chromatic_note.pitch).semitones
        departure = interval.Interval(chromatic_note.pitch, next_note.pitch).semitones
        
        if abs(approach) <= 2 and abs(departure) <= 2:
            if approach * departure > 0:  # Same direction
                return "passing"
            else:
                return "neighbor"
        elif abs(approach) > 2 and abs(departure) <= 2:
            return "appoggiatura"
        elif abs(approach) <= 2 and abs(departure) > 2:
            return "escape"
        else:
            return "free"
    
    async def _find_modal_mixture(
        self,
        score: stream.Score,
        key_context: key.Key
    ) -> List[Dict[str, Any]]:
        """Find chords borrowed from parallel mode"""
        modal_mixture_chords = []
        parallel_key = key_context.parallel
        
        # Get chords in the score
        for c in score.flatten().getElementsByClass(chord.Chord):
            # Check if chord contains notes from parallel key
            chord_pcs = set(p.pitchClass for p in c.pitches)
            parallel_pcs = set(p.pitchClass for p in parallel_key.pitches)
            original_pcs = set(p.pitchClass for p in key_context.pitches)
            
            # If chord has pitches from parallel but not original key
            borrowed_pcs = chord_pcs & parallel_pcs - original_pcs
            
            if borrowed_pcs:
                try:
                    # Analyze in parallel key
                    rn_parallel = roman.romanNumeralFromChord(c, parallel_key)
                    
                    modal_mixture_chords.append({
                        'measure': c.measureNumber,
                        'beat': c.beat,
                        'chord': c.pitchesChr,
                        'borrowed_from': str(parallel_key),
                        'function_in_parallel': str(rn_parallel),
                        'borrowed_pitches': list(borrowed_pcs)
                    })
                except:
                    pass
        
        return modal_mixture_chords
    
    async def detect_advanced_harmony(
        self,
        score: stream.Score,
        key_context: Optional[key.Key] = None
    ) -> AdvancedHarmonicAnalysis:
        """
        Detect advanced harmonic structures
        
        Args:
            score: Score to analyze
            key_context: Key context
            
        Returns:
            Advanced harmonic analysis
        """
        if not key_context:
            key_context = score.analyze('key')
        
        secondary_dominants = await self._detect_secondary_dominants(score, key_context)
        augmented_sixths = await self._detect_augmented_sixths(score, key_context)
        neapolitan_chords = await self._detect_neapolitan_chords(score, key_context)
        borrowed_chords = await self._detect_borrowed_chords(score, key_context)
        chromatic_mediants = await self._detect_chromatic_mediants(score, key_context)
        extended_tonality = await self._detect_extended_tonality(score, key_context)
        
        return AdvancedHarmonicAnalysis(
            secondary_dominants=secondary_dominants,
            augmented_sixths=augmented_sixths,
            neapolitan_chords=neapolitan_chords,
            borrowed_chords=borrowed_chords,
            chromatic_mediants=chromatic_mediants,
            extended_tonality_events=extended_tonality
        )
    
    async def _detect_secondary_dominants(
        self,
        score: stream.Score,
        key_context: key.Key
    ) -> List[Dict[str, Any]]:
        """Detect secondary dominant chords"""
        secondary_dominants = []
        
        chords = list(score.flatten().getElementsByClass(chord.Chord))
        
        for i, c in enumerate(chords):
            # Check if chord is major or dominant 7th
            if c.isMajorTriad() or c.isDominantSeventh():
                # Get root
                root = c.root()
                if not root:
                    continue
                
                # Check if it's diatonic V
                try:
                    rn = roman.romanNumeralFromChord(c, key_context)
                    if rn.romanNumeral.upper() == 'V':
                        continue  # Regular dominant, not secondary
                except:
                    pass
                
                # Look for resolution
                if i < len(chords) - 1:
                    next_chord = chords[i + 1]
                    
                    # Check if it resolves down a fifth
                    if next_chord.root():
                        resolution_interval = interval.Interval(root, next_chord.root())
                        
                        if resolution_interval.semitones == -7 or resolution_interval.semitones == 5:
                            # Likely secondary dominant
                            target_degree = None
                            
                            # Find which scale degree it tonicizes
                            for degree, pitch_class in enumerate(key_context.pitches):
                                if next_chord.root().pitchClass == pitch_class.pitchClass:
                                    target_degree = degree + 1
                                    break
                            
                            if target_degree and target_degree != 1:  # Not tonic
                                secondary_dominants.append({
                                    'measure': c.measureNumber,
                                    'beat': c.beat,
                                    'chord': c.pitchesChr,
                                    'type': 'V7' if c.isDominantSeventh() else 'V',
                                    'target_degree': target_degree,
                                    'symbol': f"V{'7' if c.isDominantSeventh() else ''}/{self._degree_to_roman(target_degree)}",
                                    'resolution': next_chord.pitchesChr,
                                    'resolution_measure': next_chord.measureNumber
                                })
        
        return secondary_dominants
    
    def _degree_to_roman(self, degree: int) -> str:
        """Convert scale degree to Roman numeral"""
        romans = ['I', 'ii', 'iii', 'IV', 'V', 'vi', 'vii°']
        if 1 <= degree <= 7:
            return romans[degree - 1]
        return str(degree)
    
    async def _detect_augmented_sixths(
        self,
        score: stream.Score,
        key_context: key.Key
    ) -> List[Dict[str, Any]]:
        """Detect augmented sixth chords"""
        aug_sixths = []
        
        for c in score.flatten().getElementsByClass(chord.Chord):
            # Aug 6 chords have characteristic interval of augmented 6th
            intervals = []
            pitches = sorted(c.pitches, key=lambda p: p.ps)
            
            for i in range(len(pitches)):
                for j in range(i + 1, len(pitches)):
                    intv = interval.Interval(pitches[i], pitches[j])
                    intervals.append(intv)
            
            # Look for augmented sixth interval
            has_aug_sixth = any(
                intv.name == 'A6' or 
                (intv.semitones == 10 and intv.diatonic.generic.value == 6)
                for intv in intervals
            )
            
            if has_aug_sixth:
                # Classify type (Italian, French, German)
                chord_type = self._classify_augmented_sixth(c, key_context)
                
                if chord_type:
                    aug_sixths.append({
                        'measure': c.measureNumber,
                        'beat': c.beat,
                        'chord': c.pitchesChr,
                        'type': chord_type,
                        'bass': c.bass().name if c.bass() else None,
                        'resolution_tendency': 'V' if chord_type else None
                    })
        
        return aug_sixths
    
    def _classify_augmented_sixth(self, chord: chord.Chord, key_context: key.Key) -> Optional[str]:
        """Classify type of augmented sixth chord"""
        # Simplified classification
        # In practice, would need to check specific scale degrees
        
        chord_size = len(chord.pitches)
        
        if chord_size == 3:
            return "Italian"
        elif chord_size == 4:
            # Check for specific intervals to distinguish French vs German
            has_perfect_fifth = any(
                intv.name == 'P5' for intv in chord.intervalVector
            )
            
            if has_perfect_fifth:
                return "German"
            else:
                return "French"
        
        return None
    
    async def _detect_neapolitan_chords(
        self,
        score: stream.Score,
        key_context: key.Key
    ) -> List[Dict[str, Any]]:
        """Detect Neapolitan sixth chords"""
        neapolitans = []
        
        # Neapolitan is bII in first inversion (usually)
        # In C major: Db major chord, usually in first inversion (F-Ab-Db)
        
        flat_two = key_context.pitches[1].transpose(-1)  # Flatten the second degree
        
        for c in score.flatten().getElementsByClass(chord.Chord):
            if c.isMajorTriad():
                # Check if root is flat two
                if c.root() and c.root().pitchClass == flat_two.pitchClass:
                    neapolitans.append({
                        'measure': c.measureNumber,
                        'beat': c.beat,
                        'chord': c.pitchesChr,
                        'inversion': c.inversion(),
                        'typical_resolution': 'V',
                        'function': 'bII' + ('6' if c.inversion() == 1 else '')
                    })
        
        return neapolitans
    
    async def _detect_borrowed_chords(
        self,
        score: stream.Score,
        key_context: key.Key
    ) -> List[Dict[str, Any]]:
        """Detect borrowed chords (modal interchange)"""
        borrowed = []
        parallel_key = key_context.parallel
        
        # Common borrowed chords in major: bIII, bVI, bVII, iv
        # Common borrowed chords in minor: I, IV, V (when major)
        
        for c in score.flatten().getElementsByClass(chord.Chord):
            try:
                # Analyze in both keys
                rn_original = roman.romanNumeralFromChord(c, key_context)
                rn_parallel = roman.romanNumeralFromChord(c, parallel_key)
                
                # Check if chord fits better in parallel key
                chord_pcs = set(p.pitchClass for p in c.pitches)
                original_pcs = set(p.pitchClass for p in key_context.pitches)
                parallel_pcs = set(p.pitchClass for p in parallel_key.pitches)
                
                # More notes from parallel than original = likely borrowed
                parallel_matches = len(chord_pcs & parallel_pcs)
                original_matches = len(chord_pcs & original_pcs)
                
                if parallel_matches > original_matches:
                    borrowed.append({
                        'measure': c.measureNumber,
                        'beat': c.beat,
                        'chord': c.pitchesChr,
                        'function_original': str(rn_original),
                        'function_parallel': str(rn_parallel),
                        'borrowed_from': 'minor' if key_context.mode == 'major' else 'major'
                    })
            except:
                pass
        
        return borrowed
    
    async def _detect_chromatic_mediants(
        self,
        score: stream.Score,
        key_context: key.Key
    ) -> List[Dict[str, Any]]:
        """Detect chromatic mediant relationships"""
        mediants = []
        chords = list(score.flatten().getElementsByClass(chord.Chord))
        
        for i in range(1, len(chords)):
            prev_chord = chords[i - 1]
            curr_chord = chords[i]
            
            if prev_chord.root() and curr_chord.root():
                # Check interval between roots
                root_interval = interval.Interval(prev_chord.root(), curr_chord.root())
                
                # Chromatic mediants are typically major/minor thirds apart
                # with minimal common tones
                if root_interval.semitones in [3, 4, 8, 9]:  # Major/minor thirds and sixths
                    # Count common tones
                    prev_pcs = set(p.pitchClass for p in prev_chord.pitches)
                    curr_pcs = set(p.pitchClass for p in curr_chord.pitches)
                    common_tones = len(prev_pcs & curr_pcs)
                    
                    # Chromatic mediants typically have 0-1 common tones
                    if common_tones <= 1:
                        mediants.append({
                            'measure': curr_chord.measureNumber,
                            'beat': curr_chord.beat,
                            'chord1': prev_chord.pitchesChr,
                            'chord2': curr_chord.pitchesChr,
                            'root_interval': root_interval.niceName,
                            'common_tones': common_tones,
                            'type': 'chromatic_mediant'
                        })
        
        return mediants
    
    async def _detect_extended_tonality(
        self,
        score: stream.Score,
        key_context: key.Key
    ) -> List[Dict[str, Any]]:
        """Detect extended tonal practices"""
        extended_events = []
        
        # Look for signs of extended tonality:
        # - Quartal harmony
        # - Added note chords (add9, add11, etc.)
        # - Polychords
        # - Non-functional progressions
        
        for c in score.flatten().getElementsByClass(chord.Chord):
            # Check for quartal harmony
            intervals = []
            pitches = sorted(c.pitches, key=lambda p: p.ps)
            
            for i in range(1, len(pitches)):
                intv = interval.Interval(pitches[i-1], pitches[i])
                intervals.append(intv.semitones)
            
            # Quartal if built primarily on fourths
            if intervals and all(i in [5, 6] for i in intervals):  # P4 or A4
                extended_events.append({
                    'measure': c.measureNumber,
                    'beat': c.beat,
                    'chord': c.pitchesChr,
                    'type': 'quartal_harmony',
                    'intervals': intervals
                })
            
            # Check for extended tertian harmony
            if len(c.pitches) > 4:
                chord_type = 'extended_tertian'
                if len(c.pitches) == 5:
                    chord_type = 'ninth_chord'
                elif len(c.pitches) == 6:
                    chord_type = 'eleventh_chord'
                elif len(c.pitches) >= 7:
                    chord_type = 'thirteenth_chord'
                
                extended_events.append({
                    'measure': c.measureNumber,
                    'beat': c.beat,
                    'chord': c.pitchesChr,
                    'type': chord_type,
                    'size': len(c.pitches)
                })
        
        return extended_events
    
    async def analyze_phrase_structure(
        self,
        score: stream.Score,
        include_motivic_analysis: bool = True
    ) -> PhraseStructure:
        """
        Analyze musical phrase structure
        
        Args:
            score: Score to analyze
            include_motivic_analysis: Whether to include motivic analysis
            
        Returns:
            Phrase structure analysis
        """
        # Detect phrase boundaries
        phrases = await self._detect_phrases(score)
        
        # Classify phrase type
        phrase_type = self._classify_phrase_structure(phrases)
        
        # Extract cadences
        cadences = await self._analyze_cadences(score, phrases)
        
        # Analyze hypermetric structure
        hypermetric = self._analyze_hypermeter(phrases)
        
        # Find elisions
        elisions = self._find_elisions(phrases, cadences)
        
        # Analyze phrase rhythm
        phrase_rhythm = self._analyze_phrase_rhythm(phrases)
        
        # Motivic analysis
        motivic_analysis = {}
        if include_motivic_analysis:
            motivic_analysis = await self._analyze_motives(score, phrases)
        
        return PhraseStructure(
            phrase_type=phrase_type,
            phrase_lengths=[p['length'] for p in phrases],
            cadences=cadences,
            motivic_analysis=motivic_analysis,
            hypermetric_structure=hypermetric,
            elisions=elisions,
            phrase_rhythm=phrase_rhythm
        )
    
    async def _detect_phrases(self, score: stream.Score) -> List[Dict[str, Any]]:
        """Detect phrase boundaries"""
        phrases = []
        measures = list(score.getElementsByClass(stream.Measure))
        
        current_phrase_start = 0
        
        for i, measure in enumerate(measures):
            # Look for phrase-ending indicators
            has_fermata = any(
                isinstance(e, expressions.Fermata)
                for e in measure.flatten().expressions
            )
            
            has_long_rest = any(
                isinstance(e, note.Rest) and e.duration.quarterLength >= 2.0
                for e in measure.flatten()
            )
            
            # Check for cadential motion (simplified)
            is_cadence = False
            if i > 0:
                # Look for V-I motion or similar
                prev_harmony = self._get_measure_harmony(measures[i-1])
                curr_harmony = self._get_measure_harmony(measure)
                
                if prev_harmony and curr_harmony:
                    # Simple cadence detection
                    if self._is_cadential_motion(prev_harmony, curr_harmony):
                        is_cadence = True
            
            # If phrase ending detected
            if has_fermata or has_long_rest or is_cadence or i == len(measures) - 1:
                phrases.append({
                    'start_measure': current_phrase_start + 1,
                    'end_measure': i + 1,
                    'length': i - current_phrase_start + 1,
                    'ending_type': 'fermata' if has_fermata else 'rest' if has_long_rest else 'cadence'
                })
                current_phrase_start = i + 1
        
        return phrases
    
    def _get_measure_harmony(self, measure: stream.Measure) -> Optional[chord.Chord]:
        """Get representative harmony from a measure"""
        chords = list(measure.getElementsByClass(chord.Chord))
        if chords:
            # Return the chord with longest duration
            return max(chords, key=lambda c: c.duration.quarterLength)
        return None
    
    def _is_cadential_motion(self, chord1: chord.Chord, chord2: chord.Chord) -> bool:
        """Simple cadence detection"""
        # Check for common cadential progressions
        if not (chord1.root() and chord2.root()):
            return False
        
        interval_between = interval.Interval(chord1.root(), chord2.root())
        
        # V-I (down P5 or up P4)
        if interval_between.semitones in [-7, 5]:
            return True
        
        # IV-I (up P5 or down P4)
        if interval_between.semitones in [7, -5]:
            return True
        
        return False
    
    def _classify_phrase_structure(self, phrases: List[Dict[str, Any]]) -> PhraseType:
        """Classify overall phrase structure"""
        if len(phrases) < 2:
            return PhraseType.ASYMMETRIC
        
        lengths = [p['length'] for p in phrases]
        
        # Period: two phrases of similar length
        if len(phrases) == 2 and abs(lengths[0] - lengths[1]) <= 1:
            return PhraseType.PERIOD
        
        # Sentence: typically 8 bars (2+2+4)
        if len(phrases) >= 3 and lengths[0] == lengths[1] == 2 and lengths[2] == 4:
            return PhraseType.SENTENCE
        
        # Double period: four phrases
        if len(phrases) == 4 and all(abs(l - lengths[0]) <= 1 for l in lengths):
            return PhraseType.DOUBLE_PERIOD
        
        # Compound structures
        if len(phrases) == 4:
            if lengths[0] == lengths[1] and lengths[2] == lengths[3]:
                return PhraseType.COMPOUND_PERIOD
        
        # Phrase group: multiple phrases without clear pattern
        if len(phrases) > 2:
            return PhraseType.PHRASE_GROUP
        
        return PhraseType.ASYMMETRIC
    
    async def _analyze_cadences(
        self,
        score: stream.Score,
        phrases: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Analyze cadences at phrase endings"""
        cadences = []
        measures = list(score.getElementsByClass(stream.Measure))
        
        for phrase in phrases:
            end_measure_idx = phrase['end_measure'] - 1
            
            if end_measure_idx < len(measures) and end_measure_idx > 0:
                # Get last two measures of phrase
                penultimate = measures[end_measure_idx - 1]
                final = measures[end_measure_idx]
                
                # Get harmonies
                pen_harmony = self._get_measure_harmony(penultimate)
                final_harmony = self._get_measure_harmony(final)
                
                if pen_harmony and final_harmony:
                    cadence_type = self._classify_cadence_type(pen_harmony, final_harmony)
                    
                    cadences.append({
                        'phrase_end': phrase['end_measure'],
                        'type': cadence_type,
                        'chords': [pen_harmony.pitchesChr, final_harmony.pitchesChr],
                        'strength': self._calculate_cadence_strength(cadence_type, final)
                    })
        
        return cadences
    
    def _classify_cadence_type(self, chord1: chord.Chord, chord2: chord.Chord) -> str:
        """Classify cadence type based on two chords"""
        # Simplified cadence classification
        # In practice would use Roman numeral analysis
        
        if not (chord1.root() and chord2.root()):
            return "inconclusive"
        
        interval_between = interval.Interval(chord1.root(), chord2.root())
        
        # Perfect authentic: V-I with both in root position
        if interval_between.semitones in [-7, 5]:
            if chord1.inversion() == 0 and chord2.inversion() == 0:
                return "perfect_authentic"
            else:
                return "imperfect_authentic"
        
        # Plagal: IV-I
        if interval_between.semitones in [7, -5]:
            return "plagal"
        
        # Half cadence: ends on V
        if chord2.isDominantSeventh() or (chord2.isMajorTriad() and chord2.root().name in ['G', 'D', 'A', 'E', 'B']):
            return "half"
        
        # Deceptive: V-vi
        if interval_between.semitones in [-9, 3]:
            return "deceptive"
        
        return "inconclusive"
    
    def _calculate_cadence_strength(self, cadence_type: str, final_measure: stream.Measure) -> float:
        """Calculate cadence strength (0-1)"""
        strength = 0.5  # Base strength
        
        # Stronger cadences
        if cadence_type == "perfect_authentic":
            strength = 1.0
        elif cadence_type == "imperfect_authentic":
            strength = 0.8
        elif cadence_type == "plagal":
            strength = 0.7
        elif cadence_type == "half":
            strength = 0.6
        elif cadence_type == "deceptive":
            strength = 0.5
        else:
            strength = 0.3
        
        # Modify based on rhythmic position
        # Cadence on downbeat is stronger
        final_chord = None
        for c in final_measure.getElementsByClass(chord.Chord):
            final_chord = c
            break
        
        if final_chord and final_chord.beat == 1:
            strength *= 1.1
        
        return min(1.0, strength)
    
    def _analyze_hypermeter(self, phrases: List[Dict[str, Any]]) -> List[int]:
        """Analyze hypermetric structure"""
        # Group phrases into hypermetric units
        hypermetric = []
        
        # Common hypermeters: 4-bar, 8-bar units
        for phrase in phrases:
            length = phrase['length']
            
            if length % 4 == 0:
                hypermetric.append(4)
            elif length % 3 == 0:
                hypermetric.append(3)
            else:
                hypermetric.append(length)
        
        return hypermetric
    
    def _find_elisions(
        self,
        phrases: List[Dict[str, Any]],
        cadences: List[Dict[str, Any]]
    ) -> List[int]:
        """Find phrase elisions (overlapping phrase endings/beginnings)"""
        elisions = []
        
        for i in range(1, len(phrases)):
            # Check if phrase starts where previous ended
            if phrases[i]['start_measure'] == phrases[i-1]['end_measure']:
                # This might be an elision
                # Check if there's a cadence at this point
                for cadence in cadences:
                    if cadence['phrase_end'] == phrases[i-1]['end_measure']:
                        if cadence['type'] not in ['perfect_authentic', 'plagal']:
                            # Weak cadence might indicate elision
                            elisions.append(phrases[i]['start_measure'])
                            break
        
        return elisions
    
    def _analyze_phrase_rhythm(self, phrases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze rhythmic aspects of phrases"""
        return {
            'regular_phrases': sum(1 for p in phrases if p['length'] in [4, 8]),
            'irregular_phrases': sum(1 for p in phrases if p['length'] not in [4, 8]),
            'average_length': sum(p['length'] for p in phrases) / len(phrases) if phrases else 0,
            'length_variety': len(set(p['length'] for p in phrases))
        }
    
    async def _analyze_motives(
        self,
        score: stream.Score,
        phrases: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze motivic content"""
        # Simplified motivic analysis
        # Look for recurring melodic/rhythmic patterns
        
        motives = {
            'melodic_motives': [],
            'rhythmic_motives': [],
            'transformations': []
        }
        
        # Extract short melodic fragments (3-5 notes)
        melodic_fragments = []
        
        for n in score.flatten().notes:
            if isinstance(n, note.Note):
                melodic_fragments.append(n)
                
                if len(melodic_fragments) >= 4:
                    # Create interval sequence
                    intervals = []
                    for i in range(1, 4):
                        intv = interval.Interval(
                            melodic_fragments[-4+i-1].pitch,
                            melodic_fragments[-4+i].pitch
                        )
                        intervals.append(intv.semitones)
                    
                    # Store as potential motive
                    motives['melodic_motives'].append({
                        'intervals': intervals,
                        'start_note': melodic_fragments[-4].pitch.name,
                        'measure': melodic_fragments[-4].measureNumber
                    })
        
        # Find recurring patterns
        # (Simplified - in practice would use more sophisticated pattern matching)
        interval_sequences = [tuple(m['intervals']) for m in motives['melodic_motives']]
        sequence_counts = Counter(interval_sequences)
        
        # Keep only recurring motives
        recurring_motives = []
        for seq, count in sequence_counts.most_common(5):
            if count > 1:
                recurring_motives.append({
                    'interval_pattern': list(seq),
                    'occurrences': count,
                    'type': 'melodic'
                })
        
        motives['melodic_motives'] = recurring_motives
        
        return motives
    
    async def analyze_dissonance_treatment(
        self,
        score: stream.Score
    ) -> Dict[str, Any]:
        """
        Analyze how dissonances are prepared and resolved
        
        Args:
            score: Score to analyze
            
        Returns:
            Dissonance treatment analysis
        """
        dissonances = []
        
        # Analyze each voice for dissonant intervals
        for part in score.parts:
            notes = list(part.flatten().notes)
            
            # Compare with other voices at each time point
            for i, n1 in enumerate(notes):
                if isinstance(n1, note.Note):
                    # Find simultaneous notes in other parts
                    simultaneous = self._find_simultaneous_notes(n1, score)
                    
                    for n2 in simultaneous:
                        if n2 != n1:
                            intv = interval.Interval(n1.pitch, n2.pitch)
                            
                            # Check if dissonant
                            if self._is_dissonant_interval(intv):
                                # Analyze preparation and resolution
                                prep = self._analyze_preparation(n1, i, notes)
                                res = self._analyze_resolution(n1, i, notes)
                                
                                dissonances.append({
                                    'measure': n1.measureNumber,
                                    'beat': n1.beat,
                                    'interval': intv.niceName,
                                    'voices': [n1.pitch.name, n2.pitch.name],
                                    'type': self._classify_dissonance_type(prep, res),
                                    'preparation': prep,
                                    'resolution': res
                                })
        
        # Summarize dissonance treatment
        dissonance_types = Counter(d['type'] for d in dissonances)
        
        return {
            'total_dissonances': len(dissonances),
            'dissonance_types': dict(dissonance_types),
            'properly_treated': sum(
                1 for d in dissonances
                if d['preparation'] != 'unprepared' or d['resolution'] != 'unresolved'
            ),
            'dissonances': dissonances[:20]  # Limit output
        }
    
    def _find_simultaneous_notes(
        self,
        target_note: note.Note,
        score: stream.Score
    ) -> List[note.Note]:
        """Find notes occurring at the same time as target"""
        simultaneous = []
        target_offset = target_note.offset
        
        for part in score.parts:
            for n in part.flatten().notes:
                if isinstance(n, note.Note):
                    if abs(n.offset - target_offset) < 0.01:  # Same offset
                        simultaneous.append(n)
        
        return simultaneous
    
    def _is_dissonant_interval(self, intv: interval.Interval) -> bool:
        """Check if interval is dissonant"""
        # Dissonant intervals: m2, M2, A4, d5, m7, M7
        dissonant_semitones = [1, 2, 6, 10, 11]
        return (intv.semitones % 12) in dissonant_semitones
    
    def _analyze_preparation(
        self,
        dissonant_note: note.Note,
        index: int,
        notes: List[note.Note]
    ) -> str:
        """Analyze how dissonance is prepared"""
        if index == 0:
            return "unprepared"
        
        prev_note = notes[index - 1]
        
        if isinstance(prev_note, note.Note):
            # Common tone preparation
            if prev_note.pitch.pitchClass == dissonant_note.pitch.pitchClass:
                return "common_tone"
            
            # Stepwise preparation
            intv = interval.Interval(prev_note.pitch, dissonant_note.pitch)
            if abs(intv.semitones) <= 2:
                return "stepwise"
            
            # Leap preparation
            return "leap"
        
        return "unprepared"
    
    def _analyze_resolution(
        self,
        dissonant_note: note.Note,
        index: int,
        notes: List[note.Note]
    ) -> str:
        """Analyze how dissonance is resolved"""
        if index >= len(notes) - 1:
            return "unresolved"
        
        next_note = notes[index + 1]
        
        if isinstance(next_note, note.Note):
            intv = interval.Interval(dissonant_note.pitch, next_note.pitch)
            
            # Stepwise resolution (preferred)
            if abs(intv.semitones) == 1:
                return "stepwise_down" if intv.semitones < 0 else "stepwise_up"
            elif abs(intv.semitones) == 2:
                return "step_down" if intv.semitones < 0 else "step_up"
            
            # Leap resolution
            return "leap"
        
        return "unresolved"
    
    def _classify_dissonance_type(self, preparation: str, resolution: str) -> str:
        """Classify type of dissonance based on treatment"""
        if preparation == "common_tone":
            if "step" in resolution:
                return "suspension"
            else:
                return "retardation"
        elif preparation == "stepwise":
            if resolution == "stepwise_down":
                return "passing_tone"
            elif resolution == "leap":
                return "escape_tone"
            else:
                return "neighbor_tone"
        elif preparation == "leap":
            if "step" in resolution:
                return "appoggiatura"
            else:
                return "free_tone"
        else:
            return "unprepared_dissonance"