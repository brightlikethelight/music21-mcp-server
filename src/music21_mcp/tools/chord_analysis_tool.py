"""
Chord Analysis Tool - Extract and analyze chord progressions
"""
import logging
from typing import Any, Dict, List, Optional

from music21 import chord, roman

from .base_tool import BaseTool

logger = logging.getLogger(__name__)


class ChordAnalysisTool(BaseTool):
    """Tool for analyzing chord progressions with Roman numeral analysis"""
    
    async def execute(self, score_id: str, include_roman_numerals: bool = True,
                     include_inversions: bool = True, segment_length: float = 0.5) -> Dict[str, Any]:
        """
        Analyze chord progressions in a score
        
        Args:
            score_id: ID of the score to analyze
            include_roman_numerals: Include Roman numeral analysis
            include_inversions: Include chord inversions in analysis
            segment_length: Segment length for chordify (in quarter notes)
        """
        # Validate inputs
        error = self.validate_inputs(score_id=score_id)
        if error:
            return self.create_error_response(error)
        
        with self.error_handling(f"Chord analysis for '{score_id}'"):
            score = self.get_score(score_id)
            
            self.report_progress(0.1, "Preparing score for chord analysis")
            
            # Chordify the score
            try:
                chords = score.chordify(removeRedundantPitches=True)
            except Exception as e:
                logger.warning(f"Chordify failed, trying flatten: {e}")
                chords = score.flatten().chordify()
            
            self.report_progress(0.3, "Extracting chord progression")
            
            # Extract chord progression
            chord_progression = []
            chord_list = list(chords.flatten().getElementsByClass(chord.Chord))
            total_chords = len(chord_list)
            
            # Get the key for Roman numeral analysis
            score_key = None
            if include_roman_numerals:
                # Try to detect key first
                try:
                    key_result = score.analyze('key')
                    score_key = key_result
                except:
                    logger.warning("Could not detect key for Roman numeral analysis")
            
            # Process each chord
            for i, ch in enumerate(chord_list):
                if i % 10 == 0:
                    self.report_progress(0.3 + (0.5 * i / total_chords), f"Processing chord {i+1}/{total_chords}")
                
                chord_info = self._analyze_chord(ch, score_key, include_inversions)
                chord_progression.append(chord_info)
            
            self.report_progress(0.8, "Analyzing harmonic rhythm")
            
            # Analyze harmonic rhythm
            harmonic_rhythm = self._analyze_harmonic_rhythm(chord_list)
            
            # Extract Roman numerals if requested
            roman_numerals = []
            if include_roman_numerals and score_key:
                self.report_progress(0.9, "Generating Roman numeral analysis")
                for ch in chord_list[:50]:  # Limit to first 50 for performance
                    try:
                        rn = roman.romanNumeralFromChord(ch, score_key)
                        roman_numerals.append(str(rn.romanNumeral))
                    except:
                        roman_numerals.append("?")
            
            self.report_progress(1.0, "Analysis complete")
            
            # Compile results
            result = self.create_success_response(
                total_chords=total_chords,
                chord_progression=chord_progression[:50],  # Limit output size
                harmonic_rhythm=harmonic_rhythm
            )
            
            if include_roman_numerals:
                result['roman_numerals'] = roman_numerals
                if score_key:
                    result['analysis_key'] = str(score_key)
            
            # Add summary statistics
            result['summary'] = self._generate_chord_summary(chord_progression)
            
            return result
    
    def validate_inputs(self, score_id: str, **kwargs) -> Optional[str]:
        """Validate input parameters"""
        return self.check_score_exists(score_id)
    
    def _analyze_chord(self, ch: chord.Chord, score_key, include_inversions: bool) -> Dict[str, Any]:
        """Analyze a single chord"""
        try:
            # Get pitch names
            pitches = [str(p) for p in ch.pitches]
            
            # Get chord symbol
            chord_symbol = ch.pitchedCommonName
            
            # Get root
            root = str(ch.root()) if ch.root() else None
            
            # Get quality
            quality = ch.quality if hasattr(ch, 'quality') else None
            
            # Basic info
            info = {
                'pitches': pitches,
                'symbol': chord_symbol,
                'root': root,
                'quality': quality,
                'offset': float(ch.offset),
                'duration': float(ch.duration.quarterLength)
            }
            
            # Add inversion if requested
            if include_inversions and ch.inversion() != 0:
                info['inversion'] = ch.inversion()
                info['bass'] = str(ch.bass())
            
            # Add Roman numeral if key is known
            if score_key:
                try:
                    rn = roman.romanNumeralFromChord(ch, score_key)
                    info['roman_numeral'] = str(rn.romanNumeral)
                    info['scale_degree'] = rn.scaleDegree
                except:
                    pass
            
            return info
            
        except Exception as e:
            logger.warning(f"Error analyzing chord: {e}")
            return {
                'pitches': [str(p) for p in ch.pitches],
                'error': 'Analysis failed'
            }
    
    def _analyze_harmonic_rhythm(self, chord_list: List[chord.Chord]) -> Dict[str, Any]:
        """Analyze the harmonic rhythm (rate of chord changes)"""
        if not chord_list:
            return {'average_duration': 0, 'changes_per_measure': 0}
        
        try:
            # Calculate average chord duration
            durations = [float(ch.duration.quarterLength) for ch in chord_list]
            avg_duration = sum(durations) / len(durations)
            
            # Estimate changes per measure (assuming 4/4)
            changes_per_measure = 4.0 / avg_duration if avg_duration > 0 else 0
            
            # Find most common durations
            duration_counts = {}
            for d in durations:
                duration_counts[d] = duration_counts.get(d, 0) + 1
            
            common_durations = sorted(duration_counts.items(), key=lambda x: x[1], reverse=True)[:3]
            
            return {
                'average_duration': round(avg_duration, 2),
                'changes_per_measure': round(changes_per_measure, 2),
                'common_durations': [{'duration': d, 'count': c} for d, c in common_durations]
            }
            
        except Exception as e:
            logger.warning(f"Error analyzing harmonic rhythm: {e}")
            return {'average_duration': 0, 'changes_per_measure': 0}
    
    def _generate_chord_summary(self, chord_progression: List[Dict]) -> Dict[str, Any]:
        """Generate summary statistics about the chord progression"""
        if not chord_progression:
            return {}
        
        try:
            # Count chord types
            chord_types = {}
            for ch in chord_progression:
                symbol = ch.get('symbol', 'unknown')
                chord_types[symbol] = chord_types.get(symbol, 0) + 1
            
            # Most common chords
            common_chords = sorted(chord_types.items(), key=lambda x: x[1], reverse=True)[:5]
            
            # Count qualities
            qualities = {}
            for ch in chord_progression:
                quality = ch.get('quality', 'unknown')
                if quality:
                    qualities[quality] = qualities.get(quality, 0) + 1
            
            return {
                'unique_chords': len(chord_types),
                'most_common_chords': [{'chord': c, 'count': n} for c, n in common_chords],
                'chord_qualities': qualities
            }
            
        except Exception as e:
            logger.warning(f"Error generating chord summary: {e}")
            return {}