"""
Voice Leading Analysis Tool - Simple voice leading analysis
Uses music21 directly for basic voice leading analysis
"""

import logging
from typing import Any, Dict, List, Optional

from music21 import interval, stream, note, chord

from .base_tool import BaseTool

logger = logging.getLogger(__name__)


class VoiceLeadingAnalysisTool(BaseTool):
    """
    Voice leading analysis tool providing:
    1. Parallel fifths/octaves detection
    2. Voice crossing identification
    3. Basic voice leading smoothness analysis
    """

    def __init__(self, score_manager: Dict[str, Any]):
        super().__init__(score_manager)

    async def execute(self, **kwargs: Any) -> Dict[str, Any]:
        """
        Analyze voice leading in a score
        
        Args:
            score_id: ID of the score to analyze
        """
        score_id = kwargs.get("score_id", "")
        
        # Validate inputs
        error = self.validate_inputs(**kwargs)
        if error:
            return self.create_error_response(error)

        with self.error_handling(f"Voice leading analysis for '{score_id}'"):
            # Get the score
            score = self.get_score(score_id)
            if not score:
                return self.create_error_response(f"Score '{score_id}' not found")

            # Analyze voice leading
            parallel_issues = self._detect_parallel_intervals(score)
            voice_crossings = self._detect_voice_crossings(score)
            smoothness = self._analyze_smoothness(score)
            
            total_issues = len(parallel_issues) + len(voice_crossings)
            
            return self.create_success_response(
                f"Voice leading analysis complete: {total_issues} issues found",
                parallel_issues=parallel_issues,
                voice_crossings=voice_crossings,
                smoothness_analysis=smoothness,
                total_issues=total_issues,
                overall_score=max(0, 100 - total_issues * 10)  # Basic scoring
            )

    def validate_inputs(self, **kwargs: Any) -> Optional[str]:
        """Validate the inputs for voice leading analysis"""
        score_id = kwargs.get("score_id", "")
        if not score_id:
            return "score_id is required"
        return None

    def _detect_parallel_intervals(self, score: stream.Score) -> List[Dict[str, Any]]:
        """Detect parallel fifths and octaves"""
        parallel_issues = []
        
        # Get all parts
        parts = score.parts
        if len(parts) < 2:
            return parallel_issues
        
        # Check consecutive chords/notes between parts
        for i in range(len(parts) - 1):
            part1 = parts[i]
            part2 = parts[i + 1]
            
            # Get notes from both parts
            notes1 = [n for n in part1.flatten().notes if isinstance(n, note.Note)]
            notes2 = [n for n in part2.flatten().notes if isinstance(n, note.Note)]
            
            # Check for parallel motion
            min_len = min(len(notes1), len(notes2))
            for j in range(min_len - 1):
                if j + 1 < min_len:
                    try:
                        # Calculate intervals
                        interval1 = interval.Interval(notes1[j], notes2[j])
                        interval2 = interval.Interval(notes1[j + 1], notes2[j + 1])
                        
                        # Check for parallel fifths or octaves
                        if (interval1.simpleName == interval2.simpleName and 
                            interval1.simpleName in ['P5', 'P8']):
                            parallel_issues.append({
                                'type': f'parallel_{interval1.simpleName}',
                                'position': j,
                                'parts': [i, i + 1],
                                'notes': [
                                    str(notes1[j].pitch),
                                    str(notes2[j].pitch),
                                    str(notes1[j + 1].pitch),
                                    str(notes2[j + 1].pitch)
                                ]
                            })
                    except:
                        continue
        
        return parallel_issues

    def _detect_voice_crossings(self, score: stream.Score) -> List[Dict[str, Any]]:
        """Detect voice crossings"""
        voice_crossings = []
        
        # Get all parts
        parts = score.parts
        if len(parts) < 2:
            return voice_crossings
        
        # Check for voice crossings between adjacent parts
        for i in range(len(parts) - 1):
            part1 = parts[i]  # Higher part
            part2 = parts[i + 1]  # Lower part
            
            # Get notes from both parts
            notes1 = [n for n in part1.flatten().notes if isinstance(n, note.Note)]
            notes2 = [n for n in part2.flatten().notes if isinstance(n, note.Note)]
            
            # Check for crossings
            min_len = min(len(notes1), len(notes2))
            for j in range(min_len):
                try:
                    if notes1[j].pitch.midi < notes2[j].pitch.midi:
                        voice_crossings.append({
                            'position': j,
                            'higher_part': i,
                            'lower_part': i + 1,
                            'higher_note': str(notes1[j].pitch),
                            'lower_note': str(notes2[j].pitch)
                        })
                except:
                    continue
        
        return voice_crossings

    def _analyze_smoothness(self, score: stream.Score) -> Dict[str, Any]:
        """Analyze voice leading smoothness"""
        smoothness = {
            'total_motion': 0,
            'stepwise_motion': 0,
            'leap_motion': 0,
            'large_leap_motion': 0,
            'average_interval_size': 0.0,
            'smoothness_score': 0.0
        }
        
        # Analyze each part
        for part in score.parts:
            notes = [n for n in part.flatten().notes if isinstance(n, note.Note)]
            
            if len(notes) < 2:
                continue
            
            total_intervals = 0
            total_semitones = 0
            
            for i in range(len(notes) - 1):
                try:
                    intv = interval.Interval(notes[i], notes[i + 1])
                    semitones = abs(intv.semitones)
                    
                    total_intervals += 1
                    total_semitones += semitones
                    
                    if semitones <= 2:
                        smoothness['stepwise_motion'] += 1
                    elif semitones <= 4:
                        smoothness['leap_motion'] += 1
                    else:
                        smoothness['large_leap_motion'] += 1
                except:
                    continue
            
            smoothness['total_motion'] = total_intervals
            if total_intervals > 0:
                smoothness['average_interval_size'] = total_semitones / total_intervals
                
                # Calculate smoothness score (higher = smoother)
                stepwise_ratio = smoothness['stepwise_motion'] / total_intervals
                leap_ratio = smoothness['leap_motion'] / total_intervals
                large_leap_ratio = smoothness['large_leap_motion'] / total_intervals
                
                smoothness['smoothness_score'] = (
                    stepwise_ratio * 1.0 +
                    leap_ratio * 0.5 +
                    large_leap_ratio * 0.2
                ) * 100
        
        return smoothness