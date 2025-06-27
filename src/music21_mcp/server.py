"""
Music21 MCP Server - Core server implementation with comprehensive music analysis capabilities
"""
from mcp.server.fastmcp import FastMCP
from music21 import (
    stream, note, chord, key, meter, tempo, pitch, interval,
    corpus, converter, analysis, roman, scale, dynamics,
    articulations, expressions, instrument, midi
)
import json
import asyncio
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from collections import Counter
import logging
from pathlib import Path
import tempfile
import os
from datetime import datetime
import hashlib
from cachetools import TTLCache, LRUCache
import aiofiles
import httpx
from enum import Enum
import re
import chardet
import traceback
from concurrent.futures import ThreadPoolExecutor
import threading

# Import core analyzers
from .core.theory_analyzer import TheoryAnalyzer, KeyDetectionMethod
from .core.rhythm_analyzer import RhythmAnalyzer

# Configure comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastMCP server
mcp = FastMCP("Music21 Analysis & Composition Server")

# Score storage with proper state management
class ScoreManager:
    """Thread-safe score storage and management"""
    def __init__(self, max_scores: int = 100):
        self.scores: Dict[str, stream.Score] = {}
        self.metadata_cache: Dict[str, Dict[str, Any]] = {}
        self.analysis_cache = TTLCache(maxsize=1000, ttl=3600)  # 1 hour TTL
        self.file_cache = LRUCache(maxsize=50)  # Cache parsed files
        self.lock = threading.RLock()
        self.max_scores = max_scores
        
    def add_score(self, score_id: str, score: stream.Score, metadata: Dict[str, Any]):
        with self.lock:
            if len(self.scores) >= self.max_scores:
                # Remove oldest score
                oldest = min(self.metadata_cache.items(), 
                           key=lambda x: x[1].get('imported_at', datetime.now()))
                del self.scores[oldest[0]]
                del self.metadata_cache[oldest[0]]
            
            self.scores[score_id] = score
            self.metadata_cache[score_id] = metadata
    
    def get_score(self, score_id: str) -> Optional[stream.Score]:
        with self.lock:
            return self.scores.get(score_id)
    
    def get_metadata(self, score_id: str) -> Optional[Dict[str, Any]]:
        with self.lock:
            return self.metadata_cache.get(score_id)
    
    def list_scores(self) -> List[Dict[str, Any]]:
        with self.lock:
            return [
                {"score_id": sid, **self.metadata_cache.get(sid, {})}
                for sid in self.scores.keys()
            ]

# Global score manager
score_manager = ScoreManager()

# Thread pool for CPU-intensive operations
executor = ThreadPoolExecutor(max_workers=4)

class FormatType(Enum):
    """Supported music file formats"""
    MIDI = "midi"
    MUSICXML = "musicxml"
    ABC = "abc"
    KERN = "kern"
    LILYPOND = "lilypond"
    MEI = "mei"
    CAPELLA = "capella"
    MUSEDATA = "musedata"
    UNKNOWN = "unknown"

@mcp.tool()
async def import_score(
    score_id: str,
    source: Union[str, Dict[str, Any]],
    source_type: str = "auto",
    encoding: str = "auto",
    validate: bool = True,
    extract_parts: bool = True,
    preserve_layout: bool = True,
    progress_callback: Optional[str] = None
) -> Dict[str, Any]:
    """
    Import a musical score from various formats with comprehensive parsing.
    
    Args:
        score_id: Unique identifier for the score
        source: File path, URL, or direct music content
        source_type: Format type ('midi', 'musicxml', 'abc', 'kern', 'auto')
        encoding: Text encoding ('auto' for detection, or specific like 'utf-8')
        validate: Whether to perform comprehensive score validation
        extract_parts: Whether to extract individual parts
        preserve_layout: Whether to preserve original layout information
        progress_callback: Optional callback ID for progress updates
    
    Returns:
        Import status with comprehensive metadata and validation results
    """
    try:
        start_time = datetime.now()
        logger.info(f"Starting import of score '{score_id}' from {type(source).__name__}")
        
        # Progress tracking
        progress = {"status": "initializing", "percent": 0}
        if progress_callback:
            await _update_progress(progress_callback, progress)
        
        # Detect encoding if needed
        if encoding == "auto" and isinstance(source, (str, dict)):
            encoding = await _detect_encoding(source)
            logger.info(f"Detected encoding: {encoding}")
        
        # Auto-detect format if needed
        if source_type == "auto":
            source_type = await _detect_format(source)
            logger.info(f"Detected format: {source_type}")
        
        progress.update({"status": "parsing", "percent": 20})
        if progress_callback:
            await _update_progress(progress_callback, progress)
        
        # Parse the score based on source type
        score = await _parse_score(source, source_type, encoding, preserve_layout)
        
        if not score:
            raise ValueError("Failed to parse score")
        
        progress.update({"status": "analyzing", "percent": 60})
        if progress_callback:
            await _update_progress(progress_callback, progress)
        
        # Extract comprehensive metadata
        metadata = await _extract_comprehensive_metadata(score)
        metadata.update({
            "score_id": score_id,
            "source_type": source_type,
            "encoding": encoding,
            "imported_at": datetime.now().isoformat(),
            "file_hash": _calculate_hash(source) if isinstance(source, str) else None
        })
        
        # Extract parts if requested
        if extract_parts and len(score.parts) > 1:
            metadata["parts"] = await _extract_parts_metadata(score)
        
        progress.update({"status": "validating", "percent": 80})
        if progress_callback:
            await _update_progress(progress_callback, progress)
        
        # Perform validation if requested
        validation_results = {}
        if validate:
            validation_results = await _validate_score_comprehensive(score)
            metadata["validation"] = validation_results
        
        # Store the score
        score_manager.add_score(score_id, score, metadata)
        
        progress.update({"status": "complete", "percent": 100})
        if progress_callback:
            await _update_progress(progress_callback, progress)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return {
            "status": "success",
            "score_id": score_id,
            "metadata": metadata,
            "validation": validation_results,
            "processing_time": processing_time,
            "message": f"Score '{score_id}' imported successfully in {processing_time:.2f} seconds"
        }
        
    except Exception as e:
        logger.error(f"Error importing score {score_id}: {str(e)}\n{traceback.format_exc()}")
        return {
            "status": "error",
            "score_id": score_id,
            "message": str(e),
            "error_type": type(e).__name__,
            "traceback": traceback.format_exc() if logger.level <= logging.DEBUG else None
        }

@mcp.tool()
async def export_score(
    score_id: str,
    format: str,
    options: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Export a score to various formats with customization options.
    
    Args:
        score_id: ID of the score to export
        format: Target format ('midi', 'musicxml', 'abc', 'lilypond', 'png', 'svg', 'audio')
        options: Format-specific export options
    
    Returns:
        Export result with file path or content
    """
    try:
        score = score_manager.get_score(score_id)
        if not score:
            raise ValueError(f"Score '{score_id}' not found")
        
        options = options or {}
        
        # Handle different export formats
        if format.lower() == "midi":
            result = await _export_midi(score, options)
        elif format.lower() in ["musicxml", "xml"]:
            result = await _export_musicxml(score, options)
        elif format.lower() == "abc":
            result = await _export_abc(score, options)
        elif format.lower() in ["png", "svg"]:
            result = await _export_image(score, format.lower(), options)
        elif format.lower() == "audio":
            result = await _export_audio(score, options)
        elif format.lower() == "lilypond":
            result = await _export_lilypond(score, options)
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        return {
            "status": "success",
            "score_id": score_id,
            "format": format,
            "result": result,
            "message": f"Score exported successfully to {format}"
        }
        
    except Exception as e:
        logger.error(f"Error exporting score {score_id}: {str(e)}")
        return {"status": "error", "message": str(e)}

async def _detect_encoding(source: Union[str, Dict[str, Any]]) -> str:
    """Detect text encoding using chardet"""
    try:
        if isinstance(source, dict) and 'content' in source:
            content = source['content'].encode() if isinstance(source['content'], str) else source['content']
            result = chardet.detect(content)
            return result['encoding'] or 'utf-8'
        elif isinstance(source, str) and os.path.exists(source):
            async with aiofiles.open(source, 'rb') as f:
                content = await f.read(10000)  # Read first 10KB
                result = chardet.detect(content)
                return result['encoding'] or 'utf-8'
    except:
        pass
    return 'utf-8'

async def _detect_format(source: Union[str, Dict[str, Any]]) -> str:
    """Intelligently detect music file format"""
    # Check file extension first
    if isinstance(source, str):
        path = Path(source)
        ext = path.suffix.lower()
        
        ext_map = {
            '.mid': FormatType.MIDI,
            '.midi': FormatType.MIDI,
            '.xml': FormatType.MUSICXML,
            '.mxl': FormatType.MUSICXML,
            '.musicxml': FormatType.MUSICXML,
            '.abc': FormatType.ABC,
            '.krn': FormatType.KERN,
            '.ly': FormatType.LILYPOND,
            '.mei': FormatType.MEI,
            '.cap': FormatType.CAPELLA,
            '.capx': FormatType.CAPELLA,
            '.md': FormatType.MUSEDATA,
            '.mus': FormatType.MUSEDATA
        }
        
        if ext in ext_map:
            return ext_map[ext].value
    
    # Check content for format signatures
    content = None
    if isinstance(source, dict) and 'content' in source:
        content = source['content']
    elif isinstance(source, str) and os.path.exists(source):
        try:
            async with aiofiles.open(source, 'r', encoding='utf-8', errors='ignore') as f:
                content = await f.read(1000)  # Read first 1KB
        except:
            # Try binary read for MIDI
            async with aiofiles.open(source, 'rb') as f:
                binary_content = await f.read(4)
                if binary_content == b'MThd':
                    return FormatType.MIDI.value
    
    if content:
        # Check for format signatures in content
        if isinstance(content, str):
            if '<?xml' in content and ('<score-partwise' in content or '<score-timewise' in content):
                return FormatType.MUSICXML.value
            elif content.strip().startswith('X:'):
                return FormatType.ABC.value
            elif '**kern' in content:
                return FormatType.KERN.value
            elif '\\version' in content or '\\score' in content:
                return FormatType.LILYPOND.value
            elif '<mei' in content:
                return FormatType.MEI.value
    
    return FormatType.UNKNOWN.value

async def _parse_score(
    source: Union[str, Dict[str, Any]],
    source_type: str,
    encoding: str,
    preserve_layout: bool
) -> stream.Score:
    """Parse score with advanced options"""
    # Run parsing in thread pool to avoid blocking
    loop = asyncio.get_event_loop()
    
    def parse_sync():
        try:
            # Configure parser options
            if source_type == FormatType.MIDI.value:
                # Special MIDI parsing options
                converter.parse.keywords['quantizePost'] = False
                converter.parse.keywords['quarterLengthDivisors'] = [4, 3, 2, 1]
            
            if isinstance(source, str) and os.path.exists(source):
                # File path
                score = converter.parse(source, format=source_type if source_type != 'auto' else None)
            elif isinstance(source, str) and source.startswith(('http://', 'https://')):
                # URL - download first
                score = converter.parse(source)
            elif isinstance(source, dict) and 'content' in source:
                # Direct content
                with tempfile.NamedTemporaryFile(
                    mode='w', 
                    suffix=f'.{source_type}', 
                    encoding=encoding,
                    delete=False
                ) as f:
                    f.write(source['content'])
                    temp_path = f.name
                
                score = converter.parse(temp_path)
                os.unlink(temp_path)
            else:
                raise ValueError("Invalid source format")
            
            # Post-processing
            if preserve_layout and hasattr(score, 'layoutScore'):
                score.layoutScore()
            
            return score
            
        except Exception as e:
            logger.error(f"Parsing error: {str(e)}")
            raise
    
    return await loop.run_in_executor(executor, parse_sync)

async def _extract_comprehensive_metadata(score: stream.Score) -> Dict[str, Any]:
    """Extract all available metadata from a score"""
    metadata = {
        # Basic metadata
        "title": getattr(score.metadata, 'title', None),
        "composer": getattr(score.metadata, 'composer', None),
        "arranger": getattr(score.metadata, 'arranger', None),
        "copyright": getattr(score.metadata, 'copyright', None),
        "date": getattr(score.metadata, 'date', None),
        
        # Score structure
        "duration_quarters": float(score.duration.quarterLength),
        "duration_seconds": float(score.seconds) if hasattr(score, 'seconds') else None,
        "measure_count": len(score.getElementsByClass(stream.Measure)),
        "part_count": len(score.parts),
        
        # Musical elements
        "time_signatures": [
            {
                "measure": ts.measureNumber,
                "signature": str(ts),
                "numerator": ts.numerator,
                "denominator": ts.denominator,
                "beat_count": ts.beatCount,
                "beat_duration": float(ts.beatDuration.quarterLength)
            }
            for ts in score.getElementsByClass(meter.TimeSignature)
        ],
        
        "key_signatures": [
            {
                "measure": ks.measureNumber,
                "key": str(ks),
                "sharps": ks.sharps,
                "mode": getattr(ks, 'mode', None)
            }
            for ks in score.getElementsByClass(key.KeySignature)
        ],
        
        "tempo_markings": [
            {
                "measure": t.measureNumber,
                "tempo": str(t),
                "bpm": t.number if hasattr(t, 'number') else None,
                "text": getattr(t, 'text', None)
            }
            for t in score.getElementsByClass(tempo.TempoIndication)
        ],
        
        # Statistics
        "note_count": len(score.flat.notes),
        "rest_count": len(score.flat.rests),
        "chord_count": len(score.flat.getElementsByClass(chord.Chord)),
        
        # Pitch information
        "ambitus": await _calculate_ambitus(score),
        "pitch_histogram": await _calculate_pitch_histogram(score),
        
        # Dynamics and articulations
        "dynamics": [str(d) for d in score.flat.getElementsByClass(dynamics.Dynamic)],
        "articulations": list(set(
            str(a) for note in score.flat.notes 
            for a in note.articulations
        )),
        
        # Instruments
        "instruments": [
            {
                "part": i,
                "name": str(part.getInstrument()),
                "midi_program": part.getInstrument().midiProgram if part.getInstrument() else None,
                "range": {
                    "lowest": str(part.lowestOffset) if hasattr(part, 'lowestOffset') else None,
                    "highest": str(part.highestOffset) if hasattr(part, 'highestOffset') else None
                }
            }
            for i, part in enumerate(score.parts)
            if part.getInstrument()
        ]
    }
    
    return metadata

async def _calculate_ambitus(score: stream.Score) -> Dict[str, Any]:
    """Calculate the pitch range of the score"""
    pitches = score.flat.pitches
    if not pitches:
        return {"lowest": None, "highest": None, "range_semitones": 0}
    
    sorted_pitches = sorted(pitches, key=lambda p: p.ps)
    lowest = sorted_pitches[0]
    highest = sorted_pitches[-1]
    
    return {
        "lowest": str(lowest),
        "lowest_midi": lowest.midi,
        "highest": str(highest),
        "highest_midi": highest.midi,
        "range_semitones": highest.midi - lowest.midi,
        "range_interval": str(interval.Interval(lowest, highest))
    }

async def _calculate_pitch_histogram(score: stream.Score) -> Dict[str, int]:
    """Calculate pitch class histogram"""
    histogram = {}
    for p in score.flat.pitches:
        pc = p.pitchClass
        pc_name = p.pitchClassString
        histogram[pc_name] = histogram.get(pc_name, 0) + 1
    return histogram

async def _extract_parts_metadata(score: stream.Score) -> List[Dict[str, Any]]:
    """Extract metadata for each part"""
    parts_data = []
    
    for i, part in enumerate(score.parts):
        part_data = {
            "index": i,
            "id": part.id,
            "name": part.partName or f"Part {i+1}",
            "abbreviation": part.partAbbreviation,
            "instrument": str(part.getInstrument()) if part.getInstrument() else None,
            "measures": len(part.getElementsByClass(stream.Measure)),
            "notes": len(part.flat.notes),
            "duration": float(part.duration.quarterLength)
        }
        
        # Calculate part-specific ambitus
        if part.flat.pitches:
            part_data["ambitus"] = await _calculate_ambitus(part)
        
        parts_data.append(part_data)
    
    return parts_data

async def _validate_score_comprehensive(score: stream.Score) -> Dict[str, Any]:
    """Perform comprehensive score validation"""
    issues = []
    warnings = []
    info = []
    
    # Time signature validation
    time_sigs = score.getElementsByClass(meter.TimeSignature)
    if not time_sigs:
        warnings.append("No time signature found")
    else:
        # Check for time signature changes
        if len(set(str(ts) for ts in time_sigs)) > 1:
            info.append(f"Multiple time signatures found: {len(set(str(ts) for ts in time_sigs))} different")
    
    # Key signature validation
    key_sigs = score.getElementsByClass(key.KeySignature)
    if not key_sigs:
        warnings.append("No key signature found")
    
    # Part consistency validation
    parts = score.parts
    if len(parts) > 1:
        durations = [part.duration.quarterLength for part in parts]
        if not all(abs(d - durations[0]) < 0.1 for d in durations):
            issues.append(f"Parts have inconsistent durations: {durations}")
    
    # Measure validation
    for part in parts:
        measures = part.getElementsByClass(stream.Measure)
        for measure in measures:
            # Check measure completeness
            if hasattr(measure, 'timeSignature') and measure.timeSignature:
                expected_duration = measure.timeSignature.barDuration.quarterLength
                actual_duration = measure.duration.quarterLength
                if abs(expected_duration - actual_duration) > 0.01:
                    warnings.append(
                        f"Part {part.partName or 'Unknown'}, Measure {measure.number}: "
                        f"Expected duration {expected_duration}, got {actual_duration}"
                    )
    
    # Instrument range validation
    for i, part in enumerate(parts):
        instr = part.getInstrument()
        if instr and hasattr(instr, 'lowestNote') and hasattr(instr, 'highestNote'):
            part_pitches = part.flat.pitches
            if part_pitches:
                lowest_pitch = min(part_pitches, key=lambda p: p.ps)
                highest_pitch = max(part_pitches, key=lambda p: p.ps)
                
                if lowest_pitch.ps < instr.lowestNote.ps:
                    warnings.append(
                        f"Part {i+1} ({instr}): Note {lowest_pitch} is below instrument range"
                    )
                if highest_pitch.ps > instr.highestNote.ps:
                    warnings.append(
                        f"Part {i+1} ({instr}): Note {highest_pitch} is above instrument range"
                    )
    
    # Performance feasibility
    for part in parts:
        # Check for impossible intervals
        notes = list(part.flat.notes)
        for i in range(len(notes) - 1):
            if notes[i].offset + notes[i].duration.quarterLength > notes[i+1].offset:
                # Overlapping notes - check interval
                intv = interval.Interval(notes[i], notes[i+1])
                if intv.semitones > 12:  # More than an octave
                    warnings.append(
                        f"Large interval ({intv}) between overlapping notes at offset {notes[i].offset}"
                    )
    
    return {
        "is_valid": len(issues) == 0,
        "issues": issues,
        "warnings": warnings,
        "info": info,
        "score_health": "healthy" if len(issues) == 0 else "invalid",
        "completeness": _calculate_completeness_score(issues, warnings)
    }

def _calculate_completeness_score(issues: List[str], warnings: List[str]) -> float:
    """Calculate a completeness score from 0 to 1"""
    # Start with perfect score
    score = 1.0
    
    # Deduct for issues (more severe)
    score -= len(issues) * 0.1
    
    # Deduct for warnings (less severe)
    score -= len(warnings) * 0.05
    
    # Ensure score stays between 0 and 1
    return max(0.0, min(1.0, score))

async def _export_midi(score: stream.Score, options: Dict[str, Any]) -> Dict[str, Any]:
    """Export score to MIDI with advanced options"""
    try:
        # MIDI export options
        velocity_map = options.get('velocity_map', 'default')
        tempo_changes = options.get('preserve_tempo_changes', True)
        program_changes = options.get('preserve_program_changes', True)
        
        # Create MIDI file
        mf = midi.translate.music21ObjectToMidiFile(score)
        
        # Apply velocity curve if specified
        if velocity_map == 'expressive':
            # Apply expressive velocity mapping
            for track in mf.tracks:
                for event in track.events:
                    if hasattr(event, 'velocity') and event.velocity:
                        # Apply expressive curve
                        event.velocity = int(event.velocity * 1.2)
                        event.velocity = max(1, min(127, event.velocity))
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.mid', delete=False) as f:
            mf.open(f.name, 'wb')
            mf.write()
            mf.close()
            
            # Read file content
            async with aiofiles.open(f.name, 'rb') as rf:
                content = await rf.read()
            
            os.unlink(f.name)
            
            return {
                "format": "midi",
                "content": content.hex(),  # Return as hex string
                "content_type": "application/octet-stream",
                "filename": f"{score.metadata.title or 'score'}.mid"
            }
            
    except Exception as e:
        logger.error(f"MIDI export error: {str(e)}")
        raise

async def _export_musicxml(score: stream.Score, options: Dict[str, Any]) -> Dict[str, Any]:
    """Export score to MusicXML with formatting options"""
    try:
        # MusicXML export options
        compress = options.get('compress', False)
        pretty_print = options.get('pretty_print', True)
        
        # Export to MusicXML
        if compress:
            suffix = '.mxl'
            content = score.write('musicxml.mxl', fp=None)
        else:
            suffix = '.xml'
            content = score.write('musicxml', fp=None)
        
        return {
            "format": "musicxml",
            "content": content,
            "content_type": "application/xml" if not compress else "application/octet-stream",
            "filename": f"{score.metadata.title or 'score'}{suffix}"
        }
        
    except Exception as e:
        logger.error(f"MusicXML export error: {str(e)}")
        raise

async def _export_abc(score: stream.Score, options: Dict[str, Any]) -> Dict[str, Any]:
    """Export score to ABC notation"""
    try:
        # ABC export options
        reference_number = options.get('reference_number', 1)
        
        # Export to ABC
        content = score.write('abc', fp=None)
        
        return {
            "format": "abc",
            "content": content,
            "content_type": "text/plain",
            "filename": f"{score.metadata.title or 'score'}.abc"
        }
        
    except Exception as e:
        logger.error(f"ABC export error: {str(e)}")
        raise

async def _export_image(score: stream.Score, format: str, options: Dict[str, Any]) -> Dict[str, Any]:
    """Export score to image format (PNG/SVG)"""
    try:
        # Image export options
        dpi = options.get('dpi', 300)
        transparent = options.get('transparent', False)
        
        # Export via LilyPond
        if format == 'png':
            fp = score.write('lily.png', fp=None, dpi=dpi)
        else:  # svg
            fp = score.write('lily.svg', fp=None)
        
        return {
            "format": format,
            "file_path": fp,
            "content_type": f"image/{format}",
            "filename": f"{score.metadata.title or 'score'}.{format}"
        }
        
    except Exception as e:
        logger.error(f"Image export error: {str(e)}")
        raise

async def _export_audio(score: stream.Score, options: Dict[str, Any]) -> Dict[str, Any]:
    """Export score to audio file"""
    try:
        # Audio export options
        soundfont = options.get('soundfont', 'default')
        format = options.get('format', 'mp3')
        
        # First export to MIDI
        mf = midi.translate.music21ObjectToMidiFile(score)
        
        with tempfile.NamedTemporaryFile(suffix='.mid', delete=False) as f:
            mf.open(f.name, 'wb')
            mf.write()
            mf.close()
            midi_path = f.name
        
        # TODO: Implement MIDI to audio conversion
        # This would require additional dependencies like FluidSynth
        
        os.unlink(midi_path)
        
        return {
            "format": format,
            "message": "Audio export requires additional setup",
            "content_type": f"audio/{format}"
        }
        
    except Exception as e:
        logger.error(f"Audio export error: {str(e)}")
        raise

async def _export_lilypond(score: stream.Score, options: Dict[str, Any]) -> Dict[str, Any]:
    """Export score to LilyPond format"""
    try:
        # Export to LilyPond
        content = score.write('lily', fp=None)
        
        return {
            "format": "lilypond",
            "content": content,
            "content_type": "text/plain",
            "filename": f"{score.metadata.title or 'score'}.ly"
        }
        
    except Exception as e:
        logger.error(f"LilyPond export error: {str(e)}")
        raise

async def _update_progress(callback_id: str, progress: Dict[str, Any]):
    """Update progress for long-running operations"""
    # In a real implementation, this would send progress updates
    # to the client via the callback mechanism
    logger.info(f"Progress [{callback_id}]: {progress}")

def _calculate_hash(source: str) -> str:
    """Calculate file hash for caching"""
    if os.path.exists(source):
        hasher = hashlib.sha256()
        with open(source, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                hasher.update(chunk)
        return hasher.hexdigest()
    return None

# Additional helper tools

@mcp.tool()
async def list_scores() -> Dict[str, Any]:
    """List all currently loaded scores"""
    try:
        scores = score_manager.list_scores()
        return {
            "status": "success",
            "count": len(scores),
            "scores": scores
        }
    except Exception as e:
        logger.error(f"Error listing scores: {str(e)}")
        return {"status": "error", "message": str(e)}

@mcp.tool()
async def get_score_info(score_id: str) -> Dict[str, Any]:
    """Get detailed information about a specific score"""
    try:
        metadata = score_manager.get_metadata(score_id)
        if not metadata:
            raise ValueError(f"Score '{score_id}' not found")
        
        return {
            "status": "success",
            "score_id": score_id,
            "metadata": metadata
        }
    except Exception as e:
        logger.error(f"Error getting score info: {str(e)}")
        return {"status": "error", "message": str(e)}

@mcp.tool()
async def delete_score(score_id: str) -> Dict[str, Any]:
    """Remove a score from memory"""
    try:
        with score_manager.lock:
            if score_id in score_manager.scores:
                del score_manager.scores[score_id]
                del score_manager.metadata_cache[score_id]
                return {
                    "status": "success",
                    "message": f"Score '{score_id}' deleted successfully"
                }
            else:
                raise ValueError(f"Score '{score_id}' not found")
    except Exception as e:
        logger.error(f"Error deleting score: {str(e)}")
        return {"status": "error", "message": str(e)}

# Initialize analyzers
theory_analyzer = TheoryAnalyzer()
rhythm_analyzer = RhythmAnalyzer()

# Theory Analysis Tools

@mcp.tool()
async def analyze_key(
    score_id: str,
    method: str = "hybrid",
    window_size: Optional[int] = None,
    confidence_threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Perform comprehensive key analysis on a score.
    
    Args:
        score_id: ID of the score to analyze
        method: Detection method ('krumhansl', 'aarden', 'bellman', 'temperley', 'simple', 'hybrid')
        window_size: Window size for local key analysis (measures)
        confidence_threshold: Minimum confidence for key detection
    
    Returns:
        Comprehensive key analysis including global key, local keys, modulations, and evidence
    """
    try:
        score = score_manager.get_score(score_id)
        if not score:
            raise ValueError(f"Score '{score_id}' not found")
        
        # Convert method string to enum
        method_enum = KeyDetectionMethod(method.lower())
        
        # Perform analysis
        result = await theory_analyzer.analyze_key(
            score, method_enum, window_size, confidence_threshold
        )
        
        return {
            "status": "success",
            "score_id": score_id,
            "key": str(result.key),
            "confidence": result.confidence,
            "method": result.method,
            "alternatives": [
                {"key": str(k), "confidence": c} for k, c in result.alternatives
            ],
            "local_keys": result.local_keys,
            "modulations": result.modulations,
            "evidence": result.evidence
        }
        
    except Exception as e:
        logger.error(f"Error analyzing key for {score_id}: {str(e)}")
        return {"status": "error", "message": str(e)}

@mcp.tool()
async def analyze_scale(
    score_id: str,
    part_index: Optional[int] = None,
    include_modes: bool = True,
    include_exotic: bool = False
) -> Dict[str, Any]:
    """
    Analyze scale patterns in a score or specific part.
    
    Args:
        score_id: ID of the score to analyze
        part_index: Optional index of specific part to analyze
        include_modes: Include modal scale detection
        include_exotic: Include exotic scales (pentatonic, blues, etc.)
    
    Returns:
        Scale analysis with possible scales and confidence scores
    """
    try:
        score = score_manager.get_score(score_id)
        if not score:
            raise ValueError(f"Score '{score_id}' not found")
        
        # Get specific part or whole score
        if part_index is not None and part_index < len(score.parts):
            analyze_stream = score.parts[part_index]
        else:
            analyze_stream = score
        
        # Perform analysis
        result = await theory_analyzer.analyze_scale(
            analyze_stream, include_modes, include_exotic
        )
        
        return {
            "status": "success",
            "score_id": score_id,
            "part_analyzed": part_index if part_index is not None else "all",
            **result
        }
        
    except Exception as e:
        logger.error(f"Error analyzing scale for {score_id}: {str(e)}")
        return {"status": "error", "message": str(e)}

@mcp.tool()
async def analyze_intervals(
    score_id: str,
    measure: int,
    note1_index: int,
    note2_index: int
) -> Dict[str, Any]:
    """
    Analyze the interval between two specific notes.
    
    Args:
        score_id: ID of the score
        measure: Measure number
        note1_index: Index of first note in measure
        note2_index: Index of second note in measure
    
    Returns:
        Detailed interval analysis
    """
    try:
        score = score_manager.get_score(score_id)
        if not score:
            raise ValueError(f"Score '{score_id}' not found")
        
        # Find the notes
        measures = score.getElementsByClass(stream.Measure)
        if measure > len(measures):
            raise ValueError(f"Measure {measure} not found")
        
        target_measure = measures[measure - 1]
        notes = list(target_measure.notes)
        
        if note1_index >= len(notes) or note2_index >= len(notes):
            raise ValueError("Note index out of range")
        
        note1 = notes[note1_index]
        note2 = notes[note2_index]
        
        # Analyze interval
        analysis = await theory_analyzer.analyze_intervals(note1, note2, detailed=True)
        
        return {
            "status": "success",
            "score_id": score_id,
            "interval": str(analysis.interval),
            "quality": analysis.quality,
            "size": analysis.size,
            "semitones": analysis.semitones,
            "cents": analysis.cents,
            "consonance": analysis.consonance,
            "enharmonic_equivalent": analysis.enharmonic_equivalent,
            "compound": analysis.compound,
            "inverted": str(analysis.inverted) if analysis.inverted else None
        }
        
    except Exception as e:
        logger.error(f"Error analyzing interval: {str(e)}")
        return {"status": "error", "message": str(e)}

@mcp.tool()
async def analyze_chord_progressions(
    score_id: str,
    analysis_type: str = "roman",
    simplify: bool = True,
    include_inversions: bool = True
) -> Dict[str, Any]:
    """
    Analyze chord progressions in a score.
    
    Args:
        score_id: ID of the score to analyze
        analysis_type: Type of analysis ('roman', 'functional', 'jazz')
        simplify: Whether to simplify complex chords
        include_inversions: Whether to include inversion information
    
    Returns:
        Chord progression analysis with Roman numerals and harmonic functions
    """
    try:
        score = score_manager.get_score(score_id)
        if not score:
            raise ValueError(f"Score '{score_id}' not found")
        
        # Get chords
        chords = list(score.flatten().getElementsByClass(chord.Chord))
        
        if not chords:
            return {
                "status": "success",
                "score_id": score_id,
                "message": "No chords found in score",
                "chord_count": 0
            }
        
        # Detect key for Roman numeral analysis
        key_obj = score.analyze('key')
        
        progression = []
        for i, c in enumerate(chords):
            chord_info = {
                "index": i,
                "measure": c.measureNumber,
                "beat": c.beat,
                "chord": c.pitchesChr,
                "root": str(c.root()) if c.root() else None,
                "quality": c.quality,
                "inversion": c.inversion() if include_inversions else None
            }
            
            # Add Roman numeral analysis
            if analysis_type == "roman" and key_obj:
                try:
                    rn = roman.romanNumeralFromChord(c, key_obj)
                    chord_info["roman_numeral"] = str(rn.romanNumeralAlone)
                    chord_info["scale_degree"] = rn.scaleDegree
                    chord_info["function"] = theory_analyzer.FUNCTION_MAP.get(rn.scaleDegree, "unknown")
                except:
                    chord_info["roman_numeral"] = "N/A"
            
            # Add jazz symbols
            elif analysis_type == "jazz":
                jazz_analysis = await theory_analyzer.analyze_chord_quality(c)
                chord_info["jazz_symbol"] = jazz_analysis["jazz_symbol"]
                chord_info["extensions"] = jazz_analysis.get("extensions", [])
            
            progression.append(chord_info)
        
        # Identify common progressions
        progression_patterns = []
        if analysis_type == "roman" and len(progression) > 1:
            for i in range(len(progression) - 1):
                if "roman_numeral" in progression[i] and "roman_numeral" in progression[i+1]:
                    pattern = f"{progression[i]['roman_numeral']} - {progression[i+1]['roman_numeral']}"
                    progression_patterns.append(pattern)
        
        return {
            "status": "success",
            "score_id": score_id,
            "key": str(key_obj) if key_obj else None,
            "chord_count": len(chords),
            "analysis_type": analysis_type,
            "progression": progression,
            "common_patterns": Counter(progression_patterns).most_common(5) if progression_patterns else []
        }
        
    except Exception as e:
        logger.error(f"Error analyzing chord progressions: {str(e)}")
        return {"status": "error", "message": str(e)}

# Rhythm Analysis Tools

@mcp.tool()
async def analyze_rhythm(
    score_id: str,
    include_patterns: bool = True,
    pattern_min_length: int = 2,
    pattern_min_occurrences: int = 3
) -> Dict[str, Any]:
    """
    Perform comprehensive rhythm analysis on a score.
    
    Args:
        score_id: ID of the score to analyze
        include_patterns: Whether to search for rhythmic patterns
        pattern_min_length: Minimum pattern length to consider
        pattern_min_occurrences: Minimum pattern occurrences
    
    Returns:
        Complete rhythm analysis including tempo, meter, patterns, complexity, and groove
    """
    try:
        score = score_manager.get_score(score_id)
        if not score:
            raise ValueError(f"Score '{score_id}' not found")
        
        # Perform analysis
        result = await rhythm_analyzer.analyze_rhythm(
            score, include_patterns, pattern_min_length, pattern_min_occurrences
        )
        
        # Format patterns for JSON
        patterns = []
        for p in result.patterns[:10]:  # Limit to top 10
            patterns.append({
                "pattern": p.pattern,
                "occurrences": p.occurrences,
                "locations": p.locations[:10],  # Limit locations
                "confidence": p.confidence,
                "type": p.pattern_type,
                "is_ostinato": p.is_ostinato
            })
        
        return {
            "status": "success",
            "score_id": score_id,
            "tempo": {
                "primary_bpm": result.tempo.primary_tempo,
                "character": result.tempo.tempo_character,
                "stability": result.tempo.tempo_stability,
                "variance": result.tempo.tempo_variance,
                "rubato_likelihood": result.tempo.rubato_likelihood,
                "tempo_changes": result.tempo.tempo_changes
            },
            "meter": {
                "primary": str(result.meter.primary_meter),
                "complexity": result.meter.metric_complexity,
                "is_compound": result.meter.is_compound,
                "is_asymmetric": result.meter.is_asymmetric,
                "is_mixed": result.meter.is_mixed_meter,
                "stability": result.meter.meter_stability,
                "changes": result.meter.meter_changes
            },
            "patterns": patterns,
            "complexity": result.complexity.value,
            "syncopation_level": result.syncopation_level,
            "groove": result.groove_analysis,
            "polyrhythms": result.polyrhythms,
            "rhythm_histogram": result.rhythm_histogram
        }
        
    except Exception as e:
        logger.error(f"Error analyzing rhythm for {score_id}: {str(e)}")
        return {"status": "error", "message": str(e)}

@mcp.tool()
async def analyze_tempo(
    score_id: str
) -> Dict[str, Any]:
    """
    Analyze tempo characteristics of a score.
    
    Args:
        score_id: ID of the score to analyze
    
    Returns:
        Detailed tempo analysis
    """
    try:
        score = score_manager.get_score(score_id)
        if not score:
            raise ValueError(f"Score '{score_id}' not found")
        
        # Use rhythm analyzer for tempo analysis
        tempo_analysis = await rhythm_analyzer._analyze_tempo(score)
        
        return {
            "status": "success",
            "score_id": score_id,
            "primary_tempo": tempo_analysis.primary_tempo,
            "average_tempo": tempo_analysis.average_tempo,
            "tempo_character": tempo_analysis.tempo_character,
            "tempo_markings": tempo_analysis.tempo_markings,
            "tempo_changes": tempo_analysis.tempo_changes,
            "tempo_stability": tempo_analysis.tempo_stability,
            "suggested_range": {
                "min": tempo_analysis.suggested_tempo_range[0],
                "max": tempo_analysis.suggested_tempo_range[1]
            },
            "rubato_likelihood": tempo_analysis.rubato_likelihood
        }
        
    except Exception as e:
        logger.error(f"Error analyzing tempo: {str(e)}")
        return {"status": "error", "message": str(e)}

@mcp.tool()
async def find_rhythmic_patterns(
    score_id: str,
    part_index: Optional[int] = None,
    min_length: int = 2,
    min_occurrences: int = 3,
    pattern_type: Optional[str] = None
) -> Dict[str, Any]:
    """
    Find specific rhythmic patterns in a score.
    
    Args:
        score_id: ID of the score to analyze
        part_index: Optional specific part to analyze
        min_length: Minimum pattern length
        min_occurrences: Minimum occurrences
        pattern_type: Optional specific pattern type to search for
    
    Returns:
        Found rhythmic patterns with locations
    """
    try:
        score = score_manager.get_score(score_id)
        if not score:
            raise ValueError(f"Score '{score_id}' not found")
        
        # Get specific part if requested
        if part_index is not None and part_index < len(score.parts):
            analyze_score = stream.Score()
            analyze_score.append(score.parts[part_index])
        else:
            analyze_score = score
        
        # Extract patterns
        patterns = await rhythm_analyzer._extract_rhythmic_patterns(
            analyze_score, min_length, min_occurrences
        )
        
        # Filter by type if specified
        if pattern_type:
            patterns = [p for p in patterns if p.pattern_type == pattern_type]
        
        # Format results
        results = []
        for pattern in patterns[:20]:  # Limit to 20 patterns
            results.append({
                "pattern": pattern.pattern,
                "pattern_notation": _rhythm_to_notation(pattern.pattern),
                "type": pattern.pattern_type,
                "occurrences": pattern.occurrences,
                "measures": pattern.locations[:20],  # Limit locations
                "confidence": pattern.confidence,
                "is_ostinato": pattern.is_ostinato
            })
        
        return {
            "status": "success",
            "score_id": score_id,
            "part_analyzed": part_index if part_index is not None else "all",
            "patterns_found": len(results),
            "patterns": results
        }
        
    except Exception as e:
        logger.error(f"Error finding rhythmic patterns: {str(e)}")
        return {"status": "error", "message": str(e)}

def _rhythm_to_notation(durations: List[float]) -> str:
    """Convert duration list to readable notation"""
    notation_map = {
        4.0: "𝅝",      # whole note
        2.0: "𝅗𝅥",     # half note
        1.0: "♩",      # quarter note
        0.5: "♪",      # eighth note
        0.25: "♬",     # sixteenth note
        0.125: "𝅘𝅥𝅯",   # thirty-second note
    }
    
    notation = []
    for dur in durations:
        # Find closest standard duration
        closest = min(notation_map.keys(), key=lambda x: abs(x - dur))
        if abs(closest - dur) < 0.05:
            notation.append(notation_map[closest])
        else:
            notation.append(f"[{dur}]")
    
    return " ".join(notation)

# Week 2: Music Theory Analysis Engine Tools

@mcp.tool()
async def identify_scale(
    score_id: str,
    start_measure: Optional[int] = None,
    end_measure: Optional[int] = None,
    confidence_threshold: float = 0.7
) -> Dict[str, Any]:
    """
    Identify scales used in melodies or passages.
    
    Args:
        score_id: ID of the score to analyze
        start_measure: Start measure for analysis (optional)
        end_measure: End measure for analysis (optional)
        confidence_threshold: Minimum confidence for scale detection
    
    Returns:
        Detected scales with confidence scores
    """
    try:
        score = score_manager.get_score(score_id)
        if not score:
            raise ValueError(f"Score '{score_id}' not found")
        
        # Extract passage if specified
        if start_measure and end_measure:
            measures = score.getElementsByClass(stream.Measure)
            passage = stream.Stream()
            for m in measures[start_measure-1:end_measure]:
                passage.append(m)
            analyze_stream = passage
        else:
            analyze_stream = score
        
        # Use theory analyzer
        result = await theory_analyzer.analyze_scale(
            analyze_stream, 
            include_modes=True,
            include_exotic=True
        )
        
        # Filter by confidence
        filtered_scales = [
            s for s in result.get('possible_scales', [])
            if s['match_score'] >= confidence_threshold
        ]
        
        return {
            "status": "success",
            "score_id": score_id,
            "measure_range": f"{start_measure or 1}-{end_measure or 'end'}",
            "detected_scales": filtered_scales,
            "best_match": result.get('best_match'),
            "pitch_content": result.get('pitch_classes', [])
        }
        
    except Exception as e:
        logger.error(f"Error identifying scale: {str(e)}")
        return {"status": "error", "message": str(e)}

@mcp.tool()
async def interval_vector(
    score_id: str,
    start_measure: Optional[int] = None,
    end_measure: Optional[int] = None
) -> Dict[str, Any]:
    """
    Calculate interval class vectors for sections.
    
    Args:
        score_id: ID of the score to analyze
        start_measure: Start measure (optional)
        end_measure: End measure (optional)
    
    Returns:
        Interval vector analysis with consonance metrics
    """
    try:
        score = score_manager.get_score(score_id)
        if not score:
            raise ValueError(f"Score '{score_id}' not found")
        
        # Import advanced analyzer
        from .core.advanced_theory import AdvancedTheoryAnalyzer
        adv_analyzer = AdvancedTheoryAnalyzer()
        
        # Extract section
        if start_measure and end_measure:
            measures = list(score.getElementsByClass(stream.Measure))
            section = stream.Stream()
            for m in measures[start_measure-1:end_measure]:
                section.append(m)
            analyze_stream = section
        else:
            analyze_stream = score
        
        # Calculate interval vector
        result = await adv_analyzer.calculate_interval_vector(analyze_stream)
        
        return {
            "status": "success",
            "score_id": score_id,
            "measure_range": f"{start_measure or 1}-{end_measure or 'end'}",
            "interval_vector": result.interval_vector,
            "total_intervals": result.total_intervals,
            "consonance_ratio": result.consonance_ratio,
            "tritone_count": result.tritone_count,
            "common_intervals": result.common_intervals,
            "z_relation": result.z_relation
        }
        
    except Exception as e:
        logger.error(f"Error calculating interval vector: {str(e)}")
        return {"status": "error", "message": str(e)}

@mcp.tool()
async def chromatic_analysis(
    score_id: str,
    include_voice_leading: bool = True
) -> Dict[str, Any]:
    """
    Identify chromatic passages and their functions.
    
    Args:
        score_id: ID of the score to analyze
        include_voice_leading: Include voice leading analysis
    
    Returns:
        Chromatic analysis with functional classifications
    """
    try:
        score = score_manager.get_score(score_id)
        if not score:
            raise ValueError(f"Score '{score_id}' not found")
        
        # Import advanced analyzer
        from .core.advanced_theory import AdvancedTheoryAnalyzer
        adv_analyzer = AdvancedTheoryAnalyzer()
        
        # Get key context
        key_context = score.analyze('key')
        
        # Analyze chromatic elements
        result = await adv_analyzer.analyze_chromatic_elements(score, key_context)
        
        response = {
            "status": "success",
            "score_id": score_id,
            "key_context": str(key_context),
            "chromatic_density": result.chromatic_density,
            "chromatic_functions": result.chromatic_functions,
            "modal_mixture_chords": result.modal_mixture_chords[:10],  # Limit output
            "chromatic_notes_count": len(result.chromatic_notes)
        }
        
        if include_voice_leading:
            response["voice_leading"] = result.chromatic_voice_leading[:10]  # Limit
        
        return response
        
    except Exception as e:
        logger.error(f"Error in chromatic analysis: {str(e)}")
        return {"status": "error", "message": str(e)}

@mcp.tool()
async def secondary_dominants(
    score_id: str
) -> Dict[str, Any]:
    """
    Identify secondary dominant chords and tonicization patterns.
    
    Args:
        score_id: ID of the score to analyze
    
    Returns:
        Secondary dominants with target degrees and resolutions
    """
    try:
        score = score_manager.get_score(score_id)
        if not score:
            raise ValueError(f"Score '{score_id}' not found")
        
        # Import advanced analyzer
        from .core.advanced_theory import AdvancedTheoryAnalyzer
        adv_analyzer = AdvancedTheoryAnalyzer()
        
        # Get key context
        key_context = score.analyze('key')
        
        # Detect advanced harmony
        harmony_result = await adv_analyzer.detect_advanced_harmony(score, key_context)
        
        return {
            "status": "success",
            "score_id": score_id,
            "key_context": str(key_context),
            "secondary_dominants": harmony_result.secondary_dominants,
            "count": len(harmony_result.secondary_dominants),
            "tonicized_degrees": list(set(
                sd['target_degree'] for sd in harmony_result.secondary_dominants
            ))
        }
        
    except Exception as e:
        logger.error(f"Error detecting secondary dominants: {str(e)}")
        return {"status": "error", "message": str(e)}

@mcp.tool()
async def phrase_structure(
    score_id: str,
    include_motives: bool = True
) -> Dict[str, Any]:
    """
    Analyze musical phrase structure (period, sentence, etc).
    
    Args:
        score_id: ID of the score to analyze
        include_motives: Include motivic analysis
    
    Returns:
        Phrase structure analysis with cadences and form
    """
    try:
        score = score_manager.get_score(score_id)
        if not score:
            raise ValueError(f"Score '{score_id}' not found")
        
        # Import advanced analyzer
        from .core.advanced_theory import AdvancedTheoryAnalyzer
        adv_analyzer = AdvancedTheoryAnalyzer()
        
        # Analyze phrase structure
        result = await adv_analyzer.analyze_phrase_structure(score, include_motives)
        
        return {
            "status": "success",
            "score_id": score_id,
            "phrase_type": result.phrase_type.value,
            "phrase_lengths": result.phrase_lengths,
            "cadences": result.cadences,
            "hypermetric_structure": result.hypermetric_structure,
            "elisions": result.elisions,
            "phrase_rhythm": result.phrase_rhythm,
            "motivic_analysis": result.motivic_analysis if include_motives else None
        }
        
    except Exception as e:
        logger.error(f"Error analyzing phrase structure: {str(e)}")
        return {"status": "error", "message": str(e)}

# Week 3: Advanced Rhythm Tools

@mcp.tool()
async def beat_strength(
    score_id: str,
    part_index: Optional[int] = None
) -> Dict[str, Any]:
    """
    Calculate metric accent patterns and beat strength.
    
    Args:
        score_id: ID of the score to analyze
        part_index: Specific part to analyze (optional)
    
    Returns:
        Beat strength analysis with metric accents
    """
    try:
        score = score_manager.get_score(score_id)
        if not score:
            raise ValueError(f"Score '{score_id}' not found")
        
        # Get time signature
        time_sigs = score.getElementsByClass(meter.TimeSignature)
        if not time_sigs:
            return {"status": "error", "message": "No time signature found"}
        
        primary_ts = time_sigs[0]
        
        # Analyze beat strength for each measure
        beat_analysis = []
        measures = list(score.getElementsByClass(stream.Measure))[:10]  # Limit to first 10
        
        for measure in measures:
            measure_beats = []
            for beat_num in range(1, int(primary_ts.numerator) + 1):
                # Find notes on this beat
                notes_on_beat = []
                for n in measure.notes:
                    if isinstance(n, (note.Note, chord.Chord)) and int(n.beat) == beat_num:
                        notes_on_beat.append(n)
                
                # Calculate strength based on meter
                if beat_num == 1:
                    strength = 1.0  # Downbeat
                elif beat_num == (primary_ts.numerator + 1) // 2:
                    strength = 0.7  # Mid-bar accent
                else:
                    strength = 0.3  # Weak beats
                
                measure_beats.append({
                    "beat": beat_num,
                    "strength": strength,
                    "notes": len(notes_on_beat),
                    "has_accent": any(
                        isinstance(a, articulations.Accent) 
                        for n in notes_on_beat 
                        for a in n.articulations
                    )
                })
            
            beat_analysis.append({
                "measure": measure.measureNumber,
                "beats": measure_beats
            })
        
        return {
            "status": "success",
            "score_id": score_id,
            "time_signature": str(primary_ts),
            "beat_pattern": rhythm_analyzer._analyze_beat_hierarchy(primary_ts),
            "measure_analysis": beat_analysis
        }
        
    except Exception as e:
        logger.error(f"Error analyzing beat strength: {str(e)}")
        return {"status": "error", "message": str(e)}

# Week 4: Integration & Unified Analysis

@mcp.tool()
async def comprehensive_analysis(
    score_id: str,
    include_advanced: bool = True
) -> Dict[str, Any]:
    """
    Run all Phase 1 analyses on a score.
    
    Args:
        score_id: ID of the score to analyze
        include_advanced: Include advanced theory analysis
    
    Returns:
        Comprehensive analysis results from all modules
    """
    try:
        score = score_manager.get_score(score_id)
        if not score:
            raise ValueError(f"Score '{score_id}' not found")
        
        results = {
            "status": "success",
            "score_id": score_id,
            "metadata": score_manager.get_metadata(score_id),
            "analyses": {}
        }
        
        # Basic theory analysis
        try:
            key_result = await analyze_key(score_id, method="hybrid")
            results["analyses"]["key"] = key_result
        except Exception as e:
            logger.error(f"Key analysis failed: {e}")
            results["analyses"]["key"] = {"error": str(e)}
        
        # Scale analysis
        try:
            scale_result = await analyze_scale(score_id, include_modes=True)
            results["analyses"]["scale"] = scale_result
        except Exception as e:
            logger.error(f"Scale analysis failed: {e}")
            results["analyses"]["scale"] = {"error": str(e)}
        
        # Chord progression
        try:
            chord_result = await analyze_chord_progressions(score_id)
            results["analyses"]["harmony"] = chord_result
        except Exception as e:
            logger.error(f"Chord analysis failed: {e}")
            results["analyses"]["harmony"] = {"error": str(e)}
        
        # Rhythm analysis
        try:
            rhythm_result = await analyze_rhythm(score_id, include_patterns=True)
            results["analyses"]["rhythm"] = rhythm_result
        except Exception as e:
            logger.error(f"Rhythm analysis failed: {e}")
            results["analyses"]["rhythm"] = {"error": str(e)}
        
        # Advanced analyses if requested
        if include_advanced:
            # Chromatic analysis
            try:
                chromatic_result = await chromatic_analysis(score_id)
                results["analyses"]["chromatic"] = chromatic_result
            except Exception as e:
                logger.error(f"Chromatic analysis failed: {e}")
                results["analyses"]["chromatic"] = {"error": str(e)}
            
            # Phrase structure
            try:
                phrase_result = await phrase_structure(score_id, include_motives=False)
                results["analyses"]["phrase_structure"] = phrase_result
            except Exception as e:
                logger.error(f"Phrase analysis failed: {e}")
                results["analyses"]["phrase_structure"] = {"error": str(e)}
        
        # Calculate analysis time
        results["analysis_time"] = datetime.now().isoformat()
        
        return results
        
    except Exception as e:
        logger.error(f"Error in comprehensive analysis: {str(e)}")
        return {"status": "error", "message": str(e)}

@mcp.tool()
async def batch_analysis(
    score_ids: List[str],
    analysis_types: List[str],
    parallel: bool = False
) -> Dict[str, Any]:
    """
    Analyze multiple scores in sequence or parallel.
    
    Args:
        score_ids: List of score IDs to analyze
        analysis_types: Types of analysis to run ('key', 'scale', 'rhythm', 'harmony')
        parallel: Run analyses in parallel (experimental)
    
    Returns:
        Batch analysis results
    """
    try:
        results = {
            "status": "success",
            "total_scores": len(score_ids),
            "analyses": {},
            "summary": {}
        }
        
        # Analysis function mapping
        analysis_functions = {
            "key": lambda sid: analyze_key(sid),
            "scale": lambda sid: analyze_scale(sid),
            "rhythm": lambda sid: analyze_rhythm(sid, include_patterns=False),
            "harmony": lambda sid: analyze_chord_progressions(sid)
        }
        
        # Run analyses
        for score_id in score_ids:
            results["analyses"][score_id] = {}
            
            for analysis_type in analysis_types:
                if analysis_type in analysis_functions:
                    try:
                        result = await analysis_functions[analysis_type](score_id)
                        results["analyses"][score_id][analysis_type] = result
                    except Exception as e:
                        results["analyses"][score_id][analysis_type] = {"error": str(e)}
        
        # Generate summary statistics
        if "key" in analysis_types:
            keys = []
            for sid in score_ids:
                if "key" in results["analyses"][sid] and "key" in results["analyses"][sid]["key"]:
                    keys.append(results["analyses"][sid]["key"]["key"])
            
            if keys:
                results["summary"]["common_keys"] = Counter(keys).most_common(3)
        
        return results
        
    except Exception as e:
        logger.error(f"Error in batch analysis: {str(e)}")
        return {"status": "error", "message": str(e)}

@mcp.tool()
async def generate_report(
    score_id: str,
    report_format: str = "summary"
) -> Dict[str, Any]:
    """
    Generate a formatted analysis report.
    
    Args:
        score_id: ID of the score
        report_format: Format type ('summary', 'detailed', 'educational')
    
    Returns:
        Formatted analysis report
    """
    try:
        # Run comprehensive analysis
        analysis = await comprehensive_analysis(score_id, include_advanced=True)
        
        if analysis["status"] != "success":
            return analysis
        
        # Generate report based on format
        report = {
            "status": "success",
            "score_id": score_id,
            "title": analysis["metadata"].get("title", "Untitled"),
            "composer": analysis["metadata"].get("composer", "Unknown"),
            "report_type": report_format,
            "generated_at": datetime.now().isoformat()
        }
        
        if report_format == "summary":
            # Executive summary
            report["summary"] = {
                "key": analysis["analyses"].get("key", {}).get("key", "Unknown"),
                "time_signature": analysis["metadata"].get("time_signatures", [{}])[0].get("signature", "Unknown"),
                "tempo": analysis["analyses"].get("rhythm", {}).get("tempo", {}).get("primary_bpm", "Unknown"),
                "measures": analysis["metadata"].get("measure_count", 0),
                "complexity": analysis["analyses"].get("rhythm", {}).get("complexity", "Unknown")
            }
            
        elif report_format == "detailed":
            # Full analysis details
            report["details"] = analysis["analyses"]
            
        elif report_format == "educational":
            # Educational explanations
            report["explanations"] = {}
            
            # Key explanation
            if "key" in analysis["analyses"]:
                key_data = analysis["analyses"]["key"]
                report["explanations"]["key"] = (
                    f"The piece is in {key_data.get('key', 'an unknown key')}. "
                    f"This was determined with {key_data.get('confidence', 0):.1%} confidence."
                )
            
            # Rhythm explanation
            if "rhythm" in analysis["analyses"]:
                rhythm_data = analysis["analyses"]["rhythm"]
                tempo = rhythm_data.get("tempo", {})
                report["explanations"]["tempo"] = (
                    f"The tempo is {tempo.get('primary_bpm', 'unknown')} BPM "
                    f"({tempo.get('character', 'moderate')}), with "
                    f"{tempo.get('stability', 0):.1%} stability."
                )
        
        return report
        
    except Exception as e:
        logger.error(f"Error generating report: {str(e)}")
        return {"status": "error", "message": str(e)}

def main():
    """Main entry point for the server"""
    import uvicorn
    logger.info("Starting Music21 MCP Server...")
    uvicorn.run("music21_mcp.server:mcp", host="0.0.0.0", port=8000, reload=True)

if __name__ == "__main__":
    main()