"""
Music21 MCP Server - Simplified and Stable Version
Provides core music analysis functionality through MCP protocol
"""
import logging
import os
import tempfile
from typing import Any, Dict, Optional

from mcp.server.fastmcp import FastMCP
from music21 import converter, corpus, note, stream

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastMCP server instance
mcp = FastMCP("Music21 Server")

# Simple in-memory score storage
score_manager = {}


@mcp.tool()
async def import_score(
    score_id: str,
    source: str,
    source_type: Optional[str] = "auto"
) -> Dict[str, Any]:
    """
    Import a musical score from various sources.
    
    Args:
        score_id: Unique identifier for the score
        source: File path, corpus path (e.g., 'bach/bwv66.6'), or note sequence
        source_type: Type of source ('file', 'corpus', 'text', 'auto')
    
    Returns:
        Import status with basic metadata
    """
    try:
        score = None
        
        # Auto-detect source type if needed
        if source_type == "auto":
            if os.path.exists(source):
                source_type = "file"
            elif '/' in source and not os.path.exists(source):
                source_type = "corpus"
            elif ' ' in source and all(
                n.replace('#', '').replace('-', '').replace('b', '').isalnum() 
                for n in source.split()
            ):
                source_type = "text"
        
        # Import based on source type
        if source_type == "file" and os.path.exists(source):
            score = converter.parse(source)
        elif source_type == "corpus":
            try:
                score = corpus.parse(source)
            except:
                return {"status": "error", "message": f"Corpus work '{source}' not found"}
        elif source_type == "text":
            # Create score from note names
            score = stream.Score()
            part = stream.Part()
            for note_str in source.split():
                try:
                    n = note.Note(note_str)
                    part.append(n)
                except:
                    return {"status": "error", "message": f"Invalid note: {note_str}"}
            score.append(part)
        else:
            # Try converter as fallback
            try:
                score = converter.parse(source)
            except:
                return {"status": "error", "message": "Could not parse source"}
        
        if score is None:
            return {"status": "error", "message": "Failed to import score"}
        
        # Store the score
        score_manager[score_id] = score
        
        # Get basic metadata
        num_notes = len(list(score.flatten().notes))
        num_measures = len(list(score.flatten().getElementsByClass('Measure')))
        num_parts = len(score.parts) if hasattr(score, 'parts') else 1
        
        return {
            "status": "success",
            "score_id": score_id,
            "num_notes": num_notes,
            "num_measures": num_measures,
            "num_parts": num_parts,
            "source_type": source_type
        }
        
    except Exception as e:
        logger.error(f"Import error: {e}")
        return {"status": "error", "message": str(e)}


@mcp.tool()
async def list_scores() -> Dict[str, Any]:
    """
    List all imported scores.
    
    Returns:
        List of score IDs with basic info
    """
    try:
        scores_list = []
        for score_id, score in score_manager.items():
            scores_list.append({
                "score_id": score_id,
                "num_notes": len(list(score.flatten().notes)),
                "num_parts": len(score.parts) if hasattr(score, 'parts') else 1
            })
        
        return {
            "status": "success",
            "scores": scores_list,
            "total_count": len(scores_list)
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@mcp.tool()
async def analyze_key(
    score_id: str,
    method: Optional[str] = "default"
) -> Dict[str, Any]:
    """
    Analyze the key of a score.
    
    Args:
        score_id: ID of the score to analyze
        method: Analysis method ('default', 'krumhansl', 'aarden')
    
    Returns:
        Key analysis with confidence score
    """
    try:
        if score_id not in score_manager:
            return {"status": "error", "message": f"Score '{score_id}' not found"}
        
        score = score_manager[score_id]
        
        # Analyze key
        if method == "default":
            key = score.analyze('key')
        else:
            key = score.analyze(f'key.{method}')
        
        # Get confidence if available
        confidence = getattr(key, 'correlationCoefficient', 0.5)
        
        # Get alternative keys
        alternatives = []
        if hasattr(key, 'alternateKeys'):
            for alt_key in key.alternateKeys[:3]:
                alt_conf = getattr(alt_key, 'correlationCoefficient', 0.3)
                alternatives.append({
                    "key": str(alt_key),
                    "confidence": float(alt_conf)
                })
        
        return {
            "status": "success",
            "score_id": score_id,
            "key": str(key),
            "confidence": float(confidence),
            "method": method,
            "alternatives": alternatives
        }
        
    except Exception as e:
        logger.error(f"Key analysis error: {e}")
        return {"status": "error", "message": str(e)}


@mcp.tool()
async def analyze_chords(
    score_id: str,
    include_roman_numerals: Optional[bool] = False
) -> Dict[str, Any]:
    """
    Analyze chords in a score.
    
    Args:
        score_id: ID of the score to analyze
        include_roman_numerals: Whether to include Roman numeral analysis
    
    Returns:
        Chord analysis with progression
    """
    try:
        if score_id not in score_manager:
            return {"status": "error", "message": f"Score '{score_id}' not found"}
        
        score = score_manager[score_id]
        
        # Use chordify to extract chords
        chordified = score.chordify()
        chords = list(chordified.flatten().getElementsByClass('Chord'))
        
        # Get chord progression (first 20 chords)
        progression = []
        for i, chord in enumerate(chords[:20]):
            chord_info = {
                "index": i,
                "pitches": [str(p) for p in chord.pitches],
                "root": str(chord.root()) if chord.root() else None,
                "quality": chord.quality if hasattr(chord, 'quality') else None,
                "measure": chord.measureNumber if hasattr(chord, 'measureNumber') else None
            }
            
            # Add Roman numeral if requested
            if include_roman_numerals:
                try:
                    key = score.analyze('key')
                    from music21 import roman
                    rn = roman.romanNumeralFromChord(chord, key)
                    chord_info["roman_numeral"] = str(rn.romanNumeralAlone)
                except:
                    chord_info["roman_numeral"] = None
            
            progression.append(chord_info)
        
        return {
            "status": "success",
            "score_id": score_id,
            "total_chords": len(chords),
            "chord_progression": progression,
            "includes_roman_numerals": include_roman_numerals
        }
        
    except Exception as e:
        logger.error(f"Chord analysis error: {e}")
        return {"status": "error", "message": str(e)}


@mcp.tool()
async def get_score_info(score_id: str) -> Dict[str, Any]:
    """
    Get detailed information about a score.
    
    Args:
        score_id: ID of the score
    
    Returns:
        Comprehensive score metadata
    """
    try:
        if score_id not in score_manager:
            return {"status": "error", "message": f"Score '{score_id}' not found"}
        
        score = score_manager[score_id]
        
        # Extract metadata
        metadata = score.metadata if hasattr(score, 'metadata') else None
        
        info = {
            "status": "success",
            "score_id": score_id,
            # Basic metadata
            "title": metadata.title if metadata and hasattr(metadata, 'title') else None,
            "composer": metadata.composer if metadata and hasattr(metadata, 'composer') else None,
            "date": str(metadata.date) if metadata and hasattr(metadata, 'date') else None,
            # Structure
            "num_parts": len(score.parts) if hasattr(score, 'parts') else 1,
            "num_measures": len(list(score.flatten().getElementsByClass('Measure'))),
            "num_notes": len(list(score.flatten().notes)),
            "duration_quarters": float(score.duration.quarterLength) if hasattr(score, 'duration') else 0,
            # Time signatures
            "time_signatures": [],
            # Key signatures
            "key_signatures": [],
            # Tempo markings
            "tempo_markings": []
        }
        
        # Get time signatures
        for ts in score.flatten().getElementsByClass('TimeSignature'):
            info["time_signatures"].append({
                "measure": ts.measureNumber if hasattr(ts, 'measureNumber') else None,
                "signature": str(ts)
            })
        
        # Get key signatures
        for ks in score.flatten().getElementsByClass('KeySignature'):
            info["key_signatures"].append({
                "measure": ks.measureNumber if hasattr(ks, 'measureNumber') else None,
                "sharps": ks.sharps
            })
        
        # Get tempo markings
        for tempo in score.flatten().getElementsByClass('MetronomeMark'):
            info["tempo_markings"].append({
                "measure": tempo.measureNumber if hasattr(tempo, 'measureNumber') else None,
                "bpm": tempo.number,
                "unit": str(tempo.referent)
            })
        
        return info
        
    except Exception as e:
        logger.error(f"Score info error: {e}")
        return {"status": "error", "message": str(e)}


@mcp.tool() 
async def export_score(
    score_id: str,
    format: str = "musicxml",
    file_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Export a score to various formats.
    
    Args:
        score_id: ID of the score to export
        format: Export format ('musicxml', 'midi', 'lilypond', 'abc', 'pdf')
        file_path: Output file path (auto-generated if not provided)
    
    Returns:
        Export status with file path
    """
    try:
        if score_id not in score_manager:
            return {"status": "error", "message": f"Score '{score_id}' not found"}
        
        score = score_manager[score_id]
        
        # Generate file path if not provided
        if not file_path:
            suffix = {
                'musicxml': '.xml',
                'midi': '.mid',
                'lilypond': '.ly',
                'abc': '.abc',
                'pdf': '.pdf'
            }.get(format, '.xml')
            
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
                file_path = f.name
        
        # Export
        score.write(format, fp=file_path)
        
        # Check if file was created
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            return {
                "status": "success",
                "score_id": score_id,
                "format": format,
                "file_path": file_path,
                "file_size": file_size
            }
        else:
            return {"status": "error", "message": "Export failed - file not created"}
        
    except Exception as e:
        logger.error(f"Export error: {e}")
        return {"status": "error", "message": str(e)}


@mcp.tool()
async def delete_score(score_id: str) -> Dict[str, Any]:
    """
    Delete a score from memory.
    
    Args:
        score_id: ID of the score to delete
    
    Returns:
        Deletion status
    """
    try:
        if score_id not in score_manager:
            return {"status": "error", "message": f"Score '{score_id}' not found"}
        
        del score_manager[score_id]
        
        return {
            "status": "success",
            "message": f"Score '{score_id}' deleted",
            "remaining_scores": len(score_manager)
        }
        
    except Exception as e:
        return {"status": "error", "message": str(e)}


def main():
    """Main entry point for the server"""
    logger.info("Starting Music21 MCP Server (Simplified Version)...")
    logger.info(f"Available tools: {', '.join(tool.name for tool in mcp.tools)}")
    
    # Run the server
    mcp.run()


if __name__ == "__main__":
    main()