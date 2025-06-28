"""
Music21 MCP Server - SIMPLIFIED VERSION
Emergency simplification for stability
"""
try:
    from mcp.server.fastmcp import FastMCP
    mcp = FastMCP("Music21 Simple Server")
except ImportError:
    # For testing without MCP
    class MockMCP:
        def tool(self):
            def decorator(func):
                return func
            return decorator
        def run(self):
            pass
    mcp = MockMCP()

from music21 import stream, converter, corpus
import os
from typing import Dict, Any, Optional, Union
import logging

# Simple logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Simple in-memory score storage
scores = {}


@mcp.tool()
async def import_score(
    score_id: str,
    source: str
) -> Dict[str, Any]:
    """
    Simple score import - handles files and corpus paths
    
    Args:
        score_id: ID to store the score
        source: File path or corpus path (e.g., 'bach/bwv66.6')
    """
    try:
        # Try as file path first
        if os.path.exists(source):
            score = converter.parse(source)
        else:
            # Try as corpus path
            try:
                score = corpus.parse(source)
            except:
                # Try as simple note list (e.g., "C4 D4 E4 F4")
                if ' ' in source and all(
                    n.replace('#', '').replace('-', '').replace('b', '').isalnum() 
                    for n in source.split()
                ):
                    # Create score manually from note names
                    score = stream.Score()
                    part = stream.Part()
                    for note_str in source.split():
                        try:
                            from music21 import note
                            n = note.Note(note_str)
                            part.append(n)
                        except:
                            # If any note fails, try converter
                            score = converter.parse(source)
                            break
                    else:
                        # All notes parsed successfully
                        score.append(part)
                else:
                    # Try as direct musical content
                    score = converter.parse(source)
        
        # Store the score
        scores[score_id] = score
        
        # Get basic info
        num_notes = len(list(score.flatten().notes))
        num_measures = len(list(score.flatten().getElementsByClass('Measure')))
        
        return {
            "status": "success",
            "score_id": score_id,
            "num_notes": num_notes,
            "num_measures": num_measures
        }
        
    except Exception as e:
        logger.error(f"Import error: {e}")
        return {"status": "error", "message": str(e)}


@mcp.tool()
async def analyze_key(score_id: str) -> Dict[str, Any]:
    """
    Simple key analysis using music21 defaults
    
    Args:
        score_id: ID of the score to analyze
    """
    try:
        if score_id not in scores:
            return {"status": "error", "message": f"Score '{score_id}' not found"}
        
        score = scores[score_id]
        
        # Simple key analysis
        key = score.analyze('key')
        
        # Get confidence if available
        confidence = 0.5  # Default confidence
        if hasattr(key, 'correlationCoefficient'):
            confidence = float(key.correlationCoefficient)
        
        return {
            "status": "success",
            "score_id": score_id,
            "key": str(key),
            "confidence": confidence
        }
        
    except Exception as e:
        logger.error(f"Key analysis error: {e}")
        return {"status": "error", "message": str(e)}


@mcp.tool()
async def analyze_chords(score_id: str) -> Dict[str, Any]:
    """
    Simple chord analysis with chordify
    
    Args:
        score_id: ID of the score to analyze
    """
    try:
        if score_id not in scores:
            return {"status": "error", "message": f"Score '{score_id}' not found"}
        
        score = scores[score_id]
        
        # Use chordify to extract chords
        chordified = score.chordify()
        chords = list(chordified.flatten().getElementsByClass('Chord'))
        
        # Get chord progression (first 10 chords)
        progression = []
        for chord in chords[:10]:
            # Get chord pitches as strings
            chord_pitches = [str(p) for p in chord.pitches]
            progression.append(chord_pitches)
        
        return {
            "status": "success",
            "score_id": score_id,
            "chord_count": len(chords),
            "sample_progression": progression
        }
        
    except Exception as e:
        logger.error(f"Chord analysis error: {e}")
        return {"status": "error", "message": str(e)}


@mcp.tool()
async def get_score_info(score_id: str) -> Dict[str, Any]:
    """
    Get basic information about a score
    
    Args:
        score_id: ID of the score
    """
    try:
        if score_id not in scores:
            return {"status": "error", "message": f"Score '{score_id}' not found"}
        
        score = scores[score_id]
        
        # Basic metadata
        info = {
            "status": "success",
            "score_id": score_id,
            "title": score.metadata.title if hasattr(score.metadata, 'title') else "Unknown",
            "composer": score.metadata.composer if hasattr(score.metadata, 'composer') else "Unknown",
            "num_parts": len(score.parts) if hasattr(score, 'parts') else 1,
            "num_measures": len(list(score.flatten().getElementsByClass('Measure'))),
            "num_notes": len(list(score.flatten().notes)),
            "duration_quarters": float(score.duration.quarterLength) if hasattr(score, 'duration') else 0
        }
        
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
    Export a score to file
    
    Args:
        score_id: ID of the score to export
        format: Export format (musicxml, midi, lilypond, abc)
        file_path: Output file path (optional)
    """
    try:
        if score_id not in scores:
            return {"status": "error", "message": f"Score '{score_id}' not found"}
        
        score = scores[score_id]
        
        # Generate file path if not provided
        if not file_path:
            import tempfile
            suffix = {
                'musicxml': '.xml',
                'midi': '.mid',
                'lilypond': '.ly',
                'abc': '.abc'
            }.get(format, '.xml')
            
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
                file_path = f.name
        
        # Export
        score.write(format, fp=file_path)
        
        return {
            "status": "success",
            "score_id": score_id,
            "format": format,
            "file_path": file_path
        }
        
    except Exception as e:
        logger.error(f"Export error: {e}")
        return {"status": "error", "message": str(e)}


def main():
    """Main entry point for the simplified server"""
    logger.info("Starting Simplified Music21 MCP Server...")
    mcp.run()


if __name__ == "__main__":
    main()