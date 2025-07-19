"""
Import Score Tool - Import musical scores from various sources
Supports files, corpus, and text notation with intelligent auto-detection
"""

import logging
import os
from typing import Any, Dict, Optional

from music21 import converter, corpus, note, stream

from .base_tool import BaseTool

logger = logging.getLogger(__name__)


class ImportScoreTool(BaseTool):
    """Tool for importing musical scores from various sources"""

    SUPPORTED_FILE_EXTENSIONS = {
        ".mid",
        ".midi",
        ".xml",
        ".musicxml",
        ".mxl",
        ".abc",
        ".krn",
        ".mei",
    }

    async def execute(self, **kwargs: Any) -> Dict[str, Any]:
        """
        Import a musical score from various sources

        Args:
            **kwargs: Keyword arguments including:
                score_id: Unique identifier for the score
                source: File path, corpus path, or note sequence
                source_type: Type of source ('file', 'corpus', 'text', 'auto')
        """
        # Extract parameters from kwargs
        score_id = kwargs.get("score_id", "")
        source = kwargs.get("source", "")
        source_type = kwargs.get("source_type", "auto")
        # Validate inputs
        error = self.validate_inputs(**kwargs)
        if error:
            return self.create_error_response(error)

        with self.error_handling(f"Import score '{score_id}'"):
            # Auto-detect source type if needed
            if source_type == "auto":
                source_type = self._detect_source_type(source)

            self.report_progress(0.1, f"Importing from {source_type}")

            # Import based on source type
            score = None

            if source_type == "file":
                score = await self._import_from_file(source)
            elif source_type == "corpus":
                score = await self._import_from_corpus(source)
            elif source_type == "text":
                score = await self._import_from_text(source)
            else:
                return self.create_error_response(f"Unknown source type: {source_type}")

            if score is None:
                return self.create_error_response("Failed to import score")

            self.report_progress(0.8, "Analyzing score metadata")

            # Store the score
            self.score_manager[score_id] = score

            # Get metadata
            metadata = self._extract_metadata(score)

            self.report_progress(1.0, "Import complete")

            return self.create_success_response(
                message=f"Successfully imported score '{score_id}' from {source_type}",
                score_id=score_id,
                source_type=source_type,
                **metadata,
            )

    def validate_inputs(self, **kwargs: Any) -> Optional[str]:
        """Validate input parameters"""
        score_id = kwargs.get("score_id", "")
        source = kwargs.get("source", "")
        source_type = kwargs.get("source_type", "auto")

        if not score_id:
            return "score_id cannot be empty"

        if not source:
            return "source cannot be empty"

        if score_id in self.score_manager:
            return f"Score with ID '{score_id}' already exists"

        if source_type not in ["auto", "file", "corpus", "text"]:
            return f"Invalid source_type: {source_type}"

        return None

    def _detect_source_type(self, source: str) -> str:
        """Auto-detect the source type"""
        # Check if it's a file path
        if os.path.exists(source):
            return "file"

        # Check if it has a file extension
        if any(source.lower().endswith(ext) for ext in self.SUPPORTED_FILE_EXTENSIONS):
            return "file"

        # Check if it looks like a corpus path
        if "/" in source and not os.path.exists(source):
            # Common corpus patterns
            if any(
                composer in source.lower()
                for composer in ["bach", "mozart", "beethoven", "chopin"]
            ):
                return "corpus"

        # Check if it looks like note text
        if " " in source:
            # Check if all tokens look like notes
            tokens = source.split()
            if all(self._is_note_like(token) for token in tokens):
                return "text"

        # Default to trying as file
        return "file"

    def _is_note_like(self, token: str) -> bool:
        """Check if a token looks like a note"""
        # Remove accidentals
        cleaned = token.replace("#", "").replace("-", "").replace("b", "")
        # Check if it's alphanumeric with letter followed by number
        if len(cleaned) >= 2:
            return cleaned[0].isalpha() and cleaned[1:].isdigit()
        return False

    async def _import_from_file(self, file_path: str) -> Optional[stream.Score]:
        """Import from a file"""
        if not os.path.exists(file_path):
            return None

        try:
            self.report_progress(0.3, "Parsing file")
            score = converter.parse(file_path)
            self.report_progress(0.7, "File parsed successfully")
            return score
        except Exception as e:
            logger.error(f"Failed to parse file {file_path}: {e}")
            return None

    async def _import_from_corpus(self, corpus_path: str) -> Optional[stream.Score]:
        """Import from music21 corpus"""
        try:
            self.report_progress(0.3, "Loading from corpus")
            score = corpus.parse(corpus_path)
            self.report_progress(0.7, "Corpus loaded successfully")
            return score
        except Exception as e:
            logger.error(f"Failed to load corpus {corpus_path}: {e}")
            return None

    async def _import_from_text(self, text: str) -> Optional[stream.Score]:
        """Import from text notation"""
        try:
            self.report_progress(0.3, "Parsing text notation")
            score = stream.Score()
            part = stream.Part()

            tokens = text.split()
            total = len(tokens)

            for i, note_str in enumerate(tokens):
                try:
                    n = note.Note(note_str)
                    part.append(n)
                    if i % 10 == 0:  # Update progress every 10 notes
                        self.report_progress(
                            0.3 + (0.4 * i / total), f"Parsing note {i+1}/{total}"
                        )
                except Exception as e:
                    logger.warning(f"Invalid note '{note_str}': {e}")
                    return None

            score.append(part)
            self.report_progress(0.7, "Text notation parsed")
            return score

        except Exception as e:
            logger.error(f"Failed to parse text notation: {e}")
            return None

    def _extract_metadata(self, score: stream.Score) -> Dict[str, Any]:
        """Extract metadata from score"""
        try:
            num_notes = len(list(score.flatten().notes))
            num_measures = len(list(score.flatten().getElementsByClass("Measure")))
            num_parts = len(score.parts) if hasattr(score, "parts") else 1

            # Get first and last notes for range
            notes = list(score.flatten().notes)
            if notes:
                lowest = min(n.pitch.midi for n in notes if hasattr(n, "pitch"))
                highest = max(n.pitch.midi for n in notes if hasattr(n, "pitch"))
                pitch_range = highest - lowest
            else:
                pitch_range = 0

            return {
                "num_notes": num_notes,
                "num_measures": num_measures,
                "num_parts": num_parts,
                "pitch_range": pitch_range,
            }
        except Exception as e:
            logger.warning(f"Error extracting metadata: {e}")
            return {"num_notes": 0, "num_measures": 0, "num_parts": 0, "pitch_range": 0}
