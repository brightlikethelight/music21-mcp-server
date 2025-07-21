"""
Import Score Tool - Import musical scores from various sources
Supports files, corpus, and text notation with intelligent auto-detection
"""

import logging
import os
from typing import Any

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

    async def execute(self, **kwargs: Any) -> dict[str, Any]:
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

            try:
                if source_type == "file":
                    score = await self._import_from_file(source)
                elif source_type == "corpus":
                    score = await self._import_from_corpus(source)
                elif source_type == "text":
                    score = await self._import_from_text(source)
                else:
                    return self.create_error_response(
                        f"Unknown source type: {source_type}"
                    )

                if score is None:
                    return self.create_error_response("Failed to import score")
            except Exception as e:
                # Return the specific error message from music21
                error_msg = str(e)
                if "Could not find" in error_msg:
                    return self.create_error_response(f"Could not find score: {source}")
                return self.create_error_response(f"Import failed: {error_msg}")

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

    def validate_inputs(self, **kwargs: Any) -> str | None:
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

    async def _import_from_file(self, file_path: str) -> stream.Score | None:
        """Import from a file"""
        if not os.path.exists(file_path):
            return None

        try:
            self.report_progress(0.3, "Parsing file")
            parsed = converter.parse(file_path)
            # Ensure we return a Score object
            if isinstance(parsed, stream.Score):
                self.report_progress(0.7, "File parsed successfully")
                return parsed
            if hasattr(parsed, "flatten"):
                # Convert to Score if it's another stream type
                score = stream.Score()
                score.append(parsed)
                self.report_progress(0.7, "File parsed and converted to Score")
                return score
            self.report_progress(0.7, "File parsed successfully")
            return None
        except Exception as e:
            logger.error(f"Failed to parse file {file_path}: {e}")
            return None

    async def _import_from_corpus(self, corpus_path: str) -> stream.Score | None:
        """Import from music21 corpus"""
        self.report_progress(0.3, "Loading from corpus")
        parsed = corpus.parse(corpus_path)
        # Ensure we return a Score object
        if isinstance(parsed, stream.Score):
            self.report_progress(0.7, "Corpus loaded successfully")
            return parsed
        if hasattr(parsed, "expandRepeats"):
            # Convert to Score if it's another stream type
            score = stream.Score()
            score.append(parsed)
            self.report_progress(0.7, "Corpus loaded and converted to Score")
            return score
        self.report_progress(0.7, "Corpus loaded successfully")
        return None

    async def _import_from_text(self, text: str) -> stream.Score | None:
        """Import from text notation"""
        try:
            self.report_progress(0.3, "Parsing text notation")

            # Check if it's tinyNotation format
            if text.strip().startswith("tinyNotation:"):
                # Use music21's tinyNotation parser
                from music21 import converter

                tiny_text = text.replace("tinyNotation:", "").strip()
                parsed = converter.parse(f"tinyNotation: {tiny_text}")
                # Ensure we return a Score object
                if isinstance(parsed, stream.Score):
                    self.report_progress(0.7, "TinyNotation parsed")
                    return parsed
                # Convert to Score if it's another stream type
                score = stream.Score()
                score.append(parsed)
                self.report_progress(0.7, "TinyNotation parsed and converted to Score")
                return score

            # Otherwise parse as space-separated notes
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
                            0.3 + (0.4 * i / total), f"Parsing note {i + 1}/{total}"
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

    def _extract_metadata(self, score: stream.Score) -> dict[str, Any]:
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
