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
                    return self.create_error_response(f"Could not find or import score: {source}")
            except Exception as e:
                # Return the specific error message from music21
                error_msg = str(e)
                if "Could not find" in error_msg:
                    return self.create_error_response(f"Could not find score: {source}")
                return self.create_error_response(f"Import failed: {error_msg}")

            self.report_progress(0.8, "Analyzing score metadata")

            # Store the score
            self.score_manager[score_id] = score

            # Get metadata asynchronously
            metadata = await self._extract_metadata(score)

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
        """Import from a file using async execution"""
        if not os.path.exists(file_path):
            return None

        try:
            # Parse file in background thread to avoid blocking event loop
            def _parse_file():
                return converter.parse(file_path)

            parsed = await self.run_with_progress(
                _parse_file,
                progress_start=0.3,
                progress_end=0.7,
                message="Parsing file"
            )

            # Ensure we return a Score object
            if isinstance(parsed, stream.Score):
                return parsed
            if hasattr(parsed, "flatten"):
                # Convert to Score if it's another stream type
                def _convert_to_score():
                    score = stream.Score()
                    score.append(parsed)
                    return score

                return await self.run_music21_operation(_convert_to_score)
            return None
        except Exception as e:
            logger.error(f"Failed to parse file {file_path}: {e}")
            return None

    async def _import_from_corpus(self, corpus_path: str) -> stream.Score | None:
        """Import from music21 corpus using async execution"""
        try:
            # Parse corpus in background thread to avoid blocking event loop
            def _parse_corpus():
                return corpus.parse(corpus_path)

            parsed = await self.run_with_progress(
                _parse_corpus,
                progress_start=0.3,
                progress_end=0.7,
                message="Loading from corpus"
            )

            # Ensure we return a Score object
            if isinstance(parsed, stream.Score):
                return parsed
            if hasattr(parsed, "expandRepeats"):
                # Convert to Score if it's another stream type
                def _convert_to_score():
                    score = stream.Score()
                    score.append(parsed)
                    return score

                return await self.run_music21_operation(_convert_to_score)
            return None
        except Exception as e:
            logger.error(f"Failed to load corpus {corpus_path}: {e}")
            return None

    async def _import_from_text(self, text: str) -> stream.Score | None:
        """Import from text notation using async execution"""
        try:
            self.report_progress(0.3, "Parsing text notation")

            # Check if it's tinyNotation format
            if text.strip().startswith("tinyNotation:"):
                tiny_text = text.replace("tinyNotation:", "").strip()

                def _parse_tiny_notation():
                    from music21 import converter
                    return converter.parse(f"tinyNotation: {tiny_text}")

                # Parse in background thread
                parsed = await self.run_music21_operation(_parse_tiny_notation)

                # Ensure we return a Score object
                if isinstance(parsed, stream.Score):
                    self.report_progress(0.7, "TinyNotation parsed")
                    return parsed

                # Convert to Score if it's another stream type
                def _convert_to_score():
                    score = stream.Score()
                    score.append(parsed)
                    return score

                result = await self.run_music21_operation(_convert_to_score)
                self.report_progress(0.7, "TinyNotation parsed and converted to Score")
                return result

            # Otherwise parse as space-separated notes
            def _parse_note_sequence():
                score = stream.Score()
                part = stream.Part()
                tokens = text.split()

                for note_str in tokens:
                    try:
                        n = note.Note(note_str)
                        part.append(n)
                    except Exception as e:
                        logger.warning(f"Invalid note '{note_str}': {e}")
                        raise ValueError(f"Invalid note: {note_str}")

                score.append(part)
                return score

            # Parse note sequence in background thread
            result = await self.run_with_progress(
                _parse_note_sequence,
                progress_start=0.3,
                progress_end=0.7,
                message="Parsing text notation"
            )
            return result

        except Exception as e:
            logger.error(f"Failed to parse text notation: {e}")
            return None

    async def _extract_metadata(self, score: stream.Score) -> dict[str, Any]:
        """Extract metadata from score using async execution"""
        def _extract_sync():
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

        return await self.run_music21_operation(_extract_sync)

