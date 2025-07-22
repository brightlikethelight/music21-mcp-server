"""
Export Score Tool - Export scores to various formats
"""

import builtins
import contextlib
import logging
import os
import tempfile
from typing import Any

from music21 import stream

from .base_tool import BaseTool

logger = logging.getLogger(__name__)


class ExportScoreTool(BaseTool):
    """Tool for exporting scores to various formats"""

    SUPPORTED_FORMATS: dict[str, dict[str, Any]] = {
        "midi": {"extensions": [".mid", ".midi"], "method": "midi"},
        "musicxml": {"extensions": [".xml", ".musicxml", ".mxl"], "method": "musicxml"},
        "abc": {"extensions": [".abc"], "method": "abc"},
        "lilypond": {"extensions": [".ly"], "method": "lily.png"},
        "lily": {"extensions": [".ly"], "method": "lily"},
        "pdf": {"extensions": [".pdf"], "method": "lily.pdf"},
        "png": {"extensions": [".png"], "method": "lily.png"},
        "braille": {"extensions": [".brl"], "method": "braille"},
        "text": {"extensions": [".txt"], "method": "text"},
    }

    async def execute(self, **kwargs: Any) -> dict[str, Any]:
        """
        Export a score to various formats

        Args:
            **kwargs: Keyword arguments including:
                score_id: ID of the score to export
                format: Export format (midi, musicxml, abc, lilypond, pdf, etc.)
                output_path: Optional custom output path
        """
        # Extract parameters from kwargs
        score_id = kwargs.get("score_id", "")
        format = kwargs.get("format", "musicxml")
        output_path = kwargs.get("output_path")
        # Validate inputs
        error = self.validate_inputs(**kwargs)
        if error:
            return self.create_error_response(error)

        with self.error_handling(f"Export '{score_id}' to {format}"):
            score = self.get_score(score_id)

            self.report_progress(0.1, f"Preparing to export as {format}")

            # Normalize format
            format = format.lower()

            # Get format info
            format_info = self.SUPPORTED_FORMATS.get(format)
            if not format_info:
                return self.create_error_response(f"Unsupported format: {format}")

            # Determine output path
            if output_path is None:
                # Create temp file with appropriate extension
                extension = format_info["extensions"][0]
                fd, output_path = tempfile.mkstemp(suffix=extension)
                os.close(fd)  # Close file descriptor, music21 will open it

            self.report_progress(0.3, f"Exporting to {os.path.basename(output_path)}")

            # Export based on format
            try:
                method = str(format_info["method"])
                success = await self._export_score(
                    score, method, output_path
                )

                if not success:
                    return self.create_error_response(f"Export to {format} failed")

                # Verify file was created
                if not os.path.exists(output_path):
                    return self.create_error_response(
                        "Export failed - file not created"
                    )

                # Get file size
                file_size = os.path.getsize(output_path)

                self.report_progress(1.0, "Export complete")

                return self.create_success_response(
                    format=format,
                    file_path=output_path,
                    file_size=file_size,
                    message=f"Successfully exported to {format}",
                )

            except Exception:
                # Clean up temp file on error
                if output_path and os.path.exists(output_path) and not output_path:
                    with contextlib.suppress(builtins.BaseException):
                        os.unlink(output_path)
                raise

    def validate_inputs(self, **kwargs: Any) -> str | None:
        """Validate input parameters"""
        score_id = kwargs.get("score_id", "")
        format = kwargs.get("format", "musicxml")

        error = self.check_score_exists(score_id)
        if error:
            return error

        if format.lower() not in self.SUPPORTED_FORMATS:
            supported = ", ".join(self.SUPPORTED_FORMATS.keys())
            return f"Unsupported format: {format}. Supported formats: {supported}"

        return None

    async def _export_score(
        self, score: stream.Score, method: str, output_path: str
    ) -> bool:
        """Export score using music21's write method"""
        try:
            # Handle special cases
            if method == "lily.pdf":
                # PDF export requires LilyPond
                self.report_progress(0.5, "Generating LilyPond file")
                score.write("lily.pdf", fp=output_path)
            elif method == "lily.png":
                # PNG export requires LilyPond
                self.report_progress(0.5, "Generating LilyPond file")
                score.write("lily.png", fp=output_path)
            elif method == "text":
                # Text export
                self.report_progress(0.5, "Generating text representation")
                with open(output_path, "w") as f:
                    f.write(str(score.flatten().show("text")))
            else:
                # Standard export
                self.report_progress(0.5, f"Writing {method} file")
                score.write(method, fp=output_path)

            self.report_progress(0.9, "Finalizing export")
            return True

        except Exception as e:
            logger.error(f"Export failed: {e}")
            # Provide helpful error messages
            if "lily" in method and "LilyPond" in str(e):
                logger.info("LilyPond not installed. Install from: http://lilypond.org")
            return False
