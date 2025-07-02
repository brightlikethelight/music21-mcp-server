"""
Advanced score parsing module with multi-format support and intelligent parsing
"""

import logging
import os
import re
import tempfile
import zipfile
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import aiofiles
import chardet
import httpx
from music21 import abcFormat, converter, environment, expressions, midi, stream

logger = logging.getLogger(__name__)

# Configure music21 environment
env = environment.Environment()


class FormatDetector:
    """Intelligent format detection for music files"""

    # File signatures (magic numbers)
    SIGNATURES = {
        "midi": [b"MThd", b"MTrk"],
        "musicxml": [b"<?xml", b"<score-partwise", b"<score-timewise"],
        "compressed_musicxml": [b"PK\x03\x04"],  # ZIP signature
        "abc": [b"X:", b"T:", b"M:", b"K:"],
        "kern": [b"**kern", b"!!!"],
        "lilypond": [b"\\version", b"\\score", b"\\relative"],
        "mei": [b"<mei", b"<?xml.*<mei"],
        "musedata": [b"@", b"$"],
        "capella": [b"CAP", b"Capella"],
    }

    # Content patterns for text-based formats
    CONTENT_PATTERNS = {
        "musicxml": re.compile(
            r"<(?:score-partwise|score-timewise|opus)", re.IGNORECASE
        ),
        "abc": re.compile(r"^[XTMKLQVW]:", re.MULTILINE),
        "kern": re.compile(r"\*\*kern|\*\*dynam|\*\*text", re.MULTILINE),
        "lilypond": re.compile(
            r"\\(?:version|score|relative|new|header)", re.MULTILINE
        ),
        "mei": re.compile(r"<mei\s+xmlns", re.IGNORECASE),
        "musedata": re.compile(r"^[@$]", re.MULTILINE),
        "romantext": re.compile(r"^(?:Time Signature:|m\d+)", re.MULTILINE),
    }

    @classmethod
    async def detect_format(cls, source: Union[str, bytes, Dict[str, Any]]) -> str:
        """
        Detect format from various source types

        Args:
            source: File path, URL, bytes, or dict with content

        Returns:
            Detected format name
        """
        # Handle different source types
        if isinstance(source, str):
            if source.startswith(("http://", "https://")):
                return await cls._detect_from_url(source)
            elif os.path.exists(source):
                return await cls._detect_from_file(source)
            else:
                # Assume it's content
                return cls._detect_from_content(source.encode())

        elif isinstance(source, bytes):
            return cls._detect_from_content(source)

        elif isinstance(source, dict):
            if "content" in source:
                content = source["content"]
                if isinstance(content, str):
                    content = content.encode()
                return cls._detect_from_content(content)
            elif "path" in source:
                return await cls._detect_from_file(source["path"])
            elif "url" in source:
                return await cls._detect_from_url(source["url"])

        return "unknown"

    @classmethod
    async def _detect_from_file(cls, filepath: str) -> str:
        """Detect format from file"""
        path = Path(filepath)

        # Check extension first
        ext_format = cls._detect_from_extension(path.suffix.lower())
        if ext_format != "unknown":
            # Verify with content
            async with aiofiles.open(filepath, "rb") as f:
                header = await f.read(1024)
                content_format = cls._detect_from_content(header)

                # If they match, we're confident
                if content_format == ext_format:
                    return ext_format

        # Fall back to content detection
        async with aiofiles.open(filepath, "rb") as f:
            content = await f.read(10240)  # Read more for better detection
            return cls._detect_from_content(content)

    @classmethod
    async def _detect_from_url(cls, url: str) -> str:
        """Detect format from URL"""
        async with httpx.AsyncClient() as client:
            # First try HEAD request to check content type
            try:
                head_response = await client.head(url)
                content_type = head_response.headers.get("content-type", "")

                if "midi" in content_type:
                    return "midi"
                elif "xml" in content_type or "musicxml" in content_type:
                    return "musicxml"
            except:
                pass

            # Get actual content
            response = await client.get(url)
            return cls._detect_from_content(response.content)

    @classmethod
    def _detect_from_content(cls, content: bytes) -> str:
        """Detect format from content bytes"""
        # Check binary signatures first
        for format_name, signatures in cls.SIGNATURES.items():
            for sig in signatures:
                if content.startswith(sig):
                    # Special handling for compressed MusicXML
                    if format_name == "compressed_musicxml":
                        # Check if it's actually a compressed MusicXML
                        try:
                            with zipfile.ZipFile(io.BytesIO(content)) as zf:
                                for name in zf.namelist():
                                    if name.endswith(".xml"):
                                        xml_content = zf.read(name)
                                        if (
                                            b"score-partwise" in xml_content
                                            or b"score-timewise" in xml_content
                                        ):
                                            return "musicxml"
                        except:
                            pass
                    else:
                        return format_name

        # Try to decode as text for pattern matching
        try:
            # Detect encoding
            detected = chardet.detect(content[:1024])
            encoding = detected["encoding"] or "utf-8"

            text_content = content.decode(encoding, errors="ignore")

            # Check text patterns
            for format_name, pattern in cls.CONTENT_PATTERNS.items():
                if pattern.search(text_content):
                    return format_name

        except Exception as e:
            logger.debug(f"Text decoding failed: {e}")

        return "unknown"

    @classmethod
    def _detect_from_extension(cls, ext: str) -> str:
        """Map file extension to format"""
        extension_map = {
            ".mid": "midi",
            ".midi": "midi",
            ".smf": "midi",
            ".xml": "musicxml",
            ".mxl": "musicxml",
            ".musicxml": "musicxml",
            ".mxml": "musicxml",
            ".abc": "abc",
            ".abcx": "abc",
            ".krn": "kern",
            ".kern": "kern",
            ".ly": "lilypond",
            ".ily": "lilypond",
            ".mei": "mei",
            ".md": "musedata",
            ".mus": "musedata",
            ".cap": "capella",
            ".capx": "capella",
            ".sib": "sibelius",
            ".mscz": "musescore",
            ".mscx": "musescore",
            ".nwc": "noteworthy",
            ".nwctxt": "noteworthy",
            ".ove": "overture",
            ".musx": "finale",
        }

        return extension_map.get(ext, "unknown")


class BaseParser(ABC):
    """Abstract base class for format-specific parsers"""

    @abstractmethod
    async def parse(
        self, content: Union[str, bytes], encoding: str = "utf-8", **options: Any
    ) -> stream.Score:
        """Parse content into a music21 Score object"""
        pass

    @abstractmethod
    def validate_content(self, content: Union[str, bytes]) -> bool:
        """Validate that content is valid for this format"""
        pass

    async def parse_with_recovery(
        self, content: Union[str, bytes], encoding: str = "utf-8", **options: Any
    ) -> Tuple[stream.Score, List[str]]:
        """Parse with error recovery, returning score and any warnings"""
        warnings: List[str] = []

        try:
            # First try normal parsing
            score = await self.parse(content, encoding, **options)
            return score, warnings

        except Exception as e:
            logger.warning(f"Initial parsing failed: {e}")
            warnings.append(f"Initial parsing failed: {str(e)}")

            # Try recovery strategies
            try:
                score = await self._recover_parse(content, encoding, **options)
                warnings.append("Score parsed with error recovery")
                return score, warnings
            except Exception as recovery_error:
                logger.error(f"Recovery parsing also failed: {recovery_error}")
                raise

    async def _recover_parse(
        self, content: Union[str, bytes], encoding: str = "utf-8", **options
    ) -> stream.Score:
        """Default recovery strategy - override in subclasses"""
        raise NotImplementedError("No recovery strategy implemented")


class MidiParser(BaseParser):
    """MIDI file parser with advanced features"""

    async def parse(
        self,
        content: Union[str, bytes],
        encoding: str = "utf-8",
        preserve_track_info: bool = True,
        quantize: bool = False,
        **options: Any,
    ) -> stream.Score:
        """Parse MIDI content with preservation of performance data"""

        # Handle file path
        if isinstance(content, str) and os.path.exists(content):
            mf = midi.MidiFile()
            mf.open(content)
            mf.read()
            mf.close()
        else:
            # Handle bytes content
            if isinstance(content, str):
                content = content.encode("latin-1")

            with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as f:
                f.write(content)
                f.flush()
                temp_path = f.name

            try:
                mf = midi.MidiFile()
                mf.open(temp_path)
                mf.read()
                mf.close()
            finally:
                os.unlink(temp_path)

        # Convert to score with options
        score = midi.translate.midiFileToStream(
            mf,
            quantizePost=quantize,
            quarterLengthDivisors=(
                options.get("quarter_length_divisors", [4, 3, 2, 1])
                if quantize
                else None
            ),
        )

        # Preserve additional MIDI data
        if preserve_track_info:
            self._preserve_midi_metadata(score, mf)

        return score

    def validate_content(self, content: Union[str, bytes]) -> bool:
        """Validate MIDI content"""
        if isinstance(content, str):
            if os.path.exists(content):
                with open(content, "rb") as f:
                    header = f.read(4)
            else:
                header = content[:4].encode("latin-1")
        else:
            header = content[:4]

        return header == b"MThd"

    def _preserve_midi_metadata(
        self, score: stream.Score, midi_file: midi.MidiFile
    ) -> None:
        """Preserve additional MIDI metadata in the score"""
        # Store MIDI-specific data
        score.metadata.custom["midi_format"] = midi_file.format
        score.metadata.custom["midi_ticks_per_quarter"] = midi_file.ticksPerQuarterNote
        score.metadata.custom["midi_track_count"] = len(midi_file.tracks)

        # Preserve track names
        track_names = []
        for track in midi_file.tracks:
            for event in track.events:
                if isinstance(event, midi.TrackNameEvent):
                    track_names.append(event.data)
                    break

        if track_names:
            score.metadata.custom["midi_track_names"] = track_names

    async def _recover_parse(
        self, content: Union[str, bytes], encoding: str = "utf-8", **options
    ) -> stream.Score:
        """MIDI recovery strategies"""
        # Try with different quantization settings
        for quantize in [True, False]:
            for divisors in [[4, 2, 1], [4, 3, 2, 1], [8, 6, 4, 3, 2, 1]]:
                try:
                    options["quarter_length_divisors"] = divisors
                    return await self.parse(
                        content, encoding, quantize=quantize, **options
                    )
                except:
                    continue

        raise ValueError("Could not parse MIDI file with any recovery strategy")


class MusicXMLParser(BaseParser):
    """MusicXML parser with full metadata preservation"""

    async def parse(
        self,
        content: Union[str, bytes],
        encoding: str = "utf-8",
        compressed: bool = False,
        **options: Any,
    ) -> stream.Score:
        """Parse MusicXML content"""

        # Handle compressed MusicXML (.mxl)
        if compressed or self._is_compressed(content):
            content = await self._decompress_mxl(content)

        # Parse with music21
        if isinstance(content, str) and os.path.exists(content):
            score = converter.parse(content, format="musicxml")
        else:
            with tempfile.NamedTemporaryFile(
                mode="w" if isinstance(content, str) else "wb",
                suffix=".xml",
                encoding=encoding if isinstance(content, str) else None,
                delete=False,
            ) as f:
                f.write(content)
                f.flush()
                temp_path = f.name

            try:
                score = converter.parse(temp_path, format="musicxml")
            finally:
                os.unlink(temp_path)

        return score

    def validate_content(self, content: Union[str, bytes]) -> bool:
        """Validate MusicXML content"""
        if isinstance(content, bytes):
            try:
                content = content.decode("utf-8", errors="ignore")
            except:
                return False

        if isinstance(content, str):
            return bool(re.search(r"<(?:score-partwise|score-timewise|opus)", content))

        return False

    def _is_compressed(self, content: Union[str, bytes]) -> bool:
        """Check if content is compressed MusicXML"""
        if isinstance(content, str) and os.path.exists(content):
            return content.endswith(".mxl")

        if isinstance(content, bytes):
            return content.startswith(b"PK")  # ZIP signature

        return False

    async def _decompress_mxl(self, content: Union[str, bytes]) -> bytes:
        """Decompress MXL (compressed MusicXML) content"""
        if isinstance(content, str) and os.path.exists(content):
            with zipfile.ZipFile(content, "r") as zf:
                # Find the main .xml file
                xml_files = [f for f in zf.namelist() if f.endswith(".xml")]
                if not xml_files:
                    raise ValueError("No XML files found in MXL archive")

                # Prefer specific names
                for preferred in ["score.xml", xml_files[0]]:
                    if preferred in xml_files:
                        return zf.read(preferred)

        # Handle bytes
        with zipfile.ZipFile(io.BytesIO(content), "r") as zf:
            xml_files = [f for f in zf.namelist() if f.endswith(".xml")]
            if xml_files:
                return zf.read(xml_files[0])

        raise ValueError("Could not extract MusicXML from compressed file")

    async def _recover_parse(
        self, content: Union[str, bytes], encoding: str = "utf-8", **options
    ) -> stream.Score:
        """MusicXML recovery strategies"""

        # Try to fix common issues
        if isinstance(content, str):
            # Fix encoding declaration
            content = re.sub(
                r'<\?xml[^>]+encoding=["\'][^"\']+["\'][^>]*\?>',
                '<?xml version="1.0" encoding="UTF-8"?>',
                content,
            )

            # Fix missing closing tags (basic)
            for tag in ["note", "measure", "part"]:
                open_count = content.count(f"<{tag}>")
                close_count = content.count(f"</{tag}>")
                if open_count > close_count:
                    content += f"</{tag}>" * (open_count - close_count)

        return await self.parse(content, encoding, **options)


class ABCParser(BaseParser):
    """ABC notation parser with ornament support"""

    # ABC ornament mappings
    ORNAMENTS = {
        "T": "trill",
        "H": "fermata",
        "L": "emphasis",
        "M": "mordent",
        "P": "pralltriller",
        "S": "segno",
        "O": "coda",
        "u": "upbow",
        "v": "downbow",
    }

    async def parse(
        self,
        content: Union[str, bytes],
        encoding: str = "utf-8",
        expand_repeats: bool = True,
        **options: Any,
    ) -> stream.Score:
        """Parse ABC notation content"""

        if isinstance(content, bytes):
            content = content.decode(encoding)

        # Pre-process ABC notation
        content = self._preprocess_abc(content)

        # Handle file or content
        if os.path.exists(content):
            ah = abcFormat.ABCHandler()
            ah.open(content)
            score = ah.stream
        else:
            # Parse from string
            ah = abcFormat.ABCHandler()
            ah.process(content)
            score = ah.stream

        # Post-process for ornaments and other ABC-specific features
        self._process_ornaments(score)

        if expand_repeats:
            score = score.expandRepeats()

        return score

    def validate_content(self, content: Union[str, bytes]) -> bool:
        """Validate ABC content"""
        if isinstance(content, bytes):
            try:
                content = content.decode("utf-8")
            except:
                return False

        if isinstance(content, str):
            # Check for ABC header fields
            lines = content.strip().split("\n")
            return any(line.startswith(("X:", "T:", "M:", "K:")) for line in lines[:10])

        return False

    def _preprocess_abc(self, content: str) -> str:
        """Pre-process ABC notation for better parsing"""
        lines = content.split("\n")
        processed_lines = []

        for line in lines:
            # Handle inline comments
            if "%" in line and not line.strip().startswith("%"):
                line = line[: line.index("%")]

            # Handle line continuations
            if line.endswith("\\"):
                line = line[:-1] + " "

            processed_lines.append(line)

        return "\n".join(processed_lines)

    def _process_ornaments(self, score: stream.Score) -> None:
        """Add ornaments based on ABC notation"""
        # ABC ornaments are typically indicated with special symbols
        # This is a simplified implementation
        for element in score.flatten():
            if hasattr(element, "lyric") and element.lyric:
                for ornament_symbol, ornament_name in self.ORNAMENTS.items():
                    if ornament_symbol in element.lyric:
                        # Add the ornament
                        if ornament_name == "trill":
                            element.expressions.append(expressions.Trill())
                        elif ornament_name == "fermata":
                            element.expressions.append(expressions.Fermata())
                        elif ornament_name == "mordent":
                            element.expressions.append(expressions.Mordent())

    async def _recover_parse(
        self, content: Union[str, bytes], encoding: str = "utf-8", **options
    ) -> stream.Score:
        """ABC recovery strategies"""

        if isinstance(content, bytes):
            content = content.decode(encoding, errors="ignore")

        # Try to fix common ABC issues
        lines = content.split("\n")
        fixed_lines = []

        has_header = False
        has_key = False

        for line in lines:
            # Ensure we have essential headers
            if line.startswith("X:"):
                has_header = True
            elif line.startswith("K:"):
                has_key = True

            fixed_lines.append(line)

        # Add missing headers
        if not has_header:
            fixed_lines.insert(0, "X:1")

        if not has_key:
            # Try to find where to insert key
            for i, line in enumerate(fixed_lines):
                if line.strip() and not line.startswith(tuple("XTMKLQVW:")):
                    fixed_lines.insert(i, "K:C")
                    break
            else:
                fixed_lines.append("K:C")

        fixed_content = "\n".join(fixed_lines)
        return await self.parse(fixed_content, encoding, **options)


class KernParser(BaseParser):
    """Kern format parser for computational musicology"""

    async def parse(
        self, content: Union[str, bytes], encoding: str = "utf-8", **options
    ) -> stream.Score:
        """Parse Kern format content"""

        if isinstance(content, bytes):
            content = content.decode(encoding)

        # music21 can handle Kern directly
        if os.path.exists(content):
            score = converter.parse(content, format="humdrum")
        else:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".krn", encoding=encoding, delete=False
            ) as f:
                f.write(content)
                f.flush()
                temp_path = f.name

            try:
                score = converter.parse(temp_path, format="humdrum")
            finally:
                os.unlink(temp_path)

        return score

    def validate_content(self, content: Union[str, bytes]) -> bool:
        """Validate Kern content"""
        if isinstance(content, bytes):
            try:
                content = content.decode("utf-8")
            except:
                return False

        if isinstance(content, str):
            lines = content.strip().split("\n")
            # Check for Kern spine indicators
            return any("**kern" in line or "**dynam" in line for line in lines[:20])

        return False

    async def _recover_parse(
        self, content: Union[str, bytes], encoding: str = "utf-8", **options
    ) -> stream.Score:
        """Kern recovery strategies"""

        if isinstance(content, bytes):
            content = content.decode(encoding, errors="ignore")

        # Ensure proper spine termination
        if "**kern" in content and "*-" not in content:
            content += "\n*-"

        return await self.parse(content, encoding, **options)


class ScoreParserFactory:
    """Factory for creating appropriate parsers"""

    PARSERS = {
        "midi": MidiParser,
        "musicxml": MusicXMLParser,
        "abc": ABCParser,
        "kern": KernParser,
        # Add more parsers as implemented
    }

    @classmethod
    def create_parser(cls, format_type: str) -> Optional[BaseParser]:
        """Create a parser for the given format"""
        parser_class = cls.PARSERS.get(format_type.lower())
        if parser_class:
            return parser_class()
        return None


class SmartScoreParser:
    """High-level parser with format detection and error recovery"""

    def __init__(self) -> None:
        self.detector = FormatDetector()
        self.factory = ScoreParserFactory()
        self._parser_cache = {}

    async def parse_score(
        self,
        source: Union[str, bytes, Dict[str, Any]],
        format_hint: Optional[str] = None,
        encoding: str = "auto",
        validate: bool = True,
        recover_errors: bool = True,
        **parser_options,
    ) -> Tuple[stream.Score, Dict[str, Any]]:
        """
        Parse a score from various sources with intelligent handling

        Args:
            source: The source to parse
            format_hint: Optional format hint
            encoding: Character encoding ('auto' for detection)
            validate: Whether to validate after parsing
            recover_errors: Whether to attempt error recovery
            **parser_options: Format-specific parser options

        Returns:
            Tuple of (parsed_score, metadata_dict)
        """
        metadata = {
            "parse_start": datetime.now(),
            "source_type": type(source).__name__,
            "format_hint": format_hint,
            "encoding_requested": encoding,
        }

        try:
            # Detect encoding if needed
            if encoding == "auto":
                encoding = await self._detect_encoding(source)
                metadata["encoding_detected"] = encoding

            # Detect format
            if format_hint:
                detected_format = format_hint
            else:
                detected_format = await self.detector.detect_format(source)
                metadata["format_detected"] = detected_format

            if detected_format == "unknown":
                raise ValueError("Could not detect file format")

            # Get appropriate parser
            parser = self._get_parser(detected_format)
            if not parser:
                raise ValueError(f"No parser available for format: {detected_format}")

            # Load content
            content = await self._load_content(source, encoding)

            # Validate content if requested
            if validate and not parser.validate_content(content):
                if not recover_errors:
                    raise ValueError(f"Invalid {detected_format} content")
                metadata["validation_failed"] = True

            # Parse with or without recovery
            if recover_errors:
                score, warnings = await parser.parse_with_recovery(
                    content, encoding, **parser_options
                )
                if warnings:
                    metadata["warnings"] = warnings
            else:
                score = await parser.parse(content, encoding, **parser_options)

            # Post-parsing validation
            if validate:
                validation_results = self._validate_parsed_score(score)
                metadata["validation"] = validation_results

            metadata["parse_end"] = datetime.now()
            metadata["parse_duration"] = (
                metadata["parse_end"] - metadata["parse_start"]
            ).total_seconds()

            return score, metadata

        except Exception as e:
            metadata["error"] = str(e)
            metadata["error_type"] = type(e).__name__
            raise

    def _get_parser(self, format_type: str) -> Optional[BaseParser]:
        """Get or create a parser instance"""
        if format_type not in self._parser_cache:
            parser = self.factory.create_parser(format_type)
            if parser:
                self._parser_cache[format_type] = parser

        return self._parser_cache.get(format_type)

    async def _detect_encoding(self, source: Union[str, bytes, Dict[str, Any]]) -> str:
        """Detect text encoding"""
        sample = None

        if isinstance(source, bytes):
            sample = source[:1024]
        elif isinstance(source, str) and os.path.exists(source):
            async with aiofiles.open(source, "rb") as f:
                sample = await f.read(1024)
        elif isinstance(source, dict) and "content" in source:
            content = source["content"]
            if isinstance(content, bytes):
                sample = content[:1024]
            elif isinstance(content, str):
                return "utf-8"  # Already decoded

        if sample:
            result = chardet.detect(sample)
            return result["encoding"] or "utf-8"

        return "utf-8"

    async def _load_content(
        self, source: Union[str, bytes, Dict[str, Any]], encoding: str
    ) -> Union[str, bytes]:
        """Load content from various sources"""

        if isinstance(source, bytes):
            return source

        elif isinstance(source, str):
            if source.startswith(("http://", "https://")):
                # Download from URL
                async with httpx.AsyncClient() as client:
                    response = await client.get(source)
                    response.raise_for_status()
                    return response.content

            elif os.path.exists(source):
                # Return file path for parsers that can handle it
                return source

            else:
                # Assume it's content
                return source

        elif isinstance(source, dict):
            if "content" in source:
                return source["content"]
            elif "path" in source:
                return source["path"]
            elif "url" in source:
                async with httpx.AsyncClient() as client:
                    response = await client.get(source["url"])
                    response.raise_for_status()
                    return response.content

        raise ValueError(f"Cannot load content from source: {type(source)}")

    def _validate_parsed_score(self, score: stream.Score) -> Dict[str, Any]:
        """Validate a parsed score"""
        validation = {"valid": True, "issues": [], "warnings": [], "stats": {}}

        # Basic structure checks
        if not score:
            validation["valid"] = False
            validation["issues"].append("Score is empty")
            return validation

        # Check for parts
        parts = score.parts
        validation["stats"]["part_count"] = len(parts)

        if len(parts) == 0:
            validation["warnings"].append("No parts found in score")

        # Check for measures
        measures = score.flatten().getElementsByClass(stream.Measure)
        validation["stats"]["measure_count"] = len(measures)

        # Check for notes
        notes = score.flatten().notes
        validation["stats"]["note_count"] = len(notes)

        if len(notes) == 0:
            validation["warnings"].append("No notes found in score")

        # Check measure consistency
        for part in parts:
            part_measures = part.getElementsByClass(stream.Measure)
            for measure in part_measures:
                if hasattr(measure, "number"):
                    expected_duration = measure.barDuration.quarterLength
                    actual_duration = measure.duration.quarterLength

                    if abs(expected_duration - actual_duration) > 0.01:
                        validation["warnings"].append(
                            f"Measure {measure.number} in part {part.id or 'unknown'}: "
                            f"duration mismatch (expected {expected_duration}, got {actual_duration})"
                        )

        return validation


# Convenience function for direct use
async def parse_score(
    source: Union[str, bytes, Dict[str, Any]], **options
) -> Tuple[stream.Score, Dict[str, Any]]:
    """
    Parse a musical score from various sources

    This is a convenience function that creates a SmartScoreParser
    and parses the score with the given options.
    """
    parser = SmartScoreParser()
    return await parser.parse_score(source, **options)


# Add import for io module
import io
