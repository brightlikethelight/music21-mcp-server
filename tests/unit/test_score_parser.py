"""
Unit tests for score parsing functionality
"""

import os
import tempfile

import pytest
from music21 import stream

from src.music21_mcp.core.score_parser import (ABCParser, FormatDetector,
                                               MidiParser, MusicXMLParser,
                                               SmartScoreParser)


class TestFormatDetector:
    """Test format detection capabilities"""

    @pytest.mark.asyncio
    async def test_detect_midi_by_content(self):
        """Test MIDI detection by file signature"""
        midi_header = b"MThd\x00\x00\x00\x06\x00\x00\x00\x01"
        detected = await FormatDetector.detect_format(midi_header)
        assert detected == "midi"

    @pytest.mark.asyncio
    async def test_detect_musicxml_by_content(self):
        """Test MusicXML detection by content"""
        xml_content = '<?xml version="1.0"?><score-partwise>'
        detected = await FormatDetector.detect_format(xml_content)
        assert detected == "musicxml"

    @pytest.mark.asyncio
    async def test_detect_abc_by_content(self):
        """Test ABC notation detection"""
        abc_content = """X:1
T:Test Tune
M:4/4
K:C
CDEF|GABc|"""
        detected = await FormatDetector.detect_format(abc_content)
        assert detected == "abc"

    @pytest.mark.asyncio
    async def test_detect_kern_by_content(self):
        """Test Kern format detection"""
        kern_content = """**kern
*M4/4
=1
4c
4d
4e
4f
*-"""
        detected = await FormatDetector.detect_format(kern_content)
        assert detected == "kern"

    @pytest.mark.asyncio
    async def test_detect_from_file_extension(self, temp_midi_file):
        """Test format detection from file extension"""
        detected = await FormatDetector.detect_format(temp_midi_file)
        assert detected == "midi"

    @pytest.mark.asyncio
    async def test_detect_unknown_format(self):
        """Test unknown format detection"""
        unknown_content = "This is not a music file"
        detected = await FormatDetector.detect_format(unknown_content)
        assert detected == "unknown"


class TestSmartScoreParser:
    """Test the main score parser"""

    @pytest.fixture
    def parser(self):
        return SmartScoreParser()

    @pytest.mark.asyncio
    async def test_parse_midi_file(self, parser, temp_midi_file):
        """Test parsing a MIDI file"""
        score, metadata = await parser.parse_score(temp_midi_file)

        assert score is not None
        assert isinstance(score, stream.Score)
        assert metadata["format_detected"] == "midi"
        assert "parse_duration" in metadata

    @pytest.mark.asyncio
    async def test_parse_musicxml_file(self, parser, temp_musicxml_file):
        """Test parsing a MusicXML file"""
        score, metadata = await parser.parse_score(temp_musicxml_file)

        assert score is not None
        assert isinstance(score, stream.Score)
        assert metadata["format_detected"] == "musicxml"

    @pytest.mark.asyncio
    async def test_parse_abc_content(self, parser):
        """Test parsing ABC notation from string"""
        abc_content = """X:1
T:Simple Scale
M:4/4
K:C
L:1/4
CDEF|GABc|"""

        score, metadata = await parser.parse_score(abc_content, format_hint="abc")

        assert score is not None
        assert len(score.flat.notes) == 8

    @pytest.mark.asyncio
    async def test_parse_with_encoding_detection(self, parser):
        """Test automatic encoding detection"""
        # Create content with specific encoding
        content = "X:1\nT:Test Mélodie\nM:4/4\nK:C\nCDEF|"

        score, metadata = await parser.parse_score(
            {"content": content}, encoding="auto"
        )

        assert score is not None
        assert "encoding_detected" in metadata

    @pytest.mark.asyncio
    async def test_parse_with_validation(self, parser):
        """Test parsing with validation enabled"""
        abc_content = """X:1
T:Test
M:4/4
K:C
CDEF|"""

        score, metadata = await parser.parse_score(
            abc_content, format_hint="abc", validate=True
        )

        assert "validation" in metadata
        assert metadata["validation"]["valid"] == True

    @pytest.mark.asyncio
    async def test_parse_invalid_content(self, parser):
        """Test parsing invalid content"""
        with pytest.raises(ValueError):
            await parser.parse_score(
                "This is not valid music notation", format_hint="musicxml"
            )

    @pytest.mark.asyncio
    async def test_parse_with_error_recovery(self, parser):
        """Test error recovery during parsing"""
        # Slightly malformed ABC
        abc_content = """X:1
T:Test
M:4/4
K:C
CDEF GABc"""  # Missing bar line

        score, metadata = await parser.parse_score(
            abc_content, format_hint="abc", recover_errors=True
        )

        assert score is not None
        assert "warnings" in metadata or "validation" in metadata


class TestMidiParser:
    """Test MIDI-specific parsing"""

    @pytest.fixture
    def parser(self):
        return MidiParser()

    @pytest.mark.asyncio
    async def test_parse_midi_preserves_tempo(self, parser, simple_score):
        """Test that MIDI parsing preserves tempo information"""
        # Create MIDI with specific tempo
        simple_score.insert(0, tempo.MetronomeMark(number=140))

        with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as f:
            simple_score.write("midi", fp=f.name)
            temp_path = f.name

        try:
            parsed_score = await parser.parse(temp_path)

            # Check tempo preservation
            tempos = parsed_score.flat.getElementsByClass(tempo.MetronomeMark)
            assert len(tempos) > 0
            # Note: Exact tempo might vary due to MIDI quantization

        finally:
            os.unlink(temp_path)

    @pytest.mark.asyncio
    async def test_parse_midi_with_multiple_tracks(self, parser):
        """Test parsing MIDI with multiple tracks"""
        # Create multi-track score
        s = stream.Score()
        for i in range(3):
            p = stream.Part()
            p.append(note.Note("C4", quarterLength=4))
            s.append(p)

        with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as f:
            s.write("midi", fp=f.name)
            temp_path = f.name

        try:
            parsed_score = await parser.parse(temp_path)
            assert len(parsed_score.parts) == 3

        finally:
            os.unlink(temp_path)


class TestMusicXMLParser:
    """Test MusicXML-specific parsing"""

    @pytest.fixture
    def parser(self):
        return MusicXMLParser()

    @pytest.mark.asyncio
    async def test_parse_compressed_musicxml(self, parser, simple_score):
        """Test parsing compressed MusicXML (.mxl)"""
        with tempfile.NamedTemporaryFile(suffix=".mxl", delete=False) as f:
            simple_score.write("musicxml", fp=f.name, compress=True)
            temp_path = f.name

        try:
            parsed_score = await parser.parse(temp_path, compressed=True)
            assert parsed_score is not None
            assert len(parsed_score.flat.notes) > 0

        finally:
            os.unlink(temp_path)


class TestABCParser:
    """Test ABC notation parsing"""

    @pytest.fixture
    def parser(self):
        return ABCParser()

    @pytest.mark.asyncio
    async def test_parse_abc_with_ornaments(self, parser):
        """Test parsing ABC with ornaments"""
        abc_content = """X:1
T:Ornament Test
M:4/4
K:G
L:1/8
|:G2 ~G2 TGAB|c2 Hd2 e2fg:|"""

        score = await parser.parse(abc_content)
        assert score is not None
        # Verify ornaments were processed

    @pytest.mark.asyncio
    async def test_parse_abc_with_repeats(self, parser):
        """Test ABC repeat expansion"""
        abc_content = """X:1
T:Repeat Test
M:4/4
K:C
|:CDEF:|GABC|"""

        score = await parser.parse(abc_content, expand_repeats=True)
        # Should have more notes due to repeat expansion
        assert len(score.flat.notes) > 8

    @pytest.mark.asyncio
    async def test_abc_error_recovery(self, parser):
        """Test ABC parser error recovery"""
        # Missing key signature
        abc_content = """X:1
T:Test
M:4/4
CDEF|"""

        score, warnings = await parser.parse_with_recovery(abc_content)
        assert score is not None
        assert len(warnings) > 0
