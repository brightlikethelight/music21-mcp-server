"""Security tests for ImportScoreTool._validate_safe_path()"""

import tempfile
from pathlib import Path

import pytest

from music21_mcp.tools import ImportScoreTool


@pytest.fixture
def tool():
    return ImportScoreTool(score_manager={})


class TestValidateSafePath:
    def test_rejects_directory_traversal(self, tool):
        with pytest.raises(ValueError, match="outside allowed directories"):
            tool._validate_safe_path("../../etc/passwd")

    def test_rejects_absolute_path_outside_allowed(self, tool):
        with pytest.raises(ValueError, match="outside allowed directories"):
            tool._validate_safe_path("/etc/passwd")

    def test_accepts_path_within_cwd(self, tool):
        cwd = Path.cwd()
        result = tool._validate_safe_path(str(cwd / "test.xml"))
        assert result == str((cwd / "test.xml").resolve())

    def test_accepts_path_within_tempdir(self, tool):
        tmp = Path(tempfile.gettempdir()).resolve()
        result = tool._validate_safe_path(str(tmp / "score.mid"))
        assert result == str((tmp / "score.mid").resolve())

    def test_rejects_traversal_via_symlink_like_path(self, tool):
        with pytest.raises(ValueError, match="outside allowed directories"):
            tool._validate_safe_path("/var/log/../../../etc/shadow")
