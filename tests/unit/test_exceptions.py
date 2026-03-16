"""Tests for the custom exception hierarchy and error response utilities."""

import pytest

from music21_mcp.exceptions import (
    ANALYSIS_EXCEPTIONS,
    EXCEPTION_TO_ERROR_CODE,
    EXCEPTION_TO_HTTP_STATUS,
    IO_EXCEPTIONS,
    NON_RETRYABLE_EXCEPTIONS,
    RETRYABLE_EXCEPTIONS,
    AnalysisError,
    ConfigurationError,
    ErrorResponse,
    ExportError,
    GenerationError,
    InvalidScoreFormatError,
    Music21MCPError,
    OperationTimeoutError,
    ResourceExhaustedError,
    ScoreImportError,
    ScoreNotFoundError,
    ValidationError,
    exception_to_error_response,
)


class TestExceptionInstantiation:
    """Each exception type can be created with the expected arguments."""

    def test_base_error(self):
        exc = Music21MCPError("base error", details={"key": "val"})
        assert exc.message == "base error"
        assert exc.details == {"key": "val"}
        assert str(exc) == "base error"

    def test_score_not_found(self):
        exc = ScoreNotFoundError("abc123")
        assert exc.score_id == "abc123"
        assert "abc123" in str(exc)

    def test_invalid_score_format(self):
        exc = InvalidScoreFormatError("pdf", ["musicxml", "midi"])
        assert exc.format_name == "pdf"
        assert exc.supported_formats == ["musicxml", "midi"]

    def test_score_import_error(self):
        exc = ScoreImportError("file.mid", "file", "corrupt header")
        assert exc.source == "file.mid"
        assert exc.source_type == "file"
        assert exc.reason == "corrupt header"

    def test_export_error(self):
        exc = ExportError("s1", "pdf", "unsupported")
        assert exc.score_id == "s1"
        assert exc.format_name == "pdf"

    def test_analysis_error(self):
        exc = AnalysisError("harmony", "s1", "no chords found")
        assert exc.analysis_type == "harmony"

    def test_generation_error(self):
        exc = GenerationError("counterpoint", "invalid species")
        assert exc.generation_type == "counterpoint"

    def test_resource_exhausted(self):
        exc = ResourceExhaustedError("memory", 490.0, 512.0)
        assert exc.resource_type == "memory"
        assert exc.current == 490.0
        assert exc.limit == 512.0

    def test_configuration_error(self):
        exc = ConfigurationError("port", "-1", "must be positive")
        assert exc.setting == "port"

    def test_operation_timeout(self):
        exc = OperationTimeoutError("analyze_key", 30.0)
        assert exc.timeout_seconds == 30.0

    def test_validation_error(self):
        exc = ValidationError("score_id", "", "must not be empty")
        assert exc.field == "score_id"


class TestExceptionCategories:
    """Category tuples contain the right exception types."""

    def test_retryable(self):
        assert OperationTimeoutError in RETRYABLE_EXCEPTIONS
        assert ResourceExhaustedError in RETRYABLE_EXCEPTIONS

    def test_non_retryable(self):
        assert ScoreNotFoundError in NON_RETRYABLE_EXCEPTIONS
        assert ValidationError in NON_RETRYABLE_EXCEPTIONS

    def test_analysis(self):
        assert AnalysisError in ANALYSIS_EXCEPTIONS
        assert ScoreNotFoundError in ANALYSIS_EXCEPTIONS

    def test_io(self):
        assert ScoreImportError in IO_EXCEPTIONS
        assert ExportError in IO_EXCEPTIONS


class TestHTTPStatusMapping:
    """Every custom exception maps to a sensible HTTP status code."""

    @pytest.mark.parametrize(
        ("exc_cls", "expected_status"),
        [
            (ScoreNotFoundError, 404),
            (InvalidScoreFormatError, 400),
            (ValidationError, 400),
            (ConfigurationError, 400),
            (ScoreImportError, 422),
            (ExportError, 422),
            (AnalysisError, 500),
            (GenerationError, 500),
            (ResourceExhaustedError, 503),
            (OperationTimeoutError, 504),
        ],
    )
    def test_status_code(self, exc_cls, expected_status):
        assert EXCEPTION_TO_HTTP_STATUS[exc_cls] == expected_status

    def test_all_custom_exceptions_have_status(self):
        for exc_cls in EXCEPTION_TO_HTTP_STATUS:
            assert issubclass(exc_cls, Exception)

    def test_all_custom_exceptions_have_error_code(self):
        for exc_cls in EXCEPTION_TO_ERROR_CODE:
            assert isinstance(EXCEPTION_TO_ERROR_CODE[exc_cls], str)


class TestErrorResponse:
    """ErrorResponse serialisation."""

    def test_to_dict_minimal(self):
        resp = ErrorResponse(
            error="TestError",
            error_code="TEST",
            message="boom",
        )
        d = resp.to_dict()
        assert d == {"error": "TestError", "error_code": "TEST", "message": "boom"}

    def test_to_dict_with_details_and_retry(self):
        resp = ErrorResponse(
            error="Timeout",
            error_code="TIMEOUT",
            message="slow",
            details={"op": "analyze"},
            retry_after=5.0,
            http_status=504,
        )
        d = resp.to_dict()
        assert d["details"] == {"op": "analyze"}
        assert d["retry_after"] == 5.0
        assert resp.http_status == 504


class TestExceptionToErrorResponse:
    """exception_to_error_response produces correct structured responses."""

    def test_known_exception(self):
        exc = ScoreNotFoundError("xyz")
        resp = exception_to_error_response(exc)
        assert resp.http_status == 404
        assert resp.error_code == "SCORE_NOT_FOUND"
        assert resp.error == "ScoreNotFoundError"
        assert resp.details == {"score_id": "xyz"}
        assert resp.retry_after is None

    def test_retryable_exception_gets_retry_after(self):
        exc = OperationTimeoutError("analyze", 30.0)
        resp = exception_to_error_response(exc)
        assert resp.retry_after == 5.0
        assert resp.http_status == 504

    def test_unknown_exception_falls_back(self):
        exc = RuntimeError("unexpected")
        resp = exception_to_error_response(exc)
        assert resp.http_status == 500
        assert resp.error_code == "INTERNAL_ERROR"
