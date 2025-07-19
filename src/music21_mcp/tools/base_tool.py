"""
Base class for all music21 MCP tools
Provides common functionality and interface contracts
"""

import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Generator
from contextlib import contextmanager
from typing import Any

logger = logging.getLogger(__name__)


class BaseTool(ABC):
    """
    Base class for all MCP tools with:
    - Clear input/output contracts
    - Comprehensive error handling
    - Progress reporting for long operations
    - Memory-efficient processing
    """

    def __init__(self, score_manager: dict[str, Any]):
        """Initialize with reference to score manager"""
        self.score_manager = score_manager
        self._progress_callback: Callable | None = None

    @property
    def scores(self) -> dict[str, Any]:
        """Provide access to scores for backward compatibility with tests"""
        return self.score_manager

    @abstractmethod
    async def execute(self, **kwargs: Any) -> dict[str, Any]:
        """
        Execute the tool operation

        Returns:
            Dict with at minimum:
            - status: 'success' or 'error'
            - message: Human-readable message
            - Additional tool-specific fields
        """
        pass

    @abstractmethod
    def validate_inputs(self, **kwargs: Any) -> str | None:
        """
        Validate input parameters

        Returns:
            None if valid, error message if invalid
        """
        pass

    def set_progress_callback(self, callback: Callable[[float, str], None]) -> None:
        """Set callback for progress reporting (percent, message)"""
        self._progress_callback = callback

    def report_progress(self, percent: float, message: str = "") -> None:
        """Report progress if callback is set"""
        if self._progress_callback:
            self._progress_callback(percent, message)

    @contextmanager
    def error_handling(self, operation: str) -> Generator[None, None, None]:
        """Context manager for consistent error handling"""
        try:
            start_time = time.time()
            yield
            duration = time.time() - start_time
            if duration > 1.0:
                logger.info(f"{operation} completed in {duration:.2f}s")
        except Exception as e:
            logger.error(f"{operation} failed: {str(e)}")
            raise

    def create_error_response(
        self, message: str, details: dict | None = None
    ) -> dict[str, Any]:
        """Create standardized error response"""
        response = {"status": "error", "message": message}
        if details:
            response.update(details)
        return response

    def create_success_response(
        self, message: str = "Operation completed successfully", **kwargs: Any
    ) -> dict[str, Any]:
        """Create standardized success response"""
        response = {"status": "success", "message": message}
        response.update(kwargs)
        return response

    def check_score_exists(self, score_id: str) -> str | None:
        """Check if score exists, return error message if not"""
        if score_id not in self.score_manager:
            return f"Score with ID '{score_id}' not found"
        return None

    def get_score(self, score_id: str) -> Any:
        """Get score from manager"""
        return self.score_manager.get(score_id)
