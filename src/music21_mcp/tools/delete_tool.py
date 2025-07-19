"""
Delete Score Tool - Remove scores from memory
"""

import logging
from typing import Any

from .base_tool import BaseTool

logger = logging.getLogger(__name__)


class DeleteScoreTool(BaseTool):
    """Tool for deleting scores from memory"""

    async def execute(self, **kwargs: Any) -> dict[str, Any]:
        """
        Delete a score or all scores from memory

        Args:
            **kwargs: Keyword arguments including:
                score_id: ID of score to delete, or '*' to delete all
        """
        # Extract parameters from kwargs
        score_id = kwargs.get("score_id", "")
        # Validate inputs
        error = self.validate_inputs(**kwargs)
        if error:
            return self.create_error_response(error)

        with self.error_handling("Delete score(s)"):
            if score_id == "*":
                # Delete all scores
                count = len(self.score_manager)

                if count == 0:
                    return self.create_success_response(
                        message="No scores to delete", deleted_count=0
                    )

                self.report_progress(0.5, f"Deleting {count} scores")

                # Clear all scores
                self.score_manager.clear()

                self.report_progress(1.0, "All scores deleted")

                return self.create_success_response(
                    message=f"Deleted all {count} scores", deleted_count=count
                )
            # Delete single score
            if score_id not in self.score_manager:
                return self.create_error_response(f"Score '{score_id}' not found")

            self.report_progress(0.5, f"Deleting score '{score_id}'")

            # Remove the score
            del self.score_manager[score_id]

            self.report_progress(1.0, "Score deleted")

            return self.create_success_response(
                message=f"Deleted score '{score_id}'", deleted_count=1
            )

    def validate_inputs(self, **kwargs: Any) -> str | None:
        """Validate input parameters"""
        score_id = kwargs.get("score_id", "")

        if not score_id:
            return "Score '' not found"

        # Allow wildcard for delete all
        if score_id == "*":
            return None

        # For single delete, no need to check existence here
        # We'll handle it in execute for better error message
        return None
