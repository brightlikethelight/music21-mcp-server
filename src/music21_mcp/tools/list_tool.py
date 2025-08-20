"""
List Scores Tool - List all imported scores with metadata
"""

import logging
from typing import Any

from .base_tool import BaseTool

logger = logging.getLogger(__name__)


class ListScoresTool(BaseTool):
    """Tool for listing all imported scores"""

    async def execute(self, **kwargs: Any) -> dict[str, Any]:
        """
        List all imported scores with their metadata

        Returns:
            Dict containing count and list of scores
        """
        with self.error_handling("List scores"):
            scores = []

            for score_id, score in self.score_manager.items():
                try:
                    # Extract basic info for each score
                    info = {
                        "score_id": score_id,  # Changed from "id" to "score_id" to match test expectations
                        "measures": len(
                            list(score.flatten().getElementsByClass("Measure"))
                        ),
                        "notes": len(list(score.flatten().notes)),
                        "parts": len(score.parts) if hasattr(score, "parts") else 1,
                    }

                    # Try to get title
                    if hasattr(score, "metadata"):
                        if score.metadata.title:
                            info["title"] = score.metadata.title
                        elif score.metadata.movementName:
                            info["title"] = score.metadata.movementName

                    scores.append(info)

                except Exception as e:
                    logger.warning(f"Error processing score {score_id}: {e}")
                    scores.append(
                        {"score_id": score_id, "error": "Failed to extract metadata"}
                    )

            # Sort by ID for consistent ordering
            scores.sort(key=lambda x: str(x.get("score_id", "")))

            return self.create_success_response(count=len(scores), scores=scores)

    def validate_inputs(self, **kwargs: Any) -> str | None:
        """No inputs to validate for list operation"""
        return None
