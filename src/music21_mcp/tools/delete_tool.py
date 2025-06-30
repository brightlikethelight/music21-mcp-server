"""
Delete Score Tool - Remove scores from memory
"""
import logging
from typing import Any, Dict, Optional

from .base_tool import BaseTool

logger = logging.getLogger(__name__)


class DeleteScoreTool(BaseTool):
    """Tool for deleting scores from memory"""
    
    async def execute(self, score_id: str) -> Dict[str, Any]:
        """
        Delete a score or all scores from memory
        
        Args:
            score_id: ID of score to delete, or '*' to delete all
        """
        # Validate inputs
        error = self.validate_inputs(score_id=score_id)
        if error:
            return self.create_error_response(error)
        
        with self.error_handling("Delete score(s)"):
            if score_id == "*":
                # Delete all scores
                count = len(self.score_manager)
                
                if count == 0:
                    return self.create_success_response(
                        message="No scores to delete",
                        deleted_count=0
                    )
                
                self.report_progress(0.5, f"Deleting {count} scores")
                
                # Clear all scores
                self.score_manager.clear()
                
                self.report_progress(1.0, "All scores deleted")
                
                return self.create_success_response(
                    message=f"Deleted all {count} scores",
                    deleted_count=count
                )
            else:
                # Delete single score
                if score_id not in self.score_manager:
                    return self.create_error_response(f"Score '{score_id}' not found")
                
                self.report_progress(0.5, f"Deleting score '{score_id}'")
                
                # Remove the score
                del self.score_manager[score_id]
                
                self.report_progress(1.0, "Score deleted")
                
                return self.create_success_response(
                    message=f"Deleted score '{score_id}'",
                    deleted_count=1
                )
    
    def validate_inputs(self, score_id: str) -> Optional[str]:
        """Validate input parameters"""
        if not score_id:
            return "score_id cannot be empty"
        
        # Allow wildcard for delete all
        if score_id == "*":
            return None
        
        # For single delete, no need to check existence here
        # We'll handle it in execute for better error message
        return None