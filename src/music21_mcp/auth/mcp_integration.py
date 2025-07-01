"""
Integration between OAuth2 authentication and MCP server
Provides secure remote access to MCP tools via OAuth2
"""
import logging
from typing import Dict, Optional, Any
from functools import wraps

from fastapi import Depends, HTTPException, Request, status
from mcp import Tool

from .security import SecurityMiddleware, require_scope
from .models import AccessToken, User
from .storage import OAuth2Storage

logger = logging.getLogger(__name__)


class MCPAuthContext:
    """Authentication context for MCP operations"""
    
    def __init__(
        self,
        user: Optional[User] = None,
        client_id: Optional[str] = None,
        access_token: Optional[AccessToken] = None,
        scope: str = "read"
    ):
        self.user = user
        self.client_id = client_id
        self.access_token = access_token
        self.scope = scope
    
    @property
    def user_id(self) -> Optional[str]:
        return self.user.user_id if self.user else None
    
    @property
    def username(self) -> Optional[str]:
        return self.user.username if self.user else None
    
    def has_scope(self, required_scope: str) -> bool:
        """Check if context has required scope"""
        if not self.access_token:
            return False
        return self.access_token.has_scope(required_scope)
    
    def has_permission(self, permission: str) -> bool:
        """Check if user has permission"""
        if not self.user:
            return False
        return self.user.has_permission(permission)


class AuthenticatedMCPServer:
    """MCP server wrapper with OAuth2 authentication"""
    
    def __init__(self, mcp_server, oauth2_storage: OAuth2Storage):
        self.mcp_server = mcp_server
        self.oauth2_storage = oauth2_storage
        self.tool_permissions: Dict[str, str] = {}  # tool_name -> required_scope
        self._setup_default_permissions()
    
    def _setup_default_permissions(self):
        """Set up default tool permissions"""
        # Read operations
        read_tools = [
            "list_scores", "get_score_info", "analyze_harmony",
            "analyze_key", "analyze_melody", "find_parallel_fifths",
            "get_notation", "search_notes", "get_part_info"
        ]
        for tool in read_tools:
            self.tool_permissions[tool] = "read"
        
        # Write operations
        write_tools = [
            "import_score", "transpose_score", "extract_parts",
            "merge_scores", "harmonize_melody", "create_counterpoint",
            "imitate_style"
        ]
        for tool in write_tools:
            self.tool_permissions[tool] = "write"
        
        # Admin operations
        admin_tools = ["clear_cache", "reset_server"]
        for tool in admin_tools:
            self.tool_permissions[tool] = "admin"
    
    def set_tool_permission(self, tool_name: str, required_scope: str):
        """Set required scope for a tool"""
        self.tool_permissions[tool_name] = required_scope
    
    async def check_tool_access(self, tool_name: str, auth_context: MCPAuthContext) -> bool:
        """Check if auth context has access to tool"""
        required_scope = self.tool_permissions.get(tool_name, "read")
        
        # Check scope
        if not auth_context.has_scope(required_scope):
            logger.warning(f"Access denied to {tool_name}: missing scope {required_scope}")
            return False
        
        # Additional permission checks for admin tools
        if required_scope == "admin" and not auth_context.has_permission("admin"):
            logger.warning(f"Access denied to {tool_name}: missing admin permission")
            return False
        
        return True
    
    async def list_tools_for_context(self, auth_context: MCPAuthContext) -> list:
        """List tools available for the given auth context"""
        available_tools = []
        
        for tool_name, tool in self.mcp_server.tools.items():
            if await self.check_tool_access(tool_name, auth_context):
                available_tools.append({
                    "name": tool_name,
                    "description": tool.description,
                    "required_scope": self.tool_permissions.get(tool_name, "read")
                })
        
        return available_tools
    
    async def execute_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        auth_context: MCPAuthContext
    ) -> Any:
        """Execute MCP tool with authentication check"""
        # Check access
        if not await self.check_tool_access(tool_name, auth_context):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions for tool: {tool_name}"
            )
        
        # Get tool
        tool = self.mcp_server.tools.get(tool_name)
        if not tool:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Tool not found: {tool_name}"
            )
        
        # Log access
        logger.info(
            f"Tool access: user={auth_context.username}, "
            f"tool={tool_name}, client={auth_context.client_id}"
        )
        
        # Execute tool
        try:
            result = await tool.execute(arguments)
            return result
        except Exception as e:
            logger.error(f"Tool execution error: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Tool execution failed: {str(e)}"
            )


def mcp_auth_required(required_scope: str = "read"):
    """Decorator for MCP endpoints requiring authentication"""
    def decorator(func):
        @wraps(func)
        async def wrapper(
            request: Request,
            security: AccessToken = Depends(SecurityMiddleware()),
            scope_check = Depends(require_scope(required_scope)),
            *args,
            **kwargs
        ):
            # Get user from access token
            storage: OAuth2Storage = request.app.state.oauth2_storage
            user = None
            if security.user_id:
                user = await storage.get_user(security.user_id)
            
            # Create auth context
            auth_context = MCPAuthContext(
                user=user,
                client_id=security.client_id,
                access_token=security,
                scope=security.scope
            )
            
            # Add auth context to kwargs
            kwargs['auth_context'] = auth_context
            
            return await func(request, *args, **kwargs)
        
        return wrapper
    return decorator


class MCPSessionManager:
    """Manages MCP sessions with OAuth2 integration"""
    
    def __init__(self, authenticated_server: AuthenticatedMCPServer):
        self.authenticated_server = authenticated_server
        self.active_sessions: Dict[str, MCPAuthContext] = {}
    
    async def create_mcp_session(
        self,
        session_id: str,
        auth_context: MCPAuthContext
    ):
        """Create MCP session with auth context"""
        self.active_sessions[session_id] = auth_context
        logger.info(f"Created MCP session: {session_id} for user {auth_context.username}")
    
    async def get_mcp_session(self, session_id: str) -> Optional[MCPAuthContext]:
        """Get MCP session auth context"""
        return self.active_sessions.get(session_id)
    
    async def remove_mcp_session(self, session_id: str):
        """Remove MCP session"""
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
            logger.info(f"Removed MCP session: {session_id}")
    
    async def execute_in_session(
        self,
        session_id: str,
        tool_name: str,
        arguments: Dict[str, Any]
    ) -> Any:
        """Execute tool in session context"""
        auth_context = await self.get_mcp_session(session_id)
        if not auth_context:
            raise ValueError(f"Invalid session: {session_id}")
        
        return await self.authenticated_server.execute_tool(
            tool_name,
            arguments,
            auth_context
        )


# Utility functions for MCP OAuth2 integration
def create_mcp_oauth2_config(
    base_url: str = "http://localhost:8000",
    client_id: str = "mcp-client",
    client_secret: Optional[str] = None,
    scopes: list = None
) -> dict:
    """Create OAuth2 configuration for MCP client"""
    return {
        "authorization_endpoint": f"{base_url}/auth/authorize",
        "token_endpoint": f"{base_url}/auth/token",
        "client_id": client_id,
        "client_secret": client_secret,
        "scope": " ".join(scopes or ["read", "write"]),
        "response_type": "code",
        "grant_type": "authorization_code",
        "code_challenge_method": "S256"
    }


async def validate_mcp_token(
    token: str,
    oauth2_storage: OAuth2Storage,
    required_scope: str = "read"
) -> MCPAuthContext:
    """Validate MCP access token and return auth context"""
    # Get access token
    access_token = await oauth2_storage.get_access_token(token)
    if not access_token or access_token.is_expired():
        raise ValueError("Invalid or expired token")
    
    # Check scope
    if not access_token.has_scope(required_scope):
        raise ValueError(f"Insufficient scope: required {required_scope}")
    
    # Get user if applicable
    user = None
    if access_token.user_id:
        user = await oauth2_storage.get_user(access_token.user_id)
    
    return MCPAuthContext(
        user=user,
        client_id=access_token.client_id,
        access_token=access_token,
        scope=access_token.scope
    )