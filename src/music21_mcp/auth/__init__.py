"""
OAuth2 and Authentication module for Music21 MCP Server
Implements OAuth 2.1 with PKCE for secure remote access
"""

from .oauth2_provider import OAuth2Provider, OAuth2Config
from .session_manager import SessionManager, SessionConfig
from .security import (
    SecurityMiddleware,
    generate_code_verifier,
    generate_code_challenge,
)
from .models import (
    AuthorizationRequest,
    TokenRequest,
    TokenResponse,
    UserSession,
    ClientRegistration,
)

__all__ = [
    "OAuth2Provider",
    "OAuth2Config",
    "SessionManager",
    "SessionConfig",
    "SecurityMiddleware",
    "generate_code_verifier",
    "generate_code_challenge",
    "AuthorizationRequest",
    "TokenRequest",
    "TokenResponse",
    "UserSession",
    "ClientRegistration",
]
