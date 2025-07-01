"""
OAuth2 and Authentication module for Music21 MCP Server
Implements OAuth 2.1 with PKCE for secure remote access
"""

from .models import (
    AuthorizationRequest,
    ClientRegistration,
    TokenRequest,
    TokenResponse,
    UserSession,
)
from .oauth2_provider import OAuth2Config, OAuth2Provider
from .security import (
    SecurityMiddleware,
    generate_code_challenge,
    generate_code_verifier,
)
from .session_manager import SessionConfig, SessionManager

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
