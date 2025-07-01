"""
OAuth2 and session models following OAuth 2.1 specification
"""

import secrets
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from pydantic import BaseModel, Field, validator


class GrantType(str, Enum):
    """OAuth2 grant types"""

    AUTHORIZATION_CODE = "authorization_code"
    CLIENT_CREDENTIALS = "client_credentials"
    REFRESH_TOKEN = "refresh_token"


class ResponseType(str, Enum):
    """OAuth2 response types"""

    CODE = "code"
    TOKEN = "token"


class TokenType(str, Enum):
    """OAuth2 token types"""

    BEARER = "Bearer"


class CodeChallengeMethod(str, Enum):
    """PKCE code challenge methods"""

    S256 = "S256"
    PLAIN = "plain"


class ClientType(str, Enum):
    """OAuth2 client types"""

    PUBLIC = "public"
    CONFIDENTIAL = "confidential"


class ClientRegistration(BaseModel):
    """OAuth2 client registration"""

    client_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    client_type: ClientType = ClientType.PUBLIC
    client_secret: Optional[str] = None
    client_name: str
    redirect_uris: List[str]
    grant_types: List[GrantType] = [GrantType.AUTHORIZATION_CODE]
    response_types: List[ResponseType] = [ResponseType.CODE]
    scope: str = "read write"
    contacts: List[str] = []
    logo_uri: Optional[str] = None
    client_uri: Optional[str] = None
    policy_uri: Optional[str] = None
    tos_uri: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None

    @validator("client_secret", always=True)
    def set_client_secret(cls, v: Optional[str], values: Dict[str, Any]) -> Optional[str]:
        if values.get("client_type") == ClientType.CONFIDENTIAL and v is None:
            return secrets.token_urlsafe(32)
        return v

    @validator("redirect_uris")
    def validate_redirect_uris(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError("At least one redirect URI is required")
        for uri in v:
            if uri == "localhost" or uri.startswith("http://localhost"):
                continue  # Allow localhost for development
            if not (uri.startswith("https://") or uri.startswith("http://")):
                raise ValueError(f"Invalid redirect URI: {uri}")
        return v


class AuthorizationRequest(BaseModel):
    """OAuth2 authorization request"""

    response_type: ResponseType
    client_id: str
    redirect_uri: str
    scope: str = "read"
    state: str
    code_challenge: str
    code_challenge_method: CodeChallengeMethod = CodeChallengeMethod.S256
    nonce: Optional[str] = None

    @validator("code_challenge")
    def validate_code_challenge(cls, v: str) -> str:
        if len(v) < 43:  # Base64 URL encoded SHA256 is 43 chars
            raise ValueError("Code challenge too short")
        return v


class AuthorizationCode(BaseModel):
    """OAuth2 authorization code"""

    code: str = Field(default_factory=lambda: secrets.token_urlsafe(32))
    client_id: str
    redirect_uri: str
    scope: str
    user_id: str
    code_challenge: str
    code_challenge_method: CodeChallengeMethod
    expires_at: datetime = Field(
        default_factory=lambda: datetime.utcnow() + timedelta(minutes=10)
    )
    used: bool = False

    def is_expired(self) -> bool:
        return datetime.utcnow() > self.expires_at

    def is_valid(self) -> bool:
        return not self.used and not self.is_expired()


class TokenRequest(BaseModel):
    """OAuth2 token request"""

    grant_type: GrantType
    code: Optional[str] = None  # For authorization_code
    redirect_uri: Optional[str] = None  # For authorization_code
    client_id: str
    client_secret: Optional[str] = None  # For confidential clients
    code_verifier: Optional[str] = None  # For PKCE
    refresh_token: Optional[str] = None  # For refresh_token grant
    scope: Optional[str] = None  # For client_credentials

    @validator("code_verifier")
    def validate_code_verifier(cls, v: Optional[str], values: Dict[str, Any]) -> Optional[str]:
        if values.get("grant_type") == GrantType.AUTHORIZATION_CODE:
            if not v:
                raise ValueError("Code verifier is required for PKCE")
            if len(v) < 43 or len(v) > 128:
                raise ValueError("Code verifier must be 43-128 characters")
        return v

    @validator("code")
    def validate_code(cls, v: Optional[str], values: Dict[str, Any]) -> Optional[str]:
        if values.get("grant_type") == GrantType.AUTHORIZATION_CODE and not v:
            raise ValueError("Code is required for authorization_code grant")
        return v


class TokenResponse(BaseModel):
    """OAuth2 token response"""

    access_token: str = Field(default_factory=lambda: secrets.token_urlsafe(32))
    token_type: TokenType = TokenType.BEARER
    expires_in: int = 3600  # 1 hour
    refresh_token: Optional[str] = Field(
        default_factory=lambda: secrets.token_urlsafe(32)
    )
    scope: str = "read"

    def to_dict(self) -> Dict:
        """Convert to OAuth2 response format"""
        return {
            "access_token": self.access_token,
            "token_type": self.token_type.value,
            "expires_in": self.expires_in,
            "refresh_token": self.refresh_token,
            "scope": self.scope,
        }


class AccessToken(BaseModel):
    """OAuth2 access token details"""

    token: str
    client_id: str
    user_id: Optional[str] = None  # None for client_credentials
    scope: str
    expires_at: datetime
    refresh_token: Optional[str] = None

    def is_expired(self) -> bool:
        return datetime.utcnow() > self.expires_at

    def has_scope(self, required_scope: str) -> bool:
        """Check if token has required scope"""
        token_scopes = set(self.scope.split())
        required_scopes = set(required_scope.split())
        return required_scopes.issubset(token_scopes)


class RefreshToken(BaseModel):
    """OAuth2 refresh token details"""

    token: str
    client_id: str
    user_id: str
    scope: str
    expires_at: datetime = Field(
        default_factory=lambda: datetime.utcnow() + timedelta(days=30)
    )

    def is_expired(self) -> bool:
        return datetime.utcnow() > self.expires_at


class UserSession(BaseModel):
    """User session information"""

    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    client_id: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_accessed: datetime = Field(default_factory=datetime.utcnow)
    expires_at: datetime = Field(
        default_factory=lambda: datetime.utcnow() + timedelta(minutes=30)
    )
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    data: Dict = Field(default_factory=dict)

    def is_expired(self) -> bool:
        return datetime.utcnow() > self.expires_at

    def refresh(self, duration_minutes: int = 30) -> None:
        """Refresh session expiration (sliding expiration)"""
        self.last_accessed = datetime.utcnow()
        self.expires_at = datetime.utcnow() + timedelta(minutes=duration_minutes)


class User(BaseModel):
    """User model for authentication"""

    user_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    username: str
    email: str
    full_name: Optional[str] = None
    active: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_login: Optional[datetime] = None
    permissions: Set[str] = Field(default_factory=set)

    def has_permission(self, permission: str) -> bool:
        return permission in self.permissions or "admin" in self.permissions


class OAuth2ServerMetadata(BaseModel):
    """OAuth2 Authorization Server Metadata (RFC 8414)"""

    issuer: str
    authorization_endpoint: str
    token_endpoint: str
    token_endpoint_auth_methods_supported: List[str] = [
        "client_secret_post",
        "client_secret_basic",
    ]
    jwks_uri: Optional[str] = None
    registration_endpoint: Optional[str] = None
    scopes_supported: List[str] = ["read", "write", "admin"]
    response_types_supported: List[str] = ["code", "token"]
    grant_types_supported: List[str] = [
        "authorization_code",
        "client_credentials",
        "refresh_token",
    ]
    code_challenge_methods_supported: List[str] = ["S256", "plain"]
    service_documentation: Optional[str] = None
    ui_locales_supported: List[str] = ["en"]


class OAuth2ProtectedResourceMetadata(BaseModel):
    """OAuth2 Protected Resource Metadata"""

    resource: str
    authorization_servers: List[str]
    scopes_supported: List[str] = ["read", "write"]
    bearer_methods_supported: List[str] = ["header", "body", "query"]
    resource_documentation: Optional[str] = None
