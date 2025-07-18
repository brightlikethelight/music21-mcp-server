"""
OAuth 2.1 with PKCE Provider for Music21 MCP Server

Implements enterprise-grade OAuth 2.1 authentication with PKCE (Proof Key for Code Exchange)
for secure MCP server access. Complies with 2024/2025 security standards.
"""

import asyncio
import base64
import hashlib
import secrets
import time
from datetime import datetime, timedelta
from typing import Dict, Optional, Any, List
from dataclasses import dataclass, asdict
import logging

from pydantic import BaseModel, Field
from fastapi import HTTPException, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

logger = logging.getLogger(__name__)


@dataclass
class OAuthToken:
    """OAuth token structure"""
    access_token: str
    token_type: str = "Bearer"
    expires_in: int = 3600
    refresh_token: Optional[str] = None
    scope: Optional[str] = None
    resource_indicator: Optional[str] = None


@dataclass
class ClientCredentials:
    """OAuth client credentials"""
    client_id: str
    client_secret: str
    redirect_uris: List[str]
    grant_types: List[str] = None
    scope: str = "music21:read music21:write"
    
    def __post_init__(self):
        if self.grant_types is None:
            self.grant_types = ["authorization_code", "refresh_token"]


class TokenRequest(BaseModel):
    """Token request validation"""
    grant_type: str = Field(..., regex="^(authorization_code|refresh_token)$")
    code: Optional[str] = None
    redirect_uri: Optional[str] = None
    client_id: str = Field(..., min_length=1, max_length=100)
    client_secret: str = Field(..., min_length=1, max_length=200)
    code_verifier: Optional[str] = None
    refresh_token: Optional[str] = None
    resource: Optional[str] = None  # RFC 8707 Resource Indicators


class AuthorizationRequest(BaseModel):
    """Authorization request validation"""
    response_type: str = Field(..., regex="^code$")
    client_id: str = Field(..., min_length=1, max_length=100)
    redirect_uri: str = Field(..., min_length=1, max_length=500)
    scope: str = Field(default="music21:read", max_length=200)
    state: Optional[str] = Field(None, max_length=128)
    code_challenge: str = Field(..., min_length=43, max_length=128)
    code_challenge_method: str = Field(..., regex="^S256$")
    resource: Optional[str] = None  # RFC 8707 Resource Indicators


class OAuth2Provider:
    """
    OAuth 2.1 with PKCE provider for MCP server authentication
    
    Features:
    - PKCE (Proof Key for Code Exchange) for security
    - Resource Indicators (RFC 8707) for token scoping
    - Refresh tokens with rotation
    - Comprehensive audit logging
    """
    
    def __init__(self, issuer: str = "https://music21-mcp.example.com"):
        self.issuer = issuer
        self.clients: Dict[str, ClientCredentials] = {}
        self.authorization_codes: Dict[str, Dict[str, Any]] = {}
        self.access_tokens: Dict[str, Dict[str, Any]] = {}
        self.refresh_tokens: Dict[str, Dict[str, Any]] = {}
        
        # Token lifetimes
        self.code_lifetime = 600  # 10 minutes
        self.access_token_lifetime = 3600  # 1 hour
        self.refresh_token_lifetime = 86400 * 7  # 7 days
        
        # Security settings
        self.max_failed_attempts = 5
        self.lockout_duration = 300  # 5 minutes
        self.failed_attempts: Dict[str, Dict[str, Any]] = {}
    
    def register_client(self, client_id: str, client_secret: str, 
                       redirect_uris: List[str], **kwargs) -> ClientCredentials:
        """Register OAuth client"""
        client = ClientCredentials(
            client_id=client_id,
            client_secret=client_secret,
            redirect_uris=redirect_uris,
            **kwargs
        )
        self.clients[client_id] = client
        
        logger.info(f"OAuth client registered: {client_id}")
        return client
    
    def generate_pkce_params(self) -> tuple[str, str]:
        """Generate PKCE parameters"""
        # Generate code verifier (43-128 chars, URL-safe)
        code_verifier = base64.urlsafe_b64encode(
            secrets.token_bytes(32)
        ).decode('utf-8').rstrip('=')
        
        # Generate code challenge (SHA256 of verifier)
        code_challenge = base64.urlsafe_b64encode(
            hashlib.sha256(code_verifier.encode('utf-8')).digest()
        ).decode('utf-8').rstrip('=')
        
        return code_verifier, code_challenge
    
    async def authorize(self, request: AuthorizationRequest) -> Dict[str, Any]:
        """
        OAuth authorization endpoint
        
        Returns authorization code for valid requests
        """
        # Validate client
        if request.client_id not in self.clients:
            raise HTTPException(
                status_code=400,
                detail="Invalid client_id"
            )
        
        client = self.clients[request.client_id]
        
        # Validate redirect URI
        if request.redirect_uri not in client.redirect_uris:
            raise HTTPException(
                status_code=400,
                detail="Invalid redirect_uri"
            )
        
        # Validate scope
        if not self._validate_scope(request.scope, client.scope):
            raise HTTPException(
                status_code=400,
                detail="Invalid scope"
            )
        
        # Generate authorization code
        code = self._generate_secure_token()
        
        # Store authorization code with metadata
        self.authorization_codes[code] = {
            "client_id": request.client_id,
            "redirect_uri": request.redirect_uri,
            "scope": request.scope,
            "code_challenge": request.code_challenge,
            "code_challenge_method": request.code_challenge_method,
            "resource": request.resource,
            "expires_at": time.time() + self.code_lifetime,
            "used": False
        }
        
        # Log authorization
        logger.info(f"Authorization code generated for client: {request.client_id}")
        
        return {
            "authorization_code": code,
            "expires_in": self.code_lifetime,
            "redirect_uri": request.redirect_uri,
            "state": request.state
        }
    
    async def token(self, request: TokenRequest) -> OAuthToken:
        """
        OAuth token endpoint
        
        Exchanges authorization code for access token
        """
        # Check for rate limiting
        await self._check_rate_limit(request.client_id)
        
        if request.grant_type == "authorization_code":
            return await self._handle_authorization_code_grant(request)
        elif request.grant_type == "refresh_token":
            return await self._handle_refresh_token_grant(request)
        else:
            raise HTTPException(
                status_code=400,
                detail="Unsupported grant_type"
            )
    
    async def _handle_authorization_code_grant(self, request: TokenRequest) -> OAuthToken:
        """Handle authorization code grant"""
        if not request.code:
            raise HTTPException(
                status_code=400,
                detail="Missing authorization code"
            )
        
        # Validate authorization code
        code_data = self.authorization_codes.get(request.code)
        if not code_data:
            await self._record_failed_attempt(request.client_id, "invalid_code")
            raise HTTPException(
                status_code=400,
                detail="Invalid authorization code"
            )
        
        # Check if code is expired
        if time.time() > code_data["expires_at"]:
            del self.authorization_codes[request.code]
            await self._record_failed_attempt(request.client_id, "expired_code")
            raise HTTPException(
                status_code=400,
                detail="Authorization code expired"
            )
        
        # Check if code was already used
        if code_data["used"]:
            del self.authorization_codes[request.code]
            await self._record_failed_attempt(request.client_id, "code_reuse")
            raise HTTPException(
                status_code=400,
                detail="Authorization code already used"
            )
        
        # Validate client
        if code_data["client_id"] != request.client_id:
            await self._record_failed_attempt(request.client_id, "client_mismatch")
            raise HTTPException(
                status_code=400,
                detail="Client ID mismatch"
            )
        
        # Validate client secret
        client = self.clients[request.client_id]
        if client.client_secret != request.client_secret:
            await self._record_failed_attempt(request.client_id, "invalid_secret")
            raise HTTPException(
                status_code=401,
                detail="Invalid client credentials"
            )
        
        # Validate redirect URI
        if request.redirect_uri != code_data["redirect_uri"]:
            await self._record_failed_attempt(request.client_id, "redirect_mismatch")
            raise HTTPException(
                status_code=400,
                detail="Redirect URI mismatch"
            )
        
        # Validate PKCE
        if not request.code_verifier:
            await self._record_failed_attempt(request.client_id, "missing_verifier")
            raise HTTPException(
                status_code=400,
                detail="Missing code_verifier"
            )
        
        if not self._validate_pkce(request.code_verifier, code_data["code_challenge"]):
            await self._record_failed_attempt(request.client_id, "invalid_verifier")
            raise HTTPException(
                status_code=400,
                detail="Invalid code_verifier"
            )
        
        # Mark code as used
        code_data["used"] = True
        
        # Generate tokens
        access_token = self._generate_secure_token()
        refresh_token = self._generate_secure_token()
        
        # Store tokens
        self.access_tokens[access_token] = {
            "client_id": request.client_id,
            "scope": code_data["scope"],
            "resource": code_data["resource"],
            "expires_at": time.time() + self.access_token_lifetime,
            "issued_at": time.time()
        }
        
        self.refresh_tokens[refresh_token] = {
            "client_id": request.client_id,
            "scope": code_data["scope"],
            "resource": code_data["resource"],
            "expires_at": time.time() + self.refresh_token_lifetime,
            "issued_at": time.time()
        }
        
        # Clean up authorization code
        del self.authorization_codes[request.code]
        
        # Log successful token issuance
        logger.info(f"Access token issued for client: {request.client_id}")
        
        return OAuthToken(
            access_token=access_token,
            expires_in=self.access_token_lifetime,
            refresh_token=refresh_token,
            scope=code_data["scope"],
            resource_indicator=code_data["resource"]
        )
    
    async def _handle_refresh_token_grant(self, request: TokenRequest) -> OAuthToken:
        """Handle refresh token grant"""
        if not request.refresh_token:
            raise HTTPException(
                status_code=400,
                detail="Missing refresh token"
            )
        
        # Validate refresh token
        refresh_data = self.refresh_tokens.get(request.refresh_token)
        if not refresh_data:
            await self._record_failed_attempt(request.client_id, "invalid_refresh")
            raise HTTPException(
                status_code=400,
                detail="Invalid refresh token"
            )
        
        # Check if refresh token is expired
        if time.time() > refresh_data["expires_at"]:
            del self.refresh_tokens[request.refresh_token]
            await self._record_failed_attempt(request.client_id, "expired_refresh")
            raise HTTPException(
                status_code=400,
                detail="Refresh token expired"
            )
        
        # Validate client
        if refresh_data["client_id"] != request.client_id:
            await self._record_failed_attempt(request.client_id, "client_mismatch")
            raise HTTPException(
                status_code=400,
                detail="Client ID mismatch"
            )
        
        # Validate client secret
        client = self.clients[request.client_id]
        if client.client_secret != request.client_secret:
            await self._record_failed_attempt(request.client_id, "invalid_secret")
            raise HTTPException(
                status_code=401,
                detail="Invalid client credentials"
            )
        
        # Generate new tokens
        access_token = self._generate_secure_token()
        new_refresh_token = self._generate_secure_token()
        
        # Store new tokens
        self.access_tokens[access_token] = {
            "client_id": request.client_id,
            "scope": refresh_data["scope"],
            "resource": refresh_data["resource"],
            "expires_at": time.time() + self.access_token_lifetime,
            "issued_at": time.time()
        }
        
        self.refresh_tokens[new_refresh_token] = {
            "client_id": request.client_id,
            "scope": refresh_data["scope"],
            "resource": refresh_data["resource"],
            "expires_at": time.time() + self.refresh_token_lifetime,
            "issued_at": time.time()
        }
        
        # Remove old refresh token (refresh token rotation)
        del self.refresh_tokens[request.refresh_token]
        
        # Log successful token refresh
        logger.info(f"Token refreshed for client: {request.client_id}")
        
        return OAuthToken(
            access_token=access_token,
            expires_in=self.access_token_lifetime,
            refresh_token=new_refresh_token,
            scope=refresh_data["scope"],
            resource_indicator=refresh_data["resource"]
        )
    
    async def validate_token(self, token: str, required_scope: str = None,
                           resource_indicator: str = None) -> Dict[str, Any]:
        """Validate access token"""
        token_data = self.access_tokens.get(token)
        
        if not token_data:
            raise HTTPException(
                status_code=401,
                detail="Invalid access token"
            )
        
        # Check if token is expired
        if time.time() > token_data["expires_at"]:
            del self.access_tokens[token]
            raise HTTPException(
                status_code=401,
                detail="Access token expired"
            )
        
        # Validate scope
        if required_scope and not self._validate_scope(required_scope, token_data["scope"]):
            raise HTTPException(
                status_code=403,
                detail="Insufficient scope"
            )
        
        # Validate resource indicator
        if resource_indicator and token_data["resource"] != resource_indicator:
            raise HTTPException(
                status_code=403,
                detail="Invalid resource indicator"
            )
        
        return token_data
    
    def _generate_secure_token(self) -> str:
        """Generate cryptographically secure token"""
        return base64.urlsafe_b64encode(secrets.token_bytes(32)).decode('utf-8').rstrip('=')
    
    def _validate_pkce(self, code_verifier: str, code_challenge: str) -> bool:
        """Validate PKCE code verifier against challenge"""
        expected_challenge = base64.urlsafe_b64encode(
            hashlib.sha256(code_verifier.encode('utf-8')).digest()
        ).decode('utf-8').rstrip('=')
        
        return expected_challenge == code_challenge
    
    def _validate_scope(self, requested_scope: str, granted_scope: str) -> bool:
        """Validate requested scope against granted scope"""
        requested_scopes = set(requested_scope.split())
        granted_scopes = set(granted_scope.split())
        
        return requested_scopes.issubset(granted_scopes)
    
    async def _check_rate_limit(self, client_id: str):
        """Check if client is rate limited"""
        if client_id in self.failed_attempts:
            attempt_data = self.failed_attempts[client_id]
            
            if attempt_data["count"] >= self.max_failed_attempts:
                if time.time() < attempt_data["locked_until"]:
                    raise HTTPException(
                        status_code=429,
                        detail="Too many failed attempts. Try again later."
                    )
                else:
                    # Reset failed attempts after lockout period
                    del self.failed_attempts[client_id]
    
    async def _record_failed_attempt(self, client_id: str, reason: str):
        """Record failed authentication attempt"""
        if client_id not in self.failed_attempts:
            self.failed_attempts[client_id] = {
                "count": 0,
                "first_attempt": time.time(),
                "last_attempt": time.time(),
                "locked_until": 0
            }
        
        attempt_data = self.failed_attempts[client_id]
        attempt_data["count"] += 1
        attempt_data["last_attempt"] = time.time()
        
        if attempt_data["count"] >= self.max_failed_attempts:
            attempt_data["locked_until"] = time.time() + self.lockout_duration
            
            logger.warning(f"Client {client_id} locked out due to failed attempts: {reason}")
        else:
            logger.warning(f"Failed authentication attempt for client {client_id}: {reason}")
    
    async def cleanup_expired_tokens(self):
        """Clean up expired tokens (should be run periodically)"""
        current_time = time.time()
        
        # Clean up expired authorization codes
        expired_codes = [
            code for code, data in self.authorization_codes.items()
            if current_time > data["expires_at"]
        ]
        for code in expired_codes:
            del self.authorization_codes[code]
        
        # Clean up expired access tokens
        expired_access = [
            token for token, data in self.access_tokens.items()
            if current_time > data["expires_at"]
        ]
        for token in expired_access:
            del self.access_tokens[token]
        
        # Clean up expired refresh tokens
        expired_refresh = [
            token for token, data in self.refresh_tokens.items()
            if current_time > data["expires_at"]
        ]
        for token in expired_refresh:
            del self.refresh_tokens[token]
        
        if expired_codes or expired_access or expired_refresh:
            logger.info(f"Cleaned up expired tokens: {len(expired_codes)} codes, "
                       f"{len(expired_access)} access tokens, {len(expired_refresh)} refresh tokens")


# FastAPI security scheme
oauth2_scheme = HTTPBearer(auto_error=False)


async def verify_oauth_token(credentials: HTTPAuthorizationCredentials = None,
                           oauth_provider: OAuth2Provider = None,
                           required_scope: str = None,
                           resource_indicator: str = None) -> Dict[str, Any]:
    """
    Verify OAuth token and return token data
    
    Usage in FastAPI:
    ```python
    @app.get("/protected")
    async def protected_endpoint(
        token_data: dict = Depends(verify_oauth_token)
    ):
        return {"message": "Access granted", "client": token_data["client_id"]}
    ```
    """
    if not credentials:
        raise HTTPException(
            status_code=401,
            detail="Missing authorization header"
        )
    
    if not oauth_provider:
        raise HTTPException(
            status_code=500,
            detail="OAuth provider not configured"
        )
    
    return await oauth_provider.validate_token(
        credentials.credentials,
        required_scope=required_scope,
        resource_indicator=resource_indicator
    )