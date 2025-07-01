"""
Security middleware and utilities for OAuth2 authentication
"""

import base64
import hashlib
import logging
import secrets
from typing import Optional, Tuple
from fastapi import HTTPException, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from .models import AccessToken
from .oauth2_provider import OAuth2Provider

logger = logging.getLogger(__name__)


class SecurityMiddleware:
    """Security middleware for OAuth2 authentication"""

    def __init__(self, oauth2_provider: OAuth2Provider):
        self.oauth2_provider = oauth2_provider
        self.bearer_scheme = HTTPBearer(auto_error=False)

    async def __call__(self, request: Request) -> Optional[AccessToken]:
        """Validate OAuth2 bearer token from request"""
        # Extract credentials
        credentials: Optional[HTTPAuthorizationCredentials] = await self.bearer_scheme(
            request
        )

        if not credentials:
            # Check for token in query parameters (less secure, for compatibility)
            token = request.query_params.get("access_token")
            if not token:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Bearer token required",
                    headers={"WWW-Authenticate": "Bearer"},
                )
        else:
            token = credentials.credentials

        # Validate token
        access_token = await self.oauth2_provider.validate_token(token)

        if not access_token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired token",
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Add token info to request state
        request.state.access_token = access_token
        request.state.user_id = access_token.user_id
        request.state.client_id = access_token.client_id
        request.state.scope = access_token.scope

        return access_token


def require_scope(required_scope: str):
    """Dependency to require specific OAuth2 scope"""

    async def scope_checker(request: Request) -> None:
        if not hasattr(request.state, "access_token"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail="No access token found"
            )

        access_token: AccessToken = request.state.access_token

        if not access_token.has_scope(required_scope):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient scope. Required: {required_scope}",
            )

    return scope_checker


def generate_code_verifier() -> str:
    """Generate PKCE code verifier (43-128 characters)"""
    # Generate 32 bytes = 256 bits of randomness
    # Base64 URL encoding gives ~43 characters
    verifier = base64.urlsafe_b64encode(secrets.token_bytes(32)).decode("utf-8")
    # Remove padding
    return verifier.rstrip("=")


def generate_code_challenge(verifier: str, method: str = "S256") -> str:
    """Generate PKCE code challenge from verifier"""
    if method == "S256":
        # SHA256 hash of verifier
        digest = hashlib.sha256(verifier.encode("utf-8")).digest()
        # Base64 URL encode without padding
        challenge = base64.urlsafe_b64encode(digest).decode("utf-8").rstrip("=")
        return challenge
    elif method == "plain":
        # Plain text (not recommended)
        return verifier
    else:
        raise ValueError(f"Unsupported code challenge method: {method}")


def verify_code_challenge(verifier: str, challenge: str, method: str = "S256") -> bool:
    """Verify PKCE code challenge"""
    expected_challenge = generate_code_challenge(verifier, method)
    return secrets.compare_digest(expected_challenge, challenge)


def parse_basic_auth(authorization: str) -> Tuple[str, str]:
    """Parse HTTP Basic Authentication header"""
    try:
        # Remove "Basic " prefix
        if not authorization.startswith("Basic "):
            raise ValueError("Invalid Basic auth format")

        encoded = authorization[6:]
        decoded = base64.b64decode(encoded).decode("utf-8")

        # Split username:password
        if ":" not in decoded:
            raise ValueError("Invalid Basic auth format")

        username, password = decoded.split(":", 1)
        return username, password
    except Exception as e:
        logger.error(f"Failed to parse Basic auth: {e}")
        raise ValueError("Invalid Basic auth header")


class RateLimiter:
    """Simple in-memory rate limiter for OAuth2 endpoints"""

    def __init__(self, max_requests: int = 100, window_seconds: int = 3600):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: dict = {}  # client_id -> list of timestamps

    async def check_rate_limit(self, client_id: str) -> bool:
        """Check if client has exceeded rate limit"""
        from datetime import datetime, timedelta

        now = datetime.utcnow()
        window_start = now - timedelta(seconds=self.window_seconds)

        # Get client's request history
        if client_id not in self.requests:
            self.requests[client_id] = []

        # Remove old requests outside window
        self.requests[client_id] = [
            ts for ts in self.requests[client_id] if ts > window_start
        ]

        # Check limit
        if len(self.requests[client_id]) >= self.max_requests:
            return False

        # Add current request
        self.requests[client_id].append(now)
        return True


class IPWhitelist:
    """IP address whitelist for additional security"""

    def __init__(self, allowed_ips: Optional[list] = None):
        self.allowed_ips = set(allowed_ips or [])
        self.allow_all = not self.allowed_ips

    def is_allowed(self, ip_address: str) -> bool:
        """Check if IP address is whitelisted"""
        if self.allow_all:
            return True

        # Handle IPv6 and IPv4
        # For IPv6, also check the /64 subnet
        if ":" in ip_address and "." not in ip_address:  # IPv6
            # Check exact match
            if ip_address in self.allowed_ips:
                return True

            # Check /64 subnet
            parts = ip_address.split(":")
            if len(parts) >= 4:
                subnet = ":".join(parts[:4]) + "::/64"
                if subnet in self.allowed_ips:
                    return True

        return ip_address in self.allowed_ips

    def add_ip(self, ip_address: str):
        """Add IP to whitelist"""
        self.allowed_ips.add(ip_address)
        self.allow_all = False

    def remove_ip(self, ip_address: str):
        """Remove IP from whitelist"""
        self.allowed_ips.discard(ip_address)
        if not self.allowed_ips:
            self.allow_all = True
