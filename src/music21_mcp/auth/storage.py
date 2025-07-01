"""
Storage interfaces and implementations for OAuth2 data
Supports both in-memory and Redis-based storage
"""
import json
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Optional
import logging

from .models import (
    ClientRegistration, AuthorizationCode, AccessToken,
    RefreshToken, User, UserSession
)

logger = logging.getLogger(__name__)


class OAuth2Storage(ABC):
    """Abstract base class for OAuth2 storage"""
    
    @abstractmethod
    async def save_client(self, client: ClientRegistration):
        """Save client registration"""
        pass
    
    @abstractmethod
    async def get_client(self, client_id: str) -> Optional[ClientRegistration]:
        """Get client by ID"""
        pass
    
    @abstractmethod
    async def save_authorization_code(self, code: AuthorizationCode):
        """Save authorization code"""
        pass
    
    @abstractmethod
    async def get_authorization_code(self, code: str) -> Optional[AuthorizationCode]:
        """Get authorization code"""
        pass
    
    @abstractmethod
    async def mark_authorization_code_used(self, code: str):
        """Mark authorization code as used"""
        pass
    
    @abstractmethod
    async def save_access_token(self, token: AccessToken):
        """Save access token"""
        pass
    
    @abstractmethod
    async def get_access_token(self, token: str) -> Optional[AccessToken]:
        """Get access token"""
        pass
    
    @abstractmethod
    async def revoke_access_token(self, token: str):
        """Revoke access token"""
        pass
    
    @abstractmethod
    async def save_refresh_token(self, token: RefreshToken):
        """Save refresh token"""
        pass
    
    @abstractmethod
    async def get_refresh_token(self, token: str) -> Optional[RefreshToken]:
        """Get refresh token"""
        pass
    
    @abstractmethod
    async def revoke_refresh_token(self, token: str):
        """Revoke refresh token"""
        pass
    
    @abstractmethod
    async def save_user(self, user: User):
        """Save user"""
        pass
    
    @abstractmethod
    async def get_user(self, user_id: str) -> Optional[User]:
        """Get user by ID"""
        pass
    
    @abstractmethod
    async def get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username"""
        pass


class SessionStorage(ABC):
    """Abstract base class for session storage"""
    
    @abstractmethod
    async def save_session(self, session: UserSession):
        """Save user session"""
        pass
    
    @abstractmethod
    async def get_session(self, session_id: str) -> Optional[UserSession]:
        """Get session by ID"""
        pass
    
    @abstractmethod
    async def delete_session(self, session_id: str):
        """Delete session"""
        pass
    
    @abstractmethod
    async def cleanup_expired_sessions(self):
        """Clean up expired sessions"""
        pass


class InMemoryOAuth2Storage(OAuth2Storage):
    """In-memory implementation of OAuth2 storage (for development)"""
    
    def __init__(self):
        self.clients: Dict[str, ClientRegistration] = {}
        self.authorization_codes: Dict[str, AuthorizationCode] = {}
        self.access_tokens: Dict[str, AccessToken] = {}
        self.refresh_tokens: Dict[str, RefreshToken] = {}
        self.users: Dict[str, User] = {}
        self.username_to_user_id: Dict[str, str] = {}
    
    async def save_client(self, client: ClientRegistration):
        self.clients[client.client_id] = client
    
    async def get_client(self, client_id: str) -> Optional[ClientRegistration]:
        return self.clients.get(client_id)
    
    async def save_authorization_code(self, code: AuthorizationCode):
        self.authorization_codes[code.code] = code
    
    async def get_authorization_code(self, code: str) -> Optional[AuthorizationCode]:
        return self.authorization_codes.get(code)
    
    async def mark_authorization_code_used(self, code: str):
        if code in self.authorization_codes:
            self.authorization_codes[code].used = True
    
    async def save_access_token(self, token: AccessToken):
        self.access_tokens[token.token] = token
    
    async def get_access_token(self, token: str) -> Optional[AccessToken]:
        return self.access_tokens.get(token)
    
    async def revoke_access_token(self, token: str):
        self.access_tokens.pop(token, None)
    
    async def save_refresh_token(self, token: RefreshToken):
        self.refresh_tokens[token.token] = token
    
    async def get_refresh_token(self, token: str) -> Optional[RefreshToken]:
        return self.refresh_tokens.get(token)
    
    async def revoke_refresh_token(self, token: str):
        self.refresh_tokens.pop(token, None)
    
    async def save_user(self, user: User):
        self.users[user.user_id] = user
        self.username_to_user_id[user.username] = user.user_id
    
    async def get_user(self, user_id: str) -> Optional[User]:
        return self.users.get(user_id)
    
    async def get_user_by_username(self, username: str) -> Optional[User]:
        user_id = self.username_to_user_id.get(username)
        if user_id:
            return self.users.get(user_id)
        return None


class InMemorySessionStorage(SessionStorage):
    """In-memory session storage (for development)"""
    
    def __init__(self):
        self.sessions: Dict[str, UserSession] = {}
    
    async def save_session(self, session: UserSession):
        self.sessions[session.session_id] = session
    
    async def get_session(self, session_id: str) -> Optional[UserSession]:
        return self.sessions.get(session_id)
    
    async def delete_session(self, session_id: str):
        self.sessions.pop(session_id, None)
    
    async def cleanup_expired_sessions(self):
        """Remove expired sessions"""
        expired = []
        for session_id, session in self.sessions.items():
            if session.is_expired():
                expired.append(session_id)
        
        for session_id in expired:
            self.sessions.pop(session_id, None)
        
        if expired:
            logger.info(f"Cleaned up {len(expired)} expired sessions")


class RedisOAuth2Storage(OAuth2Storage):
    """Redis-based OAuth2 storage for production"""
    
    def __init__(self, redis_client):
        self.redis = redis_client
        self.prefix = "oauth2:"
    
    def _key(self, type_: str, id_: str) -> str:
        """Generate Redis key"""
        return f"{self.prefix}{type_}:{id_}"
    
    async def save_client(self, client: ClientRegistration):
        key = self._key("client", client.client_id)
        await self.redis.set(key, client.json())
    
    async def get_client(self, client_id: str) -> Optional[ClientRegistration]:
        key = self._key("client", client_id)
        data = await self.redis.get(key)
        if data:
            return ClientRegistration.parse_raw(data)
        return None
    
    async def save_authorization_code(self, code: AuthorizationCode):
        key = self._key("auth_code", code.code)
        ttl = int((code.expires_at - datetime.utcnow()).total_seconds())
        await self.redis.setex(key, ttl, code.json())
    
    async def get_authorization_code(self, code: str) -> Optional[AuthorizationCode]:
        key = self._key("auth_code", code)
        data = await self.redis.get(key)
        if data:
            return AuthorizationCode.parse_raw(data)
        return None
    
    async def mark_authorization_code_used(self, code: str):
        key = self._key("auth_code", code)
        data = await self.redis.get(key)
        if data:
            auth_code = AuthorizationCode.parse_raw(data)
            auth_code.used = True
            ttl = await self.redis.ttl(key)
            await self.redis.setex(key, ttl, auth_code.json())
    
    async def save_access_token(self, token: AccessToken):
        key = self._key("access_token", token.token)
        ttl = int((token.expires_at - datetime.utcnow()).total_seconds())
        await self.redis.setex(key, ttl, token.json())
        
        # Also save token by user for lookup
        if token.user_id:
            user_tokens_key = self._key("user_tokens", token.user_id)
            await self.redis.sadd(user_tokens_key, token.token)
            await self.redis.expire(user_tokens_key, ttl)
    
    async def get_access_token(self, token: str) -> Optional[AccessToken]:
        key = self._key("access_token", token)
        data = await self.redis.get(key)
        if data:
            return AccessToken.parse_raw(data)
        return None
    
    async def revoke_access_token(self, token: str):
        # Get token to find user_id
        access_token = await self.get_access_token(token)
        
        # Delete token
        key = self._key("access_token", token)
        await self.redis.delete(key)
        
        # Remove from user's token set
        if access_token and access_token.user_id:
            user_tokens_key = self._key("user_tokens", access_token.user_id)
            await self.redis.srem(user_tokens_key, token)
    
    async def save_refresh_token(self, token: RefreshToken):
        key = self._key("refresh_token", token.token)
        ttl = int((token.expires_at - datetime.utcnow()).total_seconds())
        await self.redis.setex(key, ttl, token.json())
    
    async def get_refresh_token(self, token: str) -> Optional[RefreshToken]:
        key = self._key("refresh_token", token)
        data = await self.redis.get(key)
        if data:
            return RefreshToken.parse_raw(data)
        return None
    
    async def revoke_refresh_token(self, token: str):
        key = self._key("refresh_token", token)
        await self.redis.delete(key)
    
    async def save_user(self, user: User):
        # Save by user_id
        key = self._key("user", user.user_id)
        await self.redis.set(key, user.json())
        
        # Save username to user_id mapping
        username_key = self._key("username", user.username)
        await self.redis.set(username_key, user.user_id)
    
    async def get_user(self, user_id: str) -> Optional[User]:
        key = self._key("user", user_id)
        data = await self.redis.get(key)
        if data:
            return User.parse_raw(data)
        return None
    
    async def get_user_by_username(self, username: str) -> Optional[User]:
        # Get user_id from username
        username_key = self._key("username", username)
        user_id = await self.redis.get(username_key)
        
        if user_id:
            return await self.get_user(user_id.decode() if isinstance(user_id, bytes) else user_id)
        return None


class RedisSessionStorage(SessionStorage):
    """Redis-based session storage for production"""
    
    def __init__(self, redis_client, session_prefix: str = "session:", ttl_seconds: int = 1800):
        self.redis = redis_client
        self.prefix = session_prefix
        self.ttl = ttl_seconds  # 30 minutes default
    
    def _key(self, session_id: str) -> str:
        return f"{self.prefix}{session_id}"
    
    async def save_session(self, session: UserSession):
        key = self._key(session.session_id)
        ttl = int((session.expires_at - datetime.utcnow()).total_seconds())
        await self.redis.setex(key, ttl, session.json())
        
        # Also maintain user to session mapping
        user_sessions_key = f"{self.prefix}user:{session.user_id}"
        await self.redis.sadd(user_sessions_key, session.session_id)
        await self.redis.expire(user_sessions_key, ttl)
    
    async def get_session(self, session_id: str) -> Optional[UserSession]:
        key = self._key(session_id)
        data = await self.redis.get(key)
        
        if data:
            session = UserSession.parse_raw(data)
            
            # Implement sliding expiration
            if not session.is_expired():
                session.refresh()
                await self.save_session(session)
            
            return session
        return None
    
    async def delete_session(self, session_id: str):
        # Get session to find user_id
        session = await self.get_session(session_id)
        
        # Delete session
        key = self._key(session_id)
        await self.redis.delete(key)
        
        # Remove from user's session set
        if session:
            user_sessions_key = f"{self.prefix}user:{session.user_id}"
            await self.redis.srem(user_sessions_key, session_id)
    
    async def cleanup_expired_sessions(self):
        """Clean up expired sessions using Redis expiration"""
        # Redis handles expiration automatically with TTL
        # This method is for compatibility
        logger.info("Redis handles session expiration automatically via TTL")