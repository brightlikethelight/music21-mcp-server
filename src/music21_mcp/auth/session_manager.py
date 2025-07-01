"""
Session management for OAuth2 authenticated users
Handles session creation, validation, and cleanup with Redis support
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, Set
from contextlib import asynccontextmanager

from .models import UserSession, User
from .storage import SessionStorage, InMemorySessionStorage, RedisSessionStorage

logger = logging.getLogger(__name__)


class SessionConfig:
    """Session manager configuration"""

    def __init__(
        self,
        session_ttl_minutes: int = 30,
        max_sessions_per_user: int = 5,
        cleanup_interval_minutes: int = 15,
        enable_sliding_expiration: bool = True,
        secure_cookie: bool = True,
        same_site: str = "lax",
        http_only: bool = True,
    ):
        self.session_ttl_minutes = session_ttl_minutes
        self.max_sessions_per_user = max_sessions_per_user
        self.cleanup_interval_minutes = cleanup_interval_minutes
        self.enable_sliding_expiration = enable_sliding_expiration
        self.secure_cookie = secure_cookie
        self.same_site = same_site
        self.http_only = http_only


class SessionManager:
    """Manages user sessions with support for Redis and in-memory storage"""

    def __init__(self, config: SessionConfig, storage: Optional[SessionStorage] = None):
        self.config = config
        self.storage = storage or InMemorySessionStorage()
        self._cleanup_task: Optional[asyncio.Task] = None
        self._user_sessions: Dict[str, Set[str]] = {}  # user_id -> set of session_ids

    async def start(self):
        """Start session manager and cleanup task"""
        if self.config.cleanup_interval_minutes > 0:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            logger.info("Session cleanup task started")

    async def stop(self):
        """Stop session manager and cleanup task"""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            logger.info("Session cleanup task stopped")

    @asynccontextmanager
    async def lifespan(self):
        """Async context manager for session manager lifecycle"""
        await self.start()
        try:
            yield
        finally:
            await self.stop()

    async def create_session(
        self,
        user: User,
        client_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        data: Optional[Dict] = None,
    ) -> UserSession:
        """Create a new user session"""
        # Check session limit
        await self._enforce_session_limit(user.user_id)

        # Create session
        session = UserSession(
            user_id=user.user_id,
            client_id=client_id,
            ip_address=ip_address,
            user_agent=user_agent,
            data=data or {},
            expires_at=datetime.utcnow()
            + timedelta(minutes=self.config.session_ttl_minutes),
        )

        # Save session
        await self.storage.save_session(session)

        # Track session for user
        if user.user_id not in self._user_sessions:
            self._user_sessions[user.user_id] = set()
        self._user_sessions[user.user_id].add(session.session_id)

        logger.info(f"Created session {session.session_id} for user {user.user_id}")

        return session

    async def get_session(self, session_id: str) -> Optional[UserSession]:
        """Get and optionally refresh a session"""
        session = await self.storage.get_session(session_id)

        if not session:
            # Clean up orphaned tracking
            for user_id in self._user_sessions:
                self._user_sessions[user_id].discard(session_id)
            return None

        # Check expiration
        if session.is_expired():
            await self.delete_session(session_id)
            return None

        # Handle sliding expiration
        if self.config.enable_sliding_expiration:
            session.refresh(self.config.session_ttl_minutes)
            await self.storage.save_session(session)

        return session

    async def delete_session(self, session_id: str):
        """Delete a session"""
        session = await self.storage.get_session(session_id)

        if session:
            # Remove from user's session set
            if session.user_id in self._user_sessions:
                self._user_sessions[session.user_id].discard(session_id)

        await self.storage.delete_session(session_id)
        logger.info(f"Deleted session {session_id}")

    async def delete_user_sessions(self, user_id: str):
        """Delete all sessions for a user"""
        if user_id in self._user_sessions:
            session_ids = list(self._user_sessions[user_id])
            for session_id in session_ids:
                await self.delete_session(session_id)

        logger.info(f"Deleted all sessions for user {user_id}")

    async def update_session_data(self, session_id: str, data: Dict):
        """Update session data"""
        session = await self.get_session(session_id)

        if not session:
            raise ValueError(f"Session {session_id} not found")

        session.data.update(data)
        await self.storage.save_session(session)

    async def get_user_sessions(self, user_id: str) -> list[UserSession]:
        """Get all active sessions for a user"""
        sessions = []

        if user_id in self._user_sessions:
            for session_id in list(self._user_sessions[user_id]):
                session = await self.get_session(session_id)
                if session:
                    sessions.append(session)

        return sessions

    async def _get_user_sessions_no_refresh(self, user_id: str) -> list[UserSession]:
        """Get user sessions without refreshing them (for cleanup)"""
        sessions = []

        if user_id in self._user_sessions:
            for session_id in list(self._user_sessions[user_id]):
                session = await self.storage.get_session(session_id)
                if session and not session.is_expired():
                    sessions.append(session)
                elif session and session.is_expired():
                    # Remove expired session from tracking
                    self._user_sessions[user_id].discard(session_id)

        return sessions

    async def _enforce_session_limit(self, user_id: str):
        """Enforce maximum sessions per user"""
        # Get sessions without refreshing to preserve last_accessed times
        sessions = await self._get_user_sessions_no_refresh(user_id)

        # Check if adding one more session would exceed the limit
        if len(sessions) >= self.config.max_sessions_per_user:
            # Sort by last accessed time (oldest first)
            sessions.sort(key=lambda s: s.last_accessed)

            # Delete oldest sessions to make room for new one
            to_delete = len(sessions) - self.config.max_sessions_per_user + 1
            for i in range(to_delete):
                await self.delete_session(sessions[i].session_id)

    async def _cleanup_loop(self):
        """Background task to clean up expired sessions"""
        while True:
            try:
                await asyncio.sleep(self.config.cleanup_interval_minutes * 60)
                await self._cleanup_expired_sessions()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in session cleanup: {e}")

    async def _cleanup_expired_sessions(self):
        """Clean up expired sessions"""
        await self.storage.cleanup_expired_sessions()

        # Clean up internal tracking
        for user_id in list(self._user_sessions.keys()):
            # Validate each session
            valid_sessions = set()
            for session_id in self._user_sessions[user_id]:
                session = await self.storage.get_session(session_id)
                if session and not session.is_expired():
                    valid_sessions.add(session_id)

            if valid_sessions:
                self._user_sessions[user_id] = valid_sessions
            else:
                del self._user_sessions[user_id]

    def create_session_cookie(self, session: UserSession) -> dict:
        """Create session cookie parameters"""
        return {
            "key": "session_id",
            "value": session.session_id,
            "max_age": self.config.session_ttl_minutes * 60,
            "expires": session.expires_at,
            "path": "/",
            "secure": self.config.secure_cookie,
            "httponly": self.config.http_only,
            "samesite": self.config.same_site,
        }


class SessionValidator:
    """Validates sessions and enforces security policies"""

    def __init__(self, session_manager: SessionManager):
        self.session_manager = session_manager

    async def validate_request_session(
        self,
        session_id: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> Optional[UserSession]:
        """Validate session with additional security checks"""
        session = await self.session_manager.get_session(session_id)

        if not session:
            return None

        # Optional: Validate IP address hasn't changed
        if ip_address and session.ip_address and session.ip_address != ip_address:
            logger.warning(
                f"Session {session_id} IP mismatch: {session.ip_address} != {ip_address}"
            )
            # Could optionally reject the session here for strict security

        # Optional: Validate user agent hasn't changed significantly
        if user_agent and session.user_agent and session.user_agent != user_agent:
            # Could implement fuzzy matching for minor UA changes
            logger.warning(f"Session {session_id} user agent changed")

        return session

    async def require_fresh_session(
        self, session: UserSession, max_age_minutes: int = 5
    ) -> bool:
        """Check if session is fresh enough for sensitive operations"""
        age = datetime.utcnow() - session.last_accessed
        return age.total_seconds() <= max_age_minutes * 60
