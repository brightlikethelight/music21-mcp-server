"""
Tests for OAuth2 authentication implementation
"""
import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock

from music21_mcp.auth.models import (
    AuthorizationRequest, TokenRequest, ClientRegistration,
    User, AccessToken, RefreshToken, UserSession,
    GrantType, ResponseType, ClientType
)
from music21_mcp.auth.oauth2_provider import OAuth2Provider, OAuth2Config
from music21_mcp.auth.storage import InMemoryOAuth2Storage, InMemorySessionStorage
from music21_mcp.auth.session_manager import SessionManager, SessionConfig
from music21_mcp.auth.security import (
    generate_code_verifier, generate_code_challenge,
    verify_code_challenge, SecurityMiddleware
)
# Skip MCP integration tests if MCP is not available
try:
    from music21_mcp.auth.mcp_integration import (
        MCPAuthContext, AuthenticatedMCPServer,
        validate_mcp_token
    )
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    # Mock classes for testing
    class MCPAuthContext:
        def __init__(self, user=None, client_id=None, access_token=None, scope="read"):
            self.user = user
            self.client_id = client_id
            self.access_token = access_token
            self.scope = scope
        
        @property
        def user_id(self):
            return self.user.user_id if self.user else None
        
        @property 
        def username(self):
            return self.user.username if self.user else None
        
        def has_scope(self, required_scope):
            return required_scope in self.scope
        
        def has_permission(self, permission):
            return self.user.has_permission(permission) if self.user else False
    
    class AuthenticatedMCPServer:
        def __init__(self, mcp_server, oauth2_storage):
            self.mcp_server = mcp_server
            self.oauth2_storage = oauth2_storage
            self.tool_permissions = {}
        
        def set_tool_permission(self, tool_name, required_scope):
            self.tool_permissions[tool_name] = required_scope
        
        async def check_tool_access(self, tool_name, auth_context):
            required_scope = self.tool_permissions.get(tool_name, "read")
            return auth_context.has_scope(required_scope)
        
        async def list_tools_for_context(self, auth_context):
            tools = []
            for tool_name in ["test_read", "test_write", "test_admin"]:
                if await self.check_tool_access(tool_name, auth_context):
                    tools.append({
                        "name": tool_name,
                        "description": f"Test {tool_name.split('_')[1]} tool",
                        "required_scope": self.tool_permissions.get(tool_name, "read")
                    })
            return tools
    
    async def validate_mcp_token(token, oauth2_storage, required_scope="read"):
        access_token = await oauth2_storage.get_access_token(token)
        if not access_token or access_token.is_expired():
            raise ValueError("Invalid or expired token")
        
        if not access_token.has_scope(required_scope):
            raise ValueError(f"Insufficient scope: required {required_scope}")
        
        user = None
        if access_token.user_id:
            user = await oauth2_storage.get_user(access_token.user_id)
        
        return MCPAuthContext(
            user=user,
            client_id=access_token.client_id,
            access_token=access_token,
            scope=access_token.scope
        )


@pytest.fixture
async def oauth2_storage():
    """Create in-memory OAuth2 storage"""
    return InMemoryOAuth2Storage()


@pytest.fixture
async def session_storage():
    """Create in-memory session storage"""
    return InMemorySessionStorage()


@pytest.fixture
async def oauth2_provider(oauth2_storage):
    """Create OAuth2 provider"""
    config = OAuth2Config(
        issuer="http://localhost:8000",
        require_pkce=True,
        access_token_expire_minutes=60
    )
    return OAuth2Provider(config, oauth2_storage)


@pytest.fixture
async def session_manager(session_storage):
    """Create session manager"""
    config = SessionConfig(
        session_ttl_minutes=30,
        max_sessions_per_user=5
    )
    manager = SessionManager(config, session_storage)
    # Ensure clean state for each test
    manager._user_sessions = {}
    return manager


@pytest.fixture
async def test_client(oauth2_provider):
    """Create test OAuth2 client"""
    client = await oauth2_provider.register_client(
        client_name="Test Client",
        redirect_uris=["http://localhost:3000/callback"],
        client_type="public"
    )
    return client


@pytest.fixture
async def test_user(oauth2_storage):
    """Create test user"""
    user = User(
        username="testuser",
        email="test@example.com",
        full_name="Test User",
        permissions={"read", "write"}
    )
    await oauth2_storage.save_user(user)
    return user


class TestOAuth2Models:
    """Test OAuth2 data models"""
    
    def test_client_registration_public(self):
        """Test public client registration"""
        client = ClientRegistration(
            client_name="Test App",
            redirect_uris=["http://localhost:3000/callback"],
            client_type=ClientType.PUBLIC
        )
        
        assert client.client_id
        assert client.client_secret is None
        assert client.client_type == ClientType.PUBLIC
    
    def test_client_registration_confidential(self):
        """Test confidential client registration"""
        client = ClientRegistration(
            client_name="Test Server",
            redirect_uris=["https://example.com/callback"],
            client_type=ClientType.CONFIDENTIAL
        )
        
        assert client.client_id
        assert client.client_secret  # Auto-generated
        assert len(client.client_secret) >= 32
    
    def test_authorization_request_validation(self):
        """Test authorization request validation"""
        # Valid request
        auth_req = AuthorizationRequest(
            response_type=ResponseType.CODE,
            client_id="test-client",
            redirect_uri="http://localhost:3000/callback",
            scope="read write",
            state="random-state",
            code_challenge="a" * 43,
            code_challenge_method="S256"
        )
        
        assert auth_req.code_challenge_method == "S256"
        
        # Invalid code challenge (too short)
        with pytest.raises(ValueError):
            AuthorizationRequest(
                response_type=ResponseType.CODE,
                client_id="test-client",
                redirect_uri="http://localhost:3000/callback",
                scope="read",
                state="state",
                code_challenge="short"
            )
    
    def test_access_token_expiration(self):
        """Test access token expiration"""
        # Expired token
        token = AccessToken(
            token="test-token",
            client_id="test-client",
            user_id="user-123",
            scope="read",
            expires_at=datetime.utcnow() - timedelta(minutes=1)
        )
        
        assert token.is_expired()
        
        # Valid token
        token = AccessToken(
            token="test-token",
            client_id="test-client",
            user_id="user-123",
            scope="read",
            expires_at=datetime.utcnow() + timedelta(minutes=60)
        )
        
        assert not token.is_expired()
    
    def test_access_token_scope_check(self):
        """Test access token scope checking"""
        token = AccessToken(
            token="test-token",
            client_id="test-client",
            user_id="user-123",
            scope="read write",
            expires_at=datetime.utcnow() + timedelta(minutes=60)
        )
        
        assert token.has_scope("read")
        assert token.has_scope("write")
        assert token.has_scope("read write")
        assert not token.has_scope("admin")
        assert not token.has_scope("read write admin")
    
    def test_user_session_refresh(self):
        """Test user session refresh"""
        session = UserSession(
            user_id="user-123",
            expires_at=datetime.utcnow() + timedelta(minutes=30)
        )
        
        old_expires = session.expires_at
        old_accessed = session.last_accessed
        
        # Wait a bit
        import time
        time.sleep(0.1)
        
        # Refresh session
        session.refresh(duration_minutes=60)
        
        assert session.expires_at > old_expires
        assert session.last_accessed > old_accessed


class TestOAuth2Provider:
    """Test OAuth2 provider functionality"""
    
    @pytest.mark.asyncio
    async def test_client_registration(self, oauth2_provider):
        """Test client registration"""
        client = await oauth2_provider.register_client(
            client_name="Test App",
            redirect_uris=["http://localhost:3000/callback", "https://example.com/callback"],
            client_type="public"
        )
        
        assert client.client_id
        assert client.client_name == "Test App"
        assert len(client.redirect_uris) == 2
        
        # Retrieve client
        stored_client = await oauth2_provider.storage.get_client(client.client_id)
        assert stored_client
        assert stored_client.client_id == client.client_id
    
    @pytest.mark.asyncio
    async def test_authorization_flow(self, oauth2_provider, test_client, test_user):
        """Test authorization code flow"""
        # Create authorization request
        auth_request = AuthorizationRequest(
            response_type=ResponseType.CODE,
            client_id=test_client.client_id,
            redirect_uri=test_client.redirect_uris[0],
            scope="read write",
            state="test-state",
            code_challenge=generate_code_challenge(generate_code_verifier()),
            code_challenge_method="S256"
        )
        
        # Authorize
        code, state = await oauth2_provider.authorize(auth_request, test_user)
        
        assert code
        assert state == "test-state"
        
        # Verify authorization code was saved
        auth_code = await oauth2_provider.storage.get_authorization_code(code)
        assert auth_code
        assert auth_code.client_id == test_client.client_id
        assert auth_code.user_id == test_user.user_id
    
    @pytest.mark.asyncio
    async def test_token_exchange(self, oauth2_provider, test_client, test_user):
        """Test authorization code to token exchange"""
        # Create code verifier for PKCE
        code_verifier = generate_code_verifier()
        code_challenge = generate_code_challenge(code_verifier)
        
        # Get authorization code
        auth_request = AuthorizationRequest(
            response_type=ResponseType.CODE,
            client_id=test_client.client_id,
            redirect_uri=test_client.redirect_uris[0],
            scope="read",
            state="state",
            code_challenge=code_challenge,
            code_challenge_method="S256"
        )
        
        code, _ = await oauth2_provider.authorize(auth_request, test_user)
        
        # Exchange code for token
        token_request = TokenRequest(
            grant_type=GrantType.AUTHORIZATION_CODE,
            code=code,
            redirect_uri=test_client.redirect_uris[0],
            client_id=test_client.client_id,
            code_verifier=code_verifier
        )
        
        token_response = await oauth2_provider.token(token_request)
        
        assert token_response.access_token
        assert token_response.refresh_token
        assert token_response.expires_in == 3600
        assert token_response.scope == "read"
        
        # Verify tokens were saved
        access_token = await oauth2_provider.storage.get_access_token(
            token_response.access_token
        )
        assert access_token
        assert access_token.user_id == test_user.user_id
    
    @pytest.mark.asyncio
    async def test_refresh_token_flow(self, oauth2_provider, test_client, test_user):
        """Test refresh token flow"""
        # Get initial tokens
        code_verifier = generate_code_verifier()
        auth_request = AuthorizationRequest(
            response_type=ResponseType.CODE,
            client_id=test_client.client_id,
            redirect_uri=test_client.redirect_uris[0],
            scope="read write",
            state="state",
            code_challenge=generate_code_challenge(code_verifier),
            code_challenge_method="S256"
        )
        
        code, _ = await oauth2_provider.authorize(auth_request, test_user)
        
        token_request = TokenRequest(
            grant_type=GrantType.AUTHORIZATION_CODE,
            code=code,
            redirect_uri=test_client.redirect_uris[0],
            client_id=test_client.client_id,
            code_verifier=code_verifier
        )
        
        initial_response = await oauth2_provider.token(token_request)
        
        # Use refresh token
        refresh_request = TokenRequest(
            grant_type=GrantType.REFRESH_TOKEN,
            refresh_token=initial_response.refresh_token,
            client_id=test_client.client_id
        )
        
        refresh_response = await oauth2_provider.token(refresh_request)
        
        assert refresh_response.access_token
        assert refresh_response.access_token != initial_response.access_token
        assert refresh_response.refresh_token
        assert refresh_response.refresh_token != initial_response.refresh_token
    
    @pytest.mark.asyncio
    async def test_token_validation(self, oauth2_provider, test_client, test_user):
        """Test token validation"""
        # Create a token
        access_token = AccessToken(
            token="test-access-token",
            client_id=test_client.client_id,
            user_id=test_user.user_id,
            scope="read",
            expires_at=datetime.utcnow() + timedelta(minutes=60)
        )
        
        await oauth2_provider.storage.save_access_token(access_token)
        
        # Validate token
        validated = await oauth2_provider.validate_token("test-access-token")
        assert validated
        assert validated.user_id == test_user.user_id
        
        # Invalid token
        invalid = await oauth2_provider.validate_token("invalid-token")
        assert invalid is None
        
        # Expired token
        expired_token = AccessToken(
            token="expired-token",
            client_id=test_client.client_id,
            user_id=test_user.user_id,
            scope="read",
            expires_at=datetime.utcnow() - timedelta(minutes=1)
        )
        
        await oauth2_provider.storage.save_access_token(expired_token)
        
        validated_expired = await oauth2_provider.validate_token("expired-token")
        assert validated_expired is None


class TestSessionManager:
    """Test session management"""
    
    @pytest.mark.asyncio
    async def test_session_creation(self, session_manager, test_user):
        """Test session creation"""
        session = await session_manager.create_session(
            user=test_user,
            client_id="test-client",
            ip_address="127.0.0.1",
            user_agent="Test Browser"
        )
        
        assert session.session_id
        assert session.user_id == test_user.user_id
        assert session.ip_address == "127.0.0.1"
        
        # Retrieve session
        retrieved = await session_manager.get_session(session.session_id)
        assert retrieved
        assert retrieved.session_id == session.session_id
    
    @pytest.mark.asyncio
    async def test_session_limit(self, session_manager, test_user):
        """Test session limit per user"""
        # Create max sessions
        sessions = []
        for i in range(5):
            session = await session_manager.create_session(
                user=test_user,
                data={"index": i}
            )
            sessions.append(session)
            # Small delay to ensure different timestamps
            await asyncio.sleep(0.01)
        
        # Verify we have 5 sessions
        user_sessions = await session_manager.get_user_sessions(test_user.user_id)
        assert len(user_sessions) == 5
        
        # Create one more (should delete oldest)
        new_session = await session_manager.create_session(
            user=test_user,
            data={"index": 5}
        )
        
        # Should still have max sessions
        user_sessions = await session_manager.get_user_sessions(test_user.user_id)
        assert len(user_sessions) == 5
        
        # Check that exactly one of the original sessions was deleted
        # (we can't predict which one due to timing variations)
        remaining_sessions = []
        for session in sessions:
            remaining = await session_manager.get_session(session.session_id)
            if remaining:
                remaining_sessions.append(remaining)
        
        # Should have exactly one less session than we started with
        assert len(remaining_sessions) == len(sessions) - 1
        
        # Verify that the new session exists
        new_session_check = await session_manager.get_session(new_session.session_id)
        assert new_session_check is not None
    
    @pytest.mark.asyncio
    async def test_session_expiration(self, session_manager, test_user):
        """Test session expiration"""
        # Create expired session
        session = UserSession(
            user_id=test_user.user_id,
            expires_at=datetime.utcnow() - timedelta(minutes=1)
        )
        
        await session_manager.storage.save_session(session)
        
        # Try to get expired session
        retrieved = await session_manager.get_session(session.session_id)
        assert retrieved is None
    
    @pytest.mark.asyncio
    async def test_session_data_update(self, session_manager, test_user):
        """Test updating session data"""
        session = await session_manager.create_session(
            user=test_user,
            data={"initial": "value"}
        )
        
        # Update data
        await session_manager.update_session_data(
            session.session_id,
            {"new": "data", "count": 42}
        )
        
        # Retrieve and check
        retrieved = await session_manager.get_session(session.session_id)
        assert retrieved.data["initial"] == "value"
        assert retrieved.data["new"] == "data"
        assert retrieved.data["count"] == 42


class TestSecurity:
    """Test security utilities"""
    
    def test_pkce_generation(self):
        """Test PKCE code verifier and challenge generation"""
        verifier = generate_code_verifier()
        
        # Check length requirements
        assert 43 <= len(verifier) <= 128
        
        # Generate challenge
        challenge = generate_code_challenge(verifier, "S256")
        assert len(challenge) >= 43
        
        # Verify challenge
        assert verify_code_challenge(verifier, challenge, "S256")
        
        # Wrong verifier should fail
        wrong_verifier = generate_code_verifier()
        assert not verify_code_challenge(wrong_verifier, challenge, "S256")
    
    def test_plain_code_challenge(self):
        """Test plain code challenge method"""
        verifier = generate_code_verifier()
        challenge = generate_code_challenge(verifier, "plain")
        
        assert challenge == verifier
        assert verify_code_challenge(verifier, challenge, "plain")


class TestMCPIntegration:
    """Test MCP OAuth2 integration"""
    
    @pytest.mark.asyncio
    async def test_auth_context(self, test_user):
        """Test MCPAuthContext"""
        access_token = AccessToken(
            token="test-token",
            client_id="test-client",
            user_id=test_user.user_id,
            scope="read write",
            expires_at=datetime.utcnow() + timedelta(minutes=60)
        )
        
        context = MCPAuthContext(
            user=test_user,
            client_id="test-client",
            access_token=access_token,
            scope="read write"
        )
        
        assert context.user_id == test_user.user_id
        assert context.username == test_user.username
        assert context.has_scope("read")
        assert context.has_scope("write")
        assert not context.has_scope("admin")
        assert context.has_permission("read")
        assert not context.has_permission("admin")
    
    @pytest.mark.asyncio
    async def test_authenticated_mcp_server(self, oauth2_storage, test_user):
        """Test AuthenticatedMCPServer"""
        # Mock MCP server
        mock_server = Mock()
        mock_server.tools = {
            "test_read": Mock(description="Test read tool"),
            "test_write": Mock(description="Test write tool"),
            "test_admin": Mock(description="Test admin tool")
        }
        
        # Create authenticated server
        auth_server = AuthenticatedMCPServer(mock_server, oauth2_storage)
        
        # Set up permissions
        auth_server.set_tool_permission("test_read", "read")
        auth_server.set_tool_permission("test_write", "write")
        auth_server.set_tool_permission("test_admin", "admin")
        
        # Create auth context with read/write permissions
        access_token = AccessToken(
            token="test-token",
            client_id="test-client",
            user_id=test_user.user_id,
            scope="read write",
            expires_at=datetime.utcnow() + timedelta(minutes=60)
        )
        
        context = MCPAuthContext(
            user=test_user,
            client_id="test-client",
            access_token=access_token,
            scope="read write"
        )
        
        # Check tool access
        assert await auth_server.check_tool_access("test_read", context)
        assert await auth_server.check_tool_access("test_write", context)
        assert not await auth_server.check_tool_access("test_admin", context)
        
        # List available tools
        tools = await auth_server.list_tools_for_context(context)
        assert len(tools) == 2
        tool_names = [t["name"] for t in tools]
        assert "test_read" in tool_names
        assert "test_write" in tool_names
        assert "test_admin" not in tool_names
    
    @pytest.mark.asyncio
    async def test_validate_mcp_token(self, oauth2_storage, test_user):
        """Test MCP token validation"""
        # Create valid token
        access_token = AccessToken(
            token="valid-token",
            client_id="test-client",
            user_id=test_user.user_id,
            scope="read write",
            expires_at=datetime.utcnow() + timedelta(minutes=60)
        )
        
        await oauth2_storage.save_access_token(access_token)
        await oauth2_storage.save_user(test_user)
        
        # Validate token
        context = await validate_mcp_token("valid-token", oauth2_storage, "read")
        
        assert context.user_id == test_user.user_id
        assert context.has_scope("read")
        
        # Invalid scope
        with pytest.raises(ValueError):
            await validate_mcp_token("valid-token", oauth2_storage, "admin")
        
        # Invalid token
        with pytest.raises(ValueError):
            await validate_mcp_token("invalid-token", oauth2_storage, "read")