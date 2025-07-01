"""
Remote MCP Server with OAuth2 authentication
Provides secure remote access to Music21 MCP tools
"""
import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# Import existing MCP server
from .server import create_server, ScoreManager

# Import auth components
from .auth import (
    OAuth2Provider, OAuth2Config,
    SessionManager, SessionConfig,
    SecurityMiddleware
)
from .auth.storage import OAuth2Storage, SessionStorage, InMemoryOAuth2Storage, InMemorySessionStorage
from .auth.routes import router as auth_router
from .auth.mcp_integration import (
    AuthenticatedMCPServer, MCPSessionManager,
    mcp_auth_required, MCPAuthContext
)

# Try Redis import
try:
    import redis.asyncio as redis
    from .auth.storage import RedisOAuth2Storage, RedisSessionStorage
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None

logger = logging.getLogger(__name__)


class RemoteMCPConfig:
    """Configuration for remote MCP server"""
    
    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8000,
        base_url: str = "http://localhost:8000",
        redis_url: Optional[str] = None,
        cors_origins: list = None,
        max_file_size_mb: int = 50,
        session_ttl_minutes: int = 30,
        require_pkce: bool = True,
        enable_demo_users: bool = True
    ):
        self.host = host
        self.port = port
        self.base_url = base_url
        self.redis_url = redis_url
        self.cors_origins = cors_origins or ["http://localhost:3000"]
        self.max_file_size_mb = max_file_size_mb
        self.session_ttl_minutes = session_ttl_minutes
        self.require_pkce = require_pkce
        self.enable_demo_users = enable_demo_users


async def setup_storage(config: RemoteMCPConfig) -> tuple[OAuth2Storage, SessionStorage]:
    """Set up storage backends"""
    if config.redis_url and REDIS_AVAILABLE:
        # Use Redis for production
        redis_client = await redis.from_url(config.redis_url)
        oauth2_storage = RedisOAuth2Storage(redis_client)
        session_storage = RedisSessionStorage(redis_client)
        logger.info("Using Redis storage")
    else:
        # Use in-memory for development
        oauth2_storage = InMemoryOAuth2Storage()
        session_storage = InMemorySessionStorage()
        logger.info("Using in-memory storage")
    
    return oauth2_storage, session_storage


async def setup_demo_data(oauth2_storage: OAuth2Storage, oauth2_provider: OAuth2Provider):
    """Set up demo users and clients"""
    # Create demo users
    from .auth.models import User
    
    demo_users = [
        User(
            username="alice",
            email="alice@example.com",
            full_name="Alice Smith",
            permissions={"read", "write"}
        ),
        User(
            username="bob",
            email="bob@example.com",
            full_name="Bob Jones",
            permissions={"read"}
        ),
        User(
            username="admin",
            email="admin@example.com",
            full_name="Admin User",
            permissions={"read", "write", "admin"}
        )
    ]
    
    for user in demo_users:
        await oauth2_storage.save_user(user)
    
    # Create demo OAuth2 client
    client = await oauth2_provider.register_client(
        client_name="Demo MCP Client",
        redirect_uris=[
            "http://localhost:3000/callback",
            "http://localhost:8080/callback"
        ],
        client_type="public"
    )
    
    logger.info(f"Demo client created: {client.client_id}")
    
    # Create confidential client for server-to-server
    server_client = await oauth2_provider.register_client(
        client_name="Server MCP Client",
        redirect_uris=["urn:ietf:wg:oauth:2.0:oob"],
        client_type="confidential",
        grant_types=["client_credentials"]
    )
    
    logger.info(f"Server client created: {server_client.client_id} / {server_client.client_secret}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    config: RemoteMCPConfig = app.state.config
    
    # Set up storage
    oauth2_storage, session_storage = await setup_storage(config)
    app.state.oauth2_storage = oauth2_storage
    app.state.session_storage = session_storage
    
    # Set up OAuth2 provider
    oauth2_config = OAuth2Config(
        issuer=config.base_url,
        require_pkce=config.require_pkce,
        access_token_expire_minutes=60,
        refresh_token_expire_days=30
    )
    oauth2_provider = OAuth2Provider(oauth2_config, oauth2_storage)
    app.state.oauth2_provider = oauth2_provider
    
    # Set up session manager
    session_config = SessionConfig(
        session_ttl_minutes=config.session_ttl_minutes,
        enable_sliding_expiration=True
    )
    session_manager = SessionManager(session_config, session_storage)
    app.state.session_manager = session_manager
    
    # Start session manager
    await session_manager.start()
    
    # Set up MCP server
    score_manager = ScoreManager()
    mcp_server = create_server(score_manager)
    app.state.mcp_server = mcp_server
    
    # Set up authenticated MCP server
    authenticated_server = AuthenticatedMCPServer(mcp_server, oauth2_storage)
    app.state.authenticated_server = authenticated_server
    
    # Set up MCP session manager
    mcp_session_manager = MCPSessionManager(authenticated_server)
    app.state.mcp_session_manager = mcp_session_manager
    
    # Set up demo data if enabled
    if config.enable_demo_users:
        await setup_demo_data(oauth2_storage, oauth2_provider)
    
    logger.info("Remote MCP server started")
    
    yield
    
    # Shutdown
    await session_manager.stop()
    
    # Close Redis connection if used
    if hasattr(oauth2_storage, 'redis'):
        await oauth2_storage.redis.close()
    
    logger.info("Remote MCP server stopped")


def create_remote_app(config: Optional[RemoteMCPConfig] = None) -> FastAPI:
    """Create FastAPI app for remote MCP server"""
    config = config or RemoteMCPConfig()
    
    app = FastAPI(
        title="Music21 Remote MCP Server",
        description="Remote access to Music21 MCP tools with OAuth2 authentication",
        version="1.0.0",
        lifespan=lifespan
    )
    
    # Store config
    app.state.config = config
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"]
    )
    
    # Include auth routes
    app.include_router(auth_router)
    
    # Root endpoint
    @app.get("/")
    async def root():
        return {
            "name": "Music21 Remote MCP Server",
            "version": "1.0.0",
            "oauth2": {
                "authorization_endpoint": "/auth/authorize",
                "token_endpoint": "/auth/token",
                "metadata_endpoint": "/auth/.well-known/oauth-authorization-server"
            },
            "mcp": {
                "tools_endpoint": "/mcp/tools",
                "execute_endpoint": "/mcp/execute"
            }
        }
    
    # MCP endpoints
    @app.get("/mcp/tools")
    @mcp_auth_required("read")
    async def list_tools(auth_context: MCPAuthContext, request: Request):
        """List available MCP tools for authenticated user"""
        authenticated_server: AuthenticatedMCPServer = request.app.state.authenticated_server
        tools = await authenticated_server.list_tools_for_context(auth_context)
        return {"tools": tools}
    
    @app.post("/mcp/execute/{tool_name}")
    @mcp_auth_required("read")  # Actual permission checked per tool
    async def execute_tool(
        tool_name: str,
        arguments: dict,
        auth_context: MCPAuthContext,
        request: Request
    ):
        """Execute MCP tool with authentication"""
        authenticated_server: AuthenticatedMCPServer = request.app.state.authenticated_server
        
        try:
            result = await authenticated_server.execute_tool(
                tool_name,
                arguments,
                auth_context
            )
            return {
                "success": True,
                "result": result
            }
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Tool execution error: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    # Session-based MCP execution
    @app.post("/mcp/session/create")
    @mcp_auth_required("read")
    async def create_mcp_session(auth_context: MCPAuthContext, request: Request):
        """Create MCP session for WebSocket or long-polling"""
        import uuid
        session_id = str(uuid.uuid4())
        
        mcp_session_manager: MCPSessionManager = request.app.state.mcp_session_manager
        await mcp_session_manager.create_mcp_session(session_id, auth_context)
        
        return {
            "session_id": session_id,
            "expires_in": 3600  # 1 hour
        }
    
    @app.post("/mcp/session/{session_id}/execute/{tool_name}")
    async def execute_in_session(
        session_id: str,
        tool_name: str,
        arguments: dict,
        request: Request
    ):
        """Execute tool in existing session"""
        mcp_session_manager: MCPSessionManager = request.app.state.mcp_session_manager
        
        try:
            result = await mcp_session_manager.execute_in_session(
                session_id,
                tool_name,
                arguments
            )
            return {
                "success": True,
                "result": result
            }
        except ValueError as e:
            raise HTTPException(status_code=401, detail=str(e))
        except Exception as e:
            logger.error(f"Session execution error: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    @app.delete("/mcp/session/{session_id}")
    async def close_mcp_session(session_id: str, request: Request):
        """Close MCP session"""
        mcp_session_manager: MCPSessionManager = request.app.state.mcp_session_manager
        await mcp_session_manager.remove_mcp_session(session_id)
        return {"status": "closed"}
    
    # Health check
    @app.get("/health")
    async def health_check():
        return {
            "status": "healthy",
            "service": "music21-mcp-remote"
        }
    
    return app


def main():
    """Run remote MCP server"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Music21 Remote MCP Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--redis-url", help="Redis URL for production storage")
    parser.add_argument("--no-demo", action="store_true", help="Disable demo users")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create config
    config = RemoteMCPConfig(
        host=args.host,
        port=args.port,
        redis_url=args.redis_url,
        enable_demo_users=not args.no_demo
    )
    
    # Create app
    app = create_remote_app(config)
    
    # Run server
    uvicorn.run(
        app if args.reload else "music21_mcp.server_remote:create_remote_app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )


if __name__ == "__main__":
    main()