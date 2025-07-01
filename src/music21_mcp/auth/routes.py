"""
FastAPI routes for OAuth2 authentication endpoints
Implements authorization, token, and metadata endpoints
"""

import logging
from typing import Optional
from urllib.parse import parse_qs, urlparse

from fastapi import (APIRouter, Depends, Form, HTTPException, Query, Request,
                     Response, status)
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel

from .models import (AuthorizationRequest, ClientRegistration, GrantType,
                     ResponseType, TokenRequest, TokenResponse, User)
from .oauth2_provider import OAuth2Provider
from .security import SecurityMiddleware, parse_basic_auth, require_scope
from .session_manager import SessionManager
from .storage import OAuth2Storage

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/auth", tags=["oauth2"])


# Dependencies
async def get_oauth2_provider(request: Request) -> OAuth2Provider:
    """Get OAuth2 provider from app state"""
    return request.app.state.oauth2_provider


async def get_session_manager(request: Request) -> SessionManager:
    """Get session manager from app state"""
    return request.app.state.session_manager


async def get_current_user(
    request: Request, session_manager: SessionManager = Depends(get_session_manager)
) -> Optional[User]:
    """Get current user from session cookie"""
    session_id = request.cookies.get("session_id")
    if not session_id:
        return None

    session = await session_manager.get_session(session_id)
    if not session:
        return None

    # Get user from storage
    storage: OAuth2Storage = request.app.state.oauth2_storage
    user = await storage.get_user(session.user_id)

    return user


# Request/Response models
class LoginForm(BaseModel):
    username: str
    password: str
    remember_me: bool = False


class RegisterForm(BaseModel):
    username: str
    email: str
    password: str
    full_name: Optional[str] = None


# Routes
@router.get("/.well-known/oauth-authorization-server")
async def oauth_server_metadata(
    oauth2_provider: OAuth2Provider = Depends(get_oauth2_provider),
):
    """OAuth2 Authorization Server Metadata endpoint (RFC 8414)"""
    return oauth2_provider.get_server_metadata()


@router.get("/.well-known/oauth-protected-resource")
async def protected_resource_metadata(
    oauth2_provider: OAuth2Provider = Depends(get_oauth2_provider),
):
    """OAuth2 Protected Resource Metadata endpoint"""
    return oauth2_provider.get_protected_resource_metadata()


@router.get("/authorize")
async def authorize_get(
    response_type: ResponseType = Query(...),
    client_id: str = Query(...),
    redirect_uri: str = Query(...),
    scope: str = Query("read"),
    state: str = Query(...),
    code_challenge: str = Query(...),
    code_challenge_method: str = Query("S256"),
    nonce: Optional[str] = Query(None),
    current_user: Optional[User] = Depends(get_current_user),
):
    """OAuth2 authorization endpoint - GET method (show consent form)"""
    if not current_user:
        # Redirect to login with return URL
        return_url = f"/auth/authorize?{Request.query_params}"
        return RedirectResponse(f"/auth/login?return_url={return_url}")

    # For demo, we'll auto-approve. In production, show consent form
    html_content = f"""
    <html>
    <head>
        <title>Authorize Application</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .container {{ max-width: 500px; margin: 0 auto; }}
            .app-info {{ background: #f0f0f0; padding: 20px; border-radius: 8px; }}
            .scopes {{ margin: 20px 0; }}
            .scope-item {{ margin: 10px 0; }}
            .buttons {{ margin-top: 30px; }}
            button {{ padding: 10px 20px; margin-right: 10px; }}
            .approve {{ background: #28a745; color: white; border: none; }}
            .deny {{ background: #dc3545; color: white; border: none; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Authorization Request</h1>
            <div class="app-info">
                <p><strong>Application:</strong> {client_id}</p>
                <p><strong>Redirect URI:</strong> {redirect_uri}</p>
            </div>
            
            <div class="scopes">
                <h3>This application is requesting access to:</h3>
                <div class="scope-item">✓ {scope}</div>
            </div>
            
            <p>Logged in as: <strong>{current_user.username}</strong></p>
            
            <form method="POST" action="/auth/authorize">
                <input type="hidden" name="response_type" value="{response_type}">
                <input type="hidden" name="client_id" value="{client_id}">
                <input type="hidden" name="redirect_uri" value="{redirect_uri}">
                <input type="hidden" name="scope" value="{scope}">
                <input type="hidden" name="state" value="{state}">
                <input type="hidden" name="code_challenge" value="{code_challenge}">
                <input type="hidden" name="code_challenge_method" value="{code_challenge_method}">
                
                <div class="buttons">
                    <button type="submit" name="action" value="approve" class="approve">
                        Approve
                    </button>
                    <button type="submit" name="action" value="deny" class="deny">
                        Deny
                    </button>
                </div>
            </form>
        </div>
    </body>
    </html>
    """

    return HTMLResponse(content=html_content)


@router.post("/authorize")
async def authorize_post(
    response_type: ResponseType = Form(...),
    client_id: str = Form(...),
    redirect_uri: str = Form(...),
    scope: str = Form("read"),
    state: str = Form(...),
    code_challenge: str = Form(...),
    code_challenge_method: str = Form("S256"),
    action: str = Form(...),
    current_user: Optional[User] = Depends(get_current_user),
    oauth2_provider: OAuth2Provider = Depends(get_oauth2_provider),
):
    """OAuth2 authorization endpoint - POST method (process consent)"""
    if not current_user:
        raise HTTPException(status_code=401, detail="Not authenticated")

    if action != "approve":
        # User denied authorization
        return RedirectResponse(f"{redirect_uri}?error=access_denied&state={state}")

    try:
        # Create authorization request
        auth_request = AuthorizationRequest(
            response_type=response_type,
            client_id=client_id,
            redirect_uri=redirect_uri,
            scope=scope,
            state=state,
            code_challenge=code_challenge,
            code_challenge_method=code_challenge_method,
        )

        # Process authorization
        code, state = await oauth2_provider.authorize(auth_request, current_user)

        # Redirect with authorization code
        return RedirectResponse(f"{redirect_uri}?code={code}&state={state}")

    except ValueError as e:
        # Return error to redirect URI
        return RedirectResponse(
            f"{redirect_uri}?error=invalid_request&error_description={str(e)}&state={state}"
        )
    except Exception as e:
        logger.error(f"Authorization error: {e}")
        return RedirectResponse(f"{redirect_uri}?error=server_error&state={state}")


@router.post("/token")
async def token_endpoint(
    grant_type: GrantType = Form(...),
    code: Optional[str] = Form(None),
    redirect_uri: Optional[str] = Form(None),
    client_id: str = Form(...),
    client_secret: Optional[str] = Form(None),
    code_verifier: Optional[str] = Form(None),
    refresh_token: Optional[str] = Form(None),
    scope: Optional[str] = Form(None),
    request: Request = None,
    oauth2_provider: OAuth2Provider = Depends(get_oauth2_provider),
):
    """OAuth2 token endpoint"""
    try:
        # Handle client authentication via Basic Auth if not in form
        if not client_secret and request.headers.get("Authorization"):
            auth_header = request.headers["Authorization"]
            if auth_header.startswith("Basic "):
                username, password = parse_basic_auth(auth_header)
                if username == client_id:
                    client_secret = password

        # Create token request
        token_request = TokenRequest(
            grant_type=grant_type,
            code=code,
            redirect_uri=redirect_uri,
            client_id=client_id,
            client_secret=client_secret,
            code_verifier=code_verifier,
            refresh_token=refresh_token,
            scope=scope,
        )

        # Process token request
        token_response = await oauth2_provider.token(token_request)

        return token_response.to_dict()

    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail={"error": "invalid_request", "error_description": str(e)},
        )
    except Exception as e:
        logger.error(f"Token endpoint error: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "server_error",
                "error_description": "Internal server error",
            },
        )


@router.post("/revoke")
async def revoke_token(
    token: str = Form(...),
    token_type_hint: Optional[str] = Form(None),
    client_id: str = Form(...),
    client_secret: Optional[str] = Form(None),
    request: Request = None,
    oauth2_provider: OAuth2Provider = Depends(get_oauth2_provider),
):
    """OAuth2 token revocation endpoint (RFC 7009)"""
    try:
        # Validate client credentials
        storage: OAuth2Storage = request.app.state.oauth2_storage
        client = await storage.get_client(client_id)

        if not client:
            raise HTTPException(status_code=401, detail="Invalid client")

        if client.client_type == "confidential":
            # Handle Basic Auth
            if not client_secret and request.headers.get("Authorization"):
                auth_header = request.headers["Authorization"]
                if auth_header.startswith("Basic "):
                    username, password = parse_basic_auth(auth_header)
                    if username == client_id:
                        client_secret = password

            if not client_secret or client_secret != client.client_secret:
                raise HTTPException(
                    status_code=401, detail="Invalid client credentials"
                )

        # Revoke token
        await oauth2_provider.revoke_token(token, token_type_hint)

        return Response(status_code=200)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Revocation error: {e}")
        return Response(status_code=200)  # Always return 200 per spec


@router.get("/login", response_class=HTMLResponse)
async def login_page(return_url: Optional[str] = Query(None)):
    """Simple login page for demo purposes"""
    html_content = f"""
    <html>
    <head>
        <title>Login</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .container {{ max-width: 400px; margin: 0 auto; }}
            input {{ width: 100%; padding: 8px; margin: 8px 0; }}
            button {{ width: 100%; padding: 10px; background: #007bff; color: white; border: none; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Login</h1>
            <form method="POST" action="/auth/login">
                <input type="hidden" name="return_url" value="{return_url or '/'}">
                <input type="text" name="username" placeholder="Username" required>
                <input type="password" name="password" placeholder="Password" required>
                <label>
                    <input type="checkbox" name="remember_me"> Remember me
                </label>
                <button type="submit">Login</button>
            </form>
            <p>Demo users: alice/password, bob/password</p>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@router.post("/login")
async def login(
    username: str = Form(...),
    password: str = Form(...),
    remember_me: bool = Form(False),
    return_url: str = Form("/"),
    request: Request = None,
    session_manager: SessionManager = Depends(get_session_manager),
):
    """Process login (demo implementation)"""
    # Demo user validation - replace with real authentication
    demo_users = {
        "alice": {
            "password": "password",
            "email": "alice@example.com",
            "full_name": "Alice Smith",
        },
        "bob": {
            "password": "password",
            "email": "bob@example.com",
            "full_name": "Bob Jones",
        },
    }

    if username not in demo_users or demo_users[username]["password"] != password:
        return RedirectResponse(
            "/auth/login?error=Invalid credentials", status_code=303
        )

    # Create user object
    storage: OAuth2Storage = request.app.state.oauth2_storage
    user = await storage.get_user_by_username(username)

    if not user:
        # Create user on first login
        user = User(
            username=username,
            email=demo_users[username]["email"],
            full_name=demo_users[username]["full_name"],
        )
        await storage.save_user(user)

    # Create session
    session = await session_manager.create_session(
        user=user,
        ip_address=request.client.host,
        user_agent=request.headers.get("User-Agent"),
    )

    # Create response with session cookie
    response = RedirectResponse(return_url, status_code=303)
    cookie_params = session_manager.create_session_cookie(session)

    if remember_me:
        cookie_params["max_age"] = 30 * 24 * 60 * 60  # 30 days

    response.set_cookie(**cookie_params)

    return response


@router.post("/logout")
async def logout(
    request: Request, session_manager: SessionManager = Depends(get_session_manager)
):
    """Logout and clear session"""
    session_id = request.cookies.get("session_id")

    if session_id:
        await session_manager.delete_session(session_id)

    response = RedirectResponse("/auth/login", status_code=303)
    response.delete_cookie("session_id")

    return response


@router.post("/register/client")
async def register_client(
    client_name: str = Form(...),
    redirect_uris: str = Form(...),  # Comma-separated
    client_type: str = Form("public"),
    scope: str = Form("read write"),
    current_user: User = Depends(SecurityMiddleware()),
    oauth2_provider: OAuth2Provider = Depends(get_oauth2_provider),
):
    """Register a new OAuth2 client (requires admin scope)"""
    # Check admin permission
    if not current_user.has_permission("admin"):
        raise HTTPException(status_code=403, detail="Admin permission required")

    # Parse redirect URIs
    uris = [uri.strip() for uri in redirect_uris.split(",")]

    # Register client
    client = await oauth2_provider.register_client(
        client_name=client_name,
        redirect_uris=uris,
        client_type=client_type,
        scope=scope,
    )

    return {
        "client_id": client.client_id,
        "client_secret": client.client_secret,
        "client_type": client.client_type,
        "redirect_uris": client.redirect_uris,
    }
