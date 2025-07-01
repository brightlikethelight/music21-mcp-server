# OAuth2 Authentication Guide

This guide explains how to use OAuth2 authentication with the Music21 Remote MCP Server for secure remote access.

## Overview

The Music21 Remote MCP Server implements OAuth 2.1 with PKCE (Proof Key for Code Exchange) to provide secure authentication and authorization for remote access to MCP tools. The implementation follows industry best practices and supports:

- Authorization Code flow with PKCE (for web/mobile apps)
- Client Credentials flow (for server-to-server)
- Refresh Token rotation
- Session management with sliding expiration
- Fine-grained permissions with OAuth2 scopes

## Quick Start

### 1. Start the Remote Server

```bash
# Basic setup (in-memory storage)
music21-mcp-remote

# Production setup with Redis
music21-mcp-remote --redis-url redis://localhost:6379

# Custom configuration
music21-mcp-remote --host 0.0.0.0 --port 8000
```

### 2. Demo Users and Client

The server creates demo users and an OAuth2 client by default:

**Demo Users:**
- `alice` / `password` - Has read and write permissions
- `bob` / `password` - Has read-only permissions
- `admin` / `password` - Has admin permissions

**Demo Client:**
- Client ID is shown in server logs
- Public client (no secret required)
- Redirect URIs: `http://localhost:3000/callback`, `http://localhost:8080/callback`

### 3. OAuth2 Endpoints

- Authorization: `http://localhost:8000/auth/authorize`
- Token: `http://localhost:8000/auth/token`
- Metadata: `http://localhost:8000/auth/.well-known/oauth-authorization-server`

## Authentication Flow

### Web Application Flow (with PKCE)

1. **Generate PKCE codes:**
```python
import secrets
import hashlib
import base64

# Generate code verifier
code_verifier = base64.urlsafe_b64encode(secrets.token_bytes(32)).decode('utf-8').rstrip('=')

# Generate code challenge
code_challenge = base64.urlsafe_b64encode(
    hashlib.sha256(code_verifier.encode()).digest()
).decode('utf-8').rstrip('=')
```

2. **Redirect user to authorization endpoint:**
```
http://localhost:8000/auth/authorize?
  response_type=code&
  client_id=YOUR_CLIENT_ID&
  redirect_uri=http://localhost:3000/callback&
  scope=read write&
  state=random-state-value&
  code_challenge=YOUR_CODE_CHALLENGE&
  code_challenge_method=S256
```

3. **Exchange authorization code for tokens:**
```python
import httpx

response = httpx.post("http://localhost:8000/auth/token", data={
    "grant_type": "authorization_code",
    "code": authorization_code,
    "redirect_uri": "http://localhost:3000/callback",
    "client_id": client_id,
    "code_verifier": code_verifier
})

tokens = response.json()
access_token = tokens["access_token"]
refresh_token = tokens["refresh_token"]
```

### Server-to-Server Flow (Client Credentials)

For confidential clients only:

```python
response = httpx.post("http://localhost:8000/auth/token", 
    data={
        "grant_type": "client_credentials",
        "client_id": client_id,
        "client_secret": client_secret,
        "scope": "read write"
    }
)

access_token = response.json()["access_token"]
```

## Using MCP Tools with OAuth2

### List Available Tools

```python
headers = {"Authorization": f"Bearer {access_token}"}
response = httpx.get("http://localhost:8000/mcp/tools", headers=headers)
tools = response.json()["tools"]
```

### Execute a Tool

```python
response = httpx.post(
    "http://localhost:8000/mcp/execute/list_scores",
    headers=headers,
    json={"arguments": {}}
)

result = response.json()
```

### Session-Based Execution

For long-running operations or WebSocket-like behavior:

```python
# Create session
response = httpx.post("http://localhost:8000/mcp/session/create", headers=headers)
session_id = response.json()["session_id"]

# Execute in session
response = httpx.post(
    f"http://localhost:8000/mcp/session/{session_id}/execute/analyze_harmony",
    json={"arguments": {"score_id": "bach_invention_1"}}
)

# Close session
httpx.delete(f"http://localhost:8000/mcp/session/{session_id}")
```

## OAuth2 Scopes and Permissions

### Scopes

- `read` - Access to analysis and read-only tools
- `write` - Access to modification tools (transpose, harmonize, etc.)
- `admin` - Administrative operations

### Tool Permissions

| Tool | Required Scope |
|------|---------------|
| list_scores, get_score_info | read |
| analyze_harmony, analyze_key | read |
| import_score, transpose_score | write |
| harmonize_melody, create_counterpoint | write |
| clear_cache, reset_server | admin |

## Client Registration

Register a new OAuth2 client (requires admin access):

```python
admin_headers = {"Authorization": f"Bearer {admin_token}"}

response = httpx.post(
    "http://localhost:8000/auth/register/client",
    headers=admin_headers,
    data={
        "client_name": "My Music App",
        "redirect_uris": "https://myapp.com/callback,https://myapp.com/auth",
        "client_type": "public",  # or "confidential" for server apps
        "scope": "read write"
    }
)

client_info = response.json()
```

## Security Best Practices

### 1. Always Use HTTPS in Production
```python
# Configure with proper certificates
config = RemoteMCPConfig(
    base_url="https://api.example.com",
    # ... other settings
)
```

### 2. Implement Token Storage Securely
- Never store tokens in localStorage (use sessionStorage or secure cookies)
- Encrypt tokens at rest
- Implement token rotation

### 3. Configure CORS Properly
```python
config = RemoteMCPConfig(
    cors_origins=["https://myapp.com"],  # Specific origins only
)
```

### 4. Use Redis in Production
```python
config = RemoteMCPConfig(
    redis_url="redis://:password@redis-server:6379/0"
)
```

### 5. Implement Rate Limiting
The server includes built-in rate limiting for OAuth2 endpoints.

## Advanced Configuration

### Custom OAuth2 Configuration

```python
from music21_mcp.auth import OAuth2Config

oauth2_config = OAuth2Config(
    issuer="https://api.example.com",
    access_token_expire_minutes=30,  # Shorter for higher security
    refresh_token_expire_days=7,      # Shorter refresh window
    require_pkce=True,                # Always require PKCE
    allow_public_clients=False,       # Confidential clients only
    supported_scopes=["read", "write", "admin", "compose"]
)
```

### Session Configuration

```python
from music21_mcp.auth import SessionConfig

session_config = SessionConfig(
    session_ttl_minutes=15,           # Shorter sessions
    max_sessions_per_user=3,          # Limit concurrent sessions
    enable_sliding_expiration=True,   # Keep active sessions alive
    secure_cookie=True,               # HTTPS only
    same_site="strict"                # CSRF protection
)
```

### IP Whitelisting

```python
from music21_mcp.auth.security import IPWhitelist

whitelist = IPWhitelist([
    "192.168.1.0/24",
    "10.0.0.0/8",
    "2001:db8::/32"  # IPv6 support
])
```

## Troubleshooting

### Common Issues

1. **"Invalid client_id" error**
   - Check client ID in server logs
   - Ensure client is registered

2. **"Invalid redirect_uri" error**
   - URI must match exactly (including trailing slashes)
   - URI must be registered with client

3. **"Invalid code verifier" error**
   - Ensure verifier is 43-128 characters
   - Use same verifier for challenge and token exchange

4. **Token expired errors**
   - Implement automatic token refresh
   - Check server time synchronization

### Debug Mode

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Example: Python Client Library

```python
import httpx
import secrets
import hashlib
import base64
from urllib.parse import urlencode

class Music21MCPClient:
    def __init__(self, base_url, client_id, redirect_uri):
        self.base_url = base_url
        self.client_id = client_id
        self.redirect_uri = redirect_uri
        self.access_token = None
        self.refresh_token = None
        
    def get_authorization_url(self):
        """Get OAuth2 authorization URL with PKCE"""
        self.code_verifier = base64.urlsafe_b64encode(
            secrets.token_bytes(32)
        ).decode('utf-8').rstrip('=')
        
        code_challenge = base64.urlsafe_b64encode(
            hashlib.sha256(self.code_verifier.encode()).digest()
        ).decode('utf-8').rstrip('=')
        
        params = {
            "response_type": "code",
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "scope": "read write",
            "state": secrets.token_urlsafe(16),
            "code_challenge": code_challenge,
            "code_challenge_method": "S256"
        }
        
        return f"{self.base_url}/auth/authorize?{urlencode(params)}"
    
    def exchange_code(self, code):
        """Exchange authorization code for tokens"""
        response = httpx.post(f"{self.base_url}/auth/token", data={
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": self.redirect_uri,
            "client_id": self.client_id,
            "code_verifier": self.code_verifier
        })
        
        if response.status_code == 200:
            tokens = response.json()
            self.access_token = tokens["access_token"]
            self.refresh_token = tokens["refresh_token"]
            return True
        return False
    
    def refresh_access_token(self):
        """Refresh access token"""
        response = httpx.post(f"{self.base_url}/auth/token", data={
            "grant_type": "refresh_token",
            "refresh_token": self.refresh_token,
            "client_id": self.client_id
        })
        
        if response.status_code == 200:
            tokens = response.json()
            self.access_token = tokens["access_token"]
            self.refresh_token = tokens["refresh_token"]
            return True
        return False
    
    def list_tools(self):
        """List available MCP tools"""
        headers = {"Authorization": f"Bearer {self.access_token}"}
        response = httpx.get(f"{self.base_url}/mcp/tools", headers=headers)
        
        if response.status_code == 401:
            if self.refresh_access_token():
                return self.list_tools()
        
        return response.json()
    
    def execute_tool(self, tool_name, arguments):
        """Execute an MCP tool"""
        headers = {"Authorization": f"Bearer {self.access_token}"}
        response = httpx.post(
            f"{self.base_url}/mcp/execute/{tool_name}",
            headers=headers,
            json=arguments
        )
        
        if response.status_code == 401:
            if self.refresh_access_token():
                return self.execute_tool(tool_name, arguments)
        
        return response.json()

# Usage
client = Music21MCPClient(
    base_url="http://localhost:8000",
    client_id="your-client-id",
    redirect_uri="http://localhost:3000/callback"
)

# Get authorization URL for user
auth_url = client.get_authorization_url()
print(f"Visit: {auth_url}")

# After user authorizes and you get the code
client.exchange_code(authorization_code)

# Use the client
tools = client.list_tools()
result = client.execute_tool("analyze_harmony", {"score_id": "bach_invention_1"})
```

## Production Deployment

See the [Production Deployment Guide](./production_deployment.md) for details on:
- Docker configuration with OAuth2
- Kubernetes deployment with secrets
- TLS/SSL configuration
- Redis clustering
- Monitoring OAuth2 metrics