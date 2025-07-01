#!/usr/bin/env python3
"""
Example OAuth2 client for Music21 Remote MCP Server
Demonstrates the OAuth2 authorization code flow with PKCE
"""
import asyncio
import base64
import hashlib
import secrets
import webbrowser
from urllib.parse import urlencode, parse_qs, urlparse

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
import uvicorn


class OAuth2MCPClient:
    """Simple OAuth2 client for Music21 MCP Server"""
    
    def __init__(self, server_url="http://localhost:8000", client_id=None):
        self.server_url = server_url
        self.client_id = client_id
        self.redirect_uri = "http://localhost:8888/callback"
        self.access_token = None
        self.refresh_token = None
        self.code_verifier = None
        self.state = None
        
    async def discover_endpoints(self):
        """Discover OAuth2 endpoints from server metadata"""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.server_url}/auth/.well-known/oauth-authorization-server"
            )
            metadata = response.json()
            
            self.auth_endpoint = metadata["authorization_endpoint"]
            self.token_endpoint = metadata["token_endpoint"]
            
            print(f"Discovered endpoints:")
            print(f"  Authorization: {self.auth_endpoint}")
            print(f"  Token: {self.token_endpoint}")
    
    def generate_pkce_codes(self):
        """Generate PKCE code verifier and challenge"""
        # Generate code verifier (43-128 characters)
        self.code_verifier = base64.urlsafe_b64encode(
            secrets.token_bytes(32)
        ).decode('utf-8').rstrip('=')
        
        # Generate code challenge (SHA256)
        challenge_bytes = hashlib.sha256(self.code_verifier.encode()).digest()
        code_challenge = base64.urlsafe_b64encode(challenge_bytes).decode('utf-8').rstrip('=')
        
        return code_challenge
    
    def get_authorization_url(self, scope="read write"):
        """Generate authorization URL"""
        code_challenge = self.generate_pkce_codes()
        self.state = secrets.token_urlsafe(16)
        
        params = {
            "response_type": "code",
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "scope": scope,
            "state": self.state,
            "code_challenge": code_challenge,
            "code_challenge_method": "S256"
        }
        
        return f"{self.auth_endpoint}?{urlencode(params)}"
    
    async def exchange_code(self, code):
        """Exchange authorization code for tokens"""
        async with httpx.AsyncClient() as client:
            response = await client.post(self.token_endpoint, data={
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": self.redirect_uri,
                "client_id": self.client_id,
                "code_verifier": self.code_verifier
            })
            
            if response.status_code == 200:
                tokens = response.json()
                self.access_token = tokens["access_token"]
                self.refresh_token = tokens.get("refresh_token")
                print(f"\n✅ Successfully obtained tokens!")
                print(f"Access token: {self.access_token[:20]}...")
                return True
            else:
                print(f"\n❌ Token exchange failed: {response.text}")
                return False
    
    async def refresh_tokens(self):
        """Refresh access token using refresh token"""
        if not self.refresh_token:
            print("No refresh token available")
            return False
        
        async with httpx.AsyncClient() as client:
            response = await client.post(self.token_endpoint, data={
                "grant_type": "refresh_token",
                "refresh_token": self.refresh_token,
                "client_id": self.client_id
            })
            
            if response.status_code == 200:
                tokens = response.json()
                self.access_token = tokens["access_token"]
                self.refresh_token = tokens.get("refresh_token", self.refresh_token)
                print("✅ Tokens refreshed successfully")
                return True
            else:
                print(f"❌ Token refresh failed: {response.text}")
                return False
    
    async def list_tools(self):
        """List available MCP tools"""
        headers = {"Authorization": f"Bearer {self.access_token}"}
        
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.server_url}/mcp/tools",
                headers=headers
            )
            
            if response.status_code == 200:
                return response.json()["tools"]
            elif response.status_code == 401:
                print("Token expired, attempting refresh...")
                if await self.refresh_tokens():
                    return await self.list_tools()
            
            print(f"Error listing tools: {response.text}")
            return []
    
    async def execute_tool(self, tool_name, arguments=None):
        """Execute an MCP tool"""
        headers = {"Authorization": f"Bearer {self.access_token}"}
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.server_url}/mcp/execute/{tool_name}",
                headers=headers,
                json=arguments or {}
            )
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 401:
                print("Token expired, attempting refresh...")
                if await self.refresh_tokens():
                    return await self.execute_tool(tool_name, arguments)
            
            print(f"Error executing tool: {response.text}")
            return None


# Global client instance
oauth_client = None


# Callback server
app = FastAPI()


@app.get("/")
async def root():
    """Home page with login button"""
    return HTMLResponse("""
    <html>
    <head>
        <title>Music21 MCP OAuth2 Client</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .container { max-width: 600px; margin: 0 auto; }
            button { padding: 10px 20px; font-size: 16px; cursor: pointer; }
            .success { color: green; }
            .error { color: red; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Music21 MCP OAuth2 Client</h1>
            <p>Click the button below to authenticate with the Music21 MCP Server:</p>
            <button onclick="window.location.href='/login'">Login with OAuth2</button>
        </div>
    </body>
    </html>
    """)


@app.get("/login")
async def login():
    """Initiate OAuth2 login"""
    auth_url = oauth_client.get_authorization_url()
    # Open browser automatically
    webbrowser.open(auth_url)
    return HTMLResponse(f"""
    <html>
    <head>
        <title>Redirecting...</title>
        <meta http-equiv="refresh" content="0; url={auth_url}">
    </head>
    <body>
        <p>Redirecting to authorization server...</p>
        <p>If not redirected, <a href="{auth_url}">click here</a>.</p>
    </body>
    </html>
    """)


@app.get("/callback")
async def callback(code: str = None, state: str = None, error: str = None):
    """OAuth2 callback endpoint"""
    if error:
        return HTMLResponse(f"""
        <html>
        <body>
            <h1 class="error">Authorization Error</h1>
            <p>{error}</p>
            <a href="/">Try again</a>
        </body>
        </html>
        """)
    
    if not code:
        return HTMLResponse("""
        <html>
        <body>
            <h1 class="error">Error</h1>
            <p>No authorization code received</p>
            <a href="/">Try again</a>
        </body>
        </html>
        """)
    
    # Verify state
    if state != oauth_client.state:
        return HTMLResponse("""
        <html>
        <body>
            <h1 class="error">Security Error</h1>
            <p>Invalid state parameter</p>
            <a href="/">Try again</a>
        </body>
        </html>
        """)
    
    # Exchange code for tokens
    success = await oauth_client.exchange_code(code)
    
    if success:
        # List available tools
        tools = await oauth_client.list_tools()
        tools_html = "".join([
            f"<li><strong>{tool['name']}</strong>: {tool['description']} "
            f"(requires: {tool['required_scope']})</li>"
            for tool in tools
        ])
        
        return HTMLResponse(f"""
        <html>
        <head>
            <title>Success!</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .container {{ max-width: 800px; margin: 0 auto; }}
                .success {{ color: green; }}
                .token {{ background: #f0f0f0; padding: 10px; word-break: break-all; }}
                ul {{ line-height: 1.8; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1 class="success">✅ Authentication Successful!</h1>
                <p>You can now close this window and return to the terminal.</p>
                
                <h2>Access Token:</h2>
                <div class="token">{oauth_client.access_token[:50]}...</div>
                
                <h2>Available Tools ({len(tools)}):</h2>
                <ul>{tools_html}</ul>
                
                <p><strong>Check the terminal for the interactive demo!</strong></p>
            </div>
        </body>
        </html>
        """)
    else:
        return HTMLResponse("""
        <html>
        <body>
            <h1 class="error">Token Exchange Failed</h1>
            <p>Could not exchange authorization code for tokens</p>
            <a href="/">Try again</a>
        </body>
        </html>
        """)


async def run_interactive_demo():
    """Run interactive demo after authentication"""
    print("\n" + "="*60)
    print("🎵 Music21 MCP Client - Interactive Demo")
    print("="*60)
    
    while True:
        print("\nOptions:")
        print("1. List available tools")
        print("2. List scores in library")
        print("3. Analyze harmony")
        print("4. Get score info")
        print("5. Refresh tokens")
        print("6. Exit")
        
        choice = input("\nEnter your choice (1-6): ")
        
        if choice == "1":
            tools = await oauth_client.list_tools()
            print(f"\nAvailable tools ({len(tools)}):")
            for tool in tools:
                print(f"  - {tool['name']}: {tool['description']}")
        
        elif choice == "2":
            result = await oauth_client.execute_tool("list_scores")
            if result and result.get("success"):
                scores = result["result"].get("scores", [])
                print(f"\nScores in library ({len(scores)}):")
                for score in scores[:10]:  # Show first 10
                    print(f"  - {score['id']}: {score['title']} by {score['composer']}")
                if len(scores) > 10:
                    print(f"  ... and {len(scores) - 10} more")
        
        elif choice == "3":
            score_id = input("Enter score ID (e.g., bach_invention_1): ")
            result = await oauth_client.execute_tool(
                "analyze_harmony",
                {"score_id": score_id}
            )
            if result and result.get("success"):
                analysis = result["result"]
                print(f"\nHarmonic Analysis of {score_id}:")
                print(f"  Key: {analysis.get('key', 'Unknown')}")
                print(f"  Progression: {analysis.get('progression_summary', 'N/A')}")
        
        elif choice == "4":
            score_id = input("Enter score ID: ")
            result = await oauth_client.execute_tool(
                "get_score_info",
                {"score_id": score_id}
            )
            if result and result.get("success"):
                info = result["result"]
                print(f"\nScore Info for {score_id}:")
                for key, value in info.items():
                    print(f"  {key}: {value}")
        
        elif choice == "5":
            if await oauth_client.refresh_tokens():
                print("✅ Tokens refreshed successfully!")
            else:
                print("❌ Failed to refresh tokens")
        
        elif choice == "6":
            print("\nGoodbye! 👋")
            break
        
        else:
            print("Invalid choice, please try again.")


async def main():
    """Main function"""
    global oauth_client
    
    print("🎵 Music21 MCP OAuth2 Client Demo")
    print("="*40)
    
    # Initialize client
    server_url = input("Enter MCP server URL (default: http://localhost:8000): ").strip()
    if not server_url:
        server_url = "http://localhost:8000"
    
    oauth_client = OAuth2MCPClient(server_url)
    
    # Discover endpoints
    await oauth_client.discover_endpoints()
    
    # Get client ID
    print("\nNote: Check the server logs for the demo client ID")
    client_id = input("Enter OAuth2 client ID: ").strip()
    oauth_client.client_id = client_id
    
    # Start callback server
    print(f"\n🌐 Starting callback server on http://localhost:8888")
    print("📋 A browser window will open for authentication...")
    
    # Run server in background
    config = uvicorn.Config(app, host="localhost", port=8888, log_level="error")
    server = uvicorn.Server(config)
    
    # Start server task
    server_task = asyncio.create_task(server.serve())
    
    # Wait a bit for server to start
    await asyncio.sleep(1)
    
    # Open browser
    webbrowser.open("http://localhost:8888")
    
    # Wait for authentication
    print("\n⏳ Waiting for authentication...")
    print("   (Complete the login in your browser)")
    
    while oauth_client.access_token is None:
        await asyncio.sleep(1)
    
    # Run interactive demo
    await run_interactive_demo()
    
    # Shutdown server
    server.should_exit = True
    await server_task


if __name__ == "__main__":
    asyncio.run(main())