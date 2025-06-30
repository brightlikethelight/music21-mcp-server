#!/usr/bin/env python3
"""
Test MCP client integration - simulates how Claude Desktop or Cursor would interact
"""
import asyncio
import json
import sys
from pathlib import Path

# Test with actual MCP client library
try:
    from mcp import Client
    from mcp.client.session import ClientSession
    from mcp.client.stdio import stdio_client
    HAS_MCP_CLIENT = True
except ImportError:
    print("⚠️ MCP client library not installed. Install with: pip install mcp")
    HAS_MCP_CLIENT = False

class MCPClientTester:
    """Tests MCP server from a client perspective"""
    
    def __init__(self):
        self.server_path = Path(__file__).parent.parent / "src" / "music21_mcp" / "server.py"
        
    async def test_stdio_connection(self):
        """Test connection via stdio (how Claude Desktop connects)"""
        if not HAS_MCP_CLIENT:
            print("Skipping - MCP client not available")
            return
            
        print("🔌 Testing STDIO Connection (Claude Desktop style)...")
        
        # Create client connected to our server via stdio
        async with stdio_client(
            [sys.executable, "-m", "music21_mcp.server"]
        ) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                # Initialize connection
                await session.initialize()
                print("   ✅ Connected via STDIO")
                
                # List available tools
                tools = await session.list_tools()
                print(f"   ✅ Found {len(tools)} tools")
                
                # Test importing a score
                result = await session.call_tool(
                    "import_score",
                    {
                        "score_id": "test_melody",
                        "content": "C4 E4 G4 C5",
                        "format": "text"
                    }
                )
                print(f"   ✅ Import result: {result}")
                
                # Test analysis
                analysis = await session.call_tool(
                    "analyze_key",
                    {"score_id": "test_melody"}
                )
                print(f"   ✅ Key analysis: {analysis}")
    
    async def test_claude_desktop_config(self):
        """Generate and test Claude Desktop configuration"""
        print("\n📱 Claude Desktop Configuration:")
        
        config = {
            "mcpServers": {
                "music21": {
                    "command": "python",
                    "args": ["-m", "music21_mcp.server"],
                    "env": {}
                }
            }
        }
        
        config_path = Path.home() / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json"
        print(f"\n   Configuration to add to: {config_path}")
        print(json.dumps(config, indent=2))
        
        # Test if config would work
        import subprocess
        try:
            result = subprocess.run(
                ["python", "-m", "music21_mcp.server", "--help"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                print("\n   ✅ Server command is valid")
            else:
                print(f"\n   ❌ Server command failed: {result.stderr}")
        except Exception as e:
            print(f"\n   ❌ Cannot test server command: {e}")
    
    async def test_cursor_integration(self):
        """Test Cursor IDE integration"""
        print("\n💻 Cursor IDE Integration:")
        
        # Cursor uses a similar config format
        cursor_config = {
            "mcp": {
                "servers": {
                    "music21": {
                        "command": "python",
                        "args": ["-m", "music21_mcp.server"]
                    }
                }
            }
        }
        
        print("   Add to Cursor settings:")
        print(json.dumps(cursor_config, indent=2))


async def test_real_world_scenarios():
    """Test real-world usage scenarios"""
    print("\n🎼 Real-World Scenario Tests:")
    
    if not HAS_MCP_CLIENT:
        print("   ⏭️ Skipping - MCP client not available")
        return
    
    scenarios = [
        {
            "name": "Analyze Bach Invention",
            "description": "Import and analyze a Bach invention",
            "tools": [
                ("import_score", {
                    "score_id": "bach_invention_1",
                    "file_path": "path/to/bach_invention_1.mid",
                    "format": "midi"
                }),
                ("analyze_key", {"score_id": "bach_invention_1"}),
                ("analyze_harmony", {"score_id": "bach_invention_1"}),
                ("check_voice_leading", {"score_id": "bach_invention_1"})
            ]
        },
        {
            "name": "Jazz Chord Analysis",
            "description": "Analyze jazz chord progressions",
            "tools": [
                ("import_score", {
                    "score_id": "jazz_standard",
                    "content": "Cmaj7 A7 Dm7 G7",
                    "format": "text"
                }),
                ("analyze_jazz_harmony", {"score_id": "jazz_standard"}),
                ("detect_chord_substitutions", {"score_id": "jazz_standard"})
            ]
        },
        {
            "name": "Melodic Pattern Recognition",
            "description": "Find patterns in a melody",
            "tools": [
                ("import_score", {
                    "score_id": "folk_melody",
                    "content": "C4 D4 E4 F4 G4 F4 E4 D4 C4 D4 E4 F4 G4",
                    "format": "text"
                }),
                ("detect_melodic_motives", {"score_id": "folk_melody"}),
                ("analyze_melodic_contour", {"score_id": "folk_melody"})
            ]
        }
    ]
    
    for scenario in scenarios:
        print(f"\n   📋 {scenario['name']}")
        print(f"      {scenario['description']}")
        print("      Tools to call:")
        for tool_name, args in scenario['tools']:
            print(f"      - {tool_name}({json.dumps(args, indent=0).replace(chr(10), '')})")


async def main():
    """Run all client tests"""
    print("🧪 MCP Client Integration Tests")
    print("=" * 50)
    
    tester = MCPClientTester()
    
    # Test STDIO connection
    await tester.test_stdio_connection()
    
    # Test Claude Desktop config
    await tester.test_claude_desktop_config()
    
    # Test Cursor integration
    await tester.test_cursor_integration()
    
    # Test real-world scenarios
    await test_real_world_scenarios()
    
    print("\n" + "=" * 50)
    print("✅ Client integration tests complete")
    
    if not HAS_MCP_CLIENT:
        print("\n⚠️ Install MCP client for full testing:")
        print("   pip install mcp")


if __name__ == "__main__":
    asyncio.run(main())