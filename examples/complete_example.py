#!/usr/bin/env python3
"""
Complete End-to-End Example: Music21 MCP Server
===============================================

This example shows how to:
1. Start the MCP server
2. Connect to it using MCP client
3. Analyze music using the tools
4. Get results and use them

Prerequisites:
- pip install music21-mcp-server
- Install music21 corpus data: python -m music21.configure

For Claude Desktop integration, add to your MCP config:
{
  "music21": {
    "command": "python",
    "args": ["-m", "music21_mcp.server"],
    "env": {}
  }
}
"""

import asyncio
import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mcp.client.stdio import StdioClientTransport
from mcp.client.session import ClientSession
from mcp.client import ClientResult
from mcp.types import CallToolRequest, TextContent, Tool
import subprocess
import json
import tempfile
import os

class MusicAnalysisExample:
    """Complete example of using the Music21 MCP Server"""
    
    def __init__(self):
        self.server_process = None
        self.client_session = None
        self.transport = None
    
    async def start_server(self):
        """Start the MCP server as a subprocess"""
        print("🎵 Starting Music21 MCP Server...")
        
        # Start the server using stdio transport
        server_script = Path(__file__).parent.parent / "src" / "music21_mcp" / "server_minimal.py"
        
        self.server_process = await asyncio.create_subprocess_exec(
            sys.executable, str(server_script),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        # Create MCP client transport
        self.transport = StdioClientTransport(
            self.server_process.stdout,
            self.server_process.stdin
        )
        
        # Create client session
        self.client_session = ClientSession(self.transport)
        
        # Initialize the session
        await self.client_session.initialize()
        
        print("✅ MCP Server started successfully!")
        print(f"✅ Server PID: {self.server_process.pid}")
        
        return True
    
    async def list_available_tools(self):
        """List all available tools on the server"""
        print("\n📋 Available Music Analysis Tools:")
        print("=" * 50)
        
        # Get list of tools
        tools_response = await self.client_session.list_tools()
        
        for tool in tools_response.tools:
            print(f"🔧 {tool.name}")
            print(f"   Description: {tool.description}")
            if hasattr(tool, 'inputSchema') and tool.inputSchema:
                print(f"   Input: {tool.inputSchema}")
            print()
        
        return tools_response.tools
    
    async def demonstrate_music_analysis(self):
        """Demonstrate complete music analysis workflow"""
        print("\n🎼 Music Analysis Demonstration")
        print("=" * 50)
        
        # Step 1: Import a score from the corpus
        print("\n1️⃣ Importing a Bach chorale from corpus...")
        
        import_result = await self.client_session.call_tool(
            CallToolRequest(
                name="import_score",
                arguments={
                    "score_id": "bach_chorale",
                    "source": "bach/bwv66.6",
                    "source_type": "corpus"
                }
            )
        )
        
        if import_result.isError:
            print(f"❌ Import failed: {import_result.error}")
            return False
        
        print("✅ Successfully imported Bach chorale")
        print(f"   Result: {import_result.content[0].text}")
        
        # Step 2: Get basic score information
        print("\n2️⃣ Getting score information...")
        
        info_result = await self.client_session.call_tool(
            CallToolRequest(
                name="score_info",
                arguments={"score_id": "bach_chorale"}
            )
        )
        
        if info_result.isError:
            print(f"❌ Info failed: {info_result.error}")
        else:
            print("✅ Score information retrieved")
            info_data = json.loads(info_result.content[0].text)
            print(f"   Title: {info_data.get('title', 'Unknown')}")
            print(f"   Composer: {info_data.get('composer', 'Unknown')}")
            print(f"   Parts: {info_data.get('parts', 0)}")
            print(f"   Measures: {info_data.get('measures', 0)}")
        
        # Step 3: Analyze the key
        print("\n3️⃣ Analyzing musical key...")
        
        key_result = await self.client_session.call_tool(
            CallToolRequest(
                name="key_analysis",
                arguments={"score_id": "bach_chorale"}
            )
        )
        
        if key_result.isError:
            print(f"❌ Key analysis failed: {key_result.error}")
        else:
            print("✅ Key analysis completed")
            key_data = json.loads(key_result.content[0].text)
            print(f"   Detected key: {key_data.get('key', 'Unknown')}")
            print(f"   Confidence: {key_data.get('confidence', 0):.2f}")
        
        # Step 4: Analyze harmony
        print("\n4️⃣ Analyzing harmonic progressions...")
        
        harmony_result = await self.client_session.call_tool(
            CallToolRequest(
                name="harmony_analysis",
                arguments={"score_id": "bach_chorale"}
            )
        )
        
        if harmony_result.isError:
            print(f"❌ Harmony analysis failed: {harmony_result.error}")
        else:
            print("✅ Harmony analysis completed")
            harmony_data = json.loads(harmony_result.content[0].text)
            roman_numerals = harmony_data.get('roman_numerals', [])
            print(f"   Found {len(roman_numerals)} chords")
            
            # Show first few chords
            if roman_numerals:
                print("   First few chords:")
                for i, chord in enumerate(roman_numerals[:5]):
                    print(f"     {i+1}. {chord.get('roman_numeral', '?')} ({chord.get('chord', 'Unknown')})")
        
        # Step 5: Pattern recognition
        print("\n5️⃣ Detecting melodic patterns...")
        
        pattern_result = await self.client_session.call_tool(
            CallToolRequest(
                name="pattern_recognition",
                arguments={"score_id": "bach_chorale"}
            )
        )
        
        if pattern_result.isError:
            print(f"❌ Pattern recognition failed: {pattern_result.error}")
        else:
            print("✅ Pattern recognition completed")
            pattern_data = json.loads(pattern_result.content[0].text)
            melodic_patterns = pattern_data.get('melodic_patterns', {})
            sequences = melodic_patterns.get('sequences', [])
            motifs = melodic_patterns.get('motifs', [])
            
            print(f"   Found {len(sequences)} melodic sequences")
            print(f"   Found {len(motifs)} motivic patterns")
        
        # Step 6: List all scores
        print("\n6️⃣ Listing all loaded scores...")
        
        list_result = await self.client_session.call_tool(
            CallToolRequest(
                name="list_scores",
                arguments={}
            )
        )
        
        if list_result.isError:
            print(f"❌ List failed: {list_result.error}")
        else:
            print("✅ Score list retrieved")
            list_data = json.loads(list_result.content[0].text)
            scores = list_data.get('scores', [])
            print(f"   Total scores in memory: {len(scores)}")
            for score in scores:
                print(f"     - {score.get('id', 'Unknown')} ({score.get('title', 'Untitled')})")
        
        return True
    
    async def demonstrate_health_check(self):
        """Demonstrate server health monitoring"""
        print("\n🏥 Server Health Check")
        print("=" * 30)
        
        health_result = await self.client_session.call_tool(
            CallToolRequest(
                name="health_check",
                arguments={}
            )
        )
        
        if health_result.isError:
            print(f"❌ Health check failed: {health_result.error}")
        else:
            print("✅ Server is healthy")
            health_data = json.loads(health_result.content[0].text)
            print(f"   Status: {health_data.get('status', 'unknown')}")
            print(f"   Memory usage: {health_data.get('memory', {}).get('used_mb', 0):.1f} MB")
            print(f"   Uptime: {health_data.get('uptime_seconds', 0):.1f} seconds")
            print(f"   Architecture: {health_data.get('architecture', 'unknown')}")
    
    async def cleanup(self):
        """Clean up resources"""
        print("\n🧹 Cleaning up...")
        
        if self.client_session:
            await self.client_session.close()
        
        if self.server_process:
            self.server_process.terminate()
            await self.server_process.wait()
        
        print("✅ Cleanup completed")
    
    async def run_complete_example(self):
        """Run the complete example"""
        print("🎵 Music21 MCP Server - Complete Example")
        print("=" * 50)
        print("This example demonstrates the full capabilities of the server")
        print("including music import, analysis, and pattern recognition.")
        print()
        
        try:
            # Start server
            await self.start_server()
            
            # List available tools
            await self.list_available_tools()
            
            # Demonstrate music analysis
            await self.demonstrate_music_analysis()
            
            # Check server health
            await self.demonstrate_health_check()
            
            print("\n🎉 Example completed successfully!")
            print("\nYou can now:")
            print("- Use this server with Claude Desktop")
            print("- Integrate with VS Code extensions")
            print("- Build your own music analysis applications")
            print("- Deploy to production environments")
            
        except Exception as e:
            print(f"\n❌ Example failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        finally:
            await self.cleanup()
        
        return True

def create_claude_config():
    """Create Claude Desktop configuration"""
    config = {
        "mcpServers": {
            "music21": {
                "command": "python",
                "args": ["-m", "music21_mcp.server"],
                "env": {}
            }
        }
    }
    
    print("\n📝 Claude Desktop Configuration")
    print("=" * 40)
    print("Add this to your Claude Desktop MCP configuration:")
    print()
    print(json.dumps(config, indent=2))
    print()
    print("Configuration file locations:")
    print("- macOS: ~/Library/Application Support/Claude/claude_desktop_config.json")
    print("- Windows: %APPDATA%/Claude/claude_desktop_config.json")
    print("- Linux: ~/.config/claude_desktop_config.json")

async def main():
    """Main entry point"""
    example = MusicAnalysisExample()
    
    print("Choose an option:")
    print("1. Run complete example (recommended)")
    print("2. Show Claude Desktop configuration")
    print("3. Both")
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    if choice in ["1", "3"]:
        success = await example.run_complete_example()
        if not success:
            sys.exit(1)
    
    if choice in ["2", "3"]:
        create_claude_config()

if __name__ == "__main__":
    asyncio.run(main())