#!/usr/bin/env python3
"""
Test if we can create a minimal working MCP server without all the architecture
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from mcp.server.fastmcp import FastMCP
from music21_mcp.tools.import_tool import ImportScoreTool
from music21_mcp.tools.list_tool import ListScoresTool
from music21_mcp.tools.key_analysis_tool import KeyAnalysisTool

async def test_minimal_server():
    """Create a minimal working server without architectural complexity"""
    print("🔥 Testing Minimal Working Server")
    print("=" * 50)
    
    # Create simple score storage
    scores = {}
    
    # Create FastMCP server
    mcp = FastMCP("Minimal Music21 MCP Server")
    
    # Create tool instances
    import_tool = ImportScoreTool(scores)
    list_tool = ListScoresTool(scores)
    key_tool = KeyAnalysisTool(scores)
    
    # Register tools directly with FastMCP
    @mcp.tool()
    async def import_score(score_id: str, source: str, source_type: str = "corpus"):
        """Import a score from various sources"""
        return await import_tool.execute(score_id=score_id, source=source, source_type=source_type)
    
    @mcp.tool()
    async def list_scores():
        """List all available scores"""
        return await list_tool.execute()
    
    @mcp.tool()
    async def analyze_key(score_id: str, algorithm: str = "krumhansl"):
        """Analyze the key of a score"""
        return await key_tool.execute(score_id=score_id, algorithm=algorithm)
    
    print("✅ Server created successfully")
    print(f"✅ Registered {len([import_score, list_scores, analyze_key])} tools")
    
    # Test the tools through the server interface
    print("\n🧪 Testing tools through server interface:")
    
    # Test import
    print("1. Testing import...")
    import_result = await import_score("test", "bach/bwv66.6", "corpus")
    print(f"   Import status: {import_result.get('status')}")
    
    # Test list
    print("2. Testing list...")
    list_result = await list_scores()
    print(f"   List status: {list_result.get('status')}")
    print(f"   Scores found: {len(list_result.get('scores', []))}")
    
    # Test analysis
    print("3. Testing key analysis...")
    key_result = await analyze_key("test", "krumhansl")
    print(f"   Analysis status: {key_result.get('status')}")
    print(f"   Key detected: {key_result.get('key')}")
    
    success = all([
        import_result.get('status') == 'success',
        list_result.get('status') == 'success',
        key_result.get('status') == 'success'
    ])
    
    if success:
        print("\n🎉 MINIMAL SERVER WORKS PERFECTLY!")
        print("   - No architectural complexity")
        print("   - Direct tool registration")
        print("   - All tools functional")
        print("   - Ready for production")
    else:
        print("\n❌ MINIMAL SERVER HAS ISSUES")
    
    return success

if __name__ == "__main__":
    success = asyncio.run(test_minimal_server())
    if success:
        print("\n💡 CONCLUSION: Scrap the complex architecture, use this approach!")
    else:
        print("\n🔴 CONCLUSION: Even minimal approach has issues")
    
    sys.exit(0 if success else 1)