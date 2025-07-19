#!/usr/bin/env python3
"""
Basic Working Example - Music21 MCP Server

This example demonstrates the minimal server in action.
Unlike the broken enterprise examples, this one actually works.

Based on industry best practices from successful MCP servers.
"""

import asyncio
import sys
import os

# Add src to path so we can import our module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from music21_mcp.tools import ImportScoreTool, ListScoresTool, KeyAnalysisTool


async def main():
    """Basic example that works without MCP client"""
    print("🎵 Music21 MCP Server - Basic Example")
    print("=" * 50)
    
    # Simple score storage
    scores = {}
    
    # 1. Import a score
    print("\n1. Importing a Bach chorale...")
    import_tool = ImportScoreTool(scores)
    result = await import_tool.execute(
        score_id="bach_chorale",
        source="bach/bwv66.6",
        source_type="corpus"
    )
    print(f"✅ Import: {result.get('message', 'Success')}")
    
    # 2. List scores
    print("\n2. Listing available scores...")
    list_tool = ListScoresTool(scores)
    result = await list_tool.execute()
    print(f"✅ Found {result.get('total_count', 0)} scores")
    for score in result.get('scores', []):
        print(f"   - {score.get('id')}: {score.get('notes', 0)} notes, {score.get('measures', 0)} measures")
    
    # 3. Analyze key
    print("\n3. Analyzing key signature...")
    key_tool = KeyAnalysisTool(scores)
    result = await key_tool.execute(score_id="bach_chorale")
    print(f"✅ Key analysis: {result.get('message', 'Success')}")
    if result.get('status') == 'success':
        print(f"   Key: {result.get('key', 'Unknown')}")
        print(f"   Confidence: {result.get('confidence', 0):.2f}")
    
    print("\n🎉 Basic example completed successfully!")
    print("\nNext steps:")
    print("1. Run the MCP server: python -m music21_mcp.server_minimal")
    print("2. Connect from Claude Desktop using MCP configuration")
    print("3. Ask Claude to analyze your music!")


if __name__ == "__main__":
    asyncio.run(main())