#!/usr/bin/env python3
"""
BRUTAL REALITY CHECK: Test if ANY of the MCP tools work without the architectural overhead
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from music21_mcp.tools.import_tool import ImportScoreTool
from music21_mcp.tools.list_tool import ListScoresTool
from music21_mcp.tools.key_analysis_tool import KeyAnalysisTool

async def test_basic_functionality():
    """Test if the basic tools work without all the architectural complexity"""
    print("🔥 BRUTAL REALITY CHECK: Testing basic tool functionality")
    print("=" * 60)
    
    # Create a simple score manager (just a dict)
    scores = {}
    
    # Test 1: ImportScoreTool
    print("\n1. Testing ImportScoreTool (basic functionality)")
    try:
        import_tool = ImportScoreTool(scores)
        result = await import_tool.execute(
            score_id="test_bach",
            source="bach/bwv66.6",
            source_type="corpus"
        )
        print(f"   Status: {result.get('status')}")
        print(f"   Title: {result.get('title', 'No title')}")
        print(f"   Scores stored: {len(scores)}")
        
        if result.get('status') == 'success' and len(scores) > 0:
            print("   ✅ ImportScoreTool WORKS")
        else:
            print("   ❌ ImportScoreTool BROKEN")
            return False
    except Exception as e:
        print(f"   💥 ImportScoreTool CRASHED: {e}")
        return False
    
    # Test 2: ListScoresTool
    print("\n2. Testing ListScoresTool")
    try:
        list_tool = ListScoresTool(scores)
        result = await list_tool.execute()
        print(f"   Status: {result.get('status')}")
        print(f"   Scores found: {len(result.get('scores', []))}")
        
        if result.get('status') == 'success' and len(result.get('scores', [])) > 0:
            print("   ✅ ListScoresTool WORKS")
        else:
            print("   ❌ ListScoresTool BROKEN")
            return False
    except Exception as e:
        print(f"   💥 ListScoresTool CRASHED: {e}")
        return False
    
    # Test 3: KeyAnalysisTool
    print("\n3. Testing KeyAnalysisTool")
    try:
        key_tool = KeyAnalysisTool(scores)
        result = await key_tool.execute(
            score_id="test_bach",
            algorithm="krumhansl"
        )
        print(f"   Status: {result.get('status')}")
        print(f"   Key: {result.get('key', 'No key')}")
        print(f"   Confidence: {result.get('confidence', 'No confidence')}")
        
        if result.get('status') == 'success' and result.get('key'):
            print("   ✅ KeyAnalysisTool WORKS")
        else:
            print("   ❌ KeyAnalysisTool BROKEN")
            return False
    except Exception as e:
        print(f"   💥 KeyAnalysisTool CRASHED: {e}")
        return False
    
    print("\n🎉 ALL BASIC TOOLS WORK WITHOUT ARCHITECTURAL OVERHEAD!")
    return True

async def test_server_initialization():
    """Test what happens when we try to initialize the server"""
    print("\n" + "=" * 60)
    print("🔥 TESTING SERVER INITIALIZATION")
    print("=" * 60)
    
    try:
        from music21_mcp.server import initialize_architecture
        print("Attempting to initialize modern architecture...")
        await initialize_architecture()
        print("✅ Architecture initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Architecture initialization FAILED: {e}")
        print(f"   Error type: {type(e).__name__}")
        return False

async def test_mcp_fastmcp():
    """Test if FastMCP is even available"""
    print("\n" + "=" * 60)
    print("🔥 TESTING MCP FASTMCP AVAILABILITY")
    print("=" * 60)
    
    try:
        from mcp.server.fastmcp import FastMCP
        print("✅ FastMCP is available")
        
        # Try to create a FastMCP instance
        mcp = FastMCP("Test Server")
        print("✅ FastMCP instance created successfully")
        return True
    except ImportError as e:
        print(f"❌ FastMCP NOT AVAILABLE: {e}")
        return False
    except Exception as e:
        print(f"❌ FastMCP BROKEN: {e}")
        return False

async def main():
    print("🔥 BRUTAL REALITY CHECK FOR music21-mcp-server")
    print("Testing what actually works vs theoretical architecture")
    
    # Test basic functionality
    basic_works = await test_basic_functionality()
    
    # Test MCP availability
    mcp_works = await test_mcp_fastmcp()
    
    # Test server initialization
    server_works = await test_server_initialization()
    
    print("\n" + "=" * 60)
    print("🎯 BRUTAL REALITY CHECK SUMMARY")
    print("=" * 60)
    print(f"Basic tool functionality: {'✅ WORKS' if basic_works else '❌ BROKEN'}")
    print(f"MCP FastMCP availability: {'✅ WORKS' if mcp_works else '❌ BROKEN'}")
    print(f"Modern server architecture: {'✅ WORKS' if server_works else '❌ BROKEN'}")
    
    if basic_works and not server_works:
        print("\n💡 RECOMMENDATION: STRIP OUT THE ARCHITECTURAL COMPLEXITY")
        print("   - The basic tools work fine")
        print("   - The 'modern architecture' is broken")
        print("   - Users need working tools, not architectural patterns")
        print("   - Revert to simple, working implementation")
    elif basic_works and server_works:
        print("\n🎉 SYSTEM IS WORKING")
        print("   - All components functional")
    else:
        print("\n🔴 SYSTEM IS FUNDAMENTALLY BROKEN")
        print("   - Even basic functionality is failing")
        print("   - Major fixes required")
    
    return {
        'basic_works': basic_works,
        'mcp_works': mcp_works,
        'server_works': server_works
    }

if __name__ == "__main__":
    results = asyncio.run(main())
    sys.exit(0 if all(results.values()) else 1)