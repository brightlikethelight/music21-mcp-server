#!/usr/bin/env python3
"""
Direct comparison: What works vs what's broken
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_current_server_components():
    """Test what works and what doesn't in current server"""
    print("🔥 TESTING CURRENT SERVER COMPONENTS")
    print("=" * 50)
    
    # Test 1: Basic imports
    print("1. Testing basic imports...")
    try:
        from music21_mcp.server import ScoreManager, mcp, health_check
        print("   ✅ ScoreManager, mcp, health_check import OK")
    except Exception as e:
        print(f"   ❌ Import failed: {e}")
        return False
    
    # Test 2: ScoreManager functionality
    print("2. Testing ScoreManager...")
    try:
        manager = ScoreManager(max_scores=10)
        print("   ✅ ScoreManager created successfully")
        print(f"   ✅ Initial score count: {len(manager.scores)}")
    except Exception as e:
        print(f"   ❌ ScoreManager failed: {e}")
        return False
    
    # Test 3: FastMCP server instance
    print("3. Testing FastMCP server...")
    try:
        print(f"   ✅ MCP server name: {mcp.name}")
    except Exception as e:
        print(f"   ❌ MCP server failed: {e}")
        return False
    
    # Test 4: Health check without architecture
    print("4. Testing health check...")
    try:
        result = asyncio.run(health_check())
        print(f"   ✅ Health check status: {result.get('status')}")
        print(f"   ✅ Services initialized: {result.get('services', {}).get('container_initialized', False)}")
    except Exception as e:
        print(f"   ❌ Health check failed: {e}")
        return False
    
    return True

def test_working_alternatives():
    """Test what works when we bypass the architecture"""
    print("\n🔥 TESTING WORKING ALTERNATIVES")
    print("=" * 50)
    
    # Test 1: Direct tool creation
    print("1. Testing direct tool creation...")
    try:
        from music21_mcp.tools.import_tool import ImportScoreTool
        scores = {}
        tool = ImportScoreTool(scores)
        print("   ✅ ImportScoreTool created directly")
        print(f"   ✅ Tool name: {tool.name}")
    except Exception as e:
        print(f"   ❌ Direct tool creation failed: {e}")
        return False
    
    # Test 2: Tool execution
    print("2. Testing tool execution...")
    try:
        result = asyncio.run(tool.execute(score_id="test", source="bach/bwv66.6", source_type="corpus"))
        print(f"   ✅ Tool execution: {result.get('status')}")
        print(f"   ✅ Scores created: {len(scores)}")
    except Exception as e:
        print(f"   ❌ Tool execution failed: {e}")
        return False
    
    # Test 3: Multiple tools
    print("3. Testing multiple tools...")
    try:
        from music21_mcp.tools.list_tool import ListScoresTool
        from music21_mcp.tools.key_analysis_tool import KeyAnalysisTool
        
        list_tool = ListScoresTool(scores)
        key_tool = KeyAnalysisTool(scores)
        
        list_result = asyncio.run(list_tool.execute())
        key_result = asyncio.run(key_tool.execute(score_id="test", algorithm="krumhansl"))
        
        print(f"   ✅ List tool: {list_result.get('status')}")
        print(f"   ✅ Key tool: {key_result.get('status')}")
        print(f"   ✅ Key detected: {key_result.get('key')}")
    except Exception as e:
        print(f"   ❌ Multiple tools failed: {e}")
        return False
    
    return True

def analyze_architecture_failure():
    """Try to pinpoint exactly what's broken in the architecture"""
    print("\n🔥 ANALYZING ARCHITECTURE FAILURE")
    print("=" * 50)
    
    # Test 1: Service creation
    print("1. Testing service creation...")
    try:
        from music21_mcp.core.services import ServiceConfig, ScoreManagementService
        config = ServiceConfig(name="test", description="test")
        service = ScoreManagementService(config)
        print("   ✅ Service creation works")
    except Exception as e:
        print(f"   ❌ Service creation failed: {e}")
        return False
    
    # Test 2: Container creation
    print("2. Testing container creation...")
    try:
        from music21_mcp.core.services import ServiceContainer
        container = ServiceContainer()
        print("   ✅ Container creation works")
    except Exception as e:
        print(f"   ❌ Container creation failed: {e}")
        return False
    
    # Test 3: Service registration
    print("3. Testing service registration...")
    try:
        container.register_service(service)
        print("   ✅ Service registration works")
    except Exception as e:
        print(f"   ❌ Service registration failed: {e}")
        return False
    
    # Test 4: Service initialization (this should fail)
    print("4. Testing service initialization...")
    try:
        asyncio.run(container.initialize_all())
        print("   ✅ Service initialization works")
        return True
    except Exception as e:
        print(f"   ❌ Service initialization failed: {e}")
        print(f"   📍 This is where the architecture breaks!")
        return False

if __name__ == "__main__":
    print("🔍 COMPREHENSIVE DIAGNOSIS: What Works vs What's Broken")
    print("=" * 60)
    
    current_works = test_current_server_components()
    alternative_works = test_working_alternatives()
    architecture_issue = analyze_architecture_failure()
    
    print("\n" + "=" * 60)
    print("🎯 DIAGNOSIS SUMMARY")
    print("=" * 60)
    print(f"Current server components work: {'✅ YES' if current_works else '❌ NO'}")
    print(f"Alternative approach works: {'✅ YES' if alternative_works else '❌ NO'}")
    print(f"Architecture initialization works: {'✅ YES' if architecture_issue else '❌ NO'}")
    
    if current_works and alternative_works and not architecture_issue:
        print("\n💡 DIAGNOSIS: The tools work, the architecture is broken")
        print("   - Basic components are functional")
        print("   - Alternative approach is viable")
        print("   - Complex architecture adds no value")
        print("   - RECOMMENDATION: Use alternative approach")
    elif not current_works:
        print("\n🚨 DIAGNOSIS: Fundamental issues with imports/basic components")
    else:
        print("\n❓ DIAGNOSIS: Mixed results - needs deeper investigation")