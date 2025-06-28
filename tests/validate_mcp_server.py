#!/usr/bin/env python3
"""
CRITICAL: MCP Server Core Validation Tests
This validates the fundamental MCP server functionality before any music analysis features.
"""
import asyncio
import json
import sys
import os
from pathlib import Path
import httpx
import time
from typing import Dict, Any, List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

class MCPServerValidator:
    """Validates core MCP server functionality"""
    
    def __init__(self):
        self.base_url = "http://localhost:8000"
        self.client = httpx.AsyncClient(timeout=30.0)
        self.results = {
            "server_startup": False,
            "mcp_protocol": False,
            "tool_discovery": False,
            "basic_operations": False,
            "error_handling": False,
            "real_file_import": False
        }
    
    async def validate_all(self):
        """Run all validation tests"""
        print("🔍 MCP Server Core Validation Suite")
        print("=" * 50)
        
        # Test 1: Server startup
        await self.test_server_startup()
        
        # Test 2: MCP protocol compliance
        await self.test_mcp_protocol()
        
        # Test 3: Tool discovery
        await self.test_tool_discovery()
        
        # Test 4: Basic operations
        await self.test_basic_operations()
        
        # Test 5: Error handling
        await self.test_error_handling()
        
        # Test 6: Real file import/export
        await self.test_real_files()
        
        # Summary
        self.print_summary()
    
    async def test_server_startup(self):
        """Test if server starts and responds"""
        print("\n1️⃣ Testing Server Startup...")
        
        try:
            # Try to connect to the server
            response = await self.client.get(f"{self.base_url}/health")
            if response.status_code == 200:
                print("   ✅ Server is running and responding")
                self.results["server_startup"] = True
            else:
                print(f"   ❌ Server returned status {response.status_code}")
        except httpx.ConnectError:
            print("   ❌ Cannot connect to server at localhost:8000")
            print("   💡 Start the server with: python -m music21_mcp.server")
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    async def test_mcp_protocol(self):
        """Test MCP protocol compliance"""
        print("\n2️⃣ Testing MCP Protocol Compliance...")
        
        if not self.results["server_startup"]:
            print("   ⏭️ Skipping - server not running")
            return
        
        try:
            # Test MCP initialization
            init_request = {
                "jsonrpc": "2.0",
                "method": "initialize",
                "params": {
                    "protocolVersion": "1.0",
                    "capabilities": {}
                },
                "id": 1
            }
            
            response = await self.client.post(
                f"{self.base_url}/mcp",
                json=init_request,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                data = response.json()
                if "result" in data:
                    print("   ✅ MCP initialization successful")
                    self.results["mcp_protocol"] = True
                else:
                    print(f"   ❌ Invalid MCP response: {data}")
            else:
                print(f"   ❌ MCP request failed with status {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ MCP protocol error: {e}")
    
    async def test_tool_discovery(self):
        """Test if tools are discoverable"""
        print("\n3️⃣ Testing Tool Discovery...")
        
        if not self.results["mcp_protocol"]:
            print("   ⏭️ Skipping - MCP protocol not working")
            return
        
        try:
            # List available tools
            list_tools_request = {
                "jsonrpc": "2.0",
                "method": "tools/list",
                "id": 2
            }
            
            response = await self.client.post(
                f"{self.base_url}/mcp",
                json=list_tools_request,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                data = response.json()
                if "result" in data and "tools" in data["result"]:
                    tools = data["result"]["tools"]
                    print(f"   ✅ Found {len(tools)} tools")
                    
                    # Check for essential tools
                    tool_names = [tool["name"] for tool in tools]
                    essential_tools = [
                        "import_score",
                        "list_scores", 
                        "analyze_key",
                        "analyze_chords",
                        "export_score"
                    ]
                    
                    missing = [t for t in essential_tools if t not in tool_names]
                    if missing:
                        print(f"   ⚠️ Missing essential tools: {missing}")
                    else:
                        print("   ✅ All essential tools present")
                        self.results["tool_discovery"] = True
                        
                    # Print first 5 tools
                    print("   📋 Sample tools:")
                    for tool in tools[:5]:
                        print(f"      - {tool['name']}: {tool.get('description', 'No description')[:50]}...")
                else:
                    print(f"   ❌ Invalid tool list response: {data}")
            else:
                print(f"   ❌ Tool discovery failed with status {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ Tool discovery error: {e}")
    
    async def test_basic_operations(self):
        """Test basic score operations"""
        print("\n4️⃣ Testing Basic Operations...")
        
        if not self.results["tool_discovery"]:
            print("   ⏭️ Skipping - tools not available")
            return
        
        try:
            # Test 1: Import a simple score
            import_request = {
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {
                    "name": "import_score",
                    "arguments": {
                        "score_id": "test_score",
                        "content": "C4 D4 E4 F4 G4",
                        "format": "text"
                    }
                },
                "id": 3
            }
            
            response = await self.client.post(
                f"{self.base_url}/mcp",
                json=import_request,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                data = response.json()
                if "result" in data and data["result"].get("status") == "success":
                    print("   ✅ Score import successful")
                    
                    # Test 2: List scores
                    list_request = {
                        "jsonrpc": "2.0",
                        "method": "tools/call",
                        "params": {
                            "name": "list_scores"
                        },
                        "id": 4
                    }
                    
                    response = await self.client.post(
                        f"{self.base_url}/mcp",
                        json=list_request,
                        headers={"Content-Type": "application/json"}
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        if "result" in data and "scores" in data["result"]:
                            scores = data["result"]["scores"]
                            if any(s["score_id"] == "test_score" for s in scores):
                                print("   ✅ Score listing successful")
                                self.results["basic_operations"] = True
                            else:
                                print("   ❌ Imported score not found in list")
                        else:
                            print(f"   ❌ Invalid list response: {data}")
                    else:
                        print(f"   ❌ List scores failed with status {response.status_code}")
                else:
                    print(f"   ❌ Import failed: {data}")
            else:
                print(f"   ❌ Import request failed with status {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ Basic operations error: {e}")
    
    async def test_error_handling(self):
        """Test error handling"""
        print("\n5️⃣ Testing Error Handling...")
        
        if not self.results["basic_operations"]:
            print("   ⏭️ Skipping - basic operations not working")
            return
        
        try:
            # Test 1: Invalid score ID
            invalid_request = {
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {
                    "name": "analyze_key",
                    "arguments": {
                        "score_id": "nonexistent_score"
                    }
                },
                "id": 5
            }
            
            response = await self.client.post(
                f"{self.base_url}/mcp",
                json=invalid_request,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                data = response.json()
                if "result" in data and data["result"].get("status") == "error":
                    print("   ✅ Handles invalid score ID correctly")
                    
                    # Test 2: Invalid format
                    bad_format_request = {
                        "jsonrpc": "2.0",
                        "method": "tools/call",
                        "params": {
                            "name": "import_score",
                            "arguments": {
                                "score_id": "bad_score",
                                "content": "NOT VALID MUSIC",
                                "format": "musicxml"
                            }
                        },
                        "id": 6
                    }
                    
                    response = await self.client.post(
                        f"{self.base_url}/mcp",
                        json=bad_format_request,
                        headers={"Content-Type": "application/json"}
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        if "result" in data and data["result"].get("status") == "error":
                            print("   ✅ Handles invalid format correctly")
                            self.results["error_handling"] = True
                        else:
                            print("   ⚠️ Should have failed but didn't")
                    else:
                        print(f"   ❌ Bad format request failed with status {response.status_code}")
                else:
                    print("   ⚠️ Should have returned error status")
            else:
                print(f"   ❌ Error handling test failed with status {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ Error handling test error: {e}")
    
    async def test_real_files(self):
        """Test with real music files"""
        print("\n6️⃣ Testing Real File Import/Export...")
        
        if not self.results["basic_operations"]:
            print("   ⏭️ Skipping - basic operations not working")
            return
        
        # Create test MIDI file
        test_dir = Path(__file__).parent / "test_data"
        test_dir.mkdir(exist_ok=True)
        
        # Create a simple MIDI file using music21
        try:
            from music21 import stream, note, tempo, meter
            
            # Create a simple melody
            s = stream.Score()
            p = stream.Part()
            p.append(tempo.MetronomeMark(number=120))
            p.append(meter.TimeSignature('4/4'))
            
            # Add notes (C major scale)
            for pitch in ['C4', 'D4', 'E4', 'F4', 'G4', 'A4', 'B4', 'C5']:
                n = note.Note(pitch, quarterLength=1)
                p.append(n)
            
            s.append(p)
            
            # Save as MIDI
            midi_path = test_dir / "test_scale.mid"
            s.write('midi', fp=str(midi_path))
            print(f"   ✅ Created test MIDI file: {midi_path}")
            
            # Test importing the MIDI file
            with open(midi_path, 'rb') as f:
                midi_content = f.read()
            
            import_midi_request = {
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {
                    "name": "import_score",
                    "arguments": {
                        "score_id": "real_midi_test",
                        "file_path": str(midi_path),
                        "format": "midi"
                    }
                },
                "id": 7
            }
            
            response = await self.client.post(
                f"{self.base_url}/mcp",
                json=import_midi_request,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                data = response.json()
                if "result" in data and data["result"].get("status") == "success":
                    print("   ✅ MIDI import successful")
                    
                    # Test analysis on real file
                    analyze_request = {
                        "jsonrpc": "2.0",
                        "method": "tools/call",
                        "params": {
                            "name": "analyze_key",
                            "arguments": {
                                "score_id": "real_midi_test"
                            }
                        },
                        "id": 8
                    }
                    
                    response = await self.client.post(
                        f"{self.base_url}/mcp",
                        json=analyze_request,
                        headers={"Content-Type": "application/json"}
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        if "result" in data and "key" in data["result"]:
                            detected_key = data["result"]["key"]
                            print(f"   ✅ Analysis successful - Detected key: {detected_key}")
                            self.results["real_file_import"] = True
                        else:
                            print(f"   ❌ Analysis failed: {data}")
                    else:
                        print(f"   ❌ Analysis request failed with status {response.status_code}")
                else:
                    print(f"   ❌ MIDI import failed: {data}")
            else:
                print(f"   ❌ MIDI import request failed with status {response.status_code}")
                
        except ImportError:
            print("   ❌ music21 not installed - cannot create test files")
        except Exception as e:
            print(f"   ❌ Real file test error: {e}")
    
    def print_summary(self):
        """Print validation summary"""
        print("\n" + "=" * 50)
        print("📊 VALIDATION SUMMARY")
        print("=" * 50)
        
        total_tests = len(self.results)
        passed_tests = sum(1 for v in self.results.values() if v)
        
        for test, passed in self.results.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"{status} - {test.replace('_', ' ').title()}")
        
        print(f"\nTotal: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests == total_tests:
            print("\n🎉 All validations passed! Server is ready for use.")
        elif self.results["server_startup"]:
            print("\n⚠️ Some validations failed. Check the issues above.")
        else:
            print("\n🚨 Server is not running! Start it with:")
            print("   python -m music21_mcp.server")
    
    async def cleanup(self):
        """Clean up resources"""
        await self.client.aclose()


async def main():
    """Run validation suite"""
    validator = MCPServerValidator()
    try:
        await validator.validate_all()
    finally:
        await validator.cleanup()


if __name__ == "__main__":
    asyncio.run(main())