#!/bin/bash
# Core validation script for Music21 MCP Server

echo "🔍 Music21 MCP Server Core Validation"
echo "====================================="

# Check Python and dependencies
echo -e "\n1️⃣ Checking Environment..."
python3 --version
pip show music21 mcp fastmcp | grep "Name\|Version" || echo "❌ Missing dependencies"

# Test server startup
echo -e "\n2️⃣ Testing Server Startup..."
timeout 5 python3 -m music21_mcp.server --stdio < /dev/null && echo "✅ STDIO mode works" || echo "❌ STDIO mode failed"

# Run validation tests
echo -e "\n3️⃣ Running Core Validation Tests..."
cd tests
python3 validate_mcp_server.py

# Test with real music files
echo -e "\n4️⃣ Testing Real Music Files..."
python3 test_real_music_files.py

# Test MCP client integration
echo -e "\n5️⃣ Testing MCP Client Integration..."
python3 test_mcp_client.py

echo -e "\n====================================="
echo "Validation complete. Check results above."