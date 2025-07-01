#!/bin/bash
# Alternative setup script that avoids numpy dependency for initial testing

echo "=========================================="
echo "Music21 MCP Server Production Test Setup"
echo "(No NumPy Version)"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python version
echo -n "Checking Python version... "
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
if (( $(echo "$python_version >= 3.10" | bc -l) )); then
    echo -e "${GREEN}✓ Python $python_version${NC}"
else
    echo -e "${RED}✗ Python $python_version (need 3.10+)${NC}"
    exit 1
fi

# Create necessary directories
echo -n "Creating directories... "
mkdir -p stress_test_logs test_results test_corpus
echo -e "${GREEN}✓ Done${NC}"

# Create a minimal test that doesn't require numpy
echo "Creating numpy-free test script..."
cat > test_basic_server.py << 'EOF'
#!/usr/bin/env python3
"""
Basic server test without numpy dependencies
"""
import asyncio
import httpx
import json
import subprocess
import sys
import time
import signal
import os

def start_basic_server():
    """Start the basic server (not the resilient one)"""
    print("Starting basic Music21 MCP server...")
    
    # Start basic server
    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'
    
    process = subprocess.Popen(
        [sys.executable, '-m', 'music21_mcp.server'],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    print(f"Server started with PID: {process.pid}")
    return process

async def test_server():
    """Test basic server functionality"""
    print("\nWaiting for server to start...")
    
    # Wait for server
    for i in range(30):
        try:
            response = await httpx.AsyncClient().get('http://localhost:8000/health')
            if response.status_code == 200:
                print("✓ Server is responding")
                break
        except:
            pass
        await asyncio.sleep(1)
    else:
        print("✗ Server failed to start")
        return False
    
    print("\nTesting basic operations...")
    
    # Test listing tools
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                'http://localhost:8000/mcp/list_tools',
                json={}
            )
            tools = response.json()
            print(f"✓ Found {len(tools.get('tools', []))} tools")
    except Exception as e:
        print(f"✗ Failed to list tools: {e}")
        return False
    
    # Test simple import
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                'http://localhost:8000/mcp/call_tool',
                json={
                    "name": "import_score",
                    "arguments": {
                        "score_id": "test_melody",
                        "source": "C4 D4 E4 F4 G4",
                        "source_type": "text"
                    }
                }
            )
            result = response.json()
            print("✓ Successfully imported simple score")
    except Exception as e:
        print(f"✗ Failed to import score: {e}")
        return False
    
    return True

async def main():
    """Run basic server test"""
    server_process = None
    
    try:
        # Start server
        server_process = start_basic_server()
        
        # Give it time to initialize
        await asyncio.sleep(3)
        
        # Run tests
        success = await test_server()
        
        if success:
            print("\n✓ Basic server functionality confirmed!")
            print("\nNote: This test bypasses numpy-dependent features.")
            print("To test full resilient server, numpy issues must be resolved.")
        else:
            print("\n✗ Basic server test failed")
            
    except KeyboardInterrupt:
        print("\nTest interrupted")
    finally:
        if server_process:
            print("\nShutting down server...")
            server_process.terminate()
            server_process.wait()

if __name__ == "__main__":
    asyncio.run(main())
EOF

chmod +x test_basic_server.py

# Create config files as before
echo ""
echo "Creating test configurations..."
cat > quick_test_config.json << EOF
{
  "duration_hours": 0.5,
  "concurrent_users": 10,
  "kill_interval_minutes": 15,
  "recovery_timeout_seconds": 30
}
EOF
echo -e "${GREEN}✓ Created quick_test_config.json${NC}"

# Create a simple stress test that works with basic server
cat > simple_stress_test.py << 'EOF'
#!/usr/bin/env python3
"""
Simple stress test for basic server (no numpy required)
"""
import asyncio
import httpx
import json
import random
import time
from datetime import datetime

async def stress_client(client_id: int, duration_seconds: int):
    """Simple client that makes requests"""
    end_time = time.time() + duration_seconds
    requests_made = 0
    errors = 0
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        while time.time() < end_time:
            try:
                # Random operation
                if random.random() < 0.3:
                    # List tools
                    response = await client.post(
                        'http://localhost:8000/mcp/list_tools',
                        json={}
                    )
                else:
                    # Import score
                    notes = ' '.join([
                        f"{random.choice('CDEFGAB')}{random.randint(3, 5)}"
                        for _ in range(random.randint(4, 16))
                    ])
                    
                    response = await client.post(
                        'http://localhost:8000/mcp/call_tool',
                        json={
                            "name": "import_score",
                            "arguments": {
                                "score_id": f"test_{client_id}_{requests_made}",
                                "source": notes,
                                "source_type": "text"
                            }
                        }
                    )
                
                if response.status_code == 200:
                    requests_made += 1
                else:
                    errors += 1
                    
            except Exception as e:
                errors += 1
            
            # Random delay
            await asyncio.sleep(random.uniform(0.1, 2.0))
    
    return requests_made, errors

async def run_simple_stress_test(duration_minutes: int = 5, num_clients: int = 10):
    """Run simple stress test"""
    print(f"Starting {duration_minutes}-minute stress test with {num_clients} clients")
    
    start_time = datetime.now()
    duration_seconds = duration_minutes * 60
    
    # Create client tasks
    tasks = [
        stress_client(i, duration_seconds)
        for i in range(num_clients)
    ]
    
    # Run all clients
    results = await asyncio.gather(*tasks)
    
    # Calculate totals
    total_requests = sum(r[0] for r in results)
    total_errors = sum(r[1] for r in results)
    
    elapsed = (datetime.now() - start_time).total_seconds()
    
    print("\n" + "="*50)
    print("STRESS TEST RESULTS")
    print("="*50)
    print(f"Duration: {elapsed:.1f} seconds")
    print(f"Total Requests: {total_requests}")
    print(f"Total Errors: {total_errors}")
    print(f"Success Rate: {(total_requests/(total_requests+total_errors)*100):.1f}%")
    print(f"Requests/Second: {total_requests/elapsed:.1f}")
    print("="*50)

if __name__ == "__main__":
    import sys
    
    duration = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    clients = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    
    asyncio.run(run_simple_stress_test(duration, clients))
EOF

chmod +x simple_stress_test.py

echo ""
echo "=========================================="
echo -e "${GREEN}Setup Complete!${NC}"
echo "=========================================="
echo ""
echo -e "${YELLOW}NumPy Import Issue Detected${NC}"
echo ""
echo "To fix numpy, run:"
echo "  ./fix_numpy_env.sh"
echo ""
echo "For now, you can test basic server functionality:"
echo "  python test_basic_server.py"
echo ""
echo "Run simple stress test (no numpy required):"
echo "  # Terminal 1: Start server"
echo "  python -m music21_mcp.server"
echo ""
echo "  # Terminal 2: Run stress test"
echo "  python simple_stress_test.py 5 20  # 5 minutes, 20 clients"
echo ""
echo "Once numpy is fixed, run full production test:"
echo "  python run_production_test.py --config quick_test_config.json"
echo ""