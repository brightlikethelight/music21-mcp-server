#!/bin/bash
# Setup script for production stress test

echo "=========================================="
echo "Music21 MCP Server Production Test Setup"
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

# Check if running in virtual environment
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo -e "${YELLOW}⚠ Warning: Not in a virtual environment${NC}"
    echo "  Recommended: python3 -m venv venv && source venv/bin/activate"
fi

# Install/upgrade pip
echo -n "Upgrading pip... "
python3 -m pip install --upgrade pip > /dev/null 2>&1
echo -e "${GREEN}✓ Done${NC}"

# Check numpy first (often problematic)
echo -n "Checking numpy... "
if python3 -c "import numpy" 2>/dev/null; then
    echo -e "${GREEN}✓ numpy available${NC}"
else
    echo -e "${YELLOW}Installing numpy...${NC}"
    python3 -m pip install numpy
fi

# Install package in development mode
echo "Installing music21-mcp-server..."
pip install -e . || {
    echo -e "${RED}✗ Installation failed${NC}"
    echo "Try: pip install -r requirements.txt first"
    exit 1
}

# Quick import test
echo -n "Testing imports... "
python3 -c "
from music21_mcp.resilience import CircuitBreaker
from music21_mcp.server_resilient import ResilientMusicMCPServer
print('Imports successful')
" > /dev/null 2>&1

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ All imports working${NC}"
else
    echo -e "${RED}✗ Import errors detected${NC}"
    echo "Check that numpy is properly installed"
    exit 1
fi

# Check ports
echo -n "Checking port 8000... "
if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null ; then
    echo -e "${RED}✗ Port 8000 in use${NC}"
    echo "  Kill the process or use a different port"
else
    echo -e "${GREEN}✓ Port 8000 available${NC}"
fi

echo -n "Checking port 8001... "
if lsof -Pi :8001 -sTCP:LISTEN -t >/dev/null ; then
    echo -e "${RED}✗ Port 8001 in use${NC}"
else
    echo -e "${GREEN}✓ Port 8001 available${NC}"
fi

# Check system resources
echo ""
echo "System Resources:"
echo -n "  Memory: "
if [[ "$OSTYPE" == "darwin"* ]]; then
    total_mem=$(sysctl -n hw.memsize | awk '{print $1/1024/1024/1024}')
    free_mem=$(vm_stat | grep "Pages free" | awk '{print $3}' | sed 's/\.$//')
    free_mem=$(echo "scale=2; $free_mem * 4096 / 1024 / 1024 / 1024" | bc)
else
    total_mem=$(free -g | awk '/^Mem:/{print $2}')
    free_mem=$(free -g | awk '/^Mem:/{print $4}')
fi
echo "${free_mem}GB free of ${total_mem}GB total"

if (( $(echo "$free_mem < 4" | bc -l) )); then
    echo -e "  ${YELLOW}⚠ Warning: Less than 4GB free memory${NC}"
fi

echo -n "  Disk: "
disk_free=$(df -h . | awk 'NR==2 {print $4}')
echo "$disk_free free"

# Create test config
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
echo -e "${GREEN}✓ Created quick_test_config.json (30 minute test)${NC}"

cat > moderate_test_config.json << EOF
{
  "duration_hours": 4,
  "concurrent_users": 50,
  "kill_interval_minutes": 30,
  "recovery_timeout_seconds": 45
}
EOF
echo -e "${GREEN}✓ Created moderate_test_config.json (4 hour test)${NC}"

cat > full_test_config.json << EOF
{
  "duration_hours": 24,
  "concurrent_users": 100,
  "kill_interval_minutes": 60,
  "recovery_timeout_seconds": 60
}
EOF
echo -e "${GREEN}✓ Created full_test_config.json (24 hour test)${NC}"

# Make scripts executable
chmod +x run_production_test.py
chmod +x tests/production_stress_test.py
chmod +x tests/stress_test_monitor.py

echo ""
echo "=========================================="
echo -e "${GREEN}Setup Complete!${NC}"
echo "=========================================="
echo ""
echo "Run tests with:"
echo "  Quick test (30 min):  python run_production_test.py --config quick_test_config.json"
echo "  Moderate (4 hours):   python run_production_test.py --config moderate_test_config.json"
echo "  Full test (24 hours): python run_production_test.py --config full_test_config.json"
echo ""
echo "Or custom:"
echo "  python run_production_test.py --hours 2 --users 25"
echo ""
echo "Monitor in separate terminal:"
echo "  python tests/stress_test_monitor.py"
echo ""