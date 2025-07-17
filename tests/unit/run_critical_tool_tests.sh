#!/bin/bash
# Run unit tests for critical music21-mcp-server tools

echo "🎵 Running Music21 MCP Server Critical Tool Tests 🎵"
echo "=================================================="
echo ""

# Set up colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Change to project root
cd "$(dirname "$0")/../.." || exit

# Check if pytest is installed
if ! command -v pytest &> /dev/null; then
    echo -e "${RED}❌ pytest is not installed. Please run: pip install pytest pytest-cov pytest-asyncio${NC}"
    exit 1
fi

# Run tests for each critical tool
echo -e "${YELLOW}1. Testing ImportScoreTool...${NC}"
pytest tests/unit/test_tools/test_import_tool.py -v --tb=short

echo -e "\n${YELLOW}2. Testing KeyAnalysisTool...${NC}"
pytest tests/unit/test_tools/test_key_analysis_tool.py -v --tb=short

echo -e "\n${YELLOW}3. Testing PatternRecognitionTool...${NC}"
pytest tests/unit/test_tools/test_pattern_recognition_tool.py -v --tb=short

echo -e "\n${YELLOW}4. Testing HarmonyAnalysisTool...${NC}"
pytest tests/unit/test_tools/test_harmony_analysis_tool.py -v --tb=short

# Run all tests with coverage
echo -e "\n${YELLOW}Running all tests with coverage report...${NC}"
pytest tests/unit/test_tools/ -v \
    --cov=src/music21_mcp/tools \
    --cov-report=term-missing \
    --cov-report=html \
    --cov-config=.coveragerc

# Check if tests passed
if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}✅ All tests passed!${NC}"
    echo -e "${GREEN}📊 Coverage report generated at: htmlcov/index.html${NC}"
else
    echo -e "\n${RED}❌ Some tests failed. Please check the output above.${NC}"
    exit 1
fi

# Show coverage summary
echo -e "\n${YELLOW}Coverage Summary:${NC}"
coverage report --include="src/music21_mcp/tools/*" | grep -E "(TOTAL|import_tool|key_analysis|pattern_recognition|harmony_analysis)"

echo -e "\n${GREEN}🎉 Test run complete!${NC}"