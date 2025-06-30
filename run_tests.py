#!/usr/bin/env python3
"""
Test runner for Music21 MCP Server
Runs all tests and generates coverage report
"""
import os
import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a command and report results"""
    print(f"\n{'='*60}")
    print(f"🔧 {description}")
    print(f"{'='*60}")
    
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"❌ Failed: {description}")
        return False
    print(f"✅ Success: {description}")
    return True

def main():
    """Run all tests with coverage"""
    print("🧪 Music21 MCP Server - Test Suite")
    print("="*60)
    
    # Change to project root
    project_root = Path(__file__).parent
    os.chdir(project_root)
    
    # Check if pytest is installed
    if not run_command("python -m pytest --version", "Checking pytest installation"):
        print("\n⚠️  Please install pytest: pip install pytest pytest-asyncio pytest-cov")
        return 1
    
    # Run tests with coverage
    test_commands = [
        ("python -m pytest tests/test_server_pytest.py -v", "Running main server tests"),
        ("python -m pytest tests/test_mcp_client.py -v", "Running MCP client tests"),
        ("python -m pytest tests/unit/ -v", "Running unit tests"),
        ("python -m pytest tests/integration/ -v", "Running integration tests"),
        ("python -m pytest tests/ -v --cov=src/music21_mcp --cov-report=term-missing --cov-report=html", "Running all tests with coverage")
    ]
    
    all_passed = True
    for cmd, desc in test_commands:
        if not run_command(cmd, desc):
            all_passed = False
            # Continue running other tests even if one fails
    
    # Summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    
    if all_passed:
        print("✅ All tests passed!")
        print("\n📈 Coverage report generated in htmlcov/index.html")
        print("   Open with: open htmlcov/index.html")
        return 0
    else:
        print("❌ Some tests failed. Please check the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())