#!/usr/bin/env python3
"""
Test coverage summary for music21-mcp-server tools
Runs tests and generates coverage reports for the critical tools
"""

import subprocess
import sys
from pathlib import Path

def run_tool_tests():
    """Run unit tests for the critical tools with coverage"""
    
    # Define the critical tools to test
    critical_tools = [
        "test_import_tool.py",
        "test_key_analysis_tool.py", 
        "test_pattern_recognition_tool.py",
        "test_harmony_analysis_tool.py"
    ]
    
    test_dir = Path(__file__).parent / "test_tools"
    
    print("Running unit tests for critical music21-mcp-server tools...")
    print("=" * 60)
    
    for tool_test in critical_tools:
        test_path = test_dir / tool_test
        if test_path.exists():
            print(f"\nRunning tests for: {tool_test}")
            print("-" * 40)
            
            # Run pytest with coverage for this specific test file
            cmd = [
                sys.executable, "-m", "pytest", 
                str(test_path),
                "-v",
                "--cov=src/music21_mcp/tools",
                "--cov-report=term-missing",
                "--cov-report=html",
                "--tb=short"
            ]
            
            try:
                result = subprocess.run(cmd, capture_output=True, text=True)
                print(result.stdout)
                if result.stderr:
                    print("STDERR:", result.stderr)
                    
                if result.returncode != 0:
                    print(f"⚠️  Tests failed for {tool_test}")
                else:
                    print(f"✅ Tests passed for {tool_test}")
                    
            except Exception as e:
                print(f"❌ Error running tests for {tool_test}: {e}")
        else:
            print(f"⚠️  Test file not found: {test_path}")
    
    print("\n" + "=" * 60)
    print("Test summary complete!")
    print("\nTo run all tests at once:")
    print("  pytest tests/unit/test_tools/ -v --cov=src/music21_mcp/tools --cov-report=html")
    print("\nTo run a specific test file:")
    print("  pytest tests/unit/test_tools/test_key_analysis_tool.py -v")
    print("\nCoverage report available at: htmlcov/index.html")

if __name__ == "__main__":
    run_tool_tests()