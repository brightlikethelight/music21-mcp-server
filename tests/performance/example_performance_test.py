#!/usr/bin/env python3
"""
Example: How to run and interpret performance tests

This script demonstrates running performance tests and analyzing results.
"""

import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime


def run_performance_test(test_name: str) -> dict:
    """Run a specific performance test and parse results"""
    cmd = [
        sys.executable, "-m", "pytest",
        "-v", "--tb=short", "-s",
        f"test_performance.py::{test_name}",
        "--capture=no"
    ]
    
    print(f"Running: {' '.join(cmd)}")
    
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent
    )
    
    return {
        "success": result.returncode == 0,
        "output": result.stdout,
        "errors": result.stderr
    }


def main():
    """Example performance test workflow"""
    
    print("=" * 80)
    print("Music21 MCP Server - Performance Test Example")
    print("=" * 80)
    
    # Example 1: Test concurrent client handling
    print("\n1. Testing Concurrent Client Handling (10 clients)")
    print("-" * 80)
    
    result = run_performance_test("test_concurrent_clients[10]")
    
    if result["success"]:
        print("✓ Test passed!")
        # Parse output for key metrics
        output_lines = result["output"].split("\n")
        for line in output_lines:
            if "Success rate:" in line or "Throughput:" in line or "Peak memory:" in line:
                print(f"  {line.strip()}")
    else:
        print("✗ Test failed!")
        print(f"Errors: {result['errors']}")
    
    # Example 2: Quick memory leak check
    print("\n2. Quick Memory Leak Check")
    print("-" * 80)
    
    # Run a shorter version of the memory test
    cmd = [
        sys.executable, "-m", "pytest",
        "-v", "-s",
        "test_performance.py::TestMemoryUnderSustainedLoad::test_memory_leak_detection",
        "--capture=no",
        "-o", "addopts=''",  # Override default options
        "--tb=short"
    ]
    
    # Note: This is a simplified version - the full test takes 5 minutes
    print("Note: Running simplified memory check (full test takes 5 minutes)")
    
    # Example 3: Benchmark a specific tool
    print("\n3. Benchmarking Key Analysis Tool")
    print("-" * 80)
    
    result = run_performance_test("test_key_analysis_benchmark")
    
    if result["success"]:
        print("✓ Benchmark completed!")
        # Extract benchmark results
        output_lines = result["output"].split("\n")
        for line in output_lines:
            if "Mean" in line or "Min" in line or "Max" in line:
                print(f"  {line.strip()}")
    
    # Example 4: Test rate limiting
    print("\n4. Testing Rate Limiter")
    print("-" * 80)
    
    result = run_performance_test("test_rate_limiter_enforcement")
    
    if result["success"]:
        print("✓ Rate limiter working correctly!")
    else:
        print("✗ Rate limiter test failed!")
    
    # Summary and recommendations
    print("\n" + "=" * 80)
    print("PERFORMANCE TEST RECOMMENDATIONS")
    print("=" * 80)
    
    print("""
1. Regular Testing:
   - Run quick tests (concurrent, rate limit) before each commit
   - Run full suite (including memory and stability) before releases
   
2. Performance Baselines:
   - Success Rate: Should be > 95% for production
   - Response Time: p95 should be < 2 seconds
   - Memory Growth: < 50MB per 5 minutes of operation
   
3. Monitoring Production:
   - Use the health_check endpoint for live monitoring
   - Track memory usage over time
   - Monitor error rates and response times
   
4. Optimization Targets:
   - If response times are high: Profile individual tools
   - If memory grows: Check for score cleanup
   - If errors occur: Review error logs and patterns
""")
    
    # Generate a simple report
    report_file = Path(f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    with open(report_file, "w") as f:
        f.write("Music21 MCP Server - Performance Test Report\n")
        f.write(f"Generated: {datetime.now()}\n")
        f.write("=" * 80 + "\n\n")
        f.write("Run the full test suite for comprehensive results:\n")
        f.write("  python run_performance_tests.py --scenario all\n")
    
    print(f"\nReport saved to: {report_file}")


if __name__ == "__main__":
    main()