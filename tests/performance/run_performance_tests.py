#!/usr/bin/env python3
"""
Performance Test Runner for Music21 MCP Server

This script provides an easy way to run different performance test scenarios
and generate comprehensive reports.
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def run_test(test_name: str, args: list = None) -> dict:
    """Run a specific performance test and capture results"""
    cmd = [sys.executable, "-m", "pytest", "-v", "--tb=short"]
    
    if test_name:
        cmd.extend(["-k", test_name])
    
    if args:
        cmd.extend(args)
    
    # Add the test file
    cmd.append("test_performance.py")
    
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent
        )
        
        return {
            "command": " ".join(cmd),
            "return_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "success": result.returncode == 0
        }
    except Exception as e:
        return {
            "command": " ".join(cmd),
            "error": str(e),
            "success": False
        }


def main():
    parser = argparse.ArgumentParser(
        description="Run Music21 MCP Server performance tests"
    )
    
    parser.add_argument(
        "--scenario",
        choices=[
            "all",
            "concurrent",
            "large_scores",
            "memory",
            "benchmarks",
            "rate_limit",
            "cleanup",
            "stability"
        ],
        default="all",
        help="Performance test scenario to run"
    )
    
    parser.add_argument(
        "--clients",
        type=int,
        default=50,
        help="Number of concurrent clients for concurrent test"
    )
    
    parser.add_argument(
        "--duration",
        type=int,
        default=60,
        help="Duration in seconds for sustained tests"
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("performance_results"),
        help="Directory to save test results"
    )
    
    parser.add_argument(
        "--markers",
        help="Additional pytest markers (e.g., 'not slow')"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(exist_ok=True)
    
    # Timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Define test scenarios
    scenarios = {
        "concurrent": {
            "name": "Concurrent Client Handling",
            "test": "test_concurrent_clients",
            "description": "Tests server handling of 10, 50, and 100 concurrent clients"
        },
        "large_scores": {
            "name": "Large Score Processing",
            "test": "test_large_score_processing",
            "description": "Tests processing of orchestral scores with up to 150 instruments"
        },
        "memory": {
            "name": "Memory Leak Detection",
            "test": "test_memory_leak_detection",
            "description": "Tests for memory leaks under sustained load (5 minutes)",
            "args": ["-m", "slow"]
        },
        "benchmarks": {
            "name": "Tool Execution Benchmarks",
            "test": "TestToolExecutionBenchmarks",
            "description": "Benchmarks individual tool performance"
        },
        "rate_limit": {
            "name": "Rate Limiting",
            "test": "test_rate_limiter_enforcement",
            "description": "Tests rate limiting effectiveness"
        },
        "cleanup": {
            "name": "Resource Cleanup",
            "test": "test_resource_cleanup_after_errors",
            "description": "Tests resource cleanup after errors"
        },
        "stability": {
            "name": "24-Hour Stability Simulation",
            "test": "test_stability_simulation",
            "description": "Simulates 24-hour operation in accelerated time (10 minutes)",
            "args": ["-m", "slow"]
        }
    }
    
    # Determine which tests to run
    if args.scenario == "all":
        tests_to_run = list(scenarios.keys())
    else:
        tests_to_run = [args.scenario]
    
    # Results summary
    results = {
        "timestamp": timestamp,
        "scenario": args.scenario,
        "tests": {}
    }
    
    print("=" * 80)
    print(f"Music21 MCP Server Performance Test Suite")
    print(f"Timestamp: {timestamp}")
    print(f"Scenario: {args.scenario}")
    print("=" * 80)
    
    # Run each test
    for test_key in tests_to_run:
        scenario = scenarios[test_key]
        
        print(f"\n{'-' * 80}")
        print(f"Running: {scenario['name']}")
        print(f"Description: {scenario['description']}")
        print(f"{'-' * 80}")
        
        # Prepare test arguments
        test_args = scenario.get("args", [])
        
        # Add custom markers if specified
        if args.markers:
            test_args.extend(["-m", args.markers])
        
        # Run the test
        result = run_test(scenario["test"], test_args)
        
        # Save result
        results["tests"][test_key] = {
            "name": scenario["name"],
            "success": result["success"],
            "command": result["command"]
        }
        
        # Save detailed output
        output_file = args.output_dir / f"{timestamp}_{test_key}_output.txt"
        with open(output_file, "w") as f:
            f.write(f"Test: {scenario['name']}\n")
            f.write(f"Command: {result['command']}\n")
            f.write(f"Return Code: {result.get('return_code', 'N/A')}\n")
            f.write("\n" + "=" * 80 + "\n")
            f.write("STDOUT:\n")
            f.write(result.get("stdout", ""))
            f.write("\n" + "=" * 80 + "\n")
            f.write("STDERR:\n")
            f.write(result.get("stderr", ""))
        
        print(f"Result: {'PASSED' if result['success'] else 'FAILED'}")
        print(f"Output saved to: {output_file}")
    
    # Save summary
    summary_file = args.output_dir / f"{timestamp}_summary.json"
    with open(summary_file, "w") as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 80)
    print("PERFORMANCE TEST SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for t in results["tests"].values() if t["success"])
    total = len(results["tests"])
    
    print(f"Total Tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    print(f"Success Rate: {(passed/total*100):.1f}%")
    
    print("\nDetailed Results:")
    for test_key, test_result in results["tests"].items():
        status = "✓" if test_result["success"] else "✗"
        print(f"  {status} {test_result['name']}")
    
    print(f"\nResults saved to: {args.output_dir}")
    print(f"Summary: {summary_file}")
    
    # Exit with appropriate code
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()