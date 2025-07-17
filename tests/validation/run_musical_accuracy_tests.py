#!/usr/bin/env python3
"""
Run Musical Accuracy Tests for music21-mcp-server

This script runs comprehensive musical accuracy tests and provides
a detailed report of the results.
"""

import sys
import asyncio
import pytest
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

def run_accuracy_tests(verbose: bool = False, specific_test: str = None) -> Tuple[int, Dict]:
    """
    Run the musical accuracy tests
    
    Args:
        verbose: Whether to show detailed output
        specific_test: Run only a specific test if provided
        
    Returns:
        Tuple of (exit code, results dictionary)
    """
    test_args = [
        "-v" if verbose else "-q",
        "--tb=short",
        "-x",  # Stop on first failure
        "--no-header",
    ]
    
    # Test file paths
    test_files = [
        str(Path(__file__).parent / "test_musical_accuracy.py"),
        str(Path(__file__).parent / "test_corpus_accuracy.py"),
    ]
    
    if specific_test:
        test_args.extend(["-k", specific_test])
    
    # Add test files
    test_args.extend(test_files)
    
    # Run tests with custom plugin to collect results
    results = {
        "total_tests": 0,
        "passed": 0,
        "failed": 0,
        "skipped": 0,
        "errors": [],
        "test_details": {},
        "timestamp": datetime.now().isoformat(),
    }
    
    # Run pytest
    exit_code = pytest.main(test_args)
    
    return exit_code, results

def print_summary(results: Dict):
    """Print a summary of test results"""
    print("\n" + "="*60)
    print("MUSICAL ACCURACY TEST SUMMARY")
    print("="*60)
    print(f"Timestamp: {results['timestamp']}")
    print(f"Total Tests: {results['total_tests']}")
    print(f"Passed: {results['passed']}")
    print(f"Failed: {results['failed']}")
    print(f"Skipped: {results['skipped']}")
    
    if results['failed'] > 0:
        print("\nFailed Tests:")
        for error in results['errors']:
            print(f"  - {error}")
    
    print("\n" + "="*60)

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run musical accuracy tests for music21-mcp-server"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show detailed test output"
    )
    parser.add_argument(
        "-t", "--test",
        type=str,
        help="Run only tests matching this pattern"
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate detailed HTML report"
    )
    
    args = parser.parse_args()
    
    print("Running Musical Accuracy Tests...")
    print("This will test the musical correctness of analysis results")
    print("against known musical examples and theory rules.\n")
    
    # Run the tests
    exit_code, results = run_accuracy_tests(
        verbose=args.verbose,
        specific_test=args.test
    )
    
    # Print summary
    print_summary(results)
    
    if args.report:
        # Generate HTML report if requested
        report_path = Path(__file__).parent / "accuracy_report.html"
        print(f"\nGenerating detailed report at: {report_path}")
        # TODO: Implement HTML report generation
    
    # Return appropriate exit code
    return exit_code

if __name__ == "__main__":
    sys.exit(main())