"""
Script to run musical accuracy validation tests

This script runs the validation test suite and generates a report
on the musical accuracy of the analysis tools.
"""

import subprocess
import sys
import json
from datetime import datetime
from pathlib import Path


def run_validation_tests():
    """Run the musical accuracy validation tests"""
    print("=" * 70)
    print("MUSICAL ACCURACY VALIDATION TEST SUITE")
    print("=" * 70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Define test files
    test_files = [
        "tests/validation/test_musical_accuracy.py",
        "tests/validation/test_known_compositions.py",
    ]
    
    # Run tests with pytest
    cmd = [
        sys.executable, "-m", "pytest",
        "-v",  # Verbose output
        "-s",  # Show print statements
        "--tb=short",  # Short traceback format
        "--durations=10",  # Show 10 slowest tests
        "--cov=music21_mcp.tools",  # Coverage for tools module
        "--cov-report=term-missing",  # Show missing lines
        "--cov-report=html:htmlcov/validation",  # HTML coverage report
        *test_files
    ]
    
    print("Running command:")
    print(" ".join(cmd))
    print()
    
    # Run the tests
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    print()
    print("=" * 70)
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Exit code: {result.returncode}")
    print("=" * 70)
    
    # Generate summary report
    generate_summary_report(result.returncode)
    
    return result.returncode


def generate_summary_report(exit_code: int):
    """Generate a summary report of the validation tests"""
    report_path = Path("tests/validation/validation_report.md")
    
    with open(report_path, "w") as f:
        f.write("# Musical Accuracy Validation Report\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        if exit_code == 0:
            f.write("## Status: ✅ PASSED\n\n")
            f.write("All musical accuracy validation tests passed successfully.\n\n")
        else:
            f.write("## Status: ❌ FAILED\n\n")
            f.write("Some musical accuracy validation tests failed. See test output for details.\n\n")
        
        f.write("## Test Categories\n\n")
        f.write("### 1. Key Detection Accuracy\n")
        f.write("- Bach chorales with known keys\n")
        f.write("- Complex modulations\n")
        f.write("- Atonal music detection\n\n")
        
        f.write("### 2. Chord Progression Analysis\n")
        f.write("- Classical cadences (ii-V-I)\n")
        f.write("- Jazz progressions with extensions\n")
        f.write("- Pop chord progressions\n\n")
        
        f.write("### 3. Pattern Recognition\n")
        f.write("- Melodic sequences\n")
        f.write("- Motivic development\n")
        f.write("- Imitation in polyphony\n\n")
        
        f.write("### 4. Voice Leading Analysis\n")
        f.write("- Parallel motion detection\n")
        f.write("- Smoothness scoring\n")
        f.write("- Voice crossing detection\n\n")
        
        f.write("### 5. Harmony Analysis\n")
        f.write("- Functional harmony\n")
        f.write("- Non-chord tone identification\n")
        f.write("- Tonicization detection\n\n")
        
        f.write("### 6. Counterpoint Rules\n")
        f.write("- Species counterpoint validation\n")
        f.write("- Consonance/dissonance rules\n")
        f.write("- Proper voice independence\n\n")
        
        f.write("### 7. Style Imitation\n")
        f.write("- Baroque characteristics\n")
        f.write("- Classical period features\n")
        f.write("- Impressionistic harmony\n\n")
        
        f.write("### 8. Known Compositions\n")
        f.write("- Bach WTC Prelude in C Major\n")
        f.write("- Mozart K. 545 Sonata\n")
        f.write("- Beethoven Moonlight Sonata\n")
        f.write("- Pachelbel Canon progression\n")
        f.write("- Debussy impressionism\n")
        f.write("- Schoenberg twelve-tone\n\n")
        
        f.write("## Coverage Report\n\n")
        f.write("See `htmlcov/validation/index.html` for detailed coverage report.\n\n")
        
        f.write("## Recommendations\n\n")
        f.write("1. Review any failed tests for accuracy issues\n")
        f.write("2. Consider adding more corpus examples for validation\n")
        f.write("3. Expand test cases for edge cases in music theory\n")
        f.write("4. Validate against music theory textbook examples\n")
    
    print(f"\nValidation report generated: {report_path}")


if __name__ == "__main__":
    sys.exit(run_validation_tests())