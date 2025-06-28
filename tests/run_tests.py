#!/usr/bin/env python
"""
Test runner script for music21 MCP server
"""
import sys
import os
import subprocess

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def run_tests(test_type=None):
    """Run tests with appropriate settings"""
    cmd = ['pytest']
    
    if test_type == 'unit':
        cmd.extend(['-m', 'unit', 'tests/unit'])
    elif test_type == 'integration':
        cmd.extend(['-m', 'integration', 'tests/integration'])
    elif test_type == 'coverage':
        cmd.extend(['--cov-report=html', '--cov-report=term'])
    elif test_type == 'fast':
        cmd.extend(['-m', 'not slow'])
    
    print(f"Running: {' '.join(cmd)}")
    return subprocess.call(cmd)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run music21 MCP server tests')
    parser.add_argument(
        'type',
        nargs='?',
        choices=['all', 'unit', 'integration', 'coverage', 'fast'],
        default='all',
        help='Type of tests to run'
    )
    
    args = parser.parse_args()
    
    sys.exit(run_tests(None if args.type == 'all' else args.type))