#!/usr/bin/env python3
"""
Lightweight health check for Docker container
Does minimal imports to verify basic functionality
"""
import os
import sys
from pathlib import Path


def check_health():
    """Lightweight health check that doesn't import heavy dependencies"""
    try:
        # Check Python version
        if sys.version_info < (3, 10):
            return False, "Python version too old"

        # Check if we can import basic modules
        try:
            import asyncio
            import logging
        except ImportError as e:
            return False, f"Basic import failed: {e}"

        # Check if MCP package is available
        try:
            import mcp
        except ImportError:
            return False, "MCP package not available"

        # Check if music21_mcp package structure exists
        app_src = Path("/app/src/music21_mcp")
        if not app_src.exists():
            return False, "music21_mcp source not found"

        if not (app_src / "server.py").exists():
            return False, "server.py not found"

        if not (app_src / "tools").exists():
            return False, "tools directory not found"

        # Check write permissions for data and logs
        for directory in ["/app/data", "/app/logs"]:
            if not os.access(directory, os.W_OK):
                return False, f"No write access to {directory}"

        return True, "All checks passed"

    except Exception as e:
        return False, f"Health check error: {e}"


if __name__ == "__main__":
    success, message = check_health()
    if success:
        print("healthy")
        sys.exit(0)
    else:
        print(f"unhealthy: {message}")
        sys.exit(1)
