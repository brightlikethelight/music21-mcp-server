#!/usr/bin/env python3
"""
Music21 MCP Server Launcher
Unified entry point for all interfaces

Provides easy access to:
- MCP Server (for Claude Desktop)
- HTTP API Server (for web integration)
- CLI Tools (for command-line usage)
- Python Library (for programmatic access)

Usage Examples:
    python -m music21_mcp.launcher mcp      # Start MCP server
    python -m music21_mcp.launcher http     # Start HTTP API server
    python -m music21_mcp.launcher cli      # Enter CLI mode
    python -m music21_mcp.launcher demo     # Run demo/test
"""

import argparse
import asyncio
import sys


def show_interfaces():
    """Show all available interfaces"""
    print("🎵 Music21 Analysis - Available Interfaces")
    print("=" * 45)
    print()
    print("📡 MCP SERVER (for Claude Desktop):")
    print("   python -m music21_mcp.launcher mcp")
    print("   → Starts MCP server for AI assistant integration")
    print()
    print("🌐 HTTP API SERVER (for web integration):")
    print("   python -m music21_mcp.launcher http")
    print("   → Starts REST API server on http://localhost:8000")
    print("   → API docs at http://localhost:8000/docs")
    print()
    print("💻 CLI TOOLS (for command-line usage):")
    print("   python -m music21_mcp.launcher cli [command] [args]")
    print("   → Direct command-line access to analysis tools")
    print("   → Example: cli import chorale bach/bwv66.6 corpus")
    print()
    print("🐍 PYTHON LIBRARY (for programmatic access):")
    print("   from music21_mcp.adapters import create_sync_analyzer")
    print("   analyzer = create_sync_analyzer()")
    print("   → Direct Python API without protocol overhead")
    print()
    print("🧪 DEMO MODE:")
    print("   python -m music21_mcp.launcher demo")
    print("   → Test all interfaces and show examples")


def start_mcp_server():
    """Start MCP server"""
    try:
        from .server_minimal import main

        print("🎵 Starting MCP server for Claude Desktop...")
        main()
    except ImportError as e:
        print(f"❌ MCP server not available: {e}")
        print("Install with: pip install fastmcp")
    except Exception as e:
        print(f"❌ MCP server error: {e}")


def start_http_server():
    """Start HTTP API server"""
    try:
        import uvicorn

        from .adapters.http_adapter import create_http_server

        app = create_http_server()

        print("🎵 Music21 HTTP API Server")
        print("🌐 Starting server on http://localhost:8000")
        print("📖 API docs: http://localhost:8000/docs")
        print("🔗 Health check: http://localhost:8000/health")
        print()

        uvicorn.run(app, host="0.0.0.0", port=8000)  # noqa: S104

    except ImportError as e:
        print(f"❌ HTTP server not available: {e}")
        print("Install with: pip install fastapi uvicorn")
    except Exception as e:
        print(f"❌ HTTP server error: {e}")


async def run_cli_tools(cli_args: list):
    """Run CLI tools with arguments"""
    try:
        from .adapters.cli_adapter import main

        # Set up sys.argv for CLI parser
        original_argv = sys.argv
        sys.argv = ["music21-cli"] + cli_args

        try:
            await main()
        finally:
            sys.argv = original_argv

    except ImportError as e:
        print(f"❌ CLI tools not available: {e}")
    except Exception as e:
        print(f"❌ CLI error: {e}")


async def run_demo():
    """Run demonstration of all interfaces"""
    print("🎵 Music21 Analysis - Full Interface Demo")
    print("=" * 45)

    # Test Python Library
    print("\n🐍 Testing Python Library Interface:")
    try:
        from .adapters import create_sync_analyzer

        analyzer = create_sync_analyzer()
        status = analyzer.get_status()
        print(f"   ✅ Status: {status['status']}")
        print(f"   📊 Tools: {status['tools_available']}")

        # Test import
        result = analyzer.import_score("demo", "bach/bwv66.6", "corpus")
        print(f"   📥 Import: {result.get('status', 'unknown')}")

    except Exception as e:
        print(f"   ❌ Python library error: {e}")

    # Test HTTP API (without starting server)
    print("\n🌐 Testing HTTP API Interface:")
    try:
        from .adapters.http_adapter import HTTPAdapter

        adapter = HTTPAdapter()
        adapter.get_app()
        print("   ✅ HTTP adapter created successfully")
        print("   📡 FastAPI app ready for deployment")

    except Exception as e:
        print(f"   ❌ HTTP adapter error: {e}")

    # Test CLI Tools
    print("\n💻 Testing CLI Interface:")
    try:
        from .adapters.cli_adapter import CLIAdapter

        cli = CLIAdapter()
        cli.show_status()

    except Exception as e:
        print(f"   ❌ CLI adapter error: {e}")

    # Test MCP Adapter
    print("\n📡 Testing MCP Interface:")
    try:
        from .adapters.mcp_adapter import MCPAdapter

        mcp = MCPAdapter()
        compatibility = mcp.check_protocol_compatibility()
        print(f"   📊 MCP version: {compatibility.get('supported_version', 'unknown')}")
        print(f"   ✅ Tools: {len(mcp.get_supported_tools())}")

    except Exception as e:
        print(f"   ❌ MCP adapter error: {e}")

    print("\n🎯 Demo completed! All interfaces tested.")
    print("\nNext steps:")
    print("• For Claude Desktop: python -m music21_mcp.launcher mcp")
    print("• For web API: python -m music21_mcp.launcher http")
    print("• For CLI usage: python -m music21_mcp.launcher cli status")


def main():
    """Main launcher entry point"""
    parser = argparse.ArgumentParser(
        description="Music21 Analysis - Unified Interface Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s mcp                           # Start MCP server for Claude Desktop
  %(prog)s http                          # Start HTTP API server
  %(prog)s cli status                    # Show CLI status
  %(prog)s cli import song file.xml file # Import via CLI
  %(prog)s demo                          # Run interface demonstrations
  %(prog)s                               # Show available interfaces
        """,
    )

    parser.add_argument(
        "interface",
        nargs="?",
        choices=["mcp", "http", "cli", "demo"],
        help="Interface to launch",
    )
    parser.add_argument(
        "args", nargs="*", help="Additional arguments for the interface"
    )

    # Parse arguments
    if len(sys.argv) == 1:
        show_interfaces()
        return

    args = parser.parse_args()

    try:
        if args.interface == "mcp":
            start_mcp_server()

        elif args.interface == "http":
            start_http_server()

        elif args.interface == "cli":
            asyncio.run(run_cli_tools(args.args))

        elif args.interface == "demo":
            asyncio.run(run_demo())

        else:
            show_interfaces()

    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except Exception as e:
        print(f"❌ Launcher error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
