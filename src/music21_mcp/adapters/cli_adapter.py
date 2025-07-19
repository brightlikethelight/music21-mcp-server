#!/usr/bin/env python3
"""
CLI Adapter
Command-line interface to core music analysis service

Provides CLI access to music21 functionality independent of protocols.
Works when MCP and HTTP services fail.
"""

import argparse
import asyncio
import sys
from typing import Any

from ..services import MusicAnalysisService


class CLIAdapter:
    """
    Command-line interface adapter for music analysis service

    Provides direct CLI access to core music analysis functionality.
    Independent of network protocols - always works locally.
    """

    def __init__(self):
        """Initialize CLI adapter with core service"""
        self.core_service = MusicAnalysisService()

    async def import_score(
        self, score_id: str, source: str, source_type: str = "corpus"
    ) -> None:
        """Import a score from various sources"""
        try:
            result = await self.core_service.import_score(score_id, source, source_type)
            self._print_result("Import", result)
        except Exception as e:
            self._print_error(f"Import failed: {e}")

    async def list_scores(self) -> None:
        """List all imported scores"""
        try:
            result = await self.core_service.list_scores()
            self._print_result("Scores", result)
        except Exception as e:
            self._print_error(f"List failed: {e}")

    async def analyze_key(self, score_id: str) -> None:
        """Analyze the key signature of a score"""
        try:
            result = await self.core_service.analyze_key(score_id)
            self._print_result("Key Analysis", result)
        except Exception as e:
            self._print_error(f"Key analysis failed: {e}")

    async def analyze_harmony(
        self, score_id: str, analysis_type: str = "roman"
    ) -> None:
        """Perform harmony analysis"""
        try:
            result = await self.core_service.analyze_harmony(score_id, analysis_type)
            self._print_result("Harmony Analysis", result)
        except Exception as e:
            self._print_error(f"Harmony analysis failed: {e}")

    async def analyze_voice_leading(self, score_id: str) -> None:
        """Analyze voice leading quality"""
        try:
            result = await self.core_service.analyze_voice_leading(score_id)
            self._print_result("Voice Leading Analysis", result)
        except Exception as e:
            self._print_error(f"Voice leading analysis failed: {e}")

    async def get_score_info(self, score_id: str) -> None:
        """Get detailed information about a score"""
        try:
            result = await self.core_service.get_score_info(score_id)
            self._print_result("Score Info", result)
        except Exception as e:
            self._print_error(f"Score info failed: {e}")

    async def export_score(self, score_id: str, format: str = "musicxml") -> None:
        """Export a score to various formats"""
        try:
            result = await self.core_service.export_score(score_id, format)
            self._print_result("Export", result)
        except Exception as e:
            self._print_error(f"Export failed: {e}")

    async def recognize_patterns(
        self, score_id: str, pattern_type: str = "melodic"
    ) -> None:
        """Recognize musical patterns"""
        try:
            result = await self.core_service.recognize_patterns(score_id, pattern_type)
            self._print_result("Pattern Recognition", result)
        except Exception as e:
            self._print_error(f"Pattern recognition failed: {e}")

    def show_tools(self) -> None:
        """Show all available analysis tools"""
        tools = self.core_service.get_available_tools()
        print("🎵 Available Music Analysis Tools:")
        print("=" * 40)
        for i, tool in enumerate(tools, 1):
            print(f"{i:2d}. {tool}")
        print(f"\nTotal: {len(tools)} tools available")

    def show_status(self) -> None:
        """Show service status"""
        score_count = self.core_service.get_score_count()
        tools_count = len(self.core_service.get_available_tools())

        print("🎵 Music21 Analysis Service Status")
        print("=" * 35)
        print("Service: MusicAnalysisService")
        print(f"Tools available: {tools_count}")
        print(f"Scores loaded: {score_count}")
        print("Status: ✅ Healthy")

    def _print_result(self, operation: str, result: dict[str, Any]) -> None:
        """Print formatted result"""
        print(f"\n🎵 {operation} Result:")
        print("=" * (len(operation) + 10))

        if isinstance(result, dict):
            if result.get("status") == "success":
                print("✅ Status: Success")
                if "message" in result:
                    print(f"📝 Message: {result['message']}")

                # Print key data fields
                if "data" in result:
                    data = result["data"]
                    if isinstance(data, dict):
                        for key, value in data.items():
                            if key not in ["status", "message"]:
                                print(f"📊 {key.title()}: {value}")
                    else:
                        print(f"📊 Data: {data}")
            else:
                print(f"❌ Status: {result.get('status', 'Unknown')}")
                if "error" in result:
                    print(f"🚨 Error: {result['error']}")
        else:
            print(f"📊 Result: {result}")

        print()

    def _print_error(self, message: str) -> None:
        """Print formatted error"""
        print(f"\n❌ Error: {message}")
        print()


async def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Music21 Analysis CLI - Direct access to music analysis tools",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s status                                    # Show service status
  %(prog)s tools                                     # List available tools
  %(prog)s import bach_chorale bach/bwv66.6 corpus  # Import from corpus
  %(prog)s import my_score /path/to/file.xml file    # Import from file
  %(prog)s list                                      # List imported scores
  %(prog)s key-analysis bach_chorale                 # Analyze key
  %(prog)s harmony bach_chorale roman                # Harmony analysis
  %(prog)s voice-leading bach_chorale                # Voice leading analysis
  %(prog)s info bach_chorale                         # Score information
  %(prog)s export bach_chorale musicxml              # Export score
        """,
    )

    parser.add_argument("command", help="Command to execute")
    parser.add_argument("args", nargs="*", help="Command arguments")

    # Parse arguments
    if len(sys.argv) == 1:
        parser.print_help()
        return

    args = parser.parse_args()
    command = args.command.lower()
    cmd_args = args.args

    # Initialize CLI adapter
    cli = CLIAdapter()

    try:
        # Handle commands
        if command == "status":
            cli.show_status()

        elif command == "tools":
            cli.show_tools()

        elif command == "import":
            if len(cmd_args) < 2:
                print("❌ Usage: import <score_id> <source> [source_type]")
                return
            score_id = cmd_args[0]
            source = cmd_args[1]
            source_type = cmd_args[2] if len(cmd_args) > 2 else "corpus"
            await cli.import_score(score_id, source, source_type)

        elif command == "list":
            await cli.list_scores()

        elif command in ["key", "key-analysis"]:
            if len(cmd_args) < 1:
                print("❌ Usage: key-analysis <score_id>")
                return
            await cli.analyze_key(cmd_args[0])

        elif command == "harmony":
            if len(cmd_args) < 1:
                print("❌ Usage: harmony <score_id> [analysis_type]")
                return
            analysis_type = cmd_args[1] if len(cmd_args) > 1 else "roman"
            await cli.analyze_harmony(cmd_args[0], analysis_type)

        elif command in ["voice", "voice-leading"]:
            if len(cmd_args) < 1:
                print("❌ Usage: voice-leading <score_id>")
                return
            await cli.analyze_voice_leading(cmd_args[0])

        elif command == "info":
            if len(cmd_args) < 1:
                print("❌ Usage: info <score_id>")
                return
            await cli.get_score_info(cmd_args[0])

        elif command == "export":
            if len(cmd_args) < 1:
                print("❌ Usage: export <score_id> [format]")
                return
            format_type = cmd_args[1] if len(cmd_args) > 1 else "musicxml"
            await cli.export_score(cmd_args[0], format_type)

        elif command == "patterns":
            if len(cmd_args) < 1:
                print("❌ Usage: patterns <score_id> [pattern_type]")
                return
            pattern_type = cmd_args[1] if len(cmd_args) > 1 else "melodic"
            await cli.recognize_patterns(cmd_args[0], pattern_type)

        else:
            print(f"❌ Unknown command: {command}")
            print("Use 'tools' to see available commands")

    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
