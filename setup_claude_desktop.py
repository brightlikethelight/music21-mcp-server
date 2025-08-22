#!/usr/bin/env python3
"""
🎵 Music21 MCP Server - Automated Claude Desktop Setup Script
=============================================================

This script automatically configures Claude Desktop to work with the Music21 MCP Server,
providing a seamless setup experience with comprehensive diagnostics.

Features:
- ✅ Automatic Claude Desktop detection
- ✅ One-click MCP configuration
- ✅ Built-in diagnostics and health checks
- ✅ Detailed troubleshooting guidance
- ✅ Backup and restore functionality

Usage:
    python setup_claude_desktop.py [--check-only] [--restore-backup]
"""

import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


class Colors:
    """Terminal colors for better output formatting"""

    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BLUE = "\033[94m"
    PURPLE = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    BOLD = "\033[1m"
    END = "\033[0m"


class ClaudeDesktopSetup:
    """Automated Claude Desktop setup for Music21 MCP Server"""

    def __init__(self):
        self.platform = platform.system().lower()
        self.config_path = self._get_config_path()
        self.backup_path = None
        self.python_executable = sys.executable

    def _get_config_path(self) -> Path | None:
        """Get Claude Desktop configuration file path based on platform"""
        if self.platform == "darwin":  # macOS
            return (
                Path.home()
                / "Library/Application Support/Claude/claude_desktop_config.json"
            )
        if self.platform == "windows":
            return Path.home() / "AppData/Roaming/Claude/claude_desktop_config.json"
        if self.platform == "linux":
            return Path.home() / ".config/Claude/claude_desktop_config.json"
        return None

    def print_header(self):
        """Print colorful header"""

    def check_prerequisites(self) -> bool:
        """Check all prerequisites are met"""

        checks_passed = 0
        total_checks = 4

        # Check 1: Claude Desktop installed
        if self._check_claude_desktop():
            checks_passed += 1
        else:
            self._print_claude_install_instructions()

        # Check 2: Python installation
        if self._check_python():
            checks_passed += 1
        else:
            pass

        # Check 3: Config directory exists
        if self._check_config_directory():
            checks_passed += 1
        else:
            self._create_config_directory()
            checks_passed += 1

        # Check 4: Music21 MCP Server package
        if self._check_package_installed():
            checks_passed += 1
        else:
            self._print_install_instructions()

        return checks_passed == total_checks

    def _check_claude_desktop(self) -> bool:
        """Check if Claude Desktop is installed"""
        if self.platform == "darwin":
            return Path("/Applications/Claude.app").exists()
        if self.platform == "windows":
            # Check common installation paths
            paths = [
                Path.home() / "AppData/Local/Claude/Claude.exe",
                Path("C:/Program Files/Claude/Claude.exe"),
                Path("C:/Program Files (x86)/Claude/Claude.exe"),
            ]
            return any(p.exists() for p in paths)
        if self.platform == "linux":
            # Check if claude is in PATH or common locations
            try:
                subprocess.run(["claude", "--version"], capture_output=True, check=True)
                return True
            except (subprocess.CalledProcessError, FileNotFoundError):
                return False
        return False

    def _check_python(self) -> bool:
        """Check if Python version is compatible"""
        version = sys.version_info
        return version.major == 3 and version.minor >= 8

    def _check_config_directory(self) -> bool:
        """Check if config directory exists"""
        return self.config_path and self.config_path.parent.exists()

    def _check_package_installed(self) -> bool:
        """Check if music21-mcp-server is installed"""
        try:
            import music21_mcp

            return True
        except ImportError:
            return False

    def _create_config_directory(self):
        """Create configuration directory"""
        if self.config_path:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)

    def _print_claude_install_instructions(self):
        """Print Claude Desktop installation instructions"""
        if self.platform == "darwin" or self.platform == "windows":
            pass
        else:
            pass

    def _print_install_instructions(self):
        """Print package installation instructions"""

    def backup_existing_config(self) -> bool:
        """Backup existing configuration"""
        if not self.config_path or not self.config_path.exists():
            return True

        # Create backup with timestamp
        import datetime

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.backup_path = self.config_path.with_suffix(f".backup_{timestamp}.json")

        try:
            shutil.copy2(self.config_path, self.backup_path)
            return True
        except Exception as e:
            return False

    def create_mcp_config(self) -> dict[str, Any]:
        """Create MCP configuration for music21-mcp-server"""

        # Get the directory where the current script is located
        script_dir = Path(__file__).parent
        server_script = script_dir / "src" / "music21_mcp" / "__main__.py"

        # If that doesn't exist, try the installed package location
        if not server_script.exists():
            try:
                import music21_mcp

                package_dir = Path(music21_mcp.__file__).parent
                server_script = package_dir / "__main__.py"
            except ImportError:
                # Fallback to module execution
                server_script = None

        # Build the command
        if server_script and server_script.exists():
            command = [self.python_executable, str(server_script)]
        else:
            # Use module execution as fallback
            command = [self.python_executable, "-m", "music21_mcp"]

        config = {
            "mcpServers": {
                "music21-mcp-server": {
                    "command": command,
                    "args": [],
                    "env": {
                        "MUSIC21_MCP_LOG_LEVEL": "INFO",
                        "MUSIC21_MCP_MAX_MEMORY_MB": "512",
                        "MUSIC21_MCP_MAX_SCORES": "100",
                    },
                }
            }
        }

        return config

    def update_claude_config(self, check_only: bool = False) -> bool:
        """Update Claude Desktop configuration"""
        if check_only:
            return self._validate_existing_config()

        # Load existing config or create new
        existing_config = {}
        if self.config_path.exists():
            try:
                with open(self.config_path) as f:
                    existing_config = json.load(f)
            except Exception as e:
                existing_config = {}

        # Create new MCP config
        new_config = self.create_mcp_config()

        # Merge configurations
        if "mcpServers" not in existing_config:
            existing_config["mcpServers"] = {}

        existing_config["mcpServers"].update(new_config["mcpServers"])

        # Write updated config
        try:
            with open(self.config_path, "w") as f:
                json.dump(existing_config, f, indent=2)
            return True
        except Exception as e:
            return False

    def _validate_existing_config(self) -> bool:
        """Validate existing configuration"""
        if not self.config_path.exists():
            return False

        try:
            with open(self.config_path) as f:
                config = json.load(f)

            return bool(
                "mcpServers" in config and "music21-mcp-server" in config["mcpServers"]
            )
        except Exception as e:
            return False

    def run_diagnostics(self) -> bool:
        """Run comprehensive diagnostics"""

        diagnostics_passed = 0
        total_diagnostics = 5

        # Test 1: Config file validation
        if self._test_config_file():
            diagnostics_passed += 1
        else:
            pass

        # Test 2: Python execution
        if self._test_python_execution():
            diagnostics_passed += 1
        else:
            pass

        # Test 3: Package import
        if self._test_package_import():
            diagnostics_passed += 1
        else:
            pass

        # Test 4: Music21 corpus
        if self._test_music21_corpus():
            diagnostics_passed += 1
        else:
            diagnostics_passed += 0.5

        # Test 5: MCP server startup
        if self._test_mcp_startup():
            diagnostics_passed += 1
        else:
            pass

        return diagnostics_passed >= (total_diagnostics - 0.5)

    def _test_config_file(self) -> bool:
        """Test if config file is valid JSON"""
        try:
            with open(self.config_path) as f:
                json.load(f)
            return True
        except:
            return False

    def _test_python_execution(self) -> bool:
        """Test if Python can execute properly"""
        try:
            result = subprocess.run(
                [self.python_executable, "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return result.returncode == 0
        except:
            return False

    def _test_package_import(self) -> bool:
        """Test if the package can be imported"""
        try:
            result = subprocess.run(
                [
                    self.python_executable,
                    "-c",
                    "import music21_mcp; print('Import successful')",
                ],
                capture_output=True,
                text=True,
                timeout=15,
            )
            return result.returncode == 0 and "Import successful" in result.stdout
        except:
            return False

    def _test_music21_corpus(self) -> bool:
        """Test if music21 corpus is available"""
        try:
            result = subprocess.run(
                [
                    self.python_executable,
                    "-c",
                    "import music21; score = music21.corpus.parse('bach/bwv66.6'); print('Corpus available')",
                ],
                capture_output=True,
                text=True,
                timeout=20,
            )
            return result.returncode == 0 and "Corpus available" in result.stdout
        except:
            return False

    def _test_mcp_startup(self) -> bool:
        """Test if MCP server can start up"""
        try:
            # Create a temporary test to see if the server initializes
            result = subprocess.run(
                [
                    self.python_executable,
                    "-c",
                    """
import sys
import asyncio
try:
    from music21_mcp.services import MusicAnalysisService
    service = MusicAnalysisService()
    print(f'MCP server initialized with {len(service.get_available_tools())} tools')
except Exception as e:
    print(f'Failed: {e}')
    sys.exit(1)
""",
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            return result.returncode == 0 and "MCP server initialized" in result.stdout
        except:
            return False

    def print_setup_complete(self):
        """Print setup completion message"""
        if self.backup_path:
            pass

    def print_troubleshooting_guide(self):
        """Print troubleshooting information"""

    def restore_backup(self, backup_file: str) -> bool:
        """Restore configuration from backup"""
        backup_path = Path(backup_file)
        if not backup_path.exists():
            return False

        try:
            shutil.copy2(backup_path, self.config_path)
            return True
        except Exception as e:
            return False


def main():
    """Main setup function"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Setup Music21 MCP Server for Claude Desktop"
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check current configuration, don't modify",
    )
    parser.add_argument(
        "--restore-backup", type=str, help="Restore configuration from backup file"
    )
    parser.add_argument(
        "--skip-diagnostics", action="store_true", help="Skip diagnostic tests"
    )

    args = parser.parse_args()

    setup = ClaudeDesktopSetup()
    setup.print_header()

    # Handle restore backup
    if args.restore_backup:
        if setup.restore_backup(args.restore_backup):
            pass
        return

    # Check prerequisites
    if not setup.check_prerequisites():
        setup.print_troubleshooting_guide()
        return

    # Backup existing config
    if not args.check_only:
        if not setup.backup_existing_config():
            response = input(
                f"{Colors.YELLOW}Continue without backup? (y/N): {Colors.END}"
            )
            if response.lower() != "y":
                return

    # Update configuration
    if not setup.update_claude_config(check_only=args.check_only):
        setup.print_troubleshooting_guide()
        return

    # Run diagnostics
    if not args.skip_diagnostics:
        if not setup.run_diagnostics():
            setup.print_troubleshooting_guide()
        else:
            setup.print_setup_complete()
    else:
        setup.print_setup_complete()


if __name__ == "__main__":
    main()
