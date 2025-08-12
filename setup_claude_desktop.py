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
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional, Dict, List


class Colors:
    """Terminal colors for better output formatting"""
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    END = '\033[0m'


class ClaudeDesktopSetup:
    """Automated Claude Desktop setup for Music21 MCP Server"""
    
    def __init__(self):
        self.platform = platform.system().lower()
        self.config_path = self._get_config_path()
        self.backup_path = None
        self.python_executable = sys.executable
        
    def _get_config_path(self) -> Optional[Path]:
        """Get Claude Desktop configuration file path based on platform"""
        if self.platform == "darwin":  # macOS
            return Path.home() / "Library/Application Support/Claude/claude_desktop_config.json"
        elif self.platform == "windows":
            return Path.home() / "AppData/Roaming/Claude/claude_desktop_config.json"
        elif self.platform == "linux":
            return Path.home() / ".config/Claude/claude_desktop_config.json"
        else:
            return None
    
    def print_header(self):
        """Print colorful header"""
        print(f"{Colors.CYAN}{Colors.BOLD}")
        print("🎵 Music21 MCP Server - Claude Desktop Setup")
        print("=" * 50)
        print(f"{Colors.END}")
        print(f"{Colors.WHITE}Automated setup for AI-powered music analysis{Colors.END}")
        print()
    
    def check_prerequisites(self) -> bool:
        """Check all prerequisites are met"""
        print(f"{Colors.BLUE}📋 Checking Prerequisites{Colors.END}")
        print("-" * 30)
        
        checks_passed = 0
        total_checks = 4
        
        # Check 1: Claude Desktop installed
        if self._check_claude_desktop():
            print(f"{Colors.GREEN}✅ Claude Desktop: Installed{Colors.END}")
            checks_passed += 1
        else:
            print(f"{Colors.RED}❌ Claude Desktop: Not found{Colors.END}")
            self._print_claude_install_instructions()
        
        # Check 2: Python installation
        if self._check_python():
            print(f"{Colors.GREEN}✅ Python: {sys.version.split()[0]} (Good){Colors.END}")
            checks_passed += 1
        else:
            print(f"{Colors.RED}❌ Python: Version incompatible{Colors.END}")
        
        # Check 3: Config directory exists
        if self._check_config_directory():
            print(f"{Colors.GREEN}✅ Config Directory: {self.config_path.parent}{Colors.END}")
            checks_passed += 1
        else:
            print(f"{Colors.YELLOW}⚠️  Config Directory: Creating...{Colors.END}")
            self._create_config_directory()
            checks_passed += 1
        
        # Check 4: Music21 MCP Server package
        if self._check_package_installed():
            print(f"{Colors.GREEN}✅ Music21 MCP Server: Installed{Colors.END}")
            checks_passed += 1
        else:
            print(f"{Colors.RED}❌ Music21 MCP Server: Not installed{Colors.END}")
            self._print_install_instructions()
        
        print()
        print(f"{Colors.BOLD}Prerequisites: {checks_passed}/{total_checks} passed{Colors.END}")
        return checks_passed == total_checks
    
    def _check_claude_desktop(self) -> bool:
        """Check if Claude Desktop is installed"""
        if self.platform == "darwin":
            return Path("/Applications/Claude.app").exists()
        elif self.platform == "windows":
            # Check common installation paths
            paths = [
                Path.home() / "AppData/Local/Claude/Claude.exe",
                Path("C:/Program Files/Claude/Claude.exe"),
                Path("C:/Program Files (x86)/Claude/Claude.exe")
            ]
            return any(p.exists() for p in paths)
        elif self.platform == "linux":
            # Check if claude is in PATH or common locations
            try:
                subprocess.run(["claude", "--version"], 
                             capture_output=True, check=True)
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
        print(f"{Colors.YELLOW}")
        print("📥 Please install Claude Desktop:")
        if self.platform == "darwin":
            print("   → Download from https://claude.ai/download")
            print("   → Install the .dmg file")
        elif self.platform == "windows":
            print("   → Download from https://claude.ai/download")
            print("   → Run the installer")
        else:
            print("   → Visit https://claude.ai/download")
        print(f"{Colors.END}")
    
    def _print_install_instructions(self):
        """Print package installation instructions"""
        print(f"{Colors.YELLOW}")
        print("📦 Install Music21 MCP Server:")
        print("   pip install music21-mcp-server")
        print(f"{Colors.END}")
    
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
            print(f"{Colors.GREEN}💾 Config backed up to: {self.backup_path}{Colors.END}")
            return True
        except Exception as e:
            print(f"{Colors.RED}❌ Backup failed: {e}{Colors.END}")
            return False
    
    def create_mcp_config(self) -> Dict[str, Any]:
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
                        "MUSIC21_MCP_MAX_SCORES": "100"
                    }
                }
            }
        }
        
        return config
    
    def update_claude_config(self, check_only: bool = False) -> bool:
        """Update Claude Desktop configuration"""
        if check_only:
            return self._validate_existing_config()
        
        print(f"{Colors.BLUE}🔧 Configuring Claude Desktop{Colors.END}")
        print("-" * 35)
        
        # Load existing config or create new
        existing_config = {}
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    existing_config = json.load(f)
                print(f"{Colors.GREEN}📖 Loaded existing configuration{Colors.END}")
            except Exception as e:
                print(f"{Colors.YELLOW}⚠️  Could not load existing config: {e}{Colors.END}")
                existing_config = {}
        
        # Create new MCP config
        new_config = self.create_mcp_config()
        
        # Merge configurations
        if "mcpServers" not in existing_config:
            existing_config["mcpServers"] = {}
        
        existing_config["mcpServers"].update(new_config["mcpServers"])
        
        # Write updated config
        try:
            with open(self.config_path, 'w') as f:
                json.dump(existing_config, f, indent=2)
            print(f"{Colors.GREEN}✅ Claude Desktop configured successfully{Colors.END}")
            print(f"{Colors.WHITE}   Config file: {self.config_path}{Colors.END}")
            return True
        except Exception as e:
            print(f"{Colors.RED}❌ Failed to write config: {e}{Colors.END}")
            return False
    
    def _validate_existing_config(self) -> bool:
        """Validate existing configuration"""
        if not self.config_path.exists():
            print(f"{Colors.RED}❌ No Claude Desktop config found{Colors.END}")
            return False
        
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            
            if "mcpServers" in config and "music21-mcp-server" in config["mcpServers"]:
                print(f"{Colors.GREEN}✅ Music21 MCP Server already configured{Colors.END}")
                return True
            else:
                print(f"{Colors.YELLOW}⚠️  Music21 MCP Server not found in config{Colors.END}")
                return False
        except Exception as e:
            print(f"{Colors.RED}❌ Invalid config file: {e}{Colors.END}")
            return False
    
    def run_diagnostics(self) -> bool:
        """Run comprehensive diagnostics"""
        print(f"{Colors.PURPLE}🔍 Running Diagnostics{Colors.END}")
        print("-" * 25)
        
        diagnostics_passed = 0
        total_diagnostics = 5
        
        # Test 1: Config file validation
        if self._test_config_file():
            print(f"{Colors.GREEN}✅ Config File: Valid JSON{Colors.END}")
            diagnostics_passed += 1
        else:
            print(f"{Colors.RED}❌ Config File: Invalid{Colors.END}")
        
        # Test 2: Python execution
        if self._test_python_execution():
            print(f"{Colors.GREEN}✅ Python Execution: Working{Colors.END}")
            diagnostics_passed += 1
        else:
            print(f"{Colors.RED}❌ Python Execution: Failed{Colors.END}")
        
        # Test 3: Package import
        if self._test_package_import():
            print(f"{Colors.GREEN}✅ Package Import: Successful{Colors.END}")
            diagnostics_passed += 1
        else:
            print(f"{Colors.RED}❌ Package Import: Failed{Colors.END}")
        
        # Test 4: Music21 corpus
        if self._test_music21_corpus():
            print(f"{Colors.GREEN}✅ Music21 Corpus: Available{Colors.END}")
            diagnostics_passed += 1
        else:
            print(f"{Colors.YELLOW}⚠️  Music21 Corpus: May need configuration{Colors.END}")
            diagnostics_passed += 0.5
        
        # Test 5: MCP server startup
        if self._test_mcp_startup():
            print(f"{Colors.GREEN}✅ MCP Server: Starts correctly{Colors.END}")
            diagnostics_passed += 1
        else:
            print(f"{Colors.RED}❌ MCP Server: Startup failed{Colors.END}")
        
        print()
        print(f"{Colors.BOLD}Diagnostics: {diagnostics_passed}/{total_diagnostics} passed{Colors.END}")
        return diagnostics_passed >= (total_diagnostics - 0.5)
    
    def _test_config_file(self) -> bool:
        """Test if config file is valid JSON"""
        try:
            with open(self.config_path, 'r') as f:
                json.load(f)
            return True
        except:
            return False
    
    def _test_python_execution(self) -> bool:
        """Test if Python can execute properly"""
        try:
            result = subprocess.run([self.python_executable, "--version"], 
                                  capture_output=True, text=True, timeout=10)
            return result.returncode == 0
        except:
            return False
    
    def _test_package_import(self) -> bool:
        """Test if the package can be imported"""
        try:
            result = subprocess.run([
                self.python_executable, "-c", 
                "import music21_mcp; print('Import successful')"
            ], capture_output=True, text=True, timeout=15)
            return result.returncode == 0 and "Import successful" in result.stdout
        except:
            return False
    
    def _test_music21_corpus(self) -> bool:
        """Test if music21 corpus is available"""
        try:
            result = subprocess.run([
                self.python_executable, "-c", 
                "import music21; score = music21.corpus.parse('bach/bwv66.6'); print('Corpus available')"
            ], capture_output=True, text=True, timeout=20)
            return result.returncode == 0 and "Corpus available" in result.stdout
        except:
            return False
    
    def _test_mcp_startup(self) -> bool:
        """Test if MCP server can start up"""
        try:
            # Create a temporary test to see if the server initializes
            result = subprocess.run([
                self.python_executable, "-c", 
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
"""
            ], capture_output=True, text=True, timeout=30)
            return result.returncode == 0 and "MCP server initialized" in result.stdout
        except:
            return False
    
    def print_setup_complete(self):
        """Print setup completion message"""
        print()
        print(f"{Colors.GREEN}{Colors.BOLD}🎉 Setup Complete!{Colors.END}")
        print("-" * 20)
        print(f"{Colors.WHITE}Your Claude Desktop is now configured for AI-powered music analysis!{Colors.END}")
        print()
        print(f"{Colors.CYAN}📋 Next Steps:{Colors.END}")
        print("1. 🔄 Restart Claude Desktop")
        print("2. 🗣️  Start a new conversation")
        print("3. 🎵 Try: 'Analyze the harmony in Bach BWV 66.6'")
        print("4. 📚 Check out the Jupyter tutorials in examples/notebooks/")
        print()
        print(f"{Colors.CYAN}🔗 Resources:{Colors.END}")
        print("• Documentation: README.md")
        print("• Tutorials: examples/notebooks/quickstart_tutorial.ipynb")
        print("• Issues: https://github.com/your-repo/music21-mcp-server/issues")
        print()
        if self.backup_path:
            print(f"{Colors.YELLOW}💾 Backup saved: {self.backup_path}{Colors.END}")
        print()
    
    def print_troubleshooting_guide(self):
        """Print troubleshooting information"""
        print(f"{Colors.RED}🔧 Troubleshooting Guide{Colors.END}")
        print("-" * 25)
        print()
        print(f"{Colors.YELLOW}Common Issues:{Colors.END}")
        print("1. 🔄 Claude Desktop not seeing the server:")
        print("   → Restart Claude Desktop completely")
        print("   → Check config file syntax with: python -m json.tool claude_desktop_config.json")
        print()
        print("2. 🐍 Python/Package issues:")
        print("   → Ensure Python 3.8+: python --version")
        print("   → Reinstall package: pip install --upgrade music21-mcp-server")
        print()
        print("3. 🎵 Music21 corpus issues:")
        print("   → Run: python -c 'import music21; music21.configure.run()'")
        print("   → Restart after configuration")
        print()
        print("4. 💾 Permissions issues:")
        print("   → Check Claude config directory permissions")
        print("   → Try running as administrator (Windows) or with sudo (Linux)")
        print()
        print(f"{Colors.CYAN}🆘 Need Help?{Colors.END}")
        print("• 📖 Check documentation: README.md")
        print("• 🐛 Report issues: [GitHub Issues URL]")
        print("• 💬 Join community: [Discord/Forum URL]")
        print()
    
    def restore_backup(self, backup_file: str) -> bool:
        """Restore configuration from backup"""
        backup_path = Path(backup_file)
        if not backup_path.exists():
            print(f"{Colors.RED}❌ Backup file not found: {backup_file}{Colors.END}")
            return False
        
        try:
            shutil.copy2(backup_path, self.config_path)
            print(f"{Colors.GREEN}✅ Configuration restored from backup{Colors.END}")
            return True
        except Exception as e:
            print(f"{Colors.RED}❌ Failed to restore backup: {e}{Colors.END}")
            return False


def main():
    """Main setup function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Setup Music21 MCP Server for Claude Desktop")
    parser.add_argument("--check-only", action="store_true", 
                       help="Only check current configuration, don't modify")
    parser.add_argument("--restore-backup", type=str, 
                       help="Restore configuration from backup file")
    parser.add_argument("--skip-diagnostics", action="store_true",
                       help="Skip diagnostic tests")
    
    args = parser.parse_args()
    
    setup = ClaudeDesktopSetup()
    setup.print_header()
    
    # Handle restore backup
    if args.restore_backup:
        if setup.restore_backup(args.restore_backup):
            print(f"{Colors.GREEN}🎉 Backup restored successfully!{Colors.END}")
        return
    
    # Check prerequisites
    if not setup.check_prerequisites():
        print(f"{Colors.RED}❌ Prerequisites not met. Please address the issues above.{Colors.END}")
        setup.print_troubleshooting_guide()
        return
    
    print()
    
    # Backup existing config
    if not args.check_only:
        if not setup.backup_existing_config():
            response = input(f"{Colors.YELLOW}Continue without backup? (y/N): {Colors.END}")
            if response.lower() != 'y':
                print(f"{Colors.YELLOW}Setup cancelled by user{Colors.END}")
                return
    
    # Update configuration
    if not setup.update_claude_config(check_only=args.check_only):
        print(f"{Colors.RED}❌ Configuration failed{Colors.END}")
        setup.print_troubleshooting_guide()
        return
    
    print()
    
    # Run diagnostics
    if not args.skip_diagnostics:
        if not setup.run_diagnostics():
            print(f"{Colors.YELLOW}⚠️  Some diagnostics failed{Colors.END}")
            setup.print_troubleshooting_guide()
        else:
            setup.print_setup_complete()
    else:
        setup.print_setup_complete()


if __name__ == "__main__":
    main()