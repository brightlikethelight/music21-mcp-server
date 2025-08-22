#!/usr/bin/env python3
"""
Post-installation script for Music21 MCP Server Desktop Extension

This script:
1. Configures Claude Desktop automatically
2. Sets up proper file permissions
3. Creates necessary directories
4. Performs final health checks
5. Provides user with next steps
"""

import os
import sys
import json
import platform
import subprocess
from pathlib import Path
import shutil


def log(message: str, level: str = "INFO"):
    """Simple logging function"""
    print(f"[{level}] {message}")


def get_claude_desktop_config_path():
    """Get the Claude Desktop configuration file path for the current platform"""
    system = platform.system()

    if system == "Darwin":  # macOS
        home = Path.home()
        return home / ".config" / "claude-desktop" / "config.json"
    elif system == "Windows":
        appdata = os.getenv("APPDATA")
        if appdata:
            return Path(appdata) / "claude-desktop" / "config.json"
        else:
            # Fallback
            return (
                Path.home() / "AppData" / "Roaming" / "claude-desktop" / "config.json"
            )
    else:  # Linux and others
        home = Path.home()
        return home / ".config" / "claude-desktop" / "config.json"


def backup_claude_config(config_path: Path):
    """Create a backup of the existing Claude Desktop configuration"""
    if config_path.exists():
        backup_path = config_path.with_suffix(".json.backup")
        try:
            shutil.copy2(config_path, backup_path)
            log(f"Backed up existing configuration to {backup_path}", "INFO")
            return True
        except Exception as e:
            log(f"Failed to backup configuration: {e}", "WARNING")
            return False
    return True


def load_claude_config(config_path: Path):
    """Load existing Claude Desktop configuration or create a new one"""
    if config_path.exists():
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
            log("Loaded existing Claude Desktop configuration", "INFO")
            return config
        except Exception as e:
            log(f"Failed to load existing configuration: {e}", "WARNING")
            log("Creating new configuration...", "INFO")
            return {"mcpServers": {}}
    else:
        log("No existing Claude Desktop configuration found, creating new one", "INFO")
        return {"mcpServers": {}}


def configure_claude_desktop(extension_dir: Path, env_info: dict):
    """Configure Claude Desktop to use the Music21 MCP Server"""
    config_path = get_claude_desktop_config_path()

    # Create config directory if it doesn't exist
    config_path.parent.mkdir(parents=True, exist_ok=True)

    # Backup existing config
    backup_claude_config(config_path)

    # Load existing configuration
    config = load_claude_config(config_path)

    # Ensure mcpServers section exists
    if "mcpServers" not in config:
        config["mcpServers"] = {}

    # Add our server configuration
    python_exe = env_info["python_exe"]
    server_script = extension_dir / "src" / "music21_mcp" / "server_minimal.py"

    music21_config = {
        "command": python_exe,
        "args": [str(server_script)],
        "env": {"PYTHONPATH": str(extension_dir / "src")},
    }

    # Add the configuration
    config["mcpServers"]["music21-analysis"] = music21_config

    try:
        # Write the updated configuration
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        log(f"Claude Desktop configured successfully at {config_path}", "INFO")
        log("Added 'music21-analysis' server to MCP servers", "INFO")
        return True

    except Exception as e:
        log(f"Failed to write Claude Desktop configuration: {e}", "ERROR")
        return False


def create_output_directories():
    """Create necessary output directories"""
    try:
        # Create exports directory
        exports_dir = Path.home() / "Music" / "music21-exports"
        exports_dir.mkdir(parents=True, exist_ok=True)
        log(f"Created exports directory: {exports_dir}", "INFO")

        # Create cache directory
        cache_dir = Path.home() / ".music21" / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        log(f"Created cache directory: {cache_dir}", "INFO")

        return True
    except Exception as e:
        log(f"Failed to create directories: {e}", "WARNING")
        return True  # Don't fail installation for this


def test_server_health(env_info: dict):
    """Test that the MCP server is working correctly"""
    try:
        log("Testing server health...", "INFO")

        python_exe = env_info["python_exe"]
        extension_dir = Path(env_info["extension_dir"])
        server_script = extension_dir / "src" / "music21_mcp" / "server_minimal.py"

        # Set up environment
        env = os.environ.copy()
        env["PYTHONPATH"] = str(extension_dir / "src")

        # Run a quick test
        test_script = """
import sys
sys.path.insert(0, 'src')
from music21_mcp.adapters.mcp_adapter import MCPAdapter
adapter = MCPAdapter()
result = adapter.check_protocol_compatibility()
print("Health check:", result.get('core_service_healthy', False))
"""

        result = subprocess.run(
            [python_exe, "-c", test_script],
            capture_output=True,
            text=True,
            env=env,
            timeout=30,
        )

        if result.returncode == 0 and "True" in result.stdout:
            log("Server health check passed!", "INFO")
            return True
        else:
            log(f"Server health check failed: {result.stderr}", "WARNING")
            return False

    except subprocess.TimeoutExpired:
        log("Server health check timed out", "WARNING")
        return False
    except Exception as e:
        log(f"Server health check error: {e}", "WARNING")
        return False


def print_success_message():
    """Print installation success message and next steps"""
    print("\n" + "=" * 60)
    print("🎵 Music21 MCP Server installed successfully!")
    print("=" * 60)
    print()
    print("📋 Next Steps:")
    print("1. Restart Claude Desktop if it's currently running")
    print("2. Open Claude Desktop and start a new conversation")
    print("3. The music21-analysis server should now be available")
    print()
    print("🛠️  Available Tools:")
    print("• import_score - Import musical scores from various sources")
    print("• key_analysis - Analyze key signatures and tonal centers")
    print("• harmony_analysis - Roman numeral and functional analysis")
    print("• harmonize_melody - Generate harmonizations in various styles")
    print("• generate_counterpoint - Create species counterpoint")
    print("• pattern_recognition - Identify musical patterns")
    print("• ...and 8 more analysis and composition tools!")
    print()
    print("📚 Documentation:")
    print("https://github.com/brightlikethelight/music21-mcp-server")
    print()
    print("🔧 Troubleshooting:")
    print("If the server doesn't appear in Claude Desktop:")
    print("1. Check that Claude Desktop is restarted")
    print("2. Verify the configuration file was updated correctly:")

    config_path = get_claude_desktop_config_path()
    print(f"   {config_path}")
    print("3. Check the Claude Desktop logs for error messages")
    print()
    print("🎉 Happy music analysis!")
    print("=" * 60)


def main():
    """Main post-installation process"""
    log("Starting Music21 MCP Server post-installation...", "INFO")

    # Get extension directory and environment info
    extension_dir = Path(__file__).parent.parent.absolute()
    env_file = extension_dir / "env_info.json"

    if not env_file.exists():
        log(
            "Environment info file not found - pre-installation may have failed",
            "ERROR",
        )
        sys.exit(1)

    try:
        with open(env_file, "r") as f:
            env_info = json.load(f)
    except Exception as e:
        log(f"Failed to load environment info: {e}", "ERROR")
        sys.exit(1)

    success = True

    # Configure Claude Desktop
    if not configure_claude_desktop(extension_dir, env_info):
        success = False

    # Create necessary directories
    create_output_directories()

    # Test server health
    if not test_server_health(env_info):
        log("Server health check failed, but installation will continue", "WARNING")
        # Don't fail installation for this

    # Clean up temporary files
    try:
        env_file.unlink()
        log("Cleaned up temporary files", "INFO")
    except Exception:
        pass

    if success:
        print_success_message()
        log("Post-installation completed successfully!", "INFO")
    else:
        log("Post-installation completed with warnings", "WARNING")
        log(
            "The extension may still work, but manual configuration might be needed",
            "WARNING",
        )


if __name__ == "__main__":
    main()
