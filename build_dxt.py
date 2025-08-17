#!/usr/bin/env python3
"""
Build Desktop Extension (.dxt) for Music21 MCP Server
Creates a one-click installable package for Claude Desktop
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, Any

class DesktopExtensionBuilder:
    """Build a Desktop Extension package for easy installation"""
    
    def __init__(self):
        self.root_dir = Path(__file__).parent
        self.dist_dir = self.root_dir / "dist"
        self.dxt_dir = self.root_dir / "dxt"
        self.version = self._get_version()
        
    def _get_version(self) -> str:
        """Get version from pyproject.toml"""
        try:
            pyproject_path = self.root_dir / "pyproject.toml"
            with open(pyproject_path, 'r') as f:
                for line in f:
                    if line.startswith('version = '):
                        return line.split('"')[1]
        except Exception:
            pass
        return "1.0.0"
    
    def build(self):
        """Build the Desktop Extension package"""
        print("🎵 Building Music21 MCP Server Desktop Extension")
        print("=" * 50)
        
        # Clean and create directories
        self._prepare_directories()
        
        # Create extension manifest
        self._create_manifest()
        
        # Copy server code
        self._copy_server_code()
        
        # Create installation scripts
        self._create_install_scripts()
        
        # Bundle dependencies
        self._bundle_dependencies()
        
        # Create configuration templates
        self._create_config_templates()
        
        # Create the .dxt package
        package_path = self._create_package()
        
        print(f"\n✅ Desktop Extension created: {package_path}")
        print("\n📦 Installation Instructions:")
        print("1. Open Claude Desktop")
        print("2. Go to Settings > Extensions")
        print("3. Click 'Install Extension'")
        print(f"4. Select {package_path}")
        print("5. The Music21 MCP Server will be automatically configured!")
        
        return package_path
    
    def _prepare_directories(self):
        """Prepare build directories"""
        print("\n📁 Preparing directories...")
        
        # Clean existing build
        if self.dxt_dir.exists():
            shutil.rmtree(self.dxt_dir)
        
        # Create fresh directories
        self.dxt_dir.mkdir(exist_ok=True)
        self.dist_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.dxt_dir / "server").mkdir()
        (self.dxt_dir / "scripts").mkdir()
        (self.dxt_dir / "config").mkdir()
        (self.dxt_dir / "dependencies").mkdir()
    
    def _create_manifest(self):
        """Create the extension manifest"""
        print("📝 Creating extension manifest...")
        
        manifest = {
            "name": "music21-mcp-server",
            "version": self.version,
            "displayName": "Music21 Analysis & Generation",
            "description": "Professional music analysis and generation tools powered by MIT's music21",
            "author": {
                "name": "brightliu",
                "email": "brightliu@college.harvard.edu"
            },
            "type": "mcp-server",
            "icon": "🎵",
            "categories": ["music", "analysis", "education", "composition"],
            "requirements": {
                "claudeDesktop": ">=1.0.0",
                "python": ">=3.9"
            },
            "installation": {
                "type": "automatic",
                "steps": [
                    {
                        "action": "check_python",
                        "minVersion": "3.9"
                    },
                    {
                        "action": "install_dependencies",
                        "requirements": "requirements.txt"
                    },
                    {
                        "action": "configure_mcp",
                        "config": "config/claude_desktop.json"
                    },
                    {
                        "action": "test_connection",
                        "endpoint": "health_check"
                    }
                ]
            },
            "server": {
                "command": "python",
                "args": ["-m", "music21_mcp.server_minimal"],
                "env": {
                    "PYTHONPATH": "${extension_dir}/server/src",
                    "MUSIC21_MCP_TIMEOUT": "30"
                }
            },
            "tools": [
                {
                    "name": "import_score",
                    "description": "Import musical scores from files, URLs, or corpus"
                },
                {
                    "name": "analyze_key",
                    "description": "Detect musical key with confidence scoring"
                },
                {
                    "name": "analyze_chords",
                    "description": "Analyze chord progressions with Roman numerals"
                },
                {
                    "name": "harmonize_melody",
                    "description": "Generate harmonizations in classical, jazz, pop, or modal styles"
                },
                {
                    "name": "generate_counterpoint",
                    "description": "Create species counterpoint following traditional rules"
                },
                {
                    "name": "imitate_style",
                    "description": "Generate new music in the style of Bach, Mozart, or Chopin"
                },
                {
                    "name": "analyze_voice_leading",
                    "description": "Check voice leading quality and detect errors"
                },
                {
                    "name": "detect_patterns",
                    "description": "Find recurring melodic, rhythmic, and harmonic patterns"
                },
                {
                    "name": "export_score",
                    "description": "Export to MusicXML, MIDI, PDF, or other formats"
                }
            ],
            "quickStart": {
                "examples": [
                    {
                        "title": "Analyze a Bach Chorale",
                        "prompt": "Import Bach BWV 66.6 and analyze its key and harmony"
                    },
                    {
                        "title": "Harmonize a Melody",
                        "prompt": "Import my melody and create a classical harmonization"
                    },
                    {
                        "title": "Generate Counterpoint",
                        "prompt": "Create first species counterpoint above this cantus firmus"
                    }
                ]
            },
            "support": {
                "documentation": "https://github.com/brightliu/music21-mcp-server",
                "issues": "https://github.com/brightliu/music21-mcp-server/issues"
            }
        }
        
        manifest_path = self.dxt_dir / "manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
    
    def _copy_server_code(self):
        """Copy the server source code"""
        print("📦 Copying server code...")
        
        src_dir = self.root_dir / "src" / "music21_mcp"
        dest_dir = self.dxt_dir / "server" / "src" / "music21_mcp"
        
        # Copy Python source files
        shutil.copytree(src_dir, dest_dir, 
                       ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.pytest_cache'))
        
        # Copy pyproject.toml for dependencies
        shutil.copy2(self.root_dir / "pyproject.toml", self.dxt_dir / "server")
        
        # Create requirements.txt from pyproject.toml
        self._create_requirements_txt()
    
    def _create_requirements_txt(self):
        """Extract requirements from pyproject.toml"""
        print("📋 Creating requirements.txt...")
        
        requirements = [
            "music21>=9.1.0",
            "numpy>=1.24.0",
            "fastmcp>=0.2.0",
            "mcp>=1.0.0",
            "cachetools>=5.3.0",
            "aiofiles>=23.0.0"
        ]
        
        req_path = self.dxt_dir / "requirements.txt"
        with open(req_path, 'w') as f:
            f.write('\n'.join(requirements))
    
    def _create_install_scripts(self):
        """Create installation helper scripts"""
        print("🔧 Creating installation scripts...")
        
        # Windows install script
        windows_script = '''@echo off
echo Installing Music21 MCP Server...
echo.

REM Check Python version
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed!
    echo Please install Python 3.9 or later from python.org
    pause
    exit /b 1
)

REM Install dependencies
echo Installing dependencies...
pip install -r requirements.txt

REM Configure music21
echo Configuring music21 corpus...
python -c "from music21 import corpus; corpus.cacheMetadata()"

echo.
echo ✅ Installation complete!
echo The Music21 MCP Server is ready to use with Claude Desktop.
pause
'''
        
        # macOS/Linux install script
        unix_script = '''#!/bin/bash
echo "Installing Music21 MCP Server..."
echo

# Check Python version
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed!"
    echo "Please install Python 3.9 or later"
    exit 1
fi

# Install dependencies
echo "Installing dependencies..."
pip3 install -r requirements.txt

# Configure music21
echo "Configuring music21 corpus..."
python3 -c "from music21 import corpus; corpus.cacheMetadata()"

echo
echo "✅ Installation complete!"
echo "The Music21 MCP Server is ready to use with Claude Desktop."
'''
        
        # Save scripts
        win_path = self.dxt_dir / "scripts" / "install_windows.bat"
        with open(win_path, 'w') as f:
            f.write(windows_script)
        
        unix_path = self.dxt_dir / "scripts" / "install_unix.sh"
        with open(unix_path, 'w') as f:
            f.write(unix_script)
        os.chmod(unix_path, 0o755)
    
    def _bundle_dependencies(self):
        """Bundle critical dependencies"""
        print("📚 Bundling dependencies...")
        
        # Copy workflow templates
        docs_src = self.root_dir / "docs" / "WORKFLOW_TEMPLATES.md"
        if docs_src.exists():
            shutil.copy2(docs_src, self.dxt_dir / "WORKFLOWS.md")
        
        # Copy examples
        examples_dir = self.root_dir / "examples"
        if examples_dir.exists():
            dest_examples = self.dxt_dir / "examples"
            dest_examples.mkdir(exist_ok=True)
            for example in examples_dir.glob("*.py"):
                if example.name in ["basic_usage.py", "simple_example.py"]:
                    shutil.copy2(example, dest_examples)
    
    def _create_config_templates(self):
        """Create configuration templates"""
        print("⚙️ Creating configuration templates...")
        
        # Claude Desktop configuration
        claude_config = {
            "mcpServers": {
                "music21": {
                    "command": "python",
                    "args": ["-m", "music21_mcp.server_minimal"],
                    "env": {
                        "PYTHONPATH": "${extension_dir}/server/src"
                    }
                }
            }
        }
        
        config_path = self.dxt_dir / "config" / "claude_desktop.json"
        with open(config_path, 'w') as f:
            json.dump(claude_config, f, indent=2)
        
        # VS Code configuration
        vscode_config = {
            "mcp.servers": {
                "music21": {
                    "command": "python",
                    "args": ["-m", "music21_mcp.server_minimal"],
                    "cwd": "${extension_dir}/server"
                }
            }
        }
        
        vscode_path = self.dxt_dir / "config" / "vscode.json"
        with open(vscode_path, 'w') as f:
            json.dump(vscode_config, f, indent=2)
    
    def _create_package(self) -> Path:
        """Create the final .dxt package"""
        print("\n📦 Creating .dxt package...")
        
        package_name = f"music21-mcp-server-{self.version}.dxt"
        package_path = self.dist_dir / package_name
        
        # Create ZIP archive with .dxt extension
        with zipfile.ZipFile(package_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            # Add all files from dxt directory
            for root, dirs, files in os.walk(self.dxt_dir):
                for file in files:
                    file_path = Path(root) / file
                    arcname = file_path.relative_to(self.dxt_dir)
                    zf.write(file_path, arcname)
        
        # Create checksum
        self._create_checksum(package_path)
        
        return package_path
    
    def _create_checksum(self, package_path: Path):
        """Create SHA256 checksum for the package"""
        import hashlib
        
        sha256 = hashlib.sha256()
        with open(package_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                sha256.update(chunk)
        
        checksum_path = package_path.with_suffix('.sha256')
        with open(checksum_path, 'w') as f:
            f.write(f"{sha256.hexdigest()}  {package_path.name}\n")
        
        print(f"📝 Checksum: {sha256.hexdigest()[:16]}...")


if __name__ == "__main__":
    builder = DesktopExtensionBuilder()
    try:
        package_path = builder.build()
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Build failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)