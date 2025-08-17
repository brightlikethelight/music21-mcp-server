#!/usr/bin/env python3
"""
Build script for Music21 MCP Server Desktop Extension (.dxt)

This script creates a complete .dxt package that can be installed with one click.
The .dxt file is a ZIP archive containing all necessary files for the extension.
"""

import os
import sys
import json
import zipfile
import shutil
import tempfile
from pathlib import Path
from datetime import datetime


def log(message: str, level: str = "INFO"):
    """Simple logging function"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}")


def validate_manifest(manifest_path: Path):
    """Validate the manifest.json file"""
    try:
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        
        # Check required fields
        required_fields = ["dxt_version", "name", "version", "description", "author", "server"]
        for field in required_fields:
            if field not in manifest:
                log(f"Missing required field in manifest: {field}", "ERROR")
                return False
        
        log("Manifest validation passed", "INFO")
        return True, manifest
        
    except json.JSONDecodeError as e:
        log(f"Invalid JSON in manifest: {e}", "ERROR")
        return False, None
    except Exception as e:
        log(f"Error validating manifest: {e}", "ERROR")
        return False, None


def copy_source_files(src_dir: Path, temp_dir: Path):
    """Copy source files to temporary build directory"""
    try:
        # Copy the src directory
        src_dest = temp_dir / "src"
        shutil.copytree(src_dir / "src", src_dest)
        log(f"Copied source files to {src_dest}", "INFO")
        
        # Copy essential files
        essential_files = [
            "README.md",
            "LICENSE",
            "CHANGELOG.md",
            "pyproject.toml"
        ]
        
        for file_name in essential_files:
            src_file = src_dir / file_name
            if src_file.exists():
                shutil.copy2(src_file, temp_dir)
                log(f"Copied {file_name}", "INFO")
            else:
                log(f"Optional file not found: {file_name}", "WARNING")
        
        return True
    except Exception as e:
        log(f"Error copying source files: {e}", "ERROR")
        return False


def create_assets(temp_dir: Path):
    """Create or copy asset files (icons, screenshots, etc.)"""
    assets_dir = temp_dir / "assets"
    assets_dir.mkdir(exist_ok=True)
    
    # Create a simple text-based icon if none exists
    icon_content = """
🎵 Music21 MCP Server

This is a placeholder icon for the Music21 Analysis Server.
The actual icon would be a PNG file with musical notation or
a visual representation of music analysis tools.
"""
    
    icon_file = assets_dir / "music21-icon.txt"
    with open(icon_file, 'w') as f:
        f.write(icon_content.strip())
    
    # Create screenshot placeholders
    screenshots_dir = assets_dir / "screenshots"
    screenshots_dir.mkdir(exist_ok=True)
    
    demo_content = """
Music21 Analysis Demo Screenshot

This would show the extension in action:
- Importing a musical score
- Performing key analysis
- Displaying harmony analysis results
- Showing chord progressions
"""
    
    for screenshot_name in ["analysis-demo.txt", "harmonization-demo.txt"]:
        screenshot_file = screenshots_dir / screenshot_name
        with open(screenshot_file, 'w') as f:
            f.write(demo_content.strip())
    
    log("Created asset placeholders", "INFO")
    return True


def copy_dxt_files(dxt_dir: Path, temp_dir: Path):
    """Copy DXT-specific files (manifest, scripts, etc.)"""
    try:
        # Copy manifest.json
        manifest_src = dxt_dir / "manifest.json"
        manifest_dest = temp_dir / "manifest.json"
        shutil.copy2(manifest_src, manifest_dest)
        log("Copied manifest.json", "INFO")
        
        # Copy scripts directory
        scripts_src = dxt_dir / "scripts"
        if scripts_src.exists():
            scripts_dest = temp_dir / "scripts"
            shutil.copytree(scripts_src, scripts_dest)
            log("Copied scripts directory", "INFO")
        
        # Copy requirements.txt
        requirements_src = dxt_dir / "requirements.txt"
        if requirements_src.exists():
            requirements_dest = temp_dir / "requirements.txt"
            shutil.copy2(requirements_src, requirements_dest)
            log("Copied requirements.txt", "INFO")
        
        return True
    except Exception as e:
        log(f"Error copying DXT files: {e}", "ERROR")
        return False


def create_dxt_package(temp_dir: Path, output_file: Path, manifest: dict):
    """Create the final .dxt package (ZIP file)"""
    try:
        with zipfile.ZipFile(output_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Walk through all files in temp directory
            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    file_path = Path(root) / file
                    # Get relative path from temp_dir
                    arc_name = file_path.relative_to(temp_dir)
                    zipf.write(file_path, arc_name)
                    
            log(f"Created DXT package: {output_file}", "INFO")
            
            # Print package info
            file_size = output_file.stat().st_size
            file_size_mb = file_size / (1024 * 1024)
            log(f"Package size: {file_size_mb:.2f} MB", "INFO")
            
        return True
    except Exception as e:
        log(f"Error creating DXT package: {e}", "ERROR")
        return False


def generate_package_info(output_file: Path, manifest: dict):
    """Generate information about the created package"""
    info = {
        "package_name": output_file.name,
        "package_path": str(output_file.absolute()),
        "package_size_bytes": output_file.stat().st_size,
        "package_size_mb": round(output_file.stat().st_size / (1024 * 1024), 2),
        "created_at": datetime.now().isoformat(),
        "extension_info": {
            "name": manifest.get("name"),
            "display_name": manifest.get("display_name"),
            "version": manifest.get("version"),
            "description": manifest.get("description"),
            "author": manifest.get("author", {}).get("name"),
            "tools_count": len(manifest.get("tools", [])),
        }
    }
    
    info_file = output_file.with_suffix(".info.json")
    with open(info_file, 'w') as f:
        json.dump(info, f, indent=2)
    
    log(f"Package info saved to: {info_file}", "INFO")
    return info


def main():
    """Main build process"""
    log("Starting Music21 MCP Server DXT build process...", "INFO")
    
    # Get project root directory
    project_root = Path(__file__).parent.absolute()
    log(f"Project root: {project_root}", "INFO")
    
    # Paths
    dxt_dir = project_root / "dxt"
    manifest_path = dxt_dir / "manifest.json"
    
    # Validate manifest
    if not manifest_path.exists():
        log(f"Manifest not found: {manifest_path}", "ERROR")
        sys.exit(1)
    
    valid, manifest = validate_manifest(manifest_path)
    if not valid:
        log("Manifest validation failed", "ERROR")
        sys.exit(1)
    
    # Create output directory
    output_dir = project_root / "dist"
    output_dir.mkdir(exist_ok=True)
    
    # Generate output filename
    name = manifest["name"]
    version = manifest["version"]
    output_file = output_dir / f"{name}-{version}.dxt"
    
    # Remove existing package if it exists
    if output_file.exists():
        output_file.unlink()
        log(f"Removed existing package: {output_file}", "INFO")
    
    # Create temporary build directory
    with tempfile.TemporaryDirectory() as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        log(f"Using temporary directory: {temp_dir}", "INFO")
        
        # Copy all necessary files
        if not copy_source_files(project_root, temp_dir):
            log("Failed to copy source files", "ERROR")
            sys.exit(1)
        
        if not copy_dxt_files(dxt_dir, temp_dir):
            log("Failed to copy DXT files", "ERROR")
            sys.exit(1)
        
        if not create_assets(temp_dir):
            log("Failed to create assets", "ERROR")
            sys.exit(1)
        
        # Create the DXT package
        if not create_dxt_package(temp_dir, output_file, manifest):
            log("Failed to create DXT package", "ERROR")
            sys.exit(1)
    
    # Generate package information
    package_info = generate_package_info(output_file, manifest)
    
    # Print success message
    print("\n" + "="*60)
    print("🎉 DXT Package Build Successful!")
    print("="*60)
    print(f"📦 Package: {output_file.name}")
    print(f"📂 Location: {output_file.parent}")
    print(f"📏 Size: {package_info['package_size_mb']} MB")
    print(f"🎵 Extension: {manifest['display_name']} v{manifest['version']}")
    print(f"🛠️  Tools: {len(manifest.get('tools', []))} available")
    print()
    print("📋 Installation Instructions:")
    print("1. Double-click the .dxt file to install")
    print("2. OR drag and drop it onto Claude Desktop")
    print("3. OR use the Claude Desktop extension manager")
    print()
    print("🔗 Share this file with users for one-click installation!")
    print("="*60)


if __name__ == "__main__":
    main()