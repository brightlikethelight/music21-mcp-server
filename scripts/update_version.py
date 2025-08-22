#!/usr/bin/env python3
"""
Version update script for semantic-release.
Updates version in pyproject.toml and __init__.py files.
"""

import sys
import re
import tomllib
from pathlib import Path


def update_pyproject_toml(version: str) -> None:
    """Update version in pyproject.toml"""
    pyproject_path = Path("pyproject.toml")

    if not pyproject_path.exists():
        print(f"Error: {pyproject_path} not found")
        sys.exit(1)

    # Read current content
    content = pyproject_path.read_text(encoding="utf-8")

    # Update [project] version
    content = re.sub(
        r'^version = "[^"]*"', f'version = "{version}"', content, flags=re.MULTILINE
    )

    # Update [tool.poetry] version
    content = re.sub(
        r'^version = "[^"]*"', f'version = "{version}"', content, flags=re.MULTILINE
    )

    # Write updated content
    pyproject_path.write_text(content, encoding="utf-8")
    print(f"✅ Updated version in {pyproject_path} to {version}")


def update_init_py(version: str) -> None:
    """Update version in __init__.py"""
    init_path = Path("src/music21_mcp/__init__.py")

    # Read current content or create if empty
    if init_path.exists():
        content = init_path.read_text(encoding="utf-8")
    else:
        content = ""

    # Check if __version__ already exists
    if "__version__" in content:
        # Update existing version
        content = re.sub(
            r'__version__ = "[^"]*"', f'__version__ = "{version}"', content
        )
    else:
        # Add version to file
        version_line = f'__version__ = "{version}"\n'
        if content and not content.endswith("\n"):
            content += "\n"
        content += version_line

    # Write updated content
    init_path.write_text(content, encoding="utf-8")
    print(f"✅ Updated version in {init_path} to {version}")


def verify_version_consistency(version: str) -> None:
    """Verify that versions are consistent across files"""
    try:
        # Verify pyproject.toml
        with open("pyproject.toml", "rb") as f:
            pyproject_data = tomllib.load(f)
            pyproject_version = pyproject_data["project"]["version"]
            if pyproject_version != version:
                print(
                    f"❌ Version mismatch in pyproject.toml: {pyproject_version} != {version}"
                )
                sys.exit(1)

        # Verify __init__.py
        init_path = Path("src/music21_mcp/__init__.py")
        if init_path.exists():
            content = init_path.read_text(encoding="utf-8")
            version_match = re.search(r'__version__ = "([^"]*)"', content)
            if version_match:
                init_version = version_match.group(1)
                if init_version != version:
                    print(
                        f"❌ Version mismatch in __init__.py: {init_version} != {version}"
                    )
                    sys.exit(1)

        print(f"✅ All versions consistent: {version}")

    except Exception as e:
        print(f"❌ Error verifying version consistency: {e}")
        sys.exit(1)


def main():
    """Main function"""
    if len(sys.argv) != 2:
        print("Usage: python update_version.py <version>")
        sys.exit(1)

    version = sys.argv[1].strip()

    if not version:
        print("Error: Version cannot be empty")
        sys.exit(1)

    # Remove 'v' prefix if present
    if version.startswith("v"):
        version = version[1:]

    print(f"🔄 Updating version to {version}")

    try:
        update_pyproject_toml(version)
        update_init_py(version)
        verify_version_consistency(version)
        print(f"🎉 Successfully updated all versions to {version}")
    except Exception as e:
        print(f"❌ Error updating version: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
