#!/usr/bin/env python3
"""
Release Preparation Script for Music21 MCP Server

Automates the release process:
1. Runs all tests
2. Checks code quality
3. Updates version numbers
4. Builds distribution
5. Validates package
"""

import os
import re
import subprocess
import sys
from pathlib import Path


def run_command(cmd: str, check: bool = True) -> subprocess.CompletedProcess:
    """Run a shell command and return the result"""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    if check and result.returncode != 0:
        print(f"Error: {result.stderr}")
        sys.exit(1)

    return result


def get_current_version() -> str:
    """Get current version from pyproject.toml"""
    pyproject_path = Path("pyproject.toml")
    content = pyproject_path.read_text()

    match = re.search(r'version = "([^"]+)"', content)
    if match:
        return match.group(1)

    raise ValueError("Could not find version in pyproject.toml")


def update_version(new_version: str):
    """Update version in all relevant files"""
    files_to_update = [
        ("pyproject.toml", r'version = "[^"]+"', f'version = "{new_version}"'),
        (
            "src/music21_mcp/__init__.py",
            r'__version__ = "[^"]+"',
            f'__version__ = "{new_version}"',
        ),
    ]

    for file_path, pattern, replacement in files_to_update:
        path = Path(file_path)
        if path.exists():
            content = path.read_text()
            updated = re.sub(pattern, replacement, content)
            path.write_text(updated)
            print(f"Updated version in {file_path}")


def run_tests():
    """Run all tests"""
    print("\n📋 Running tests...")
    result = run_command("python -m pytest tests/ -v", check=False)

    if result.returncode != 0:
        print("⚠️  Some tests failed. Continue anyway? (y/N)")
        if input().lower() != "y":
            sys.exit(1)


def check_code_quality():
    """Run code quality checks"""
    print("\n🔍 Checking code quality...")

    # Run ruff
    result = run_command("ruff check src/", check=False)
    if result.returncode != 0:
        print("⚠️  Ruff found issues. Auto-fix? (Y/n)")
        if input().lower() != "n":
            run_command("ruff check --fix src/")

    # Run mypy
    result = run_command("mypy src/", check=False)
    if result.returncode != 0:
        print("⚠️  Type checking failed. Continue anyway? (y/N)")
        if input().lower() != "y":
            sys.exit(1)


def build_package():
    """Build the distribution package"""
    print("\n📦 Building package...")

    # Clean previous builds
    run_command("rm -rf dist/ build/ *.egg-info", check=False)

    # Build with poetry
    run_command("poetry build")

    # Check the built package
    run_command("twine check dist/*")


def validate_package():
    """Validate the package can be installed"""
    print("\n✅ Validating package...")

    # Create a temporary virtual environment
    run_command("python -m venv test_env", check=False)

    # Install the package
    wheel_file = list(Path("dist").glob("*.whl"))[0]
    result = run_command(f"test_env/bin/pip install {wheel_file}", check=False)

    if result.returncode == 0:
        # Test import
        result = run_command(
            'test_env/bin/python -c "import music21_mcp; print(music21_mcp.__version__)"',
            check=False,
        )

        if result.returncode == 0:
            print(f"✅ Package installed successfully: {result.stdout.strip()}")
        else:
            print("❌ Package import failed!")

    # Cleanup
    run_command("rm -rf test_env", check=False)


def create_release_notes(version: str):
    """Create release notes template"""
    print("\n📝 Creating release notes...")

    template = f"""# Release v{version}

## 🎉 Highlights
- 

## ✨ New Features
- 

## 🐛 Bug Fixes
- 

## 🔧 Improvements
- 

## 📚 Documentation
- 

## 🔄 Breaking Changes
- None

## 📦 Installation
```bash
pip install music21-mcp-server=={version}
```

## 🙏 Contributors
Thank you to all contributors!
"""

    notes_path = Path(f"RELEASE_NOTES_v{version}.md")
    notes_path.write_text(template)
    print(f"Created {notes_path}")
    print("Please edit the release notes before creating the GitHub release.")


def main():
    """Main release preparation workflow"""
    print("🚀 Music21 MCP Server Release Preparation")
    print("=" * 50)

    # Get current version
    current_version = get_current_version()
    print(f"Current version: {current_version}")

    # Ask for new version
    print("\nEnter new version (or press Enter to keep current):")
    new_version = input().strip() or current_version

    if new_version != current_version:
        update_version(new_version)
        print(f"Updated version to {new_version}")

    # Run checks
    run_tests()
    check_code_quality()

    # Build package
    build_package()
    validate_package()

    # Create release notes
    create_release_notes(new_version)

    print("\n" + "=" * 50)
    print("✅ Release preparation complete!")
    print("\nNext steps:")
    print("1. Review and edit the release notes")
    print(
        "2. Commit changes: git add -A && git commit -m 'Release v{}'".format(
            new_version
        )
    )
    print("3. Create tag: git tag v{}".format(new_version))
    print("4. Push: git push && git push --tags")
    print("5. Create GitHub release with the release notes")
    print("\nThe GitHub Actions workflow will automatically publish to PyPI.")


if __name__ == "__main__":
    main()
