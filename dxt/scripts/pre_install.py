#!/usr/bin/env python3
"""
Pre-installation script for Music21 MCP Server Desktop Extension

This script:
1. Checks system requirements
2. Verifies Python version compatibility
3. Prepares the environment for installation
4. Downloads and installs Python dependencies
"""

import os
import sys
import subprocess
import platform
import json
import tempfile
import shutil
from pathlib import Path


def log(message: str, level: str = "INFO"):
    """Simple logging function"""
    print(f"[{level}] {message}")


def check_python_version():
    """Check if Python version meets requirements"""
    version = sys.version_info
    if version.major != 3 or version.minor < 10:
        log(f"Python 3.10+ required, found {version.major}.{version.minor}", "ERROR")
        return False
    log(
        f"Python version {version.major}.{version.minor}.{version.micro} is compatible",
        "INFO",
    )
    return True


def check_disk_space(required_mb=500):
    """Check available disk space"""
    try:
        statvfs = os.statvfs(".")
        free_bytes = statvfs.f_frsize * statvfs.f_bavail
        free_mb = free_bytes / (1024 * 1024)

        if free_mb < required_mb:
            log(
                f"Insufficient disk space: {free_mb:.1f}MB available, {required_mb}MB required",
                "ERROR",
            )
            return False

        log(f"Disk space check passed: {free_mb:.1f}MB available", "INFO")
        return True
    except Exception as e:
        log(f"Could not check disk space: {e}", "WARNING")
        return True  # Don't fail installation for this


def install_pip_if_missing():
    """Ensure pip is available"""
    try:
        import pip

        log("pip is available", "INFO")
        return True
    except ImportError:
        log("pip not found, attempting to install...", "WARNING")
        try:
            # Try to install pip using ensurepip
            subprocess.check_call([sys.executable, "-m", "ensurepip", "--upgrade"])
            log("pip installed successfully", "INFO")
            return True
        except subprocess.CalledProcessError as e:
            log(f"Failed to install pip: {e}", "ERROR")
            return False


def create_virtual_environment(extension_dir: Path):
    """Create a virtual environment for the extension"""
    venv_path = extension_dir / "venv"

    if venv_path.exists():
        log("Virtual environment already exists, removing old one...", "INFO")
        shutil.rmtree(venv_path)

    try:
        log("Creating virtual environment...", "INFO")
        subprocess.check_call([sys.executable, "-m", "venv", str(venv_path)])

        # Determine the correct python executable path
        if platform.system() == "Windows":
            python_exe = venv_path / "Scripts" / "python.exe"
            pip_exe = venv_path / "Scripts" / "pip.exe"
        else:
            python_exe = venv_path / "bin" / "python"
            pip_exe = venv_path / "bin" / "pip"

        if not python_exe.exists():
            raise FileNotFoundError(f"Python executable not found at {python_exe}")

        # Upgrade pip in the virtual environment
        log("Upgrading pip in virtual environment...", "INFO")
        subprocess.check_call(
            [str(python_exe), "-m", "pip", "install", "--upgrade", "pip"]
        )

        log(f"Virtual environment created at {venv_path}", "INFO")
        return venv_path, python_exe, pip_exe

    except subprocess.CalledProcessError as e:
        log(f"Failed to create virtual environment: {e}", "ERROR")
        raise
    except Exception as e:
        log(f"Unexpected error creating virtual environment: {e}", "ERROR")
        raise


def install_dependencies(pip_exe: Path, requirements_file: Path):
    """Install Python dependencies"""
    if not requirements_file.exists():
        log(f"Requirements file not found: {requirements_file}", "ERROR")
        return False

    try:
        log("Installing Python dependencies...", "INFO")
        cmd = [str(pip_exe), "install", "-r", str(requirements_file)]

        # Run with real-time output
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
        )

        for line in iter(process.stdout.readline, ""):
            print(f"    {line.rstrip()}")

        process.wait()

        if process.returncode != 0:
            log(
                f"Failed to install dependencies (exit code: {process.returncode})",
                "ERROR",
            )
            return False

        log("Dependencies installed successfully", "INFO")
        return True

    except Exception as e:
        log(f"Error installing dependencies: {e}", "ERROR")
        return False


def configure_music21():
    """Configure music21 library"""
    try:
        log("Configuring music21...", "INFO")

        # Import music21 to trigger configuration
        subprocess.check_call(
            [sys.executable, "-c", "import music21; music21.configure.run()"]
        )

        log("music21 configured successfully", "INFO")
        return True

    except subprocess.CalledProcessError as e:
        log(f"Failed to configure music21: {e}", "WARNING")
        # Don't fail installation for this
        return True
    except Exception as e:
        log(f"Unexpected error configuring music21: {e}", "WARNING")
        return True


def main():
    """Main pre-installation process"""
    log("Starting Music21 MCP Server pre-installation...", "INFO")

    # Get extension directory
    extension_dir = Path(__file__).parent.parent.absolute()
    log(f"Extension directory: {extension_dir}", "INFO")

    # Check system requirements
    if not check_python_version():
        sys.exit(1)

    if not check_disk_space():
        sys.exit(1)

    if not install_pip_if_missing():
        sys.exit(1)

    try:
        # Create virtual environment
        venv_path, python_exe, pip_exe = create_virtual_environment(extension_dir)

        # Install dependencies
        requirements_file = extension_dir / "requirements.txt"
        if not install_dependencies(pip_exe, requirements_file):
            sys.exit(1)

        # Configure music21
        configure_music21()

        # Save environment info for post-install script
        env_info = {
            "venv_path": str(venv_path),
            "python_exe": str(python_exe),
            "pip_exe": str(pip_exe),
            "extension_dir": str(extension_dir),
        }

        env_file = extension_dir / "env_info.json"
        with open(env_file, "w") as f:
            json.dump(env_info, f, indent=2)

        log("Pre-installation completed successfully!", "INFO")

    except Exception as e:
        log(f"Pre-installation failed: {e}", "ERROR")
        sys.exit(1)


if __name__ == "__main__":
    main()
