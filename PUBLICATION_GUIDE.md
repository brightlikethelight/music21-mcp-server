# PyPI Publication Guide for music21-mcp-server

This guide documents the complete process for publishing music21-mcp-server to PyPI, enabling users to install it with `pip install music21-mcp-server`.

## 📋 Pre-Publication Checklist

### 1. Version Management
- [ ] Update version in `pyproject.toml` (both `[project]` and `[tool.poetry]` sections)
- [ ] Follow semantic versioning (MAJOR.MINOR.PATCH)
- [ ] Update `CHANGELOG.md` with new version details
- [ ] Ensure version matches across all configuration files

### 2. Quality Assurance
- [ ] Run full test suite: `pytest tests/`
- [ ] Check code coverage: `pytest --cov=src/music21_mcp`
- [ ] Run linting: `ruff check src/`
- [ ] Run security scan: `bandit -r src/`
- [ ] Run dependency audit: `pip-audit`

### 3. Documentation Updates
- [ ] Update README.md with any new features
- [ ] Verify installation instructions work
- [ ] Update API documentation if needed
- [ ] Check all example code still works

### 4. Build and Validation
- [ ] Clean previous builds: `rm -rf dist/`
- [ ] Build packages: `python -m build`
- [ ] Validate packages: `python -m twine check dist/*`
- [ ] Test local installation: `pip install dist/*.whl`

## 🔧 Build Process

### Required Tools
```bash
# Install/upgrade build tools
pip install --upgrade build twine
```

### Build Commands
```bash
# Clean previous builds
rm -rf dist/ build/ *.egg-info/

# Build both wheel and source distribution
python -m build

# Validate packages
python -m twine check dist/*
```

### Expected Output
```
Successfully built music21_mcp_server-X.Y.Z.tar.gz and music21_mcp_server-X.Y.Z-py3-none-any.whl
Checking dist/music21_mcp_server-X.Y.Z-py3-none-any.whl: PASSED
Checking dist/music21_mcp_server-X.Y.Z.tar.gz: PASSED
```

## 🚀 Publication Process

### Method 1: Using the Publication Script (Recommended)
```bash
# Use the automated script
./publish_to_pypi.sh
```

The script provides interactive options:
1. **Test PyPI** (recommended first) - Safe testing environment
2. **Production PyPI** - Public release
3. **Check packages only** - Validation without upload

### Method 2: Manual Publication

#### Step 1: Test PyPI (Recommended)
```bash
# Upload to Test PyPI first
python -m twine upload --repository testpypi dist/*

# Test installation from Test PyPI
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ music21-mcp-server
```

#### Step 2: Production PyPI
```bash
# Upload to production PyPI
python -m twine upload dist/*
```

## 🔐 Authentication Setup

### PyPI Account Setup
1. Create accounts at:
   - Test PyPI: https://test.pypi.org/account/register/
   - Production PyPI: https://pypi.org/account/register/

2. Generate API tokens:
   - Test PyPI: https://test.pypi.org/manage/account/token/
   - Production PyPI: https://pypi.org/manage/account/token/

### Configure Authentication

Create `~/.pypirc`:
```ini
[testpypi]
  username = __token__
  password = <your-test-pypi-token>

[pypi]
  username = __token__
  password = <your-production-pypi-token>
```

Or use environment variables:
```bash
export TWINE_USERNAME=__token__
export TWINE_PASSWORD=<your-token>
```

## 📦 Package Configuration

### pyproject.toml Key Sections

The package is configured in `pyproject.toml` with these critical sections:

```toml
[project]
name = "music21-mcp-server"
version = "1.0.0"  # Update this for new releases
description = "Professional multi-interface music analysis server built on music21"
readme = "README.md"
requires-python = ">=3.10"
license = {text = "MIT"}
authors = [{name = "Bright Liu", email = "brightliu@college.harvard.edu"}]

# Entry points for CLI tools
[tool.poetry.scripts]
music21-analysis = "music21_mcp.launcher:main"
music21-mcp = "music21_mcp.server_minimal:main"
music21-http = "music21_mcp.adapters.http_adapter:main"
music21-cli = "music21_mcp.adapters.cli_adapter:main"
```

### MANIFEST.in
The `MANIFEST.in` file controls which files are included in the source distribution:
- ✅ Includes: README, docs, examples, configuration files
- ❌ Excludes: Tests, build artifacts, development files

## 🧪 Testing Installation

### Test Package Installation
```bash
# Create fresh virtual environment
python -m venv test_env
source test_env/bin/activate  # Linux/Mac
# test_env\Scripts\activate     # Windows

# Install from Test PyPI
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ music21-mcp-server

# Test basic functionality
python -c "from music21_mcp import launcher; print('Import successful')"
music21-analysis --help
```

### Test Entry Points
```bash
# Test all CLI entry points
music21-mcp --help
music21-http --help
music21-cli --help
music21-analysis --help
```

## 📋 Post-Publication Checklist

### Immediate Tasks
- [ ] Verify package appears on PyPI: https://pypi.org/project/music21-mcp-server/
- [ ] Test `pip install music21-mcp-server` in fresh environment
- [ ] Update project README with installation instructions
- [ ] Create GitHub release with version tag

### Documentation Updates
- [ ] Update repository README with PyPI installation instructions
- [ ] Update examples to use pip-installed package
- [ ] Submit to MCP Registry (if applicable)
- [ ] Announce on relevant platforms

### Monitoring
- [ ] Monitor PyPI download statistics
- [ ] Watch for user issues and bug reports
- [ ] Track compatibility with dependency updates

## 🚨 Troubleshooting

### Common Issues

#### Build Failures
```bash
# ModuleNotFoundError during build
pip install poetry-core>=1.0.0

# Missing files in package
# Check MANIFEST.in includes needed files
```

#### Upload Failures
```bash
# 403 Forbidden
# Check API token is correct and has upload permissions

# 400 Bad Request - Version already exists
# Increment version number in pyproject.toml
```

#### Installation Issues
```bash
# Dependency conflicts
# Check dependency versions in pyproject.toml

# Entry points not working
# Verify [tool.poetry.scripts] section
```

### Version Conflicts
If a version already exists on PyPI:
1. Increment version in `pyproject.toml`
2. Rebuild packages: `python -m build`
3. Re-upload: `python -m twine upload dist/*`

### Recovery from Failed Upload
```bash
# Clean and rebuild
rm -rf dist/ build/ *.egg-info/
python -m build
python -m twine check dist/*
python -m twine upload dist/*
```

## 📚 Additional Resources

- [Python Packaging User Guide](https://packaging.python.org/)
- [PyPI Help](https://pypi.org/help/)
- [Twine Documentation](https://twine.readthedocs.io/)
- [Semantic Versioning](https://semver.org/)

## 🎯 Quick Reference Commands

```bash
# Complete publication workflow
rm -rf dist/
python -m build
python -m twine check dist/*
python -m twine upload --repository testpypi dist/*  # Test first
python -m twine upload dist/*                        # Production
```

## 🔄 Automated CI/CD (Future Enhancement)

Consider setting up GitHub Actions for automated publishing:
```yaml
# .github/workflows/publish.yml
name: Publish to PyPI
on:
  release:
    types: [published]
jobs:
  publish:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
      - run: pip install build twine
      - run: python -m build
      - run: python -m twine upload dist/*
        env:
          TWINE_USERNAME: __token__
          TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}
```

---

**Status**: Ready for PyPI publication! All components are properly configured and tested.