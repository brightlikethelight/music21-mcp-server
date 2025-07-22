# Release Process Guide

This document outlines the comprehensive release process for music21-mcp-server, including PyPI publishing, GitHub releases, and version management.

## 🎯 Overview

The release process is fully automated using semantic-release and GitHub Actions, with multiple safety checks and testing stages:

1. **Semantic Release** - Automated versioning based on conventional commits
2. **PyPI Publishing** - Secure publishing to PyPI with verification
3. **GitHub Releases** - Automated release notes and asset uploads
4. **Test PyPI** - Pre-release testing environment

## 🚀 Release Types

### Automatic Releases (Recommended)

Releases are triggered automatically on every push to `main` branch using semantic-release:

- **Patch** (1.0.0 → 1.0.1): `fix:` commits
- **Minor** (1.0.0 → 1.1.0): `feat:` commits  
- **Major** (1.0.0 → 2.0.0): `BREAKING CHANGE:` in commit body

### Manual Releases

You can also trigger releases manually via GitHub Actions workflow dispatch.

## 📝 Commit Convention

Use [Conventional Commits](https://www.conventionalcommits.org/) for automatic version bumping:

```bash
# Patch release (bug fixes)
git commit -m "fix: resolve harmony analysis tool memory leak"

# Minor release (new features)
git commit -m "feat: add Jazz harmonization algorithm"

# Major release (breaking changes)
git commit -m "feat!: redesign MCP server interface

BREAKING CHANGE: The MCP server now uses FastMCP 3.0 which changes the tool registration format"
```

## 🔧 Setup Requirements

### 1. PyPI API Token

Add your PyPI API token to GitHub repository secrets:

1. Generate token at https://pypi.org/manage/account/token/
2. Add as repository secret: `PYPI_API_TOKEN`
3. Scope: "Entire account" or specific to this project

### 2. Test PyPI API Token (Optional)

For testing releases on Test PyPI:

1. Generate token at https://test.pypi.org/manage/account/token/
2. Add as repository secret: `TEST_PYPI_API_TOKEN`

### 3. GitHub Token

GitHub Actions automatically provides `GITHUB_TOKEN` - no setup needed.

## 🎛️ Release Workflows

### Main Release Workflow (`.github/workflows/release.yml`)

**Trigger**: Push to `main` branch or manual dispatch

**Steps**:
1. ✅ **Validation** - Check repository state, branch, changelog
2. 🔄 **Semantic Release** - Generate version, update files, create tag
3. 🏗️ **Build** - Create wheel and source distributions  
4. ✔️ **Verify** - Check distribution, test installation
5. 📦 **Publish** - Upload to PyPI with verification
6. 🏷️ **GitHub Release** - Create release with notes and assets
7. 📢 **Notify** - Success/failure notifications

### Test PyPI Workflow (`.github/workflows/test-pypi.yml`)

**Trigger**: Manual dispatch only

**Purpose**: Test releases before production deployment

**Usage**:
```bash
# Via GitHub web interface:
# Actions → Test PyPI Release → Run workflow
# - Version: 1.2.0  
# - Publish to Test PyPI: true
```

### Release Drafter (`.github/workflows/release-drafter.yml`)

**Trigger**: Push/PR to `main` branch

**Purpose**: Automatically draft release notes based on PRs and commits

## 📋 Pre-Release Checklist

Before triggering a release, ensure:

- [ ] All tests pass in CI
- [ ] CHANGELOG.md is updated (semantic-release handles this)
- [ ] Version in pyproject.toml is correct (semantic-release handles this)
- [ ] No security vulnerabilities detected
- [ ] Documentation is up to date
- [ ] All breaking changes are documented

## 🧪 Testing a Release

### 1. Test PyPI Dry Run

```bash
# Trigger test workflow via GitHub UI or:
gh workflow run test-pypi.yml -f version=1.2.0-rc.1 -f publish_to_test_pypi=true
```

### 2. Local Testing

```bash
# Build locally
python -m build

# Check distribution
python -m twine check dist/*

# Test install in clean environment
python -m venv test_env
source test_env/bin/activate
pip install dist/*.whl
music21-analysis --help
deactivate && rm -rf test_env
```

## 📦 Post-Release Verification

After a successful release:

1. **Verify PyPI**: Check https://pypi.org/project/music21-mcp-server/
2. **Test Installation**: `pip install music21-mcp-server`
3. **GitHub Release**: Verify https://github.com/brightlikethelight/music21-mcp-server/releases
4. **Documentation**: Update if necessary

## 🔍 Troubleshooting

### Common Issues

**Release not created after push to main**:
- Check commit messages follow conventional format
- Verify no `[skip ci]` in commit messages
- Check workflow logs for errors

**PyPI upload fails**:
- Verify `PYPI_API_TOKEN` secret is set correctly
- Check token permissions and expiry
- Ensure version doesn't already exist on PyPI

**Distribution build fails**:
- Check pyproject.toml syntax: `python -c "import tomllib; tomllib.load(open('pyproject.toml', 'rb'))"`
- Verify all source files are included
- Check dependencies are installable

**Test failures**:
- Run tests locally: `pytest tests/ -v`
- Check test coverage: `pytest tests/ --cov=src/music21_mcp --cov-report=term-missing`
- Verify all imports work: `python -c "import music21_mcp; print(music21_mcp.__version__)"`

### Debug Commands

```bash
# Check semantic-release dry run
semantic-release version --noop

# Validate build
python -m build --wheel --sdist .
python -m twine check dist/*

# Test package contents
python -c "
import zipfile
with zipfile.ZipFile('dist/music21_mcp_server-1.0.0-py3-none-any.whl', 'r') as z:
    print(z.namelist())
"
```

## 🔒 Security Considerations

- API tokens are stored as repository secrets (encrypted)
- Workflows only run from main branch on original repository
- All uploads are verified before publishing
- Security scanning runs on every build
- Dependencies are audited with pip-audit and safety

## 📊 Release Metrics

Track release success with:

- **Build Time**: Monitor CI/CD performance
- **Test Coverage**: Maintain 80%+ coverage
- **Security**: Zero high-severity vulnerabilities
- **Download Stats**: Monitor PyPI download statistics
- **Issue Reports**: Track post-release issues

## 🤝 Contributing to Releases

1. **Use conventional commits** for automatic versioning
2. **Update CHANGELOG.md** for manual entries (optional)
3. **Add tests** for all new features
4. **Update documentation** for breaking changes
5. **Review security** implications

## 📚 References

- [Semantic Release](https://python-semantic-release.readthedocs.io/)
- [Conventional Commits](https://www.conventionalcommits.org/)
- [PyPI Publishing](https://packaging.python.org/en/latest/guides/publishing-package-distribution-releases-using-github-actions-ci-cd-workflows/)
- [GitHub Actions](https://docs.github.com/en/actions)