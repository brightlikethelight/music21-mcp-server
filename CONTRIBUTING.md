# Contributing to Music21 MCP Server

We love your input! We want to make contributing to Music21 MCP Server as easy and transparent as possible, whether it's:

- Reporting a bug
- Discussing the current state of the code
- Submitting a fix
- Proposing new features
- Becoming a maintainer

## Development Process

We use GitHub to host code, to track issues and feature requests, as well as accept pull requests.

### Branch Protection

Our `main` branch is protected with the following rules:
- ✅ Pull request reviews required (minimum 1 approval)
- ✅ Status checks must pass before merging
- ✅ Branches must be up to date before merging
- ✅ Administrators included in restrictions
- ❌ Force pushes disabled
- ❌ Branch deletion disabled

## Development Setup

### Prerequisites

- Python 3.10 or higher
- Git
- GitHub CLI (optional but recommended)

### Initial Setup

1. **Fork the Repository**
   ```bash
   gh repo fork brightlikethelight/music21-mcp-server --clone
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -e ".[dev]"
   # or using uv
   uv pip install -e ".[dev]"
   ```

4. **Install Pre-commit Hooks**
   ```bash
   pre-commit install
   ```

## Code Style Guidelines

We use automated tools to ensure consistent code style:

### Formatting
- **Tool**: Ruff formatter
- **Command**: `uv run ruff format .`
- **Auto-fix**: `uv run ruff check --fix`

### Linting
- **Tool**: Ruff linter
- **Command**: `uv run ruff check .`
- **Configuration**: See `pyproject.toml`

### Type Checking
- **Tool**: MyPy
- **Command**: `uv run mypy .`
- **Strict mode**: Enabled for new code

### Import Sorting
- Use grouped imports
- Order: standard library, third-party, local
- Example:
  ```python
  # Standard library
  import os
  from pathlib import Path
  
  # Third-party
  import music21
  from fastmcp import FastMCP
  
  # Local
  from music21_mcp.tools import ImportScoreTool
  ```

## Testing Requirements

### Coverage Requirements
- **Minimum Coverage**: 80%
- **Target Coverage**: >80%

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src/music21_mcp --cov-report=term-missing

# Run with coverage threshold (will fail if below 80%)
pytest tests/ --cov=src/music21_mcp --cov-fail-under=74

# Run specific test file
pytest tests/unit/test_tools_unit.py

# Run with verbose output
pytest tests/ -v
```

### Writing Tests

1. **Test Location**: Place tests in `tests/unit/` or `tests/integration/`
2. **Naming**: Use descriptive names like `test_import_musicxml_success()`
3. **Fixtures**: Use pytest fixtures for common setup
4. **Mocking**: Mock external dependencies appropriately
5. **Coverage**: Ensure new features have tests

Example test:
```python
import pytest
from music21_mcp.tools import ImportScoreTool

@pytest.fixture
def import_tool():
    """Create ImportScoreTool instance for testing"""
    return ImportScoreTool()

@pytest.mark.asyncio
async def test_import_musicxml_success(import_tool, tmp_path):
    """Test successful MusicXML import"""
    # Create test file
    test_file = tmp_path / "test.xml"
    test_file.write_text("<score>...</score>")
    
    # Execute import
    result = await import_tool.execute(file_path=str(test_file))
    
    # Assertions
    assert result["status"] == "success"
    assert "score_id" in result
```

## Pull Request Process

### 1. Create Feature Branch
```bash
git checkout -b feature/amazing-feature
# or
git checkout -b fix/issue-123
```

### 2. Make Your Changes
- Write code following style guidelines
- Add tests for new functionality
- Update documentation if needed

### 3. Commit Your Changes
We use [Conventional Commits](https://www.conventionalcommits.org/):

```bash
# Features
git commit -m "feat: Add melodic pattern analysis"

# Bug fixes
git commit -m "fix: Resolve MIDI export encoding issue"

# Documentation
git commit -m "docs: Update API documentation"

# Style changes
git commit -m "style: Format code with ruff"

# Refactoring
git commit -m "refactor: Simplify harmony analysis logic"

# Tests
git commit -m "test: Add tests for voice leading"

# Chores
git commit -m "chore: Update dependencies"
```

### 4. Run Pre-commit Checks
```bash
# Run linting
uv run ruff check .

# Run formatting
uv run ruff format .

# Run type checking
uv run mypy .

# Run tests with coverage
pytest tests/ --cov=src/music21_mcp --cov-fail-under=74
```

### 5. Push Your Branch
```bash
git push origin feature/amazing-feature
```

### 6. Create Pull Request
```bash
gh pr create --title "feat: Add amazing feature" --body "Description of changes"
```

Or create via GitHub UI with:
- Clear title following conventional commit format
- Detailed description of changes
- Link to related issues
- Screenshots/examples if applicable

## Required CI/CD Checks

Your PR must pass all automated checks:

1. **Lint and Type Check**
   - Ruff linting passes
   - Ruff formatting check passes
   - MyPy type checking passes

2. **Tests**
   - All unit tests pass
   - All integration tests pass
   - Coverage remains above 80%

3. **Security Scan**
   - Bandit security scan passes
   - pip-audit vulnerability check passes

4. **Documentation**
   - Documentation builds successfully

## Code Review Guidelines

### For Contributors
- Respond to review feedback promptly
- Push new commits (don't force push)
- Mark conversations as resolved when addressed
- Request re-review when ready

### For Reviewers
- Review promptly (within 48 hours if possible)
- Provide constructive feedback
- Approve when satisfied
- Use "Request changes" sparingly

## Reporting Issues

### Bug Reports
Use the bug report template and include:
- Clear description of the issue
- Steps to reproduce
- Expected vs actual behavior
- Environment details (OS, Python version)
- Error messages/stack traces

### Feature Requests
Use the feature request template and include:
- Problem you're trying to solve
- Proposed solution
- Alternative solutions considered
- Additional context

## Discord Notifications

To get real-time updates on CI/CD pipeline status:

1. See [Webhook Setup Guide](.github/webhook-config.md)
2. Run setup script: `./scripts/setup-webhook.sh`
3. Test webhook: `./scripts/test-webhook.sh`

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Questions?

Feel free to:
- Open an issue for questions
- Contact the maintainer: brightliu@college.harvard.edu
- Check existing issues and pull requests

## Recognition

Contributors will be recognized in:
- Release notes
- Project documentation

Thank you for contributing to Music21 MCP Server! 🎵