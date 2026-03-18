# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.0] - 2026-03-14

### Added
- GitHub branch protection rules for main branch
- Discord webhook integration for CI/CD notifications
- Comprehensive webhook documentation and setup scripts
- Test coverage increased to 84%+
- Advanced CI/CD monitoring capabilities
- Webhook test scripts for validation
- Lazy adapter loading to prevent startup crashes from missing optional deps
- `python -m music21_mcp` entry point
- PEP 621 `[project.scripts]` entry points

### Fixed
- Fixed 54 test failures in unit tests
- Resolved import errors across test modules
- Fixed linting issues (single quotes to double quotes)
- Fixed variable naming convention violations
- Corrected import sorting and formatting issues
- Fixed MemoryManager constructor parameters
- Fixed ParallelProcessor.process_batch() parameters
- Moved 5 standalone verification scripts out of tests/ (were failing as pytest tests)
- Fixed mypy error from redundant `# type: ignore` on aiofiles import
- Fixed music21 `.flat` deprecation warnings (replaced with `.flatten()`)
- Removed deprecated `event_loop` pytest fixture (handled by pytest-asyncio config)
- Fixed 18 ruff violations (E722, F841, B904)

### Changed
- Updated test modules to use get_logger() instead of direct logger import
- Improved test assertions for tool attributes
- Enhanced CI/CD pipeline stability
- Refactored observability module imports
- Enabled stricter ruff rules: E722 (bare except), F841 (unused variable), B904 (raise without from)
- Removed redundant `starlette` dependency (transitive from fastapi)
- Deduplicated `[tool.poetry]` metadata with `[project]` (PEP 621)
- Fixed dependabot Docker path and added fastmcp ignore rule

### Security
- Implemented webhook secret management
- Added branch protection to prevent force pushes
- Required status checks before merging
- Enforced administrator restrictions

## [0.9.0] - 2025-08-25

### Added
- Multi-interface architecture (MCP, HTTP API, CLI, Python Library)
- 13 comprehensive music analysis tools
- FastMCP integration for Model Context Protocol support
- HTTP API with FastAPI backend
- Command-line interface for automation
- Python library for direct programming access

### Features
- Import/Export support for MusicXML, MIDI, ABC, Lilypond
- Key analysis with multiple algorithms (Krumhansl, Aarden, Bellman-Budge)
- Harmony analysis with Roman numerals and chord progression detection
- Voice leading analysis with parallel motion detection
- Pattern recognition for melodic, rhythmic, and harmonic patterns
- Bach chorale and jazz style harmonization
- Species counterpoint generation (1-5)
- Style imitation and generation
- Score manipulation (transposition, time stretching, orchestration)

### Infrastructure
- GitHub Actions CI/CD pipeline
- Multi-version Python testing (3.10, 3.11, 3.12)
- Security scanning with bandit and pip-audit
- Performance benchmarks
- Release readiness checks
- Automated PyPI publication workflow

## [0.8.0] - 2025-08-24

### Initial Release
- Core music21 integration
- Basic MCP server implementation
- Initial test suite with 75% coverage
- Documentation framework
- Basic CI/CD setup

[Unreleased]: https://github.com/brightlikethelight/music21-mcp-server/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/brightlikethelight/music21-mcp-server/compare/v0.9.0...v1.0.0
[0.9.0]: https://github.com/brightlikethelight/music21-mcp-server/compare/v0.8.0...v0.9.0
[0.8.0]: https://github.com/brightlikethelight/music21-mcp-server/releases/tag/v0.8.0