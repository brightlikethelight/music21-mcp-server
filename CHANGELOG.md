# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial release of music21-mcp-server
- 13 comprehensive music analysis tools
- Multiple interface support: MCP Server, HTTP API, CLI, Python Library
- Protocol-independent architecture for maximum flexibility
- Comprehensive test suite with pytest
- CI/CD pipeline with GitHub Actions
- Security scanning with bandit and pip-audit
- Type checking with mypy
- Code formatting with Ruff

### Changed
- Removed all CI/CD hacks and "reality-based testing"
- Updated coverage requirement to 80% minimum
- Fixed test infrastructure for proper assertions

### Fixed
- Response format consistency across all tools
- Test parameter mismatches
- Import dependencies and compatibility issues

### Security
- Implemented security scanning that actually fails on vulnerabilities
- Added safety vulnerability scanner

## [1.0.0] - TBD

Initial public release.

[Unreleased]: https://github.com/brightlikethelight/music21-mcp-server/compare/v1.0.0...HEAD