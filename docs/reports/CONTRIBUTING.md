# Contributing to Music21 MCP Server

Thank you for your interest in contributing to the Music21 MCP Server! This document provides guidelines for contributing to the project.

## How to Contribute

### Reporting Issues
1. Check if the issue already exists
2. Create a new issue with a clear title and description
3. Include steps to reproduce the problem
4. Mention your Python version and OS

### Submitting Pull Requests
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Run tests locally (`pytest tests/`)
6. Commit your changes (`git commit -m 'feat: Add amazing feature'`)
7. Push to your branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

### Code Style
- Follow PEP 8 guidelines
- Use type hints where appropriate
- Add docstrings to all functions and classes
- Keep functions focused and under 50 lines
- Use meaningful variable names

### Testing
- Write tests for all new features
- Ensure all tests pass before submitting PR
- Aim for high test coverage
- Include both unit and integration tests

### Documentation
- Update README.md if adding new features
- Add docstrings to new functions
- Update examples if relevant
- Keep documentation clear and concise

## Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/music21-mcp-server.git
   cd music21-mcp-server
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

4. Run tests:
   ```bash
   pytest tests/
   ```

## Commit Message Guidelines

Follow conventional commits:
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `style:` Code style changes
- `refactor:` Code refactoring
- `test:` Test additions/changes
- `chore:` Maintenance tasks

## Questions?

Feel free to open an issue for any questions about contributing!