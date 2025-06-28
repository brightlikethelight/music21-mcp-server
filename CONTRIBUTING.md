# Contributing to Music21 MCP Server

Thank you for your interest in contributing to the Music21 MCP Server! This guide will help you get started with contributing to our simplified, stable music analysis server.

## 🎯 Project Philosophy

Our project prioritizes:

1. **Simplicity over Complexity** - Keep it simple and maintainable
2. **Reliability over Features** - 100% success rate on what we support
3. **Clear Documentation** - Everything should be well-documented
4. **Testing First** - All changes must include tests
5. **User Experience** - Easy to use and understand

## 🚀 Quick Start for Contributors

### Prerequisites

- Python 3.8 or higher
- Git
- Basic understanding of music theory (helpful but not required)
- Familiarity with the music21 library (we'll help you learn!)

### Development Setup

```bash
# 1. Fork the repository on GitHub
# 2. Clone your fork
git clone https://github.com/YOUR_USERNAME/music21-mcp-server.git
cd music21-mcp-server

# 3. Set up the development environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 4. Install dependencies
pip install -e .
pip install music21

# 5. Install development dependencies
pip install pytest black mypy flake8 pre-commit

# 6. Set up pre-commit hooks
pre-commit install

# 7. Run tests to ensure everything works
python tests/test_simplified_final.py
```

## 📋 How to Contribute

### Types of Contributions We Welcome

1. **Bug Fixes** - Fix issues in existing functionality
2. **Documentation Improvements** - Enhance docs, add examples
3. **Test Coverage** - Add more comprehensive tests
4. **Performance Improvements** - Optimize existing code
5. **Feature Requests** - Propose new features (simple ones first!)
6. **Examples** - Real-world usage examples

### What We're NOT Looking For (Yet)

- Complex new analysis features (wait for Phase 2)
- Breaking API changes
- Dependencies that aren't essential
- Features that compromise stability

## 🔄 Development Workflow

### 1. Create an Issue First

Before starting work:

1. Check if an issue already exists
2. If not, create a new issue describing:
   - What you want to add/fix
   - Why it's needed
   - How you plan to implement it
3. Wait for maintainer feedback before starting

### 2. Branch Naming Convention

```bash
# Bug fixes
git checkout -b fix/issue-123-chord-analysis-bug

# Features  
git checkout -b feature/add-tempo-detection

# Documentation
git checkout -b docs/improve-api-examples

# Tests
git checkout -b test/add-edge-case-coverage
```

### 3. Making Changes

#### Code Style Guidelines

We follow these standards:

```bash
# Format code with black
black src/ tests/

# Type checking with mypy
mypy src/

# Linting with flake8
flake8 src/ tests/

# All checks must pass before submitting
```

#### Code Quality Rules

1. **Keep it Simple**: If it's complex, it probably doesn't belong in the simplified version
2. **Add Tests**: Every change must include tests
3. **Document Everything**: Add docstrings and comments
4. **Error Handling**: Graceful error handling, no crashes
5. **Consistent API**: Follow existing patterns

#### Example: Adding a New Tool

```python
@mcp.tool()
async def my_new_tool(
    score_id: str,
    param1: str,
    param2: Optional[int] = None
) -> Dict[str, Any]:
    """
    Brief description of what this tool does.
    
    Args:
        score_id: ID of the score to analyze
        param1: Description of parameter
        param2: Optional parameter with default
    
    Returns:
        Dictionary with status and results
        
    Raises:
        None - returns error status instead of raising
    """
    try:
        # Check if score exists
        if score_id not in score_manager.scores:
            return {"status": "error", "message": f"Score '{score_id}' not found"}
        
        score = score_manager.scores[score_id]
        
        # Your implementation here
        result = do_something(score, param1, param2)
        
        return {
            "status": "success",
            "score_id": score_id,
            "result": result
        }
        
    except Exception as e:
        logger.error(f"My new tool error: {e}")
        return {"status": "error", "message": str(e)}
```

### 4. Testing Requirements

#### Test Types Required

1. **Unit Tests**: Test individual functions
2. **Integration Tests**: Test tool interactions
3. **Edge Case Tests**: Test error conditions
4. **Example Tests**: Test documentation examples

#### Test Template

```python
async def test_my_new_tool():
    """Test my_new_tool with various inputs"""
    # Setup
    await import_score("test_score", "bach/bwv66.6")
    
    # Test normal operation
    result = await my_new_tool("test_score", "param_value")
    assert result["status"] == "success"
    assert "result" in result
    
    # Test edge cases
    error_result = await my_new_tool("nonexistent", "param_value")
    assert error_result["status"] == "error"
    
    # Cleanup
    await delete_score("test_score")
```

#### Running Tests

```bash
# Run basic tests
python tests/test_simplified_final.py

# Run comprehensive tests
python tests/test_simplified_comprehensive.py

# Run specific test file
python -m pytest tests/test_my_new_feature.py -v

# Run with coverage
python -m pytest tests/ --cov=src/music21_mcp --cov-report=html
```

### 5. Documentation Requirements

Every contribution must include:

1. **Docstrings**: For all functions and classes
2. **API Documentation**: Update docs/simplified-api.md if needed
3. **Example Usage**: Include in docstring or examples/
4. **README Updates**: If adding major features

#### Documentation Template

```python
async def example_function(param1: str, param2: int = 5) -> Dict[str, Any]:
    """
    One-line description of what this function does.
    
    Longer description if needed. Explain the purpose,
    any important details, and how it fits into the system.
    
    Args:
        param1: Description of parameter with type info
        param2: Optional parameter, defaults to 5
    
    Returns:
        Dictionary containing:
        - status: "success" or "error"
        - result: The actual result data
        - message: Error message if status is "error"
    
    Example:
        >>> result = await example_function("test", 10)
        >>> print(result["status"])
        "success"
    """
```

## 🧪 Testing Guidelines

### Test Categories

1. **Core Functionality Tests** (`tests/test_core.py`)
   - Import, export, basic analysis
   - Must maintain 100% pass rate

2. **Edge Case Tests** (`tests/test_edge_cases.py`)
   - Error conditions
   - Invalid inputs
   - Empty/malformed data

3. **Integration Tests** (`tests/test_integration.py`)
   - Multiple tool workflows
   - Real-world scenarios

4. **Performance Tests** (`tests/test_performance.py`)
   - Speed benchmarks
   - Memory usage
   - Large file handling

### Test Writing Best Practices

```python
# Good test structure
async def test_feature_name():
    """Test description explaining what's being tested"""
    
    # Setup - create test data
    setup_data = create_test_score()
    
    # Execute - run the function
    result = await function_under_test(setup_data)
    
    # Assert - check results
    assert result["status"] == "success"
    assert result["expected_field"] == expected_value
    
    # Cleanup - remove test data
    cleanup_test_data()

# Test error conditions
async def test_feature_name_error_handling():
    """Test that errors are handled gracefully"""
    
    result = await function_under_test("invalid_input")
    assert result["status"] == "error"
    assert "message" in result
    assert len(result["message"]) > 0
```

## 📝 Pull Request Process

### Before Submitting

1. **Run All Tests**: Ensure everything passes
2. **Update Documentation**: Keep docs current
3. **Check Code Style**: Run black, mypy, flake8
4. **Test Examples**: Verify all examples still work
5. **Update CHANGELOG.md**: Add your changes

### PR Template

When submitting a PR, include:

```markdown
## Description
Brief description of what this PR does.

## Type of Change
- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Testing
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing unit tests pass locally with my changes
- [ ] I have tested this with real music files

## Documentation
- [ ] I have updated the documentation accordingly
- [ ] I have added examples for new features

## Checklist
- [ ] My code follows the style guidelines of this project
- [ ] I have performed a self-review of my own code
- [ ] My changes generate no new warnings
- [ ] Any dependent changes have been merged and published
```

### Review Process

1. **Automated Checks**: CI will run tests and code quality checks
2. **Maintainer Review**: A maintainer will review your code
3. **Feedback**: Address any requested changes
4. **Approval**: Once approved, your PR will be merged

## 🐛 Bug Reports

### How to Report Bugs

1. **Check Existing Issues**: Search for existing reports
2. **Use the Bug Template**: Fill out all sections
3. **Provide Examples**: Include code that reproduces the bug
4. **Include Environment**: Python version, OS, music21 version

### Bug Report Template

```markdown
## Bug Description
A clear and concise description of what the bug is.

## To Reproduce
Steps to reproduce the behavior:
1. Import score '...'
2. Call function '...'
3. See error

## Expected Behavior
What you expected to happen.

## Actual Behavior
What actually happened. Include error messages.

## Code Example
```python
# Minimal code that reproduces the issue
await import_score("test", "bach/bwv66.6")
result = await problematic_function("test")
```

## Environment
- OS: [e.g. macOS 12.0, Ubuntu 20.04, Windows 10]
- Python Version: [e.g. 3.9.7]
- music21 Version: [e.g. 8.1.0]
- Server Version: [e.g. v2.0]
```

## 💡 Feature Requests

### Before Requesting Features

1. **Check Roadmap**: See if it's already planned
2. **Consider Complexity**: Does it fit the simplified philosophy?
3. **Propose Gradually**: Start with simple version

### Feature Request Template

```markdown
## Feature Description
Clear description of the feature you'd like.

## Problem Solved
What problem does this solve? Who would use it?

## Proposed Solution
How should this work? Include API design.

## Alternatives Considered
Other ways to solve this problem.

## Implementation Notes
Any technical considerations or challenges.

## Complexity Assessment
- [ ] Simple (fits current philosophy)
- [ ] Medium (needs careful consideration)
- [ ] Complex (probably Phase 2+)
```

## 🏆 Recognition

### Contributors

We recognize contributors in several ways:

1. **GitHub Contributors List**: Automatic recognition
2. **CHANGELOG.md**: Major contributions listed
3. **Documentation Credits**: Listed in relevant docs
4. **Special Thanks**: In README for significant contributions

### Contribution Types We Value

- **Code Contributors**: Bug fixes, features, optimizations
- **Documentation Contributors**: Docs, examples, tutorials
- **Testing Contributors**: Better test coverage, edge cases
- **Community Contributors**: Issues, discussions, feedback
- **Design Contributors**: API design, architecture suggestions

## 📞 Getting Help

### Where to Ask Questions

1. **GitHub Discussions**: General questions and discussions
2. **GitHub Issues**: Bug reports and feature requests
3. **Code Comments**: Implementation questions
4. **Documentation**: Check docs first!

### Maintainer Response Time

- **Critical Bugs**: Within 24 hours
- **General Issues**: Within 1 week
- **Feature Requests**: Within 2 weeks
- **PRs**: Within 1 week for initial review

## 📜 License

By contributing to this project, you agree that your contributions will be licensed under the MIT License.

## 🙏 Thank You!

Every contribution, no matter how small, helps make this project better. Whether you're fixing a typo, adding a test, or implementing a new feature, your help is appreciated!

Remember: **Simple, Stable, and Well-Tested** is our motto. Let's build something amazing together! 🎵