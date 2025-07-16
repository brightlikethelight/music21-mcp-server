# Contributing to Music21 MCP Server

Thank you for your interest in contributing to the Music21 MCP Server! This document provides guidelines for contributing to this production-ready Model Context Protocol server.

## 🎯 **How to Contribute**

We welcome contributions in the following areas:

### **Code Contributions**
- 🐛 **Bug fixes** - Help us improve reliability
- ✨ **New features** - Expand music analysis capabilities  
- ⚡ **Performance improvements** - Optimize algorithms and processing
- 🧪 **Test coverage** - Add tests for existing functionality
- 📚 **Documentation** - Improve guides and API documentation

### **Music Theory Contributions**
- 🎼 **New analysis algorithms** - Implement additional music theory concepts
- 🎵 **Style imitation models** - Add support for new musical styles
- 🎹 **Counterpoint rules** - Enhance species counterpoint generation
- 🎸 **Instrument support** - Extend capabilities for specific instruments

### **Infrastructure Improvements**
- 🔧 **CI/CD enhancements** - Improve automated testing and deployment
- 🐳 **Docker optimizations** - Better containerization and orchestration
- 📊 **Monitoring tools** - Add metrics and observability features
- 🔐 **Security hardening** - Enhance authentication and authorization

---

## 🚀 **Getting Started**

### **Prerequisites**
- Python 3.10 or higher
- Git for version control
- Basic knowledge of music theory (helpful but not required)
- Familiarity with MCP (Model Context Protocol)

### **Development Setup**

1. **Fork and Clone**
   ```bash
   git clone https://github.com/yourusername/music21-mcp-server.git
   cd music21-mcp-server
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -e ".[dev]"
   ```

4. **Install Pre-commit Hooks**
   ```bash
   pre-commit install
   ```

5. **Run Tests**
   ```bash
   pytest tests/ -v
   ```

6. **Start Development Server**
   ```bash
   python -m music21_mcp.server --dev
   ```

---

## 📋 **Development Guidelines**

### **Code Style**
- **Formatting**: Use `black` for code formatting
- **Imports**: Organize with `isort`
- **Linting**: Follow `flake8` and `pylint` recommendations
- **Type Hints**: Add type annotations for all functions
- **Docstrings**: Use Google-style docstrings

### **Testing Standards**
- **Coverage**: Maintain >95% test coverage
- **Unit Tests**: Test individual functions and classes
- **Integration Tests**: Test MCP protocol interactions
- **Performance Tests**: Benchmark music analysis operations
- **Music Theory Tests**: Validate algorithmic correctness

### **Git Workflow**
1. **Create Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make Changes**
   - Write code with appropriate tests
   - Follow code style guidelines
   - Update documentation as needed

3. **Commit Changes**
   ```bash
   git add .
   git commit -m "feat: add new harmony analysis algorithm"
   ```

4. **Push and Create PR**
   ```bash
   git push origin feature/your-feature-name
   ```

### **Commit Message Convention**
Use [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add new counterpoint generation algorithm
fix: resolve memory leak in score caching
docs: update API documentation for key analysis
test: add integration tests for harmony tools
perf: optimize chord progression analysis
refactor: improve error handling in auth module
```

---

## 🎼 **Music Theory Guidelines**

### **Algorithm Implementation**
- **Accuracy**: Ensure theoretical correctness
- **Performance**: Optimize for real-time processing
- **Flexibility**: Support various musical styles and periods
- **Documentation**: Explain theoretical basis and limitations

### **Testing Music Algorithms**
- **Known Examples**: Test against textbook examples
- **Edge Cases**: Handle unusual harmonic progressions
- **Style Variations**: Test across different musical periods
- **Performance**: Benchmark against large scores

### **Adding New Analysis Tools**
1. **Research**: Study relevant music theory literature
2. **Design**: Plan algorithm approach and API
3. **Implement**: Code with thorough error handling
4. **Test**: Validate with musical examples
5. **Document**: Explain usage and limitations

---

## 🧪 **Testing Your Changes**

### **Local Testing**
```bash
# Run all tests
pytest tests/ -v --cov=music21_mcp

# Run specific test category
pytest tests/test_harmony.py -v
pytest tests/test_counterpoint.py -v
pytest tests/test_mcp_integration.py -v

# Test with real MCP client
python scripts/test_mcp_client.py
```

### **Music Theory Validation**
```bash
# Test harmony analysis
python -m music21_mcp.tools.harmony_analysis_tool examples/bach_chorale.xml

# Test counterpoint generation
python -m music21_mcp.tools.counterpoint_tool --species 1 --cantus_firmus examples/cf.xml

# Validate against known examples
python scripts/validate_music_theory.py
```

### **Performance Testing**
```bash
# Benchmark analysis operations
python scripts/benchmark_analysis.py

# Test with large scores
python scripts/test_large_scores.py
```

---

## 📚 **Documentation Standards**

### **Code Documentation**
- **Function Docstrings**: Describe parameters, returns, and music theory context
- **Class Documentation**: Explain purpose and usage patterns
- **Module Documentation**: Overview of functionality and examples

### **API Documentation**
- **MCP Tools**: Document all available tools and parameters
- **Examples**: Provide musical examples and expected outputs
- **Theory Background**: Explain relevant music theory concepts

### **User Guides**
- **Getting Started**: Installation and basic usage
- **Music Theory Guide**: Explanation of implemented algorithms
- **Advanced Usage**: Complex workflows and customization

---

## 🚨 **Issue Reporting**

### **Bug Reports**
Please include:
- **Environment**: Python version, OS, music21 version
- **Musical Input**: Score or example that triggers the bug
- **Expected Behavior**: What should happen theoretically
- **Actual Behavior**: What actually happens
- **Error Messages**: Full stack traces
- **Reproduction Steps**: Minimal example to reproduce

### **Feature Requests**
Please provide:
- **Music Theory Justification**: Why is this feature important?
- **Use Cases**: How would this be used in practice?
- **Proposed API**: How should the feature be exposed?
- **Implementation Ideas**: Suggestions for approach
- **References**: Academic sources or standards

---

## 📖 **Resources**

### **Music Theory References**
- [Tonal Harmony](https://www.amazon.com/Tonal-Harmony-Stefan-Kostka/dp/0078025141) by Kostka & Payne
- [Species Counterpoint](https://www.amazon.com/Species-Counterpoint-Peter-Schubert/dp/0195162803) by Peter Schubert
- [The Jazz Theory Book](https://www.amazon.com/Jazz-Theory-Book-Mark-Levine/dp/1883217040) by Mark Levine

### **Technical References**
- [music21 Documentation](https://web.mit.edu/music21/)
- [Model Context Protocol Specification](https://modelcontextprotocol.io/)
- [FastMCP Documentation](https://fastmcp.readthedocs.io/)

### **Development Tools**
- [Python Type Checking](https://mypy.readthedocs.io/)
- [Testing with pytest](https://docs.pytest.org/)
- [Code Formatting with black](https://black.readthedocs.io/)

---

## 🏆 **Recognition**

Contributors will be:
- **Listed in CONTRIBUTORS.md** with their contributions
- **Mentioned in release notes** for significant features
- **Credited in academic papers** if applicable
- **Invited to maintainer role** for sustained contributions

---

## 📞 **Getting Help**

- **Discussions**: Use GitHub Discussions for questions
- **Issues**: Create GitHub Issues for bugs and features
- **Email**: Contact [brightliu@college.harvard.edu](mailto:brightliu@college.harvard.edu) for urgent matters
- **Music Theory Questions**: Feel free to ask about theoretical aspects

---

## 📜 **Code of Conduct**

We are committed to providing a welcoming and inclusive environment for all contributors. Please:

- **Be respectful** of different viewpoints and experiences
- **Use welcoming language** and be patient with newcomers
- **Focus on collaboration** and constructive feedback
- **Respect music theory traditions** while encouraging innovation
- **Give credit** where credit is due

---

## 🎉 **Thank You!**

Your contributions help make music theory more accessible through AI tooling. Whether you're fixing a small bug, adding a major feature, or improving documentation, every contribution matters!

For questions or suggestions about this contributing guide, please open an issue or discussion.

**Happy coding and music making!** 🎵✨