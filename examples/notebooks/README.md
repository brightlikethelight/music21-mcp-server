# 📚 Interactive Jupyter Notebooks

Welcome to the Music21 MCP Server interactive tutorials! These hands-on notebooks get you from zero to expert in minutes.

## 🚀 Quick Start

### 1. Install Requirements
```bash
pip install music21-mcp-server jupyter
```

### 2. Launch Jupyter
```bash
cd examples/notebooks
jupyter notebook
```

### 3. Open the Tutorial
Click on `quickstart_tutorial.ipynb` and follow along!

## 📖 Available Notebooks

### 🎵 [`quickstart_tutorial.ipynb`](./quickstart_tutorial.ipynb) - **START HERE**
**⏱️ 5 minutes | 🎯 Beginner**

Get productive immediately! This tutorial covers:
- ✅ Import and analyze Bach chorales
- ✅ Explore harmony and voice leading  
- ✅ Generate musical variations
- ✅ Monitor system performance

Perfect for your first experience with the system.

### 🎼 `advanced_analysis.ipynb` *(Coming Soon)*
**⏱️ 15 minutes | 🎯 Intermediate**

Deep dive into advanced features:
- Pattern recognition and motif analysis
- Cross-compositional style comparison
- Custom analysis pipelines
- Performance optimization

### 🤖 `ai_integration.ipynb` *(Coming Soon)*
**⏱️ 10 minutes | 🎯 Advanced**

Integration with AI systems:
- Claude Desktop MCP setup
- Automated composition analysis
- AI-powered music generation
- Research workflow automation

## 🛠️ Setup Tips

### First Time Setup
```bash
# Install with all dependencies
pip install music21-mcp-server[audio,visualization]

# Configure music21 (one-time setup)
python -c "import music21; music21.configure.run()"
```

### Quick Health Check
```python
from music21_mcp.services import MusicAnalysisService
service = MusicAnalysisService()
print(f"✅ {len(service.get_available_tools())} tools ready!")
```

### Troubleshooting
- **Import errors**: Ensure `pip install music21-mcp-server`
- **Corpus not found**: Run `python -c "import music21; music21.configure.run()"`
- **Performance issues**: Check memory with `service.get_memory_usage()`

## 🎯 Learning Path

1. **Start Here**: `quickstart_tutorial.ipynb` (5 min)
2. **Explore**: Try different Bach chorales and composers
3. **Integrate**: Set up Claude Desktop MCP connection
4. **Build**: Create your own analysis workflows
5. **Share**: Contribute your notebooks back to the community

## 💡 Pro Tips

- **Keyboard shortcuts**: `Shift+Enter` to run cells
- **Quick restart**: `Kernel → Restart & Run All`
- **Save often**: `Ctrl+S` or `Cmd+S`
- **Get help**: `service.get_available_tools()` lists all functions

## 🔗 Related Resources

- [📖 Main Documentation](../../docs/)
- [🌐 Web API Examples](../web_integration/)
- [⚙️ Claude Desktop Setup](../claude_desktop_setup.md)
- [🔧 Advanced Configuration](../../docs/guides/)

## 🤝 Contributing

Have a great notebook idea? We'd love to include it!

1. Create your notebook following our style
2. Test it thoroughly (should run error-free)
3. Add clear markdown explanations
4. Submit a pull request

---

**Happy learning! 🎵✨**

*These notebooks showcase the power of combining traditional music theory with modern AI capabilities.*