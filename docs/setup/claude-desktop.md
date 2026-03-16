# 🎵 Claude Desktop Setup - AI-Powered Music Analysis

**⏱️ 2 minutes | 🎯 Zero configuration required**

This automated setup script connects the Music21 MCP Server to Claude Desktop, enabling AI-powered music analysis conversations.

## 🚀 Quick Setup

### 1. One-Command Setup
```bash
python scripts/setup_claude_desktop.py
```

That's it! The script automatically:
- ✅ Detects your Claude Desktop installation
- ✅ Creates the MCP configuration
- ✅ Runs comprehensive diagnostics
- ✅ Backs up any existing config

### 2. Restart Claude Desktop
Close and reopen Claude Desktop to load the new configuration.

### 3. Test Your Setup
Start a conversation in Claude Desktop and try:

```
🎵 Analyze the harmony in Bach BWV 66.6
```

You should see Claude analyzing the Bach chorale using music21!

## 🔧 Advanced Usage

### Check Current Configuration
```bash
python scripts/setup_claude_desktop.py --check-only
```

### Skip Diagnostics (Faster)
```bash
python scripts/setup_claude_desktop.py --skip-diagnostics
```

### Restore from Backup
```bash
python scripts/setup_claude_desktop.py --restore-backup backup_file.json
```

## 🛠️ Manual Setup (Alternative)

If you prefer manual configuration, add this to your Claude Desktop config:

**macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
**Windows**: `%APPDATA%/Claude/claude_desktop_config.json`
**Linux**: `~/.config/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "music21-mcp-server": {
      "command": ["python", "-m", "music21_mcp"],
      "args": [],
      "env": {
        "MUSIC21_MCP_LOG_LEVEL": "INFO",
        "MUSIC21_MCP_MAX_MEMORY_MB": "512",
        "MUSIC21_MCP_MAX_SCORES": "100"
      }
    }
  }
}
```

## 🎯 What You Can Do

Once configured, you can ask Claude Desktop to:

### 🎼 Analyze Compositions
- *"What's the harmonic progression in Mozart K331?"*
- *"Find parallel fifths in this Bach chorale"*
- *"Analyze the voice leading in Chopin Op. 28 No. 4"*

### 🔍 Explore Music Theory
- *"What makes a Neapolitan sixth chord special?"*
- *"Show me examples of augmented sixth chords"*
- *"How do jazz chord progressions work?"*

### 📊 Generate Reports
- *"Create a detailed analysis report for this piece"*
- *"Compare the harmonic language of Bach vs. Mozart"*
- *"What are the statistical characteristics of this composer's style?"*

## 🔍 Troubleshooting

### Common Issues

**🔄 Claude Desktop doesn't see the server**
- Restart Claude Desktop completely
- Check config file syntax: `python -m json.tool claude_desktop_config.json`

**🐍 Python/Package Issues**
- Ensure Python 3.8+: `python --version`
- Reinstall: `pip install --upgrade music21-mcp-server`

**🎵 Music21 Corpus Issues**
- Run: `python -c "import music21; music21.configure.run()"`
- Restart after configuration

**💾 Permissions Issues**
- Check Claude config directory permissions
- Try running as administrator (Windows) or with sudo (Linux)

### Getting Help

- 📖 **Documentation**: See main README.md
- 🐛 **Issues**: Report problems on GitHub
- 💬 **Community**: Join our Discord/Forum
- ⚡ **Quick Test**: Run `python scripts/setup_claude_desktop.py --check-only`

## 🎉 Success Indicators

You'll know setup worked when:
- ✅ Setup script shows "5/5 diagnostics passed"
- ✅ Claude Desktop restart completes without errors
- ✅ Claude can respond to music analysis requests
- ✅ You see music21 corpus data in responses

## 📚 Next Steps

1. **Try the Tutorial**: `examples/notebooks/quickstart_tutorial.ipynb`
2. **Explore Examples**: Browse the examples directory
3. **Read Documentation**: Check out the full API reference
4. **Join Community**: Share your musical discoveries!

---

**🎵 Ready to explore music with AI? Let's make some beautiful analysis! ✨**