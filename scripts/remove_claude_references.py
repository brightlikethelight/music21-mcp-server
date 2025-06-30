#!/usr/bin/env python3
"""
Remove Claude references from the codebase
This is a safer alternative to rewriting Git history
"""
import os
import re
from pathlib import Path


def clean_file(filepath):
    """Remove Claude references from a single file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Remove Claude-specific references
        replacements = [
            (r'🤖 Generated with \[Claude Code\].*\n?', ''),
            (r'Co-Authored-By: Claude.*\n?', ''),
            (r'claude\.ai/code', 'github.com'),
            (r'Claude Desktop', 'MCP client'),
            (r'Claude Code', 'AI assistant'),
            (r'CLAUDE\.md', 'PROJECT.md'),
            (r'claude_desktop_config\.json', 'mcp_config.json'),
        ]
        
        for pattern, replacement in replacements:
            content = re.sub(pattern, replacement, content, flags=re.IGNORECASE)
        
        # Only write if content changed
        if content != original_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        return False
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return False

def main():
    """Clean Claude references from all text files"""
    print("🧹 Removing Claude references from codebase...")
    
    # Define file extensions to process
    extensions = {'.py', '.md', '.yml', '.yaml', '.json', '.txt', '.sh'}
    
    # Files to skip
    skip_files = {'clean_git_history.sh', 'remove_claude_references.py'}
    
    changed_files = []
    
    # Walk through all files
    for root, dirs, files in os.walk('.'):
        # Skip hidden directories and __pycache__
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
        
        for file in files:
            if file in skip_files:
                continue
                
            filepath = Path(root) / file
            
            # Check if it's a text file we should process
            if filepath.suffix in extensions:
                if clean_file(filepath):
                    changed_files.append(str(filepath))
    
    # Handle special renames
    if os.path.exists('CLAUDE.md'):
        os.rename('CLAUDE.md', 'PROJECT.md')
        changed_files.append('Renamed CLAUDE.md to PROJECT.md')
        print("📝 Renamed CLAUDE.md to PROJECT.md")
    
    if os.path.exists('claude_desktop_config.json'):
        os.rename('claude_desktop_config.json', 'mcp_config.example.json')
        changed_files.append('Renamed claude_desktop_config.json to mcp_config.example.json')
        print("📝 Renamed claude_desktop_config.json to mcp_config.example.json")
    
    # Summary
    print(f"\n✅ Cleaned {len(changed_files)} files:")
    for file in changed_files[:10]:  # Show first 10
        print(f"   - {file}")
    if len(changed_files) > 10:
        print(f"   ... and {len(changed_files) - 10} more")
    
    print("\n📌 Next steps:")
    print("   1. Review the changes: git diff")
    print("   2. Commit the cleanup: git commit -am 'chore: Remove AI assistant references'")
    print("   3. Push normally: git push")

if __name__ == "__main__":
    main()