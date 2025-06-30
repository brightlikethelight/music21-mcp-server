#!/bin/bash
#
# Script to clean Git history - removes Claude references from commit messages
# WARNING: This rewrites history! Only use on branches that haven't been shared.
#

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}⚠️  WARNING: This script will rewrite Git history!${NC}"
echo "This should only be done if:"
echo "1. The repository hasn't been shared/pushed yet, OR"
echo "2. You're willing to force-push and coordinate with all contributors"
echo
read -p "Do you want to continue? (yes/no): " -r
if [[ ! $REPLY =~ ^[Yy]es$ ]]; then
    echo "Aborted."
    exit 0
fi

# Create backup branch
echo -e "\n${GREEN}Creating backup branch...${NC}"
git branch backup-before-cleanup-$(date +%Y%m%d-%H%M%S)

# Create a filter script to remove Claude references
cat > /tmp/git-filter-msg.sh << 'EOF'
#!/bin/bash
# Remove Claude-related lines from commit messages
sed '/🤖 Generated with \[Claude Code\]/d' | \
sed '/Co-Authored-By: Claude/d' | \
sed '/CLAUDE DESKTOP/d' | \
sed 's/claude\.ai\/code/github.com/g' | \
sed 's/Claude Desktop/MCP client/g' | \
sed 's/CLAUDE\.md/PROJECT.md/g'
EOF

chmod +x /tmp/git-filter-msg.sh

# Use git filter-branch to clean commit messages
echo -e "\n${GREEN}Cleaning commit messages...${NC}"
git filter-branch -f --msg-filter /tmp/git-filter-msg.sh -- --all

# Clean up
rm /tmp/git-filter-msg.sh

# Remove CLAUDE.md and rename references
if [ -f "CLAUDE.md" ]; then
    echo -e "\n${GREEN}Renaming CLAUDE.md to PROJECT.md...${NC}"
    git mv CLAUDE.md PROJECT.md 2>/dev/null || mv CLAUDE.md PROJECT.md
    
    # Update references in files
    find . -type f -name "*.md" -o -name "*.py" | while read file; do
        if [[ -f "$file" ]]; then
            sed -i.bak 's/CLAUDE\.md/PROJECT.md/g' "$file" 2>/dev/null && rm "${file}.bak"
        fi
    done
    
    git add -A
    git commit -m "docs: Rename project documentation file" || true
fi

# Remove claude_desktop_config.json from tracking
if [ -f "claude_desktop_config.json" ]; then
    echo -e "\n${GREEN}Moving claude_desktop_config.json to example...${NC}"
    mv claude_desktop_config.json claude_desktop_config.example.json
    echo "claude_desktop_config.json" >> .gitignore
    git add -A
    git commit -m "chore: Convert config to example file" || true
fi

echo -e "\n${GREEN}✅ Git history cleaned!${NC}"
echo
echo "Next steps:"
echo "1. Review the changes: git log --oneline"
echo "2. If everything looks good, force push: git push --force-with-lease"
echo "3. If something went wrong, restore from backup: git reset --hard backup-before-cleanup-*"
echo
echo -e "${YELLOW}⚠️  Remember to coordinate with any other contributors before force-pushing!${NC}"