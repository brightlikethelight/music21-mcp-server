#!/bin/bash
# 🚀 PyPI Publication Script for music21-mcp-server
# This script publishes the package to PyPI, making it installable via pip

set -e  # Exit on error

echo "🎵 Music21 MCP Server - PyPI Publication Script"
echo "=============================================="
echo ""

# Check if dist directory exists
if [ ! -d "dist" ]; then
    echo "❌ Error: dist/ directory not found. Run 'python -m build' first."
    exit 1
fi

# Check if twine is installed
if ! command -v twine &> /dev/null; then
    echo "❌ Error: twine not installed. Run 'pip install twine' first."
    exit 1
fi

# Function to check package validity
check_packages() {
    echo "📦 Checking package validity..."
    python -m twine check dist/*
    if [ $? -eq 0 ]; then
        echo "✅ Package validation passed!"
    else
        echo "❌ Package validation failed. Fix issues before publishing."
        exit 1
    fi
}

# Function to publish to Test PyPI
publish_test() {
    echo ""
    echo "🧪 Publishing to Test PyPI..."
    echo "----------------------------"
    
    read -p "Do you have Test PyPI credentials configured? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "📝 Configure Test PyPI credentials:"
        echo "   1. Create account at https://test.pypi.org/account/register/"
        echo "   2. Create API token at https://test.pypi.org/manage/account/token/"
        echo "   3. Create ~/.pypirc with:"
        echo ""
        echo "[testpypi]"
        echo "  username = __token__"
        echo "  password = <your-test-pypi-token>"
        echo ""
        exit 1
    fi
    
    echo "📤 Uploading to Test PyPI..."
    python -m twine upload --repository testpypi dist/*
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ Successfully published to Test PyPI!"
        echo ""
        echo "🧪 Test installation with:"
        echo "   pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ music21-mcp-server"
        echo ""
        echo "📦 View package at:"
        echo "   https://test.pypi.org/project/music21-mcp-server/"
        echo ""
    else
        echo "❌ Test PyPI upload failed"
        exit 1
    fi
}

# Function to publish to Production PyPI
publish_prod() {
    echo ""
    echo "🚀 Publishing to Production PyPI..."
    echo "-----------------------------------"
    
    read -p "⚠️  This will make the package publicly available. Continue? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Publication cancelled."
        exit 0
    fi
    
    read -p "Do you have PyPI credentials configured? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "📝 Configure PyPI credentials:"
        echo "   1. Create account at https://pypi.org/account/register/"
        echo "   2. Create API token at https://pypi.org/manage/account/token/"
        echo "   3. Add to ~/.pypirc:"
        echo ""
        echo "[pypi]"
        echo "  username = __token__"
        echo "  password = <your-pypi-token>"
        echo ""
        exit 1
    fi
    
    echo "📤 Uploading to PyPI..."
    python -m twine upload dist/*
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "🎉 Successfully published to PyPI!"
        echo ""
        echo "✅ Users can now install with:"
        echo "   pip install music21-mcp-server"
        echo ""
        echo "📦 View package at:"
        echo "   https://pypi.org/project/music21-mcp-server/"
        echo ""
        echo "🌟 Next steps:"
        echo "   1. Update README.md with pip install instructions"
        echo "   2. Create GitHub release with v1.0.0 tag"
        echo "   3. Submit to MCP Registry"
        echo "   4. Share on social media!"
    else
        echo "❌ PyPI upload failed"
        exit 1
    fi
}

# Main menu
echo "📋 Select publication target:"
echo "   1) Test PyPI (recommended first)"
echo "   2) Production PyPI (public release)"
echo "   3) Check packages only"
echo "   4) Exit"
echo ""
read -p "Enter choice [1-4]: " choice

case $choice in
    1)
        check_packages
        publish_test
        ;;
    2)
        check_packages
        publish_prod
        ;;
    3)
        check_packages
        ;;
    4)
        echo "Exiting..."
        exit 0
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "✨ Publication script complete!"