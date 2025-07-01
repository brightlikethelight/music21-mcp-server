#!/bin/bash
# Fix numpy/scipy environment issues on macOS

echo "=========================================="
echo "Fixing NumPy Environment Issues"
echo "=========================================="

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Check if in conda environment
if [[ -z "$CONDA_DEFAULT_ENV" ]]; then
    echo -e "${RED}Not in a conda environment!${NC}"
    echo "Please activate your conda environment first:"
    echo "  conda activate cs109b"
    exit 1
fi

echo "Current environment: $CONDA_DEFAULT_ENV"
echo ""

# Option 1: Try to fix with conda-forge
echo "Option 1: Reinstalling numpy/scipy from conda-forge..."
conda install -c conda-forge numpy scipy -y

# Test numpy
echo -n "Testing numpy import... "
if python -c "import numpy; print('numpy version:', numpy.__version__)" 2>/dev/null; then
    echo -e "${GREEN}✓ Success!${NC}"
    exit 0
else
    echo -e "${RED}✗ Still failing${NC}"
fi

# Option 2: Install gfortran
echo ""
echo "Option 2: Installing gfortran..."
if [[ "$OSTYPE" == "darwin"* ]]; then
    if command -v brew &> /dev/null; then
        echo "Installing gfortran via homebrew..."
        brew install gcc gfortran
    else
        echo -e "${YELLOW}Homebrew not found. Install it first:${NC}"
        echo '/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'
    fi
fi

# Option 3: Complete reinstall
echo ""
echo "Option 3: Complete package reinstall..."
echo "This will remove and reinstall numpy, scipy, and related packages"
read -p "Continue? (y/n) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    # Remove packages
    pip uninstall -y numpy scipy matplotlib
    
    # Clean conda cache
    conda clean --all -y
    
    # Reinstall with specific versions known to work
    conda install -c conda-forge python=3.10 numpy=1.23.5 scipy=1.10.1 matplotlib=3.7.1 -y
    
    # Test again
    echo -n "Testing numpy after reinstall... "
    if python -c "import numpy; print('numpy version:', numpy.__version__)" 2>/dev/null; then
        echo -e "${GREEN}✓ Success!${NC}"
    else
        echo -e "${RED}✗ Still failing${NC}"
    fi
fi

echo ""
echo "If still having issues, try creating a fresh environment:"
echo "  conda create -n music21_test python=3.10 -y"
echo "  conda activate music21_test"
echo "  pip install -e ."
echo ""