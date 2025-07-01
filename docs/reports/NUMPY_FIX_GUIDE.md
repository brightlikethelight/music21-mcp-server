# NumPy Fix Guide - Quick Solutions

## The Problem
Your conda environment has a broken numpy installation where it can't find `libgfortran.5.dylib`. This is blocking the resilience tests.

## Solution Options (Try in Order)

### Option 1: Quick Fix with Conda-Forge (Usually Works)
```bash
# In your cs109b environment
conda install -c conda-forge numpy scipy matplotlib -y
```

### Option 2: Create Fresh Environment (Most Reliable)
```bash
# Create new environment
conda create -n music21_prod python=3.10 -y
conda activate music21_prod

# Install from conda-forge
conda install -c conda-forge numpy scipy matplotlib -y

# Install the package
pip install -e .

# Test
python -c "import numpy; print('NumPy working!')"
```

### Option 3: Fix Current Environment
```bash
# Run the fix script
./fix_numpy_env.sh
```

### Option 4: Use pip in Fresh Venv (Bypass Conda)
```bash
# Create Python venv (not conda)
python3 -m venv venv_test
source venv_test/bin/activate

# Install everything with pip
pip install --upgrade pip
pip install numpy scipy matplotlib
pip install -e .
```

## Testing Without NumPy (While Fixing)

I've created alternatives that work without numpy:

### 1. Basic Server Test
```bash
# Run simple server test
python test_basic_server.py
```

### 2. Basic Resilient Server
```bash
# Terminal 1: Start basic resilient server
python -m music21_mcp.server_basic_resilient

# Terminal 2: Test it
curl http://localhost:8000/health/detailed
```

### 3. Simple Stress Test
```bash
# Terminal 1: Keep server running

# Terminal 2: Run stress test
python simple_stress_test.py 5 20  # 5 minutes, 20 clients
```

## Once NumPy is Fixed

Run the full production test:
```bash
# Quick 30-minute test
python run_production_test.py --config quick_test_config.json

# Or custom test
python run_production_test.py --hours 2 --users 25
```

## Why This Happens

1. **Conda environments on macOS** sometimes have library path issues
2. **NumPy needs Fortran libraries** for its linear algebra operations
3. **Different NumPy versions** have different library dependencies

## Verification

Test if numpy is working:
```python
python -c "
import numpy as np
import scipy
import matplotlib
print('✓ All imports successful')
print(f'NumPy version: {np.__version__}')
print(f'Array test: {np.array([1,2,3]).sum()}')
"
```

## Emergency Workaround

If you need to demo resilience features NOW without fixing numpy:

1. Use the `server_basic_resilient.py` which has circuit breakers and rate limiting
2. It supports the core tools (import, export, key analysis, etc.)
3. Missing only the advanced analyzers that require numpy

The basic resilient server still demonstrates:
- ✅ Circuit breakers
- ✅ Rate limiting  
- ✅ Health checks
- ✅ Metrics
- ✅ Recovery simulation
- ✅ Basic stress testing

Just not the full chaos engineering suite.