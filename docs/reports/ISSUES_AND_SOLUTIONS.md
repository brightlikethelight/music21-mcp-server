# Issues and Solutions - Production Test

## Current Issues

### 1. ❌ NumPy/SciPy Import Error
**Problem**: `ModuleNotFoundError: No module named 'numpy'` and scipy
**Impact**: Can't use advanced analyzers (harmony, patterns, counterpoint)

**Solutions**:
```bash
# Option A: Create fresh conda environment (RECOMMENDED)
conda create -n music21_fresh python=3.10 -y
conda activate music21_fresh
conda install -c conda-forge numpy scipy matplotlib -y
pip install -e .

# Option B: Fix current environment
conda activate cs109b
conda update -n base conda
conda install -c conda-forge numpy scipy matplotlib -y

# Option C: Use system Python with venv
/usr/bin/python3 -m venv fresh_venv
source fresh_venv/bin/activate
pip install numpy scipy matplotlib
pip install -e .
```

### 2. ❌ Port 8000 Already in Use
**Problem**: Previous server instance still running
**Impact**: Can't start new server

**Solution**:
```bash
# Find and kill process on port 8000
lsof -ti:8000 | xargs kill -9

# Or find process manually
lsof -i :8000
# Then kill with PID
kill -9 <PID>
```

### 3. ❌ Insufficient Memory (2.7GB available, 4GB required)
**Problem**: System has only 2.7GB free memory
**Impact**: Full production test can't run

**Solutions**:
```bash
# Option A: Use lightweight test (RECOMMENDED)
python run_production_test_lite.py --minutes 10 --users 10

# Option B: Free up memory
# - Close Chrome/browsers
# - Close other applications
# - Check memory usage:
ps aux | sort -nr -k 4 | head -10

# Option C: Modify memory requirements
# Edit run_production_test.py line ~150:
# Change: if available_gb < 4:
# To:     if available_gb < 2:
```

## Quick Testing Path (Works Now!)

### 1. Basic Server Test
```bash
# Simple functionality test
python test_basic_server.py
```

### 2. Basic Resilient Server
```bash
# Terminal 1: Start server
python -m music21_mcp.server_basic_resilient

# Terminal 2: Test resilience features
curl http://localhost:8000/health/detailed

# Terminal 3: Simple stress test
python simple_stress_test.py 5 20
```

### 3. Lightweight Production Test
```bash
# 10-minute test with reduced requirements
python run_production_test_lite.py --minutes 10 --users 10
```

## What Each Test Demonstrates

### Basic Resilient Server (`server_basic_resilient.py`)
✅ **Works without numpy** - Uses only core Python libraries
- Circuit breakers (prevent cascading failures)
- Rate limiting (50 req/s with burst of 100)
- Health checks and metrics
- Recovery simulation endpoint
- Basic tools (import, export, key analysis, etc.)

### Lightweight Production Test (`run_production_test_lite.py`)
✅ **Works with 2GB RAM** - Reduced requirements
- Pre-flight checks (Python, modules, ports, memory)
- Server startup and health monitoring
- Circuit breaker testing
- Rate limiting verification
- Recovery simulation
- Basic load testing
- Metrics collection

### Simple Stress Test (`simple_stress_test.py`)
✅ **No dependencies** - Pure Python
- Concurrent client simulation
- Random operations (import scores, list tools)
- Success rate calculation
- Requests per second measurement

## Full Production Test Prerequisites

To run the full 24-hour production test with chaos engineering:

1. **Fix NumPy** (see solutions above)
2. **Free up memory** to have 4GB+ available
3. **Ensure ports are free** (8000, 8001)
4. **Run setup**:
   ```bash
   ./setup_production_test.sh
   ```
5. **Start test**:
   ```bash
   python run_production_test.py --hours 4 --users 50
   ```

## Current Capabilities (Without NumPy)

### Working Tools ✅
- `import_score` - Import from text, file, URL
- `list_scores` - List loaded scores
- `key_analysis` - Analyze key signatures
- `chord_analysis` - Basic chord detection
- `score_info` - Get score metadata
- `export_score` - Export to various formats
- `delete_score` - Remove scores

### Resilience Features ✅
- **Circuit Breakers** - Isolate failing components
- **Rate Limiting** - Prevent overload (configurable)
- **Health Checks** - `/health` and `/health/detailed`
- **Metrics** - Prometheus-compatible `/metrics`
- **Recovery** - Simulated via `/test/simulate_recovery`
- **Memory Management** - Basic GC and monitoring

### Not Available (Need NumPy) ❌
- Harmony analysis (Roman numerals, progressions)
- Pattern recognition (motifs, sequences)
- Voice leading analysis
- Counterpoint generation
- Style imitation
- Advanced statistical analysis

## Recommended Testing Sequence

1. **Start with basic test** (5 minutes):
   ```bash
   python test_basic_server.py
   ```

2. **Try lightweight production test** (10 minutes):
   ```bash
   python run_production_test_lite.py --minutes 10
   ```

3. **Monitor with simple dashboard**:
   ```bash
   # While test is running
   watch -n 1 'curl -s http://localhost:8000/health/detailed | python -m json.tool'
   ```

4. **Check metrics**:
   ```bash
   curl http://localhost:8000/metrics
   ```

## Summary

- **NumPy issue** blocks advanced features but basic resilience works
- **Memory constraint** (2.7GB) requires using lightweight tests
- **Port conflicts** easily fixed with process kill
- **Basic resilient server** demonstrates core production features
- **Lightweight tests** validate resilience without full chaos

The system can still demonstrate:
- ✅ Automatic failure detection
- ✅ Circuit breaker activation
- ✅ Rate limiting under load
- ✅ Health monitoring
- ✅ Basic recovery mechanisms
- ✅ Graceful degradation

Just not the full chaos engineering suite with network failures and hourly process kills.