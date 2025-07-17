# Performance Testing Guide for Music21 MCP Server

## Overview

This guide describes the comprehensive performance testing suite for the Music21 MCP Server. The tests are designed to validate the server's ability to handle production workloads, including concurrent clients, large scores, sustained load, and error recovery.

## Test Scenarios

### 1. Concurrent Client Handling

Tests the server's ability to handle multiple simultaneous clients.

**Test Parameters:**
- Client counts: 10, 50, 100
- Each client performs: import → analyze → export → delete
- Measures: throughput, response times, success rate

**Run Command:**
```bash
pytest tests/performance/test_performance.py::TestConcurrentClients -v
```

### 2. Large Score Processing

Tests processing of orchestral scores with many instruments.

**Test Parameters:**
- Score sizes: 20×100, 50×200, 100×300, 150×500 (instruments × measures)
- Operations: score info, key analysis, harmony analysis, export
- Measures: processing time, memory usage, success rate

**Run Command:**
```bash
pytest tests/performance/test_performance.py::TestLargeScoreProcessing -v
```

### 3. Memory Usage Under Sustained Load

Detects memory leaks during extended operation.

**Test Parameters:**
- Duration: 5 minutes continuous operation
- Rate: 10 operations per second
- Tracks: memory growth, allocation patterns

**Run Command:**
```bash
pytest tests/performance/test_performance.py::TestMemoryUnderSustainedLoad -v -m slow
```

### 4. Tool Execution Benchmarks

Benchmarks individual tool performance.

**Tools Tested:**
- Import tool (corpus loading)
- Key analysis tool
- Harmony analysis tool
- Export tool (MusicXML)

**Run Command:**
```bash
pytest tests/performance/test_performance.py::TestToolExecutionBenchmarks -v
```

### 5. Rate Limiting Effectiveness

Validates rate limiting implementation.

**Test Parameters:**
- Burst capacity validation
- Sustained rate enforcement
- Token replenishment accuracy

**Run Command:**
```bash
pytest tests/performance/test_performance.py::TestRateLimiting -v
```

### 6. Resource Cleanup Verification

Ensures proper resource cleanup after errors.

**Test Parameters:**
- 50 operations with intentional errors
- Tracks: memory growth, file descriptors, score cleanup

**Run Command:**
```bash
pytest tests/performance/test_performance.py::TestResourceCleanup -v
```

### 7. 24-Hour Stability Simulation

Simulates extended operation in accelerated time.

**Test Parameters:**
- Real duration: 10 minutes
- Simulated time: 24 hours
- Variable load patterns (business hours, evening, night)

**Run Command:**
```bash
pytest tests/performance/test_performance.py::TestStabilitySimulation -v -m slow
```

## Running Performance Tests

### Quick Test Suite

Run all fast performance tests:
```bash
python tests/performance/run_performance_tests.py --scenario all
```

### Individual Scenarios

Run specific test scenarios:
```bash
# Concurrent clients only
python tests/performance/run_performance_tests.py --scenario concurrent

# Memory leak detection
python tests/performance/run_performance_tests.py --scenario memory

# Stability simulation
python tests/performance/run_performance_tests.py --scenario stability
```

### Custom Parameters

```bash
# Test with 200 concurrent clients
pytest tests/performance/test_performance.py::test_concurrent_clients -v --clients 200

# Extended memory test (10 minutes)
pytest tests/performance/test_performance.py::test_memory_leak_detection -v --duration 600
```

## Performance Metrics

### Key Metrics Collected

1. **Response Times**
   - Min, max, mean, median
   - 95th and 99th percentiles
   - Per-tool breakdowns

2. **Throughput**
   - Requests per second
   - Success rate percentage
   - Error categorization

3. **Resource Usage**
   - Memory (current, peak, growth)
   - CPU utilization
   - File descriptors
   - Thread count

4. **Stability Indicators**
   - Recovery time after failures
   - Memory leak rate (MB/minute)
   - Long-term success rate

## Performance Baselines

Expected performance characteristics:

| Metric | Acceptable | Good | Excellent |
|--------|------------|------|-----------|
| Success Rate | ≥ 90% | ≥ 95% | ≥ 99% |
| Response Time (p95) | < 5s | < 2s | < 1s |
| Memory Growth | < 100MB/5min | < 50MB/5min | < 10MB/5min |
| Concurrent Clients | 50 | 100 | 200+ |
| Recovery Time | < 60s | < 30s | < 10s |

## Interpreting Results

### Success Criteria

Tests pass when:
- Success rate exceeds threshold (typically 90%)
- Memory growth is within limits
- Response times meet SLA requirements
- No resource leaks detected

### Common Issues

1. **High Memory Usage**
   - Check for score cleanup
   - Verify garbage collection
   - Look for circular references

2. **Slow Response Times**
   - Profile CPU-intensive operations
   - Check for blocking I/O
   - Verify async implementation

3. **Rate Limiting Issues**
   - Validate token bucket implementation
   - Check time synchronization
   - Verify burst capacity

## Continuous Performance Testing

### Integration with CI/CD

Add to GitHub Actions:
```yaml
- name: Run Performance Tests
  run: |
    python tests/performance/run_performance_tests.py \
      --scenario concurrent \
      --output-dir ${{ github.workspace }}/perf-results
    
- name: Upload Performance Results
  uses: actions/upload-artifact@v3
  with:
    name: performance-results
    path: perf-results/
```

### Performance Regression Detection

Compare results against baselines:
```python
# In your CI pipeline
current_results = load_performance_results()
baseline = load_baseline_results()

for metric in ['response_time_p95', 'memory_growth', 'success_rate']:
    if current_results[metric] > baseline[metric] * 1.1:  # 10% regression
        raise PerformanceRegression(f"{metric} degraded by >10%")
```

## Troubleshooting

### Out of Memory Errors

1. Reduce concurrent clients
2. Enable memory profiling
3. Check for score accumulation
4. Review large score handling

### Timeouts

1. Increase timeout values
2. Check rate limiting
3. Profile slow operations
4. Verify async execution

### Inconsistent Results

1. Ensure clean state between runs
2. Control system load
3. Use consistent test data
4. Check for race conditions

## Advanced Usage

### Memory Profiling

Enable detailed memory profiling:
```bash
python -m memory_profiler tests/performance/test_performance.py::test_memory_leak_detection
```

### Custom Load Patterns

Create custom load scenarios:
```python
class CustomLoadTest(PerformanceTestBase):
    async def test_custom_pattern(self):
        metrics = PerformanceMetrics("custom_load")
        
        # Implement your load pattern
        for hour in range(24):
            load = calculate_load_for_hour(hour)
            await self.generate_load(load, metrics)
```

### Performance Monitoring

Export metrics for monitoring:
```python
# Export to Prometheus format
for metric_name, value in summary.items():
    print(f"music21_mcp_{metric_name} {value}")
```

## Best Practices

1. **Run Regularly**
   - Include in CI/CD pipeline
   - Schedule nightly runs
   - Test before releases

2. **Maintain Baselines**
   - Track historical performance
   - Update after optimizations
   - Document significant changes

3. **Isolate Tests**
   - Run on dedicated hardware
   - Control background processes
   - Use consistent environments

4. **Document Findings**
   - Record anomalies
   - Track optimizations
   - Share with team

## Contact

For questions or issues with performance testing:
- Create an issue in the repository
- Contact: brightliu@college.harvard.edu