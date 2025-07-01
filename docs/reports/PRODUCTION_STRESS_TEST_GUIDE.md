# Production Stress Test Guide

## Overview

The Music21 MCP Server production stress test simulates extreme real-world conditions to ensure the server can handle production workloads with 99.9% uptime and automatic recovery from failures.

## Test Requirements

### The Challenge
- Run continuously for 24 hours
- Handle 100 concurrent users
- Process random large files (up to 50MB)
- Survive simulated network failures
- Operate under memory pressure
- Recover from process kills every hour
- Self-recover within 60 seconds from any failure

### Success Criteria
- ✅ 95%+ request success rate
- ✅ Average recovery time < 60 seconds  
- ✅ No manual intervention required
- ✅ Graceful degradation under load
- ✅ Automatic resource cleanup

## Architecture

### 1. **Resilient Server** (`server_resilient.py`)
Production-hardened server with:
- Circuit breakers for fault isolation
- Rate limiting to prevent overload
- Resource pooling for efficiency
- Memory guards to prevent OOM
- Health checks and metrics
- Automatic recovery strategies
- Graceful shutdown handling

### 2. **Stress Test Suite** (`production_stress_test.py`)
Simulates production chaos:
- 100 concurrent virtual users
- Random operations (create, analyze, export)
- Large file generation (up to 50MB)
- Network failure injection
- Memory pressure simulation
- Hourly process kills
- Continuous metrics collection

### 3. **Monitoring Dashboard** (`stress_test_monitor.py`)
Real-time terminal UI showing:
- Server health status
- Request rates and success rates
- Memory and CPU usage
- Circuit breaker states
- Recent alerts and warnings
- Performance graphs

### 4. **Test Orchestrator** (`run_production_test.py`)
Manages the entire test:
- Pre-flight checks
- Server startup
- Monitor launch
- Stress test execution
- Result collection
- Report generation

## Running the Test

### Quick Start (Default 24-hour test)
```bash
python run_production_test.py
```

### Custom Configuration
```bash
# 4-hour test with 50 users
python run_production_test.py --hours 4 --users 50

# Aggressive test with kills every 30 minutes
python run_production_test.py --kill-interval 30

# Test with stricter recovery requirement (30 seconds)
python run_production_test.py --recovery-timeout 30
```

### Configuration Options
- `--hours`: Test duration (default: 24)
- `--users`: Concurrent users (default: 100)
- `--kill-interval`: Minutes between process kills (default: 60)
- `--recovery-timeout`: Max recovery time in seconds (default: 60)
- `--no-monitor`: Disable monitoring dashboard
- `--config`: Path to JSON config file

## What to Expect

### During the Test

1. **Startup Phase (0-5 minutes)**
   - Server initialization
   - Health check verification
   - Users ramping up gradually
   - Initial metric collection

2. **Steady State (5 minutes - N hours)**
   - Consistent request load
   - Periodic chaos injection:
     - Network failures every 30-300 seconds
     - Memory pressure spikes
     - Process kills every hour
   - Automatic recovery attempts

3. **Monitoring Dashboard**
   ```
   ═══════════════════════════════════════════════════════════
        Music21 MCP Server - Production Stress Test Monitor
   ═══════════════════════════════════════════════════════════
   
   SERVER METRICS
     Status: HEALTHY                    Uptime: 3:45:22
     Total Requests: 1,234,567         RPS: 42.3      Active: 12
     Success Rate: 96.8%               Errors: 39,422
     Memory: 2,341MB
   
   SYSTEM METRICS  
     CPU: 67.3% (8 cores)
     Memory: 45.2% (8.7GB free)
     Disk: 23.1% (412.3GB free)
     Network - Sent: 1,234.5MB, Recv: 5,678.9MB
     Server Process - PID: 12345, CPU: 34.2%, Memory: 2,341MB
   
   RECENT ALERTS
     14:32:15 [WARNING] High error rate: 6.2%
     14:45:03 [WARNING] Circuit breaker open: tool_PatternRecognition
     15:00:00 [WARNING] Scheduled process kill
     15:00:42 [INFO] Recovery successful in 42.1s
   ```

### Common Scenarios

#### 1. Process Kill and Recovery
```
15:00:00 - Process killed (scheduled)
15:00:01 - Health monitor detects failure
15:00:02 - Recovery procedure initiated
15:00:05 - Server process restarted
15:00:15 - Port responding
15:00:25 - Health checks passing
15:00:42 - Full recovery complete
```

#### 2. Memory Pressure
```
High memory detected: 3,500MB
- Caches cleared automatically
- Old scores removed
- Heavy requests rejected temporarily
- Memory reduced to safe levels
```

#### 3. Circuit Breaker Activation
```
PatternRecognition tool experiencing failures
- Circuit breaker opens after 5 failures
- Requests rejected for 60 seconds
- Half-open state for testing
- Closes after 2 successful requests
```

## Interpreting Results

### Success Indicators
- ✅ Overall success rate ≥ 95%
- ✅ All recoveries < 60 seconds
- ✅ No manual interventions
- ✅ Health checks mostly passing
- ✅ Memory stays under limits

### Warning Signs
- ⚠️ Success rate 90-95%
- ⚠️ Some recoveries > 45 seconds
- ⚠️ Multiple circuit breakers open
- ⚠️ Memory approaching limits
- ⚠️ High error rates during peaks

### Failure Indicators
- ❌ Success rate < 90%
- ❌ Recovery failed or > 60 seconds
- ❌ Server crashes without recovery
- ❌ Memory exhaustion
- ❌ Cascading failures

## Final Report

After the test completes, find the report in:
```
test_results/production_test_YYYYMMDD_HHMMSS.json
```

Example summary:
```
================================================================================
PRODUCTION TEST SUMMARY
================================================================================
Duration: 24:00:15
Result: PASSED

Success Rate: 96.42%
Total Requests: 8,234,567
Failed Requests: 293,421
Total Recoveries: 24
Avg Recovery Time: 38.7s

Full report saved to: test_results/production_test_20240130_120000.json
================================================================================
```

## Troubleshooting

### Server Won't Start
- Check ports 8000 and 8001 are free
- Verify Python environment has all dependencies
- Check available memory (need 4GB+)

### High Failure Rate
- Review circuit breaker states
- Check for memory leaks
- Analyze error patterns in logs
- Verify network connectivity

### Slow Recovery
- Check system resources during recovery
- Review recovery logs for bottlenecks
- Ensure cleanup procedures complete
- Verify health check sensitivity

### Monitor Issues
- Terminal too small (need 80x40 minimum)
- Use `--no-monitor` flag if issues persist
- Check `stress_test_logs/monitor_*.log`

## Best Practices

1. **Before Running**
   - Close unnecessary applications
   - Ensure stable network connection
   - Have 10GB+ free disk space
   - Run shorter test first (2-4 hours)

2. **During Test**
   - Don't modify server code
   - Monitor system resources
   - Check logs for anomalies
   - Note any manual interventions

3. **After Test**
   - Analyze failure patterns
   - Review recovery times
   - Check for memory leaks
   - Document any issues

## Advanced Usage

### Custom Chaos Scenarios
Edit `production_stress_test.py` to add:
- Database connection failures
- Disk space exhaustion
- CPU throttling
- Network partitions

### Performance Tuning
Adjust in `server_config.json`:
```json
{
  "max_concurrent_requests": 200,
  "memory_limit_mb": 8192,
  "request_timeout": 600,
  "max_request_size": 209715200
}
```

### Integration with CI/CD
```yaml
# .github/workflows/stress-test.yml
- name: Run 4-hour stress test
  run: |
    python run_production_test.py \
      --hours 4 \
      --users 50 \
      --recovery-timeout 30
  timeout-minutes: 250
```

## Summary

This production stress test ensures the Music21 MCP Server can handle real-world production conditions with automatic recovery and graceful degradation. A passing test indicates the server is ready for production deployment with high confidence in reliability and performance.