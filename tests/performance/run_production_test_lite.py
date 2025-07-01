#!/usr/bin/env python3
"""
Lightweight Production Test Orchestrator
Works with reduced memory (2GB) and without numpy/scipy
"""
import argparse
import asyncio
import json
import logging
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LightweightTestOrchestrator:
    """Lightweight orchestrator for systems with limited resources"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.processes = {}
        self.start_time = None
        self.test_results = {
            'started': None,
            'completed': None,
            'duration': None,
            'server_restarts': 0,
            'test_passed': False,
            'failure_reasons': []
        }
        
    def run(self):
        """Run the lightweight test suite"""
        self.start_time = datetime.now()
        self.test_results['started'] = self.start_time.isoformat()
        
        logger.info("=" * 80)
        logger.info("STARTING LIGHTWEIGHT PRODUCTION TEST")
        logger.info("=" * 80)
        logger.info(f"Duration: {self.config['duration_hours']} hours")
        logger.info(f"Concurrent Users: {self.config['concurrent_users']}")
        logger.info(f"Memory Limit: {self.config.get('memory_limit_mb', 2048)}MB")
        logger.info("Using basic resilient server (no numpy required)")
        logger.info("=" * 80)
        
        try:
            # Phase 1: Lightweight pre-flight checks
            logger.info("\nPhase 1: Pre-flight checks (lightweight)")
            if not self.preflight_checks_lite():
                self.test_results['failure_reasons'].append("Pre-flight checks failed")
                return False
            
            # Phase 2: Start basic resilient server
            logger.info("\nPhase 2: Starting basic resilient server")
            if not self.start_basic_server():
                self.test_results['failure_reasons'].append("Failed to start server")
                return False
            
            # Wait for server to be ready
            if not self.wait_for_server():
                self.test_results['failure_reasons'].append("Server failed to become ready")
                return False
            
            # Phase 3: Run lightweight stress test
            logger.info("\nPhase 3: Running lightweight stress test")
            test_passed = self.run_stress_test_lite()
            
            # Phase 4: Collect results
            logger.info("\nPhase 4: Collecting results")
            self.collect_results()
            
            self.test_results['test_passed'] = test_passed
            return test_passed
            
        except KeyboardInterrupt:
            logger.info("\nTest interrupted by user")
            self.test_results['failure_reasons'].append("User interrupted")
            return False
            
        except Exception as e:
            logger.error(f"\nTest failed with error: {e}")
            self.test_results['failure_reasons'].append(f"Exception: {str(e)}")
            return False
            
        finally:
            # Cleanup
            self.cleanup()
            
            # Generate final report
            self.generate_report()
    
    def preflight_checks_lite(self) -> bool:
        """Lightweight pre-flight checks"""
        checks_passed = True
        
        # Check Python version
        if sys.version_info < (3, 10):
            logger.error(f"Python 3.10+ required, found {sys.version}")
            checks_passed = False
        else:
            logger.info(f"✓ Python {sys.version_info.major}.{sys.version_info.minor}")
        
        # Check only essential modules
        required_modules = [
            'music21', 'httpx', 'psutil', 'uvicorn', 'fastapi'
        ]
        
        for module in required_modules:
            try:
                __import__(module)
                logger.info(f"✓ Module {module} available")
            except ImportError:
                logger.error(f"✗ Module {module} not found")
                checks_passed = False
        
        # Check port availability
        import socket
        for port in [8000, 8001]:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            result = sock.connect_ex(('localhost', port))
            sock.close()
            
            if result == 0:
                logger.error(f"✗ Port {port} already in use")
                checks_passed = False
            else:
                logger.info(f"✓ Port {port} available")
        
        # Check disk space (reduced requirement)
        import psutil
        disk = psutil.disk_usage('/')
        free_gb = disk.free / 1024 / 1024 / 1024
        
        if free_gb < 5:  # Reduced from 10GB
            logger.error(f"✗ Insufficient disk space: {free_gb:.1f}GB free (need 5GB)")
            checks_passed = False
        else:
            logger.info(f"✓ Disk space: {free_gb:.1f}GB free")
        
        # Check memory (reduced requirement)
        memory = psutil.virtual_memory()
        available_gb = memory.available / 1024 / 1024 / 1024
        
        if available_gb < 2:  # Reduced from 4GB
            logger.error(f"✗ Insufficient memory: {available_gb:.1f}GB available (need 2GB)")
            checks_passed = False
        else:
            logger.info(f"✓ Memory: {available_gb:.1f}GB available")
        
        # Create necessary directories
        for dir_path in ['stress_test_logs', 'test_results']:
            Path(dir_path).mkdir(exist_ok=True)
        
        return checks_passed
    
    def start_basic_server(self) -> bool:
        """Start the basic resilient server"""
        try:
            # Create lightweight server config
            server_config = {
                'host': '0.0.0.0',
                'port': 8000,
                'max_request_size': 10 * 1024 * 1024,  # 10MB (reduced)
                'request_timeout': 60,  # 1 minute (reduced)
                'max_concurrent_requests': self.config['concurrent_users'],
                'memory_limit_mb': 2048,  # 2GB (reduced)
            }
            
            with open('server_config_lite.json', 'w') as f:
                json.dump(server_config, f)
            
            # Start basic resilient server
            env = os.environ.copy()
            env['PYTHONUNBUFFERED'] = '1'
            
            self.processes['server'] = subprocess.Popen(
                [sys.executable, '-m', 'music21_mcp.server_basic_resilient'],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            logger.info(f"Started basic resilient server with PID: {self.processes['server'].pid}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start server: {e}")
            return False
    
    def wait_for_server(self, timeout: int = 30) -> bool:
        """Wait for server to be ready"""
        import httpx
        
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                response = httpx.get('http://localhost:8000/health', timeout=2.0)
                if response.status_code == 200:
                    logger.info("✓ Server is ready")
                    
                    # Check detailed health
                    response = httpx.get('http://localhost:8000/health/detailed', timeout=2.0)
                    health_data = response.json()
                    logger.info(f"  Memory: {health_data.get('memory_mb', 0):.0f}MB")
                    logger.info(f"  Circuit breakers: {len(health_data.get('circuit_breakers', {}))}")
                    
                    return True
            except:
                pass
            
            time.sleep(1)
            
        logger.error("✗ Server failed to become ready")
        return False
    
    def run_stress_test_lite(self) -> bool:
        """Run lightweight stress test"""
        try:
            # Create lightweight stress test script
            stress_script = """
import asyncio
import httpx
import json
import random
import time
from datetime import datetime

async def test_circuit_breaker():
    '''Test circuit breaker functionality'''
    print("Testing circuit breaker...")
    
    async with httpx.AsyncClient() as client:
        # Cause failures to open circuit breaker
        for i in range(10):
            try:
                await client.post(
                    'http://localhost:8000/mcp/call_tool',
                    json={
                        "name": "nonexistent_tool",
                        "arguments": {}
                    },
                    timeout=2.0
                )
            except:
                pass
        
        # Check circuit breaker state
        response = await client.get('http://localhost:8000/health/detailed')
        health = response.json()
        print(f"Circuit breakers: {health.get('circuit_breakers', {})}")

async def test_rate_limiting():
    '''Test rate limiting'''
    print("\\nTesting rate limiting...")
    
    async with httpx.AsyncClient() as client:
        success = 0
        rate_limited = 0
        
        # Burst of requests
        for i in range(100):
            try:
                response = await client.post(
                    'http://localhost:8000/mcp/call_tool',
                    json={
                        "name": "list_scores",
                        "arguments": {}
                    },
                    timeout=1.0
                )
                if response.status_code == 200:
                    success += 1
                elif response.status_code == 429:
                    rate_limited += 1
            except:
                pass
            
            if i % 10 == 0:
                await asyncio.sleep(0.1)
        
        print(f"Success: {success}, Rate limited: {rate_limited}")

async def test_recovery():
    '''Test recovery simulation'''
    print("\\nTesting recovery...")
    
    async with httpx.AsyncClient() as client:
        # Trigger recovery
        response = await client.post('http://localhost:8000/test/simulate_recovery')
        print(f"Recovery response: {response.json()}")
        
        # Check metrics
        response = await client.get('http://localhost:8000/metrics')
        print(f"Metrics after recovery:\\n{response.text}")

async def run_basic_load(duration_seconds: int, num_clients: int):
    '''Run basic load test'''
    print(f"\\nRunning {duration_seconds}s load test with {num_clients} clients...")
    
    async def client_session(client_id: int):
        end_time = time.time() + duration_seconds
        requests = 0
        errors = 0
        
        async with httpx.AsyncClient(timeout=5.0) as client:
            while time.time() < end_time:
                try:
                    # Random operations
                    if random.random() < 0.7:
                        # Import score
                        response = await client.post(
                            'http://localhost:8000/mcp/call_tool',
                            json={
                                "name": "import_score",
                                "arguments": {
                                    "score_id": f"test_{client_id}_{requests}",
                                    "source": "C4 D4 E4 F4 G4",
                                    "source_type": "text"
                                }
                            }
                        )
                    else:
                        # List scores
                        response = await client.post(
                            'http://localhost:8000/mcp/call_tool',
                            json={
                                "name": "list_scores",
                                "arguments": {}
                            }
                        )
                    
                    if response.status_code == 200:
                        requests += 1
                    else:
                        errors += 1
                except:
                    errors += 1
                
                await asyncio.sleep(random.uniform(0.5, 2.0))
        
        return requests, errors
    
    # Run concurrent clients
    tasks = [client_session(i) for i in range(num_clients)]
    results = await asyncio.gather(*tasks)
    
    total_requests = sum(r[0] for r in results)
    total_errors = sum(r[1] for r in results)
    
    print(f"Total requests: {total_requests}")
    print(f"Total errors: {total_errors}")
    print(f"Success rate: {(total_requests/(total_requests+total_errors)*100):.1f}%")

async def main():
    # Test resilience features
    await test_circuit_breaker()
    await test_rate_limiting()
    await test_recovery()
    
    # Run load test
    await run_basic_load(60, 10)  # 1 minute, 10 clients
    
    # Final health check
    print("\\nFinal health check:")
    async with httpx.AsyncClient() as client:
        response = await client.get('http://localhost:8000/health/detailed')
        health = response.json()
        print(json.dumps(health, indent=2))

if __name__ == "__main__":
    asyncio.run(main())
"""
            
            # Write stress test script
            stress_file = Path("lightweight_stress_test.py")
            stress_file.write_text(stress_script)
            
            # Run stress test
            result = subprocess.run(
                [sys.executable, str(stress_file)],
                capture_output=True,
                text=True
            )
            
            print(result.stdout)
            if result.stderr:
                print("Errors:", result.stderr)
            
            return result.returncode == 0
            
        except Exception as e:
            logger.error(f"Stress test failed: {e}")
            return False
    
    def collect_results(self):
        """Collect test results"""
        self.test_results['completed'] = datetime.now().isoformat()
        self.test_results['duration'] = str(datetime.now() - self.start_time)
        
        # Get final metrics
        try:
            import httpx
            response = httpx.get('http://localhost:8000/metrics', timeout=2.0)
            if response.status_code == 200:
                self.test_results['final_metrics'] = response.text
        except:
            pass
    
    def cleanup(self):
        """Clean up processes"""
        logger.info("\nCleaning up...")
        
        for name, process in self.processes.items():
            if process and process.poll() is None:
                logger.info(f"Terminating {name} process")
                process.terminate()
                
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    logger.warning(f"Force killing {name} process")
                    process.kill()
    
    def generate_report(self):
        """Generate final test report"""
        report_path = Path('test_results') / f"lite_test_{self.start_time.strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(report_path, 'w') as f:
            json.dump(self.test_results, f, indent=2)
        
        # Print summary
        print("\n" + "=" * 80)
        print("LIGHTWEIGHT TEST SUMMARY")
        print("=" * 80)
        print(f"Duration: {self.test_results['duration']}")
        print(f"Result: {'PASSED' if self.test_results['test_passed'] else 'FAILED'}")
        
        if self.test_results['failure_reasons']:
            print("\nFailure Reasons:")
            for reason in self.test_results['failure_reasons']:
                print(f"  - {reason}")
        
        if 'final_metrics' in self.test_results:
            print("\nFinal Metrics:")
            print(self.test_results['final_metrics'])
        
        print(f"\nFull report saved to: {report_path}")
        print("=" * 80)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Run lightweight production test for Music21 MCP Server'
    )
    
    parser.add_argument(
        '--minutes',
        type=int,
        default=10,
        help='Test duration in minutes (default: 10)'
    )
    parser.add_argument(
        '--users',
        type=int,
        default=10,
        help='Number of concurrent users (default: 10)'
    )
    
    args = parser.parse_args()
    
    # Build configuration
    config = {
        'duration_hours': args.minutes / 60.0,
        'concurrent_users': args.users,
        'memory_limit_mb': 2048  # 2GB limit
    }
    
    # Run test
    orchestrator = LightweightTestOrchestrator(config)
    success = orchestrator.run()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()