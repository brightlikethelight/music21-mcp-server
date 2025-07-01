#!/usr/bin/env python3
"""
Production Test Orchestrator
Runs the 24-hour production stress test with monitoring and reporting
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


class ProductionTestOrchestrator:
    """Orchestrates the complete production test"""
    
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
        """Run the complete test suite"""
        self.start_time = datetime.now()
        self.test_results['started'] = self.start_time.isoformat()
        
        logger.info("=" * 80)
        logger.info("STARTING PRODUCTION STRESS TEST")
        logger.info("=" * 80)
        logger.info(f"Duration: {self.config['duration_hours']} hours")
        logger.info(f"Concurrent Users: {self.config['concurrent_users']}")
        logger.info(f"Kill Interval: {self.config['kill_interval_minutes']} minutes")
        logger.info(f"Recovery Timeout: {self.config['recovery_timeout_seconds']} seconds")
        logger.info("=" * 80)
        
        try:
            # Phase 1: Pre-flight checks
            logger.info("\nPhase 1: Pre-flight checks")
            if not self.preflight_checks():
                self.test_results['failure_reasons'].append("Pre-flight checks failed")
                return False
            
            # Phase 2: Start resilient server
            logger.info("\nPhase 2: Starting resilient server")
            if not self.start_resilient_server():
                self.test_results['failure_reasons'].append("Failed to start server")
                return False
            
            # Wait for server to be ready
            if not self.wait_for_server():
                self.test_results['failure_reasons'].append("Server failed to become ready")
                return False
            
            # Phase 3: Start monitoring
            logger.info("\nPhase 3: Starting monitoring")
            if self.config.get('enable_monitor', True):
                self.start_monitor()
            
            # Phase 4: Run stress test
            logger.info("\nPhase 4: Running stress test")
            test_passed = self.run_stress_test()
            
            # Phase 5: Collect results
            logger.info("\nPhase 5: Collecting results")
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
    
    def preflight_checks(self) -> bool:
        """Run pre-flight checks"""
        checks_passed = True
        
        # Check Python version
        if sys.version_info < (3, 10):
            logger.error(f"Python 3.10+ required, found {sys.version}")
            checks_passed = False
        
        # Check required modules
        required_modules = [
            'music21', 'mcp', 'fastmcp', 'numpy', 'scipy',
            'httpx', 'psutil', 'uvicorn'
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
        
        # Check disk space
        import psutil
        disk = psutil.disk_usage('/')
        free_gb = disk.free / 1024 / 1024 / 1024
        
        if free_gb < 10:
            logger.error(f"✗ Insufficient disk space: {free_gb:.1f}GB free (need 10GB)")
            checks_passed = False
        else:
            logger.info(f"✓ Disk space: {free_gb:.1f}GB free")
        
        # Check memory
        memory = psutil.virtual_memory()
        available_gb = memory.available / 1024 / 1024 / 1024
        
        if available_gb < 4:
            logger.error(f"✗ Insufficient memory: {available_gb:.1f}GB available (need 4GB)")
            checks_passed = False
        else:
            logger.info(f"✓ Memory: {available_gb:.1f}GB available")
        
        # Create necessary directories
        for dir_path in ['stress_test_logs', 'test_results', 'test_corpus']:
            Path(dir_path).mkdir(exist_ok=True)
        
        return checks_passed
    
    def start_resilient_server(self) -> bool:
        """Start the resilient server"""
        try:
            # Create server config
            server_config = {
                'host': '0.0.0.0',
                'port': 8000,
                'max_request_size': 100 * 1024 * 1024,
                'request_timeout': 300,
                'max_concurrent_requests': self.config['concurrent_users'] * 2,
                'memory_limit_mb': 4096,
                'enable_metrics': True,
                'health_check_port': 8001
            }
            
            with open('server_config.json', 'w') as f:
                json.dump(server_config, f)
            
            # Start server process
            env = os.environ.copy()
            env['PYTHONUNBUFFERED'] = '1'
            
            self.processes['server'] = subprocess.Popen(
                [sys.executable, '-m', 'music21_mcp.server_resilient'],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            logger.info(f"Started resilient server with PID: {self.processes['server'].pid}")
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
                    return True
            except:
                pass
            
            time.sleep(1)
            
        logger.error("✗ Server failed to become ready")
        return False
    
    def start_monitor(self):
        """Start the monitoring dashboard"""
        try:
            # Start monitor in a new terminal
            if sys.platform == 'darwin':  # macOS
                subprocess.Popen([
                    'osascript', '-e',
                    f'tell app "Terminal" to do script "cd {os.getcwd()} && {sys.executable} tests/stress_test_monitor.py"'
                ])
            elif sys.platform == 'linux':
                subprocess.Popen([
                    'gnome-terminal', '--',
                    sys.executable, 'tests/stress_test_monitor.py'
                ])
            else:  # Windows
                subprocess.Popen([
                    'start', 'cmd', '/k',
                    sys.executable, 'tests/stress_test_monitor.py'
                ], shell=True)
            
            logger.info("Started monitoring dashboard")
            
        except Exception as e:
            logger.warning(f"Could not start monitor in new terminal: {e}")
            logger.info("Run monitor manually: python tests/stress_test_monitor.py")
    
    def run_stress_test(self) -> bool:
        """Run the stress test"""
        try:
            # Build stress test command
            cmd = [
                sys.executable,
                'tests/production_stress_test.py',
                '--hours', str(self.config['duration_hours']),
                '--users', str(self.config['concurrent_users']),
                '--kill-interval', str(self.config['kill_interval_minutes']),
                '--recovery-timeout', str(self.config['recovery_timeout_seconds'])
            ]
            
            # Run stress test
            self.processes['stress_test'] = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            logger.info(f"Started stress test with PID: {self.processes['stress_test'].pid}")
            
            # Stream output
            for line in iter(self.processes['stress_test'].stdout.readline, ''):
                if line:
                    print(line.rstrip())
            
            # Wait for completion
            return_code = self.processes['stress_test'].wait()
            
            return return_code == 0
            
        except Exception as e:
            logger.error(f"Stress test failed: {e}")
            return False
    
    def collect_results(self):
        """Collect test results"""
        self.test_results['completed'] = datetime.now().isoformat()
        self.test_results['duration'] = str(datetime.now() - self.start_time)
        
        # Find latest stress test report
        log_dir = Path('stress_test_logs')
        report_files = list(log_dir.glob('stress_test_report_*.json'))
        
        if report_files:
            latest_report = max(report_files, key=lambda p: p.stat().st_mtime)
            
            with open(latest_report) as f:
                stress_results = json.load(f)
            
            self.test_results['stress_test_results'] = stress_results
            
            # Check if test passed
            if stress_results['metrics']['success_rate'] >= 95.0:
                logger.info("✓ Success rate >= 95%")
            else:
                logger.error(f"✗ Success rate too low: {stress_results['metrics']['success_rate']:.2f}%")
                self.test_results['failure_reasons'].append("Success rate below 95%")
            
            if stress_results['metrics']['avg_recovery_time'] <= self.config['recovery_timeout_seconds']:
                logger.info("✓ Recovery time within limit")
            else:
                logger.error(f"✗ Recovery too slow: {stress_results['metrics']['avg_recovery_time']:.1f}s")
                self.test_results['failure_reasons'].append("Recovery time exceeded limit")
    
    def cleanup(self):
        """Clean up processes"""
        logger.info("\nCleaning up...")
        
        for name, process in self.processes.items():
            if process and process.poll() is None:
                logger.info(f"Terminating {name} process")
                process.terminate()
                
                # Give it time to shut down gracefully
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    logger.warning(f"Force killing {name} process")
                    process.kill()
    
    def generate_report(self):
        """Generate final test report"""
        report_path = Path('test_results') / f"production_test_{self.start_time.strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(report_path, 'w') as f:
            json.dump(self.test_results, f, indent=2)
        
        # Print summary
        print("\n" + "=" * 80)
        print("PRODUCTION TEST SUMMARY")
        print("=" * 80)
        print(f"Duration: {self.test_results['duration']}")
        print(f"Result: {'PASSED' if self.test_results['test_passed'] else 'FAILED'}")
        
        if self.test_results['failure_reasons']:
            print("\nFailure Reasons:")
            for reason in self.test_results['failure_reasons']:
                print(f"  - {reason}")
        
        if 'stress_test_results' in self.test_results:
            results = self.test_results['stress_test_results']
            print(f"\nSuccess Rate: {results['metrics']['success_rate']:.2f}%")
            print(f"Total Requests: {results['metrics']['total_requests']:,}")
            print(f"Failed Requests: {results['metrics']['failed_requests']:,}")
            print(f"Total Recoveries: {results['metrics']['total_recoveries']}")
            print(f"Avg Recovery Time: {results['metrics']['avg_recovery_time']:.1f}s")
        
        print(f"\nFull report saved to: {report_path}")
        print("=" * 80)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Run production stress test for Music21 MCP Server'
    )
    
    parser.add_argument(
        '--hours',
        type=float,
        default=24,
        help='Test duration in hours (default: 24)'
    )
    parser.add_argument(
        '--users',
        type=int,
        default=100,
        help='Number of concurrent users (default: 100)'
    )
    parser.add_argument(
        '--kill-interval',
        type=int,
        default=60,
        help='Process kill interval in minutes (default: 60)'
    )
    parser.add_argument(
        '--recovery-timeout',
        type=int,
        default=60,
        help='Recovery timeout in seconds (default: 60)'
    )
    parser.add_argument(
        '--no-monitor',
        action='store_true',
        help='Disable monitoring dashboard'
    )
    parser.add_argument(
        '--config',
        type=str,
        help='Path to config file'
    )
    
    args = parser.parse_args()
    
    # Build configuration
    config = {
        'duration_hours': args.hours,
        'concurrent_users': args.users,
        'kill_interval_minutes': args.kill_interval,
        'recovery_timeout_seconds': args.recovery_timeout,
        'enable_monitor': not args.no_monitor
    }
    
    # Load config file if provided
    if args.config and os.path.exists(args.config):
        with open(args.config) as f:
            config.update(json.load(f))
    
    # Run test
    orchestrator = ProductionTestOrchestrator(config)
    success = orchestrator.run()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()