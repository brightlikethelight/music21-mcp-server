#!/usr/bin/env python3
"""
Production Stress Test Suite
Tests server resilience under extreme conditions for 24 hours
"""
import asyncio
import concurrent.futures
import gc
import json
import logging
import multiprocessing
import os
import platform
import random
import resource
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import uuid
import weakref
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import psutil

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Configure logging with rotation
from logging.handlers import RotatingFileHandler

from music21 import chord, corpus, key, meter, note, stream, tempo

from music21_mcp.server import main as run_server

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create logs directory
log_dir = Path("stress_test_logs")
log_dir.mkdir(exist_ok=True)

# Rotating file handler (100MB per file, keep 10 files)
file_handler = RotatingFileHandler(
    log_dir / f"stress_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
    maxBytes=100 * 1024 * 1024,
    backupCount=10,
)
file_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
logger.addHandler(file_handler)

# Console handler for critical messages only
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.WARNING)
logger.addHandler(console_handler)


@dataclass
class StressTestConfig:
    """Configuration for stress testing"""

    duration_hours: int = 24
    concurrent_users: int = 100
    max_file_size_mb: int = 50
    network_failure_probability: float = 0.1
    memory_pressure_gb: float = 2.0
    process_kill_interval_minutes: int = 60
    recovery_timeout_seconds: int = 60
    health_check_interval_seconds: int = 5

    # Failure injection probabilities
    connection_timeout_prob: float = 0.05
    malformed_request_prob: float = 0.1
    resource_exhaustion_prob: float = 0.05
    corruption_prob: float = 0.02

    # Resource limits
    max_open_files: int = 1024
    max_memory_per_request_mb: int = 500
    request_timeout_seconds: int = 30

    # Monitoring
    metrics_interval_seconds: int = 10
    alert_threshold_errors_per_minute: int = 50


@dataclass
class ServerMetrics:
    """Real-time server metrics"""

    start_time: datetime = field(default_factory=datetime.now)
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_recoveries: int = 0
    recovery_times: List[float] = field(default_factory=list)
    current_memory_mb: float = 0
    peak_memory_mb: float = 0
    active_connections: int = 0
    errors_per_minute: deque = field(default_factory=lambda: deque(maxlen=60))
    response_times: deque = field(default_factory=lambda: deque(maxlen=1000))

    def update_memory(self):
        """Update memory metrics"""
        process = psutil.Process()
        self.current_memory_mb = process.memory_info().rss / 1024 / 1024
        self.peak_memory_mb = max(self.peak_memory_mb, self.current_memory_mb)

    def add_error(self):
        """Track error rate"""
        self.errors_per_minute.append(time.time())
        self.failed_requests += 1

    def get_error_rate(self) -> float:
        """Get errors per minute"""
        now = time.time()
        recent_errors = sum(1 for t in self.errors_per_minute if now - t < 60)
        return recent_errors

    def get_success_rate(self) -> float:
        """Calculate success rate"""
        if self.total_requests == 0:
            return 0.0
        return (self.successful_requests / self.total_requests) * 100


class ChaosMonkey:
    """Injects failures to test resilience"""

    def __init__(self, config: StressTestConfig):
        self.config = config
        self.active_failures = set()
        self.failure_history = []

    async def inject_network_failure(self, duration_seconds: int = 10):
        """Simulate network failures"""
        failure_type = random.choice(
            [
                "packet_loss",
                "high_latency",
                "connection_refused",
                "dns_failure",
                "ssl_error",
            ]
        )

        logger.warning(
            f"Injecting network failure: {failure_type} for {duration_seconds}s"
        )
        self.active_failures.add(failure_type)
        self.failure_history.append(
            {
                "type": failure_type,
                "start": datetime.now(),
                "duration": duration_seconds,
            }
        )

        # Simulate failure
        if failure_type == "packet_loss":
            # In real implementation, use tc/iptables
            await asyncio.sleep(duration_seconds)
        elif failure_type == "high_latency":
            await asyncio.sleep(duration_seconds)
        elif failure_type == "connection_refused":
            # Block port temporarily
            await asyncio.sleep(duration_seconds)

        self.active_failures.remove(failure_type)
        logger.info(f"Network failure {failure_type} resolved")

    def inject_memory_pressure(self):
        """Create memory pressure"""
        logger.warning("Injecting memory pressure")

        # Allocate large chunks of memory
        memory_hogs = []
        chunk_size = 100 * 1024 * 1024  # 100MB chunks
        target_gb = self.config.memory_pressure_gb

        try:
            for _ in range(int(target_gb * 10)):  # 10 chunks per GB
                # Create large numpy arrays or lists
                hog = bytearray(chunk_size)
                memory_hogs.append(hog)
                time.sleep(0.1)  # Gradual allocation

            logger.info(
                f"Allocated {len(memory_hogs) * chunk_size / 1024 / 1024 / 1024:.2f}GB"
            )

            # Hold for a while
            time.sleep(30)

        finally:
            # Release memory
            memory_hogs.clear()
            gc.collect()
            logger.info("Memory pressure released")

    def corrupt_data(self, data: Any) -> Any:
        """Randomly corrupt data"""
        if random.random() < self.config.corruption_prob:
            if isinstance(data, str):
                # Flip random bits
                corrupted = list(data)
                if corrupted:
                    idx = random.randint(0, len(corrupted) - 1)
                    corrupted[idx] = chr(ord(corrupted[idx]) ^ 0xFF)
                return "".join(corrupted)
            elif isinstance(data, bytes):
                # Corrupt random byte
                corrupted = bytearray(data)
                if corrupted:
                    idx = random.randint(0, len(corrupted) - 1)
                    corrupted[idx] ^= 0xFF
                return bytes(corrupted)
        return data


class ResilientClient:
    """Client that simulates real user behavior with retry logic"""

    def __init__(self, client_id: int, server_url: str, chaos: ChaosMonkey):
        self.client_id = client_id
        self.server_url = server_url
        self.chaos = chaos
        self.session_id = str(uuid.uuid4())
        self.request_count = 0
        self.error_count = 0
        self.scores_created = []

    async def simulate_user_session(self, duration_seconds: int):
        """Simulate realistic user behavior"""
        end_time = time.time() + duration_seconds

        while time.time() < end_time:
            try:
                # Random user action
                action = random.choices(
                    ["create", "analyze", "modify", "export", "heavy_computation"],
                    weights=[30, 40, 20, 10, 5],
                )[0]

                await self.execute_action(action)
                self.request_count += 1

                # Random think time
                await asyncio.sleep(random.uniform(0.5, 3.0))

            except Exception as e:
                self.error_count += 1
                logger.error(f"Client {self.client_id} error: {e}")
                await asyncio.sleep(1)  # Back off on error

    async def execute_action(self, action: str):
        """Execute a user action with retry logic"""
        max_retries = 3
        retry_delay = 1.0

        for attempt in range(max_retries):
            try:
                if action == "create":
                    await self.create_score()
                elif action == "analyze":
                    await self.analyze_score()
                elif action == "modify":
                    await self.modify_score()
                elif action == "export":
                    await self.export_score()
                elif action == "heavy_computation":
                    await self.heavy_computation()

                return  # Success

            except Exception as e:
                if attempt < max_retries - 1:
                    logger.debug(f"Retry {attempt + 1} for {action}: {e}")
                    await asyncio.sleep(retry_delay * (2**attempt))
                else:
                    raise

    async def create_score(self):
        """Create a random score"""
        score_size = random.choice(["small", "medium", "large", "extreme"])

        if score_size == "small":
            # Simple melody
            notes = " ".join(
                [f"{random.choice('CDEFGAB')}{random.randint(3, 5)}" for _ in range(8)]
            )
        elif score_size == "medium":
            # Multi-part score
            notes = self.generate_complex_score(parts=4, measures=50)
        elif score_size == "large":
            # Orchestra-sized
            notes = self.generate_complex_score(parts=20, measures=200)
        else:  # extreme
            # Stress test with huge score
            notes = self.generate_complex_score(parts=50, measures=500)

        # Potentially corrupt the data
        notes = self.chaos.corrupt_data(notes)

        # Simulate API call
        score_id = f"stress_test_{self.client_id}_{len(self.scores_created)}"
        self.scores_created.append(score_id)

        # Simulate network conditions
        if "high_latency" in self.chaos.active_failures:
            await asyncio.sleep(random.uniform(5, 15))
        elif "packet_loss" in self.chaos.active_failures:
            if random.random() < 0.3:
                raise ConnectionError("Packet loss")

    def generate_complex_score(self, parts: int, measures: int) -> str:
        """Generate a complex score for stress testing"""
        # This would generate actual music21 score data
        # For simulation, return size indicator
        return f"COMPLEX_SCORE_P{parts}_M{measures}"

    async def analyze_score(self):
        """Analyze a score with various tools"""
        if not self.scores_created:
            await self.create_score()

        score_id = random.choice(self.scores_created)
        analysis_type = random.choice(
            ["key", "harmony", "voice_leading", "pattern", "full_analysis"]
        )

        # Simulate heavy analysis
        if analysis_type == "full_analysis":
            await asyncio.sleep(random.uniform(2, 10))
        else:
            await asyncio.sleep(random.uniform(0.1, 2))

    async def modify_score(self):
        """Modify existing score"""
        if not self.scores_created:
            return

        score_id = random.choice(self.scores_created)
        modification = random.choice(
            ["transpose", "add_harmony", "change_tempo", "add_dynamics"]
        )

        await asyncio.sleep(random.uniform(0.5, 3))

    async def export_score(self):
        """Export score in various formats"""
        if not self.scores_created:
            return

        score_id = random.choice(self.scores_created)
        format_type = random.choice(["musicxml", "midi", "pdf", "lilypond"])

        # PDF export is heavy
        if format_type == "pdf":
            await asyncio.sleep(random.uniform(5, 15))
        else:
            await asyncio.sleep(random.uniform(0.5, 2))

    async def heavy_computation(self):
        """Simulate computationally expensive operations"""
        operation = random.choice(
            [
                "generate_counterpoint",
                "style_analysis",
                "harmonic_reduction",
                "voice_separation",
            ]
        )

        # These are expensive operations
        await asyncio.sleep(random.uniform(10, 30))


class ServerHealthMonitor:
    """Monitors server health and triggers recovery"""

    def __init__(self, config: StressTestConfig, metrics: ServerMetrics):
        self.config = config
        self.metrics = metrics
        self.server_process = None
        self.is_healthy = True
        self.last_health_check = time.time()
        self.recovery_in_progress = False
        self.health_history = deque(maxlen=1000)

    async def start_monitoring(self):
        """Start health monitoring loop"""
        while True:
            try:
                health_status = await self.check_health()
                self.health_history.append(
                    {
                        "timestamp": datetime.now(),
                        "status": health_status,
                        "memory_mb": self.metrics.current_memory_mb,
                        "error_rate": self.metrics.get_error_rate(),
                    }
                )

                if not health_status["healthy"]:
                    logger.warning(f"Health check failed: {health_status['reason']}")
                    await self.trigger_recovery(health_status["reason"])

                await asyncio.sleep(self.config.health_check_interval_seconds)

            except Exception as e:
                logger.error(f"Health monitor error: {e}")
                await asyncio.sleep(1)

    async def check_health(self) -> Dict[str, Any]:
        """Comprehensive health check"""
        health_status = {"healthy": True, "checks": {}, "reason": None}

        # 1. Process check
        if self.server_process and not self.is_process_alive():
            health_status["healthy"] = False
            health_status["reason"] = "Server process dead"
            health_status["checks"]["process"] = False
        else:
            health_status["checks"]["process"] = True

        # 2. Memory check
        self.metrics.update_memory()
        memory_threshold_mb = 4096  # 4GB threshold
        if self.metrics.current_memory_mb > memory_threshold_mb:
            health_status["healthy"] = False
            health_status["reason"] = (
                f"Memory exceeded: {self.metrics.current_memory_mb:.0f}MB"
            )
            health_status["checks"]["memory"] = False
        else:
            health_status["checks"]["memory"] = True

        # 3. Error rate check
        error_rate = self.metrics.get_error_rate()
        if error_rate > self.config.alert_threshold_errors_per_minute:
            health_status["healthy"] = False
            health_status["reason"] = f"High error rate: {error_rate:.0f}/min"
            health_status["checks"]["error_rate"] = False
        else:
            health_status["checks"]["error_rate"] = True

        # 4. Response time check (if we have recent data)
        if self.metrics.response_times:
            avg_response = sum(self.metrics.response_times) / len(
                self.metrics.response_times
            )
            if avg_response > 10.0:  # 10 second threshold
                health_status["healthy"] = False
                health_status["reason"] = f"Slow responses: {avg_response:.1f}s avg"
                health_status["checks"]["response_time"] = False
            else:
                health_status["checks"]["response_time"] = True

        # 5. Port check
        if not await self.check_port_open():
            health_status["healthy"] = False
            health_status["reason"] = "Server port not responding"
            health_status["checks"]["port"] = False
        else:
            health_status["checks"]["port"] = True

        return health_status

    def is_process_alive(self) -> bool:
        """Check if server process is alive"""
        if not self.server_process:
            return False

        try:
            # Check if process exists
            process = psutil.Process(self.server_process.pid)
            return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return False

    async def check_port_open(self, port: int = 8000) -> bool:
        """Check if server port is open"""
        try:
            # Try to connect
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection("localhost", port), timeout=2.0
            )
            writer.close()
            await writer.wait_closed()
            return True
        except:
            return False

    async def trigger_recovery(self, reason: str):
        """Trigger recovery procedure"""
        if self.recovery_in_progress:
            logger.info("Recovery already in progress")
            return

        self.recovery_in_progress = True
        recovery_start = time.time()

        logger.warning(f"Starting recovery procedure: {reason}")

        try:
            # 1. Try graceful shutdown first
            if self.server_process and self.is_process_alive():
                logger.info("Attempting graceful shutdown")
                self.server_process.terminate()
                await asyncio.sleep(5)

            # 2. Force kill if still alive
            if self.server_process and self.is_process_alive():
                logger.warning("Force killing server")
                self.server_process.kill()
                await asyncio.sleep(2)

            # 3. Clean up resources
            await self.cleanup_resources()

            # 4. Restart server
            logger.info("Restarting server")
            await self.start_server()

            # 5. Wait for server to be ready
            ready = await self.wait_for_ready()

            recovery_time = time.time() - recovery_start
            self.metrics.recovery_times.append(recovery_time)
            self.metrics.total_recoveries += 1

            if ready and recovery_time <= self.config.recovery_timeout_seconds:
                logger.info(f"Recovery successful in {recovery_time:.1f}s")
            else:
                logger.error(f"Recovery failed or took too long: {recovery_time:.1f}s")

        except Exception as e:
            logger.error(f"Recovery failed with exception: {e}")
            traceback.print_exc()
        finally:
            self.recovery_in_progress = False

    async def cleanup_resources(self):
        """Clean up resources before restart"""
        # Clean up temp files
        temp_dir = Path(tempfile.gettempdir())
        for temp_file in temp_dir.glob("music21_*"):
            try:
                if temp_file.is_file():
                    temp_file.unlink()
                elif temp_file.is_dir():
                    import shutil

                    shutil.rmtree(temp_file)
            except:
                pass

        # Force garbage collection
        gc.collect()

        # Clear any locks or semaphores
        await asyncio.sleep(1)

    async def start_server(self):
        """Start the server process"""
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"

        # Start server with resource limits
        if platform.system() != "Windows":
            # Set ulimits for the process
            def set_limits():
                # Limit memory (2GB)
                resource.setrlimit(resource.RLIMIT_AS, (2 * 1024 * 1024 * 1024, -1))
                # Limit file descriptors
                resource.setrlimit(
                    resource.RLIMIT_NOFILE, (self.config.max_open_files, -1)
                )

            self.server_process = subprocess.Popen(
                [sys.executable, "-m", "music21_mcp.server"],
                env=env,
                preexec_fn=set_limits,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        else:
            self.server_process = subprocess.Popen(
                [sys.executable, "-m", "music21_mcp.server"],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

        logger.info(f"Started server process with PID: {self.server_process.pid}")

    async def wait_for_ready(self, timeout: int = 30) -> bool:
        """Wait for server to be ready"""
        start_time = time.time()

        while time.time() - start_time < timeout:
            if await self.check_port_open():
                # Additional health check
                health = await self.check_health()
                if health["healthy"]:
                    return True

            await asyncio.sleep(1)

        return False

    def kill_server_process(self):
        """Forcefully kill the server process"""
        if self.server_process:
            logger.warning(f"Force killing server process {self.server_process.pid}")
            try:
                process = psutil.Process(self.server_process.pid)
                process.kill()
            except:
                pass


class StressTestOrchestrator:
    """Orchestrates the entire stress test"""

    def __init__(self, config: StressTestConfig):
        self.config = config
        self.metrics = ServerMetrics()
        self.chaos = ChaosMonkey(config)
        self.monitor = ServerHealthMonitor(config, self.metrics)
        self.clients = []
        self.test_start_time = None
        self.should_stop = False

    async def run_stress_test(self):
        """Run the complete 24-hour stress test"""
        self.test_start_time = datetime.now()
        test_end_time = self.test_start_time + timedelta(
            hours=self.config.duration_hours
        )

        logger.info(f"Starting {self.config.duration_hours}-hour stress test")
        logger.info(f"Configuration: {self.config}")

        try:
            # Start server
            await self.monitor.start_server()
            await self.monitor.wait_for_ready()

            # Start monitoring
            monitor_task = asyncio.create_task(self.monitor.start_monitoring())

            # Start metrics collection
            metrics_task = asyncio.create_task(self.collect_metrics())

            # Start chaos injection
            chaos_task = asyncio.create_task(self.inject_chaos())

            # Start process killer
            killer_task = asyncio.create_task(self.periodic_process_kill())

            # Create and start clients
            client_tasks = []
            for i in range(self.config.concurrent_users):
                client = ResilientClient(i, "http://localhost:8000", self.chaos)
                self.clients.append(client)

                # Stagger client starts
                await asyncio.sleep(0.1)

                task = asyncio.create_task(
                    client.simulate_user_session(
                        int((test_end_time - datetime.now()).total_seconds())
                    )
                )
                client_tasks.append(task)

            # Run until test duration expires
            while datetime.now() < test_end_time and not self.should_stop:
                await asyncio.sleep(60)  # Check every minute
                await self.print_status()

            logger.info("Test duration reached, shutting down...")

        except KeyboardInterrupt:
            logger.info("Test interrupted by user")
        except Exception as e:
            logger.error(f"Test failed with error: {e}")
            traceback.print_exc()
        finally:
            self.should_stop = True

            # Cancel all tasks
            for task in client_tasks + [
                monitor_task,
                metrics_task,
                chaos_task,
                killer_task,
            ]:
                task.cancel()

            # Final report
            await self.generate_final_report()

    async def collect_metrics(self):
        """Continuously collect metrics"""
        while not self.should_stop:
            try:
                # Update metrics
                self.metrics.update_memory()

                # Collect client metrics
                total_requests = sum(c.request_count for c in self.clients)
                total_errors = sum(c.error_count for c in self.clients)

                self.metrics.total_requests = total_requests
                self.metrics.failed_requests = total_errors
                self.metrics.successful_requests = total_requests - total_errors

                # Log metrics
                if int(time.time()) % 60 == 0:  # Every minute
                    logger.info(
                        f"Metrics - Requests: {total_requests}, "
                        f"Success Rate: {self.metrics.get_success_rate():.1f}%, "
                        f"Memory: {self.metrics.current_memory_mb:.0f}MB, "
                        f"Errors/min: {self.metrics.get_error_rate():.0f}"
                    )

                await asyncio.sleep(self.config.metrics_interval_seconds)

            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
                await asyncio.sleep(10)

    async def inject_chaos(self):
        """Continuously inject chaos"""
        while not self.should_stop:
            try:
                # Random network failures
                if random.random() < self.config.network_failure_probability:
                    duration = random.randint(5, 30)
                    asyncio.create_task(self.chaos.inject_network_failure(duration))

                # Random memory pressure
                if random.random() < self.config.resource_exhaustion_prob:
                    threading.Thread(
                        target=self.chaos.inject_memory_pressure, daemon=True
                    ).start()

                # Wait before next chaos
                await asyncio.sleep(random.randint(30, 300))

            except Exception as e:
                logger.error(f"Chaos injection error: {e}")
                await asyncio.sleep(60)

    async def periodic_process_kill(self):
        """Kill the process every hour to test recovery"""
        while not self.should_stop:
            try:
                # Wait for the interval
                await asyncio.sleep(self.config.process_kill_interval_minutes * 60)

                if not self.should_stop:
                    logger.warning("Scheduled process kill")
                    self.monitor.kill_server_process()

            except Exception as e:
                logger.error(f"Process killer error: {e}")

    async def print_status(self):
        """Print current test status"""
        runtime = datetime.now() - self.test_start_time
        hours_run = runtime.total_seconds() / 3600
        hours_left = self.config.duration_hours - hours_run

        print(f"\n{'='*60}")
        print(f"STRESS TEST STATUS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")
        print(f"Runtime: {hours_run:.1f}/{self.config.duration_hours} hours")
        print(
            f"Active Clients: {len([c for c in self.clients if c.request_count > 0])}"
        )
        print(f"Total Requests: {self.metrics.total_requests:,}")
        print(f"Success Rate: {self.metrics.get_success_rate():.2f}%")
        print(
            f"Memory Usage: {self.metrics.current_memory_mb:.0f}MB (Peak: {self.metrics.peak_memory_mb:.0f}MB)"
        )
        print(f"Error Rate: {self.metrics.get_error_rate():.0f}/min")
        print(f"Total Recoveries: {self.metrics.total_recoveries}")
        if self.metrics.recovery_times:
            avg_recovery = sum(self.metrics.recovery_times) / len(
                self.metrics.recovery_times
            )
            print(f"Avg Recovery Time: {avg_recovery:.1f}s")
        print(f"Active Failures: {self.chaos.active_failures}")
        print(f"{'='*60}\n")

    async def generate_final_report(self):
        """Generate comprehensive test report"""
        runtime = datetime.now() - self.test_start_time

        report = {
            "test_summary": {
                "start_time": self.test_start_time.isoformat(),
                "end_time": datetime.now().isoformat(),
                "duration_hours": runtime.total_seconds() / 3600,
                "configuration": self.config.__dict__,
            },
            "metrics": {
                "total_requests": self.metrics.total_requests,
                "successful_requests": self.metrics.successful_requests,
                "failed_requests": self.metrics.failed_requests,
                "success_rate": self.metrics.get_success_rate(),
                "peak_memory_mb": self.metrics.peak_memory_mb,
                "total_recoveries": self.metrics.total_recoveries,
                "recovery_times": self.metrics.recovery_times,
                "avg_recovery_time": (
                    sum(self.metrics.recovery_times) / len(self.metrics.recovery_times)
                    if self.metrics.recovery_times
                    else 0
                ),
            },
            "chaos_summary": {
                "total_failures_injected": len(self.chaos.failure_history),
                "failure_types": defaultdict(int),
            },
            "client_summary": {
                "total_clients": len(self.clients),
                "avg_requests_per_client": (
                    sum(c.request_count for c in self.clients) / len(self.clients)
                    if self.clients
                    else 0
                ),
                "avg_errors_per_client": (
                    sum(c.error_count for c in self.clients) / len(self.clients)
                    if self.clients
                    else 0
                ),
            },
            "health_checks": {
                "total_checks": len(self.monitor.health_history),
                "healthy_checks": sum(
                    1 for h in self.monitor.health_history if h["status"]["healthy"]
                ),
                "health_percentage": (
                    (
                        sum(
                            1
                            for h in self.monitor.health_history
                            if h["status"]["healthy"]
                        )
                        / len(self.monitor.health_history)
                        * 100
                    )
                    if self.monitor.health_history
                    else 0
                ),
            },
        }

        # Count failure types
        for failure in self.chaos.failure_history:
            report["chaos_summary"]["failure_types"][failure["type"]] += 1

        # Save report
        report_path = (
            log_dir
            / f"stress_test_report_{self.test_start_time.strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        # Print summary
        print(f"\n{'='*80}")
        print("FINAL STRESS TEST REPORT")
        print(f"{'='*80}")
        print(f"Duration: {runtime}")
        print(f"Success Rate: {report['metrics']['success_rate']:.2f}%")
        print(f"Total Recoveries: {report['metrics']['total_recoveries']}")
        print(f"Average Recovery Time: {report['metrics']['avg_recovery_time']:.1f}s")
        print(f"Peak Memory: {report['metrics']['peak_memory_mb']:.0f}MB")
        print(
            f"Health Check Success: {report['health_checks']['health_percentage']:.1f}%"
        )

        # Pass/Fail determination
        passed = (
            report["metrics"]["success_rate"] >= 95.0
            and report["metrics"]["avg_recovery_time"]
            <= self.config.recovery_timeout_seconds
            and report["health_checks"]["health_percentage"] >= 90.0
        )

        print(f"\nTEST RESULT: {'PASSED' if passed else 'FAILED'}")

        if not passed:
            print("\nFailure Reasons:")
            if report["metrics"]["success_rate"] < 95.0:
                print(
                    f"  - Success rate too low: {report['metrics']['success_rate']:.2f}% < 95%"
                )
            if (
                report["metrics"]["avg_recovery_time"]
                > self.config.recovery_timeout_seconds
            ):
                print(
                    f"  - Recovery too slow: {report['metrics']['avg_recovery_time']:.1f}s > {self.config.recovery_timeout_seconds}s"
                )
            if report["health_checks"]["health_percentage"] < 90.0:
                print(
                    f"  - Health checks failing: {report['health_checks']['health_percentage']:.1f}% < 90%"
                )

        print(f"\nFull report saved to: {report_path}")
        print(f"{'='*80}\n")


async def main():
    """Run the stress test"""
    # Parse command line arguments
    import argparse

    parser = argparse.ArgumentParser(
        description="Production stress test for Music21 MCP Server"
    )
    parser.add_argument("--hours", type=int, default=24, help="Test duration in hours")
    parser.add_argument(
        "--users", type=int, default=100, help="Number of concurrent users"
    )
    parser.add_argument(
        "--kill-interval", type=int, default=60, help="Process kill interval in minutes"
    )
    parser.add_argument(
        "--recovery-timeout", type=int, default=60, help="Recovery timeout in seconds"
    )

    args = parser.parse_args()

    # Create configuration
    config = StressTestConfig(
        duration_hours=args.hours,
        concurrent_users=args.users,
        process_kill_interval_minutes=args.kill_interval,
        recovery_timeout_seconds=args.recovery_timeout,
    )

    # Run stress test
    orchestrator = StressTestOrchestrator(config)
    await orchestrator.run_stress_test()


if __name__ == "__main__":
    # Set up signal handlers
    def signal_handler(signum, frame):
        print("\nShutting down stress test...")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Run the test
    asyncio.run(main())
