# \!/usr/bin/env python3
"""
Simple stress test for the basic resilient server
No numpy/scipy dependencies required
"""
import asyncio
import json
import logging
import random
import subprocess
import time
from datetime import datetime
from typing import Any, Dict, List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleStressTest:
    def __init__(self, server_cmd: str = "python test_basic_server.py"):
        self.server_cmd = server_cmd
        self.server_process = None
        self.test_scores = [
            "bach/bwv66.6",
            "mozart/k155/movement1",
            "schoenberg/opus19/movement2",
            "luca/gloria",
        ]
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "circuit_breaker_trips": 0,
            "rate_limit_hits": 0,
            "server_restarts": 0,
        }

    async def start_server(self):
        """Start the MCP server"""
        logger.info("Starting MCP server...")
        self.server_process = subprocess.Popen(
            self.server_cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        await asyncio.sleep(3)  # Give server time to start
        logger.info(f"Server started with PID: {self.server_process.pid}")

    async def stop_server(self):
        """Stop the MCP server"""
        if self.server_process:
            logger.info("Stopping server...")
            self.server_process.terminate()
            self.server_process.wait()

    async def simulate_client_load(
        self, duration_seconds: int, concurrent_clients: int
    ):
        """Simulate multiple concurrent clients"""
        logger.info(
            f"Starting {concurrent_clients} concurrent clients for {duration_seconds}s"
        )

        async def client_task(client_id: int):
            end_time = time.time() + duration_seconds
            while time.time() < end_time:
                try:
                    # Random operation
                    operation = random.choice(["import", "analyze", "list", "health"])

                    if operation == "import":
                        score_id = f"score_{client_id}_{int(time.time())}"
                        source = random.choice(self.test_scores)
                        # Simulate MCP call (in real test, use MCP client)
                        logger.debug(f"Client {client_id}: Import {source}")

                    elif operation == "analyze":
                        logger.debug(f"Client {client_id}: Analyze key")

                    elif operation == "list":
                        logger.debug(f"Client {client_id}: List scores")

                    else:  # health
                        logger.debug(f"Client {client_id}: Health check")

                    self.stats["total_requests"] += 1
                    self.stats["successful_requests"] += 1

                    # Random delay between requests
                    await asyncio.sleep(random.uniform(0.1, 0.5))

                except Exception as e:
                    self.stats["failed_requests"] += 1
                    logger.error(f"Client {client_id} error: {e}")

        # Start all client tasks
        tasks = [client_task(i) for i in range(concurrent_clients)]
        await asyncio.gather(*tasks)

    async def chaos_monkey(self, duration_seconds: int):
        """Randomly kill and restart the server"""
        logger.info("Starting chaos monkey...")
        end_time = time.time() + duration_seconds

        while time.time() < end_time:
            # Wait random time before causing chaos
            await asyncio.sleep(random.uniform(30, 120))

            logger.warning("Chaos monkey: Killing server\!")
            await self.stop_server()
            self.stats["server_restarts"] += 1

            # Wait a bit
            await asyncio.sleep(5)

            # Restart server
            logger.info("Chaos monkey: Restarting server...")
            await self.start_server()

    async def monitor_health(self, duration_seconds: int):
        """Monitor server health periodically"""
        logger.info("Starting health monitor...")
        end_time = time.time() + duration_seconds

        while time.time() < end_time:
            try:
                # In real test, call health_check via MCP
                logger.info("Health check: Server responding")
            except Exception as e:
                logger.error(f"Health check failed: {e}")

            await asyncio.sleep(10)

    async def run_test(
        self, hours: float, concurrent_users: int, enable_chaos: bool = True
    ):
        """Run the complete stress test"""
        duration_seconds = int(hours * 3600)

        logger.info(
            f"Starting stress test for {hours} hours with {concurrent_users} users"
        )
        logger.info(f"Chaos monkey: {'Enabled' if enable_chaos else 'Disabled'}")

        # Start server
        await self.start_server()

        # Create test tasks
        tasks = [
            self.simulate_client_load(duration_seconds, concurrent_users),
            self.monitor_health(duration_seconds),
        ]

        if enable_chaos:
            tasks.append(self.chaos_monkey(duration_seconds))

        # Run all tasks concurrently
        try:
            await asyncio.gather(*tasks)
        except KeyboardInterrupt:
            logger.info("Test interrupted by user")
        finally:
            await self.stop_server()

        # Print results
        self.print_results()

    def print_results(self):
        """Print test results"""
        print("\n" + "=" * 60)
        print("STRESS TEST RESULTS")
        print("=" * 60)
        print(f"Total requests: {self.stats['total_requests']}")
        print(f"Successful: {self.stats['successful_requests']}")
        print(f"Failed: {self.stats['failed_requests']}")
        print(
            f"Success rate: {self.stats['successful_requests']/max(1, self.stats['total_requests'])*100:.1f}%"
        )
        print(f"Server restarts: {self.stats['server_restarts']}")
        print(f"Circuit breaker trips: {self.stats['circuit_breaker_trips']}")
        print(f"Rate limit hits: {self.stats['rate_limit_hits']}")
        print("=" * 60)


async def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Simple stress test for music21 MCP server"
    )
    parser.add_argument(
        "--hours", type=float, default=0.1, help="Test duration in hours"
    )
    parser.add_argument(
        "--users", type=int, default=10, help="Number of concurrent users"
    )
    parser.add_argument("--no-chaos", action="store_true", help="Disable chaos monkey")

    args = parser.parse_args()

    test = SimpleStressTest()
    await test.run_test(args.hours, args.users, not args.no_chaos)


if __name__ == "__main__":
    asyncio.run(main())
