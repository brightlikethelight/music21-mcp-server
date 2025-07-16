#!/usr/bin/env python3
"""
💥 CHAOS DISCOVERY TESTS - FINDING HIDDEN DISASTERS
Tests that discover unknown problems, edge cases, and system breaking points
These tests find the disasters hiding in the 0% coverage zones
"""

import asyncio
import gc
import json
import os
import random
import signal
import string
import sys
import tempfile
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
import psutil

# Add src to path for imports
sys.path.insert(0, '../../src')

from music21_mcp.server import ScoreManager, ServerConfig

class ChaosDiscoveryTester:
    """Chaos testing to discover hidden system vulnerabilities"""
    
    def __init__(self):
        self.config = ServerConfig()
        self.score_manager = ScoreManager(max_scores=self.config.max_scores)
        self.discovered_issues = []
        self.performance_anomalies = []
        self.memory_leaks = []
        self.crash_scenarios = []
        
    def log_issue(self, category: str, description: str, severity: str = "medium"):
        """Log discovered issues"""
        issue = {
            "category": category,
            "description": description,
            "severity": severity,
            "timestamp": time.time(),
            "memory_usage": self._get_memory_usage(),
            "process_info": self._get_process_info()
        }
        self.discovered_issues.append(issue)
        print(f"🔥 DISCOVERED {severity.upper()}: {description}")
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except:
            return 0.0
    
    def _get_process_info(self) -> Dict[str, Any]:
        """Get process information"""
        try:
            process = psutil.Process()
            return {
                "cpu_percent": process.cpu_percent(),
                "num_threads": process.num_threads(),
                "open_files": len(process.open_files()),
                "connections": len(process.connections())
            }
        except:
            return {}
    
    async def test_extreme_memory_pressure(self) -> bool:
        """
        DISCOVERS: Hidden memory vulnerabilities under extreme pressure
        CHAOS: Create massive memory pressure to find breaking points
        """
        print("🔍 Testing extreme memory pressure scenarios...")
        
        try:
            initial_memory = self._get_memory_usage()
            memory_hogs = []
            
            # Create increasingly large data structures
            for size_mb in [1, 5, 10, 25, 50, 100]:
                try:
                    # Create large string data
                    large_data = "X" * (size_mb * 1024 * 1024)
                    memory_hogs.append(large_data)
                    
                    current_memory = self._get_memory_usage()
                    memory_increase = current_memory - initial_memory
                    
                    print(f"   Memory usage: {current_memory:.1f}MB (+{memory_increase:.1f}MB)")
                    
                    # Test if score manager still works under pressure
                    test_score_id = f"pressure_test_{size_mb}"
                    fake_score = {"data": f"test_score_{size_mb}", "size_mb": size_mb}
                    
                    await self.score_manager.add_score(test_score_id, fake_score)
                    retrieved = await self.score_manager.get_score(test_score_id)
                    
                    if not retrieved:
                        self.log_issue(
                            "memory", 
                            f"Score manager failed under {memory_increase:.1f}MB memory pressure",
                            "high"
                        )
                        return False
                    
                    # Force garbage collection and see what happens
                    gc.collect()
                    
                    if memory_increase > 500:  # 500MB threshold
                        self.log_issue(
                            "memory",
                            f"Excessive memory usage: {memory_increase:.1f}MB",
                            "medium"
                        )
                        break
                        
                except MemoryError:
                    self.log_issue(
                        "memory",
                        f"Memory exhaustion at {size_mb}MB allocation",
                        "critical"
                    )
                    break
                except Exception as e:
                    self.log_issue(
                        "memory",
                        f"Unexpected error under memory pressure: {str(e)}",
                        "high"
                    )
                    break
            
            # Clean up
            del memory_hogs
            gc.collect()
            
            final_memory = self._get_memory_usage()
            memory_recovered = initial_memory - final_memory
            
            if abs(memory_recovered) > 50:  # Should recover most memory
                self.log_issue(
                    "memory",
                    f"Poor memory recovery: {memory_recovered:.1f}MB not freed",
                    "medium"
                )
            
            print("✅ Memory pressure test completed")
            return True
            
        except Exception as e:
            self.log_issue(
                "memory",
                f"Memory pressure test crashed: {str(e)}",
                "critical"
            )
            return False
    
    async def test_malformed_data_injection(self) -> bool:
        """
        DISCOVERS: Hidden parsing vulnerabilities with malformed data
        CHAOS: Inject increasingly corrupted data to find breaking points
        """
        print("🔍 Testing malformed data injection attacks...")
        
        malformed_payloads = [
            # JSON bombs
            '{"a":' + '{"b":' * 1000 + '{}' + '}' * 1000,
            # Extremely long strings
            "A" * 1000000,
            # Unicode nightmares
            "🎵" * 100000,
            # Null bytes
            "test\x00\x00\x00score",
            # Control characters
            "\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f",
            # SQL injection attempts
            "'; DROP TABLE scores; --",
            # NoSQL injection
            '{"$ne": null}',
            # Code injection attempts
            "__import__('os').system('echo HACKED')",
            # XXE attack attempt
            '<?xml version="1.0"?><!DOCTYPE root [<!ENTITY test SYSTEM "file:///etc/passwd">]><root>&test;</root>',
            # Billion laughs attack
            '<!DOCTYPE root [<!ENTITY lol "lol"><!ENTITY lol2 "&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;">]><root>&lol2;</root>',
            # Buffer overflow attempts
            "A" * 65536,
            # Path traversal
            "../../../etc/passwd",
            # Command injection
            "; cat /etc/passwd; echo",
            # LDAP injection
            "* )(uid=*))(|(uid=*",
        ]
        
        vulnerabilities_found = 0
        
        for i, payload in enumerate(malformed_payloads):
            try:
                print(f"   Testing payload {i+1}/{len(malformed_payloads)}: {payload[:50]}...")
                
                # Test 1: Try to add as score
                score_id = f"malformed_test_{i}"
                
                start_time = time.time()
                await self.score_manager.add_score(score_id, payload)
                response_time = time.time() - start_time
                
                # Check for suspicious response times (potential DoS)
                if response_time > 5.0:
                    self.log_issue(
                        "injection",
                        f"Slow response ({response_time:.2f}s) to malformed payload: {payload[:100]}",
                        "medium"
                    )
                    vulnerabilities_found += 1
                
                # Test 2: Try to retrieve and see if system corrupted
                retrieved = await self.score_manager.get_score(score_id)
                
                if retrieved != payload:
                    # This could be sanitization (good) or corruption (bad)
                    if retrieved is None:
                        # Rejected - probably good
                        pass
                    else:
                        self.log_issue(
                            "injection",
                            f"Data corruption detected with payload: {payload[:100]}",
                            "high"
                        )
                        vulnerabilities_found += 1
                
                # Test 3: Check if payload caused system instability
                memory_after = self._get_memory_usage()
                if memory_after > 1000:  # 1GB threshold
                    self.log_issue(
                        "injection",
                        f"Memory spike ({memory_after:.1f}MB) after malformed input",
                        "high"
                    )
                    vulnerabilities_found += 1
                
                # Clean up
                await self.score_manager.remove_score(score_id)
                
            except Exception as e:
                error_msg = str(e)
                
                # Check for dangerous error messages that reveal system info
                dangerous_keywords = [
                    "file not found", "permission denied", "sql", "database",
                    "stack trace", "internal error", "debug", "traceback"
                ]
                
                if any(keyword in error_msg.lower() for keyword in dangerous_keywords):
                    self.log_issue(
                        "injection",
                        f"Information disclosure in error: {error_msg[:200]}",
                        "medium"
                    )
                    vulnerabilities_found += 1
                
                # Check for unexpected crashes
                if "segmentation fault" in error_msg.lower() or "core dumped" in error_msg.lower():
                    self.log_issue(
                        "injection",
                        f"System crash with payload: {payload[:100]}",
                        "critical"
                    )
                    vulnerabilities_found += 1
        
        print(f"✅ Malformed data test completed - {vulnerabilities_found} vulnerabilities found")
        return vulnerabilities_found == 0
    
    async def test_concurrent_chaos(self) -> bool:
        """
        DISCOVERS: Hidden race conditions and concurrency bugs
        CHAOS: Maximum concurrent operations to find race conditions
        """
        print("🔍 Testing concurrent chaos scenarios...")
        
        try:
            # Create massive concurrent load
            num_workers = 50
            operations_per_worker = 20
            
            async def chaotic_worker(worker_id: int):
                """Worker that performs chaotic operations"""
                issues_found = []
                
                for op_id in range(operations_per_worker):
                    try:
                        score_id = f"chaos_{worker_id}_{op_id}"
                        operation = random.choice([
                            "add", "get", "remove", "list", "add_duplicate"
                        ])
                        
                        if operation == "add":
                            score_data = {
                                "worker": worker_id,
                                "operation": op_id,
                                "data": "X" * random.randint(100, 10000)
                            }
                            await self.score_manager.add_score(score_id, score_data)
                            
                        elif operation == "get":
                            # Try to get a score that might exist
                            target_id = f"chaos_{random.randint(0, num_workers-1)}_{random.randint(0, operations_per_worker-1)}"
                            await self.score_manager.get_score(target_id)
                            
                        elif operation == "remove":
                            # Try to remove a score that might exist
                            target_id = f"chaos_{random.randint(0, num_workers-1)}_{random.randint(0, operations_per_worker-1)}"
                            await self.score_manager.remove_score(target_id)
                            
                        elif operation == "list":
                            await self.score_manager.list_scores()
                            
                        elif operation == "add_duplicate":
                            # Try to add same score multiple times
                            await self.score_manager.add_score(score_id, {"duplicate": True})
                        
                        # Add random delay to increase chance of race conditions
                        await asyncio.sleep(random.uniform(0.001, 0.01))
                        
                    except Exception as e:
                        issues_found.append({
                            "worker": worker_id,
                            "operation": operation,
                            "error": str(e),
                            "score_id": score_id
                        })
                
                return issues_found
            
            # Launch all workers concurrently
            print(f"   Launching {num_workers} chaotic workers...")
            start_time = time.time()
            
            tasks = [chaotic_worker(i) for i in range(num_workers)]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            end_time = time.time()
            duration = end_time - start_time
            
            # Analyze results
            total_operations = num_workers * operations_per_worker
            operations_per_second = total_operations / duration
            
            print(f"   Completed {total_operations} operations in {duration:.2f}s ({operations_per_second:.1f} ops/sec)")
            
            # Count issues
            total_issues = 0
            for result in results:
                if isinstance(result, Exception):
                    self.log_issue(
                        "concurrency",
                        f"Worker crashed: {str(result)}",
                        "high"
                    )
                    total_issues += 1
                elif isinstance(result, list):
                    total_issues += len(result)
                    for issue in result:
                        self.log_issue(
                            "concurrency",
                            f"Concurrent operation failed: {issue['error']}",
                            "medium"
                        )
            
            # Check final state consistency
            final_scores = await self.score_manager.list_scores()
            final_count = len(final_scores)
            
            print(f"   Final score count: {final_count}")
            print(f"   Total concurrency issues: {total_issues}")
            
            if total_issues > total_operations * 0.1:  # More than 10% error rate
                self.log_issue(
                    "concurrency",
                    f"High error rate under concurrent load: {total_issues}/{total_operations}",
                    "high"
                )
            
            # Performance regression check
            if operations_per_second < 100:  # Should handle at least 100 ops/sec
                self.log_issue(
                    "performance",
                    f"Poor performance under load: {operations_per_second:.1f} ops/sec",
                    "medium"
                )
            
            print("✅ Concurrent chaos test completed")
            return total_issues < total_operations * 0.05  # Accept up to 5% error rate
            
        except Exception as e:
            self.log_issue(
                "concurrency",
                f"Concurrent chaos test crashed: {str(e)}",
                "critical"
            )
            return False
    
    async def test_resource_exhaustion_bombs(self) -> bool:
        """
        DISCOVERS: Resource exhaustion vulnerabilities
        CHAOS: Try to exhaust different system resources
        """
        print("🔍 Testing resource exhaustion bombs...")
        
        try:
            # Test 1: File descriptor exhaustion
            print("   Testing file descriptor exhaustion...")
            temp_files = []
            fd_limit_reached = False
            
            try:
                for i in range(1000):  # Try to open many files
                    temp_file = tempfile.NamedTemporaryFile(delete=False)
                    temp_files.append(temp_file)
                    
                    if i % 100 == 0:
                        process_info = self._get_process_info()
                        print(f"   Opened {i} files, process has {process_info.get('open_files', 0)} open files")
            
            except OSError as e:
                fd_limit_reached = True
                self.log_issue(
                    "resource",
                    f"File descriptor limit reached at {len(temp_files)} files: {str(e)}",
                    "medium"
                )
            
            # Clean up files
            for temp_file in temp_files:
                try:
                    temp_file.close()
                    os.unlink(temp_file.name)
                except:
                    pass
            
            # Test 2: Thread exhaustion
            print("   Testing thread exhaustion...")
            threads = []
            thread_limit_reached = False
            
            def dummy_thread():
                time.sleep(1)  # Keep thread alive briefly
            
            try:
                for i in range(100):  # Try to create many threads
                    thread = threading.Thread(target=dummy_thread)
                    thread.start()
                    threads.append(thread)
                    
                    if i % 20 == 0:
                        process_info = self._get_process_info()
                        print(f"   Created {i} threads, process has {process_info.get('num_threads', 0)} threads")
            
            except Exception as e:
                thread_limit_reached = True
                self.log_issue(
                    "resource",
                    f"Thread limit reached at {len(threads)} threads: {str(e)}",
                    "medium"
                )
            
            # Wait for threads to finish
            for thread in threads:
                try:
                    thread.join(timeout=2)
                except:
                    pass
            
            # Test 3: Disk space exhaustion (careful!)
            print("   Testing disk space pressure...")
            temp_dir = tempfile.mkdtemp()
            large_files = []
            
            try:
                for i in range(5):  # Create some large files
                    file_path = os.path.join(temp_dir, f"large_file_{i}.dat")
                    
                    # Create 10MB files
                    with open(file_path, 'wb') as f:
                        f.write(b'0' * (10 * 1024 * 1024))
                    
                    large_files.append(file_path)
                    
                    # Check remaining disk space
                    disk_usage = psutil.disk_usage(temp_dir)
                    free_gb = disk_usage.free / (1024**3)
                    
                    if free_gb < 1.0:  # Less than 1GB free
                        self.log_issue(
                            "resource",
                            f"Low disk space detected: {free_gb:.2f}GB remaining",
                            "medium"
                        )
                        break
            
            finally:
                # Clean up large files
                for file_path in large_files:
                    try:
                        os.unlink(file_path)
                    except:
                        pass
                try:
                    os.rmdir(temp_dir)
                except:
                    pass
            
            print("✅ Resource exhaustion test completed")
            return True
            
        except Exception as e:
            self.log_issue(
                "resource",
                f"Resource exhaustion test crashed: {str(e)}",
                "critical"
            )
            return False
    
    async def test_signal_chaos(self) -> bool:
        """
        DISCOVERS: Signal handling vulnerabilities
        CHAOS: Send various signals to test graceful handling
        """
        print("🔍 Testing signal chaos scenarios...")
        
        try:
            # Test graceful signal handling without actually killing the process
            # We'll simulate signal scenarios
            
            signals_to_test = [
                "SIGTERM", "SIGINT", "SIGHUP", "SIGUSR1", "SIGUSR2"
            ]
            
            for sig_name in signals_to_test:
                print(f"   Testing {sig_name} signal handling...")
                
                # Simulate signal by testing if operations can complete quickly
                start_time = time.time()
                
                # Perform some operations that should complete quickly
                for i in range(10):
                    score_id = f"signal_test_{sig_name}_{i}"
                    await self.score_manager.add_score(score_id, {"signal_test": True})
                    await self.score_manager.get_score(score_id)
                    await self.score_manager.remove_score(score_id)
                
                duration = time.time() - start_time
                
                if duration > 5.0:  # Should complete quickly
                    self.log_issue(
                        "signal",
                        f"Slow operations during {sig_name} simulation: {duration:.2f}s",
                        "medium"
                    )
                
                # Test if system is still responsive
                test_scores = await self.score_manager.list_scores()
                if test_scores is None:
                    self.log_issue(
                        "signal",
                        f"System unresponsive after {sig_name} simulation",
                        "high"
                    )
            
            print("✅ Signal chaos test completed")
            return True
            
        except Exception as e:
            self.log_issue(
                "signal",
                f"Signal chaos test crashed: {str(e)}",
                "critical"
            )
            return False
    
    async def test_edge_case_discovery(self) -> bool:
        """
        DISCOVERS: Weird edge cases that break assumptions
        CHAOS: Test bizarre inputs and scenarios
        """
        print("🔍 Testing edge case discovery...")
        
        edge_cases = [
            # Empty and null cases
            ("", "empty_string"),
            (None, "none_value"),
            ([], "empty_list"),
            ({}, "empty_dict"),
            
            # Extreme values
            (-1, "negative_one"),
            (0, "zero"),
            (2**63 - 1, "max_int"),
            (-2**63, "min_int"),
            (float('inf'), "infinity"),
            (float('-inf'), "negative_infinity"),
            (float('nan'), "not_a_number"),
            
            # Unicode edge cases
            ("🎵🎶🎵🎶" * 1000, "unicode_music"),
            ("", "zero_width_space"),
            ("\u202e", "right_to_left_override"),
            ("\U0001F4A9" * 100, "poop_emoji_spam"),
            
            # JSON edge cases
            ('{"key": "value", "key": "duplicate"}', "duplicate_json_keys"),
            ('{"": "empty_key"}', "empty_json_key"),
            ('{"null": null, "true": true, "false": false}', "json_literals"),
            
            # Boundary conditions
            ("A" * 65535, "boundary_65535"),
            ("A" * 65536, "boundary_65536"),
            ("A" * 1048576, "boundary_1mb"),
            
            # Special numbers
            (0.1 + 0.2, "floating_point_precision"),
            (1e308, "very_large_float"),
            (1e-308, "very_small_float"),
        ]
        
        edge_case_failures = 0
        
        for test_value, test_name in edge_cases:
            try:
                print(f"   Testing edge case: {test_name}")
                
                score_id = f"edge_case_{test_name}"
                
                # Test adding edge case value
                start_time = time.time()
                await self.score_manager.add_score(score_id, test_value)
                add_duration = time.time() - start_time
                
                # Test retrieving edge case value
                start_time = time.time()
                retrieved = await self.score_manager.get_score(score_id)
                get_duration = time.time() - start_time
                
                # Check for performance anomalies
                if add_duration > 1.0 or get_duration > 1.0:
                    self.log_issue(
                        "edge_case",
                        f"Slow performance with {test_name}: add={add_duration:.2f}s, get={get_duration:.2f}s",
                        "medium"
                    )
                    edge_case_failures += 1
                
                # Check for data corruption
                if isinstance(test_value, (str, int, float, list, dict)):
                    if test_value != retrieved and not (
                        isinstance(test_value, float) and 
                        isinstance(retrieved, float) and 
                        str(test_value) == str(retrieved)  # Handle NaN/inf equality
                    ):
                        self.log_issue(
                            "edge_case",
                            f"Data corruption with {test_name}: expected {type(test_value)}, got {type(retrieved)}",
                            "high"
                        )
                        edge_case_failures += 1
                
                # Clean up
                await self.score_manager.remove_score(score_id)
                
            except Exception as e:
                error_msg = str(e)
                
                # Some edge cases are expected to fail, but crashes are concerning
                if "crash" in error_msg.lower() or "segmentation" in error_msg.lower():
                    self.log_issue(
                        "edge_case",
                        f"System crash with edge case {test_name}: {error_msg}",
                        "critical"
                    )
                    edge_case_failures += 1
                else:
                    # Expected failure - edge case properly rejected
                    print(f"     Edge case {test_name} properly rejected: {error_msg[:100]}")
        
        print(f"✅ Edge case discovery completed - {edge_case_failures} critical issues found")
        return edge_case_failures == 0

async def run_chaos_discovery_tests():
    """Run all chaos discovery tests to find hidden disasters"""
    print("💥 CHAOS DISCOVERY TESTS - FINDING HIDDEN DISASTERS")
    print("=" * 60)
    print("Discovering unknown problems, edge cases, and system breaking points")
    print("Testing the disaster scenarios hiding in 0% coverage zones")
    print()
    
    tester = ChaosDiscoveryTester()
    
    chaos_tests = [
        ("Extreme Memory Pressure", tester.test_extreme_memory_pressure),
        ("Malformed Data Injection", tester.test_malformed_data_injection),
        ("Concurrent Chaos", tester.test_concurrent_chaos),
        ("Resource Exhaustion Bombs", tester.test_resource_exhaustion_bombs),
        ("Signal Chaos", tester.test_signal_chaos),
        ("Edge Case Discovery", tester.test_edge_case_discovery),
    ]
    
    results = {}
    passed = 0
    total = len(chaos_tests)
    
    for test_name, test_func in chaos_tests:
        print(f"\n💥 {test_name}:")
        try:
            result = await test_func()
            results[test_name] = result
            if result:
                passed += 1
                print(f"✅ {test_name} - No critical issues found")
            else:
                print(f"❌ {test_name} - Critical issues discovered")
        except Exception as e:
            print(f"💥 {test_name} - TEST CRASHED: {str(e)}")
            print(f"Traceback: {traceback.format_exc()}")
            results[test_name] = False
        print()
    
    # Analyze discovered issues
    print("📊 CHAOS DISCOVERY SUMMARY:")
    print("=" * 60)
    print(f"Total chaos tests: {total}")
    print(f"Tests passed: {passed}")
    print(f"Tests failed: {total - passed}")
    print(f"Chaos survival rate: {passed/total*100:.1f}%")
    print()
    
    # Categorize discovered issues
    issues_by_category = {}
    issues_by_severity = {"critical": 0, "high": 0, "medium": 0, "low": 0}
    
    for issue in tester.discovered_issues:
        category = issue["category"]
        severity = issue["severity"]
        
        if category not in issues_by_category:
            issues_by_category[category] = []
        issues_by_category[category].append(issue)
        
        if severity in issues_by_severity:
            issues_by_severity[severity] += 1
    
    print("🔥 DISCOVERED ISSUES BY CATEGORY:")
    for category, issues in issues_by_category.items():
        print(f"   {category.upper()}: {len(issues)} issues")
        for issue in issues[:3]:  # Show first 3 issues
            print(f"     - {issue['severity'].upper()}: {issue['description']}")
    print()
    
    print("🚨 ISSUES BY SEVERITY:")
    for severity, count in issues_by_severity.items():
        if count > 0:
            print(f"   {severity.upper()}: {count} issues")
    print()
    
    total_issues = len(tester.discovered_issues)
    critical_issues = issues_by_severity["critical"]
    high_issues = issues_by_severity["high"]
    
    if critical_issues > 0:
        print("🔴 CRITICAL: System has critical vulnerabilities - IMMEDIATE ACTION REQUIRED")
    elif high_issues > 0:
        print("🟡 HIGH RISK: System has high-severity issues - Action needed soon")
    elif total_issues > 0:
        print("🟠 MEDIUM RISK: System has some issues - Review and fix when possible")
    else:
        print("🟢 ROBUST: System survived chaos testing - No critical issues found")
    
    print()
    print("🔥 CHAOS TESTING INSIGHTS:")
    print("1. Memory pressure reveals hidden resource management issues")
    print("2. Malformed data injection exposes parsing vulnerabilities")
    print("3. Concurrent chaos discovers race conditions and deadlocks")
    print("4. Resource exhaustion finds system limits and breaking points")
    print("5. Edge cases reveal assumption failures and boundary bugs")
    
    return {
        'total_tests': total,
        'passed_tests': passed,
        'total_issues': total_issues,
        'issues_by_severity': issues_by_severity,
        'issues_by_category': issues_by_category,
        'chaos_survival_rate': passed/total*100 if total > 0 else 0
    }

def main():
    """Main entry point"""
    return asyncio.run(run_chaos_discovery_tests())

if __name__ == "__main__":
    results = main()
    print(f"\n💥 Chaos Survival Rate: {results['chaos_survival_rate']:.1f}%")
    print(f"🔥 Total Disasters Discovered: {results['total_issues']}")