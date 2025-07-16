#!/usr/bin/env python3
"""
📁💥 FILE SYSTEM CHAOS & CONCURRENCY BOMB TESTS
Tests file system failures, I/O disasters, and sophisticated concurrency bombs
These tests find the disasters hiding in file operations and parallel processing
"""

import asyncio
import concurrent.futures
import gc
import os
import random
import shutil
import string
import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
import psutil

# Add src to path for imports
sys.path.insert(0, 'src')

from music21_mcp.server import ScoreManager, ServerConfig

class FileSystemChaosTester:
    """File system and concurrency chaos testing"""
    
    def __init__(self):
        self.config = ServerConfig()
        self.score_manager = ScoreManager(max_scores=self.config.max_scores)
        self.discovered_issues = []
        self.temp_dirs = []
        self.test_files = []
        
    def log_issue(self, category: str, description: str, severity: str = "medium"):
        """Log discovered issues"""
        issue = {
            "category": category,
            "description": description,
            "severity": severity,
            "timestamp": time.time(),
            "memory_usage": self._get_memory_usage(),
            "thread_count": threading.active_count()
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
    
    def cleanup(self):
        """Clean up test resources"""
        for temp_dir in self.temp_dirs:
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
            except:
                pass
        
        for test_file in self.test_files:
            try:
                os.unlink(test_file)
            except:
                pass
    
    async def test_corrupted_file_scenarios(self) -> bool:
        """
        DISCOVERS: File corruption handling vulnerabilities
        CHAOS: Create and test with various corrupted file scenarios
        """
        print("🔍 Testing corrupted file scenarios...")
        
        try:
            temp_dir = tempfile.mkdtemp()
            self.temp_dirs.append(temp_dir)
            
            corruption_scenarios = [
                ("empty_file", b""),
                ("null_bytes", b"\x00" * 1000),
                ("random_binary", os.urandom(10000)),
                ("partial_text", b"incomplete json {\"key\": \"val"),
                ("mixed_encoding", "test🎵".encode('utf-8') + b"\xff\xfe\x00"),
                ("extremely_long_line", b"A" * 1000000),
                ("control_characters", bytes(range(32))),
                ("unicode_bom", b"\xef\xbb\xbf" + "test".encode('utf-8')),
                ("zip_bomb_attempt", b"PK\x03\x04" + b"\x00" * 1000),
                ("fake_musicxml", b"<?xml version='1.0'?><score-partwise>" + b"X" * 10000),
            ]
            
            failures = 0
            
            for scenario_name, data in corruption_scenarios:
                try:
                    print(f"   Testing {scenario_name}...")
                    
                    # Create corrupted file
                    file_path = os.path.join(temp_dir, f"{scenario_name}.test")
                    with open(file_path, 'wb') as f:
                        f.write(data)
                    
                    self.test_files.append(file_path)
                    
                    # Test 1: Try to read file directly
                    start_time = time.time()
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read(10000)  # Limit read size
                    except Exception as e:
                        # Expected for binary files
                        pass
                    
                    read_time = time.time() - start_time
                    
                    if read_time > 2.0:
                        self.log_issue(
                            "file_io",
                            f"Slow file read for {scenario_name}: {read_time:.2f}s",
                            "medium"
                        )
                        failures += 1
                    
                    # Test 2: Try to store file path as score data
                    try:
                        score_id = f"file_test_{scenario_name}"
                        score_data = {
                            "file_path": file_path,
                            "scenario": scenario_name,
                            "data_preview": str(data[:100])
                        }
                        
                        await self.score_manager.add_score(score_id, score_data)
                        retrieved = await self.score_manager.get_score(score_id)
                        
                        if not retrieved:
                            self.log_issue(
                                "file_io",
                                f"Failed to store file reference for {scenario_name}",
                                "medium"
                            )
                            failures += 1
                        
                        # Clean up
                        await self.score_manager.remove_score(score_id)
                        
                    except Exception as e:
                        self.log_issue(
                            "file_io",
                            f"Error handling corrupted file {scenario_name}: {str(e)}",
                            "high"
                        )
                        failures += 1
                
                except Exception as e:
                    self.log_issue(
                        "file_io",
                        f"Critical error with {scenario_name}: {str(e)}",
                        "critical"
                    )
                    failures += 1
            
            print(f"✅ Corrupted file test completed - {failures} issues found")
            return failures == 0
            
        except Exception as e:
            self.log_issue(
                "file_io",
                f"Corrupted file test crashed: {str(e)}",
                "critical"
            )
            return False
    
    async def test_file_permission_chaos(self) -> bool:
        """
        DISCOVERS: File permission handling vulnerabilities
        CHAOS: Test various file permission scenarios
        """
        print("🔍 Testing file permission chaos...")
        
        try:
            temp_dir = tempfile.mkdtemp()
            self.temp_dirs.append(temp_dir)
            
            permission_tests = [
                ("read_only_file", 0o444),
                ("write_only_file", 0o222),
                ("no_permissions", 0o000),
                ("execute_only", 0o111),
            ]
            
            failures = 0
            
            for test_name, permissions in permission_tests:
                try:
                    print(f"   Testing {test_name}...")
                    
                    # Create test file with specific permissions
                    file_path = os.path.join(temp_dir, f"{test_name}.txt")
                    with open(file_path, 'w') as f:
                        f.write(f"Test file for {test_name}")
                    
                    os.chmod(file_path, permissions)
                    self.test_files.append(file_path)
                    
                    # Test reading the file
                    try:
                        with open(file_path, 'r') as f:
                            content = f.read()
                        
                        if permissions & 0o444 == 0:  # Should not be readable
                            self.log_issue(
                                "permissions",
                                f"Read succeeded on unreadable file: {test_name}",
                                "medium"
                            )
                            failures += 1
                    
                    except PermissionError:
                        # Expected for files without read permission
                        pass
                    except Exception as e:
                        self.log_issue(
                            "permissions",
                            f"Unexpected error reading {test_name}: {str(e)}",
                            "medium"
                        )
                        failures += 1
                    
                    # Test writing to the file
                    try:
                        with open(file_path, 'a') as f:
                            f.write("additional content")
                        
                        if permissions & 0o222 == 0:  # Should not be writable
                            self.log_issue(
                                "permissions",
                                f"Write succeeded on unwritable file: {test_name}",
                                "medium"
                            )
                            failures += 1
                    
                    except PermissionError:
                        # Expected for files without write permission
                        pass
                    except Exception as e:
                        self.log_issue(
                            "permissions",
                            f"Unexpected error writing {test_name}: {str(e)}",
                            "medium"
                        )
                        failures += 1
                
                except Exception as e:
                    self.log_issue(
                        "permissions",
                        f"Critical error with {test_name}: {str(e)}",
                        "high"
                    )
                    failures += 1
            
            print(f"✅ File permission test completed - {failures} issues found")
            return failures == 0
            
        except Exception as e:
            self.log_issue(
                "permissions",
                f"File permission test crashed: {str(e)}",
                "critical"
            )
            return False
    
    async def test_concurrent_file_bombs(self) -> bool:
        """
        DISCOVERS: Concurrent file operation vulnerabilities
        CHAOS: Simultaneous file operations from multiple threads
        """
        print("🔍 Testing concurrent file operation bombs...")
        
        try:
            temp_dir = tempfile.mkdtemp()
            self.temp_dirs.append(temp_dir)
            
            num_workers = 20
            operations_per_worker = 50
            
            def file_chaos_worker(worker_id: int) -> List[str]:
                """Worker that performs chaotic file operations"""
                issues = []
                
                for op_id in range(operations_per_worker):
                    try:
                        file_name = f"worker_{worker_id}_op_{op_id}.txt"
                        file_path = os.path.join(temp_dir, file_name)
                        
                        operation = random.choice([
                            "create", "write", "read", "delete", "rename", "copy"
                        ])
                        
                        if operation == "create":
                            with open(file_path, 'w') as f:
                                f.write(f"Worker {worker_id} operation {op_id}")
                        
                        elif operation == "write":
                            # Try to write to existing file
                            existing_files = [
                                f for f in os.listdir(temp_dir) 
                                if f.endswith('.txt')
                            ]
                            if existing_files:
                                target_file = os.path.join(temp_dir, random.choice(existing_files))
                                try:
                                    with open(target_file, 'a') as f:
                                        f.write(f"\\nAppended by worker {worker_id}")
                                except:
                                    pass  # Expected conflicts
                        
                        elif operation == "read":
                            # Try to read existing file
                            existing_files = [
                                f for f in os.listdir(temp_dir) 
                                if f.endswith('.txt')
                            ]
                            if existing_files:
                                target_file = os.path.join(temp_dir, random.choice(existing_files))
                                try:
                                    with open(target_file, 'r') as f:
                                        content = f.read()
                                except:
                                    pass  # Expected conflicts
                        
                        elif operation == "delete":
                            # Try to delete existing file
                            existing_files = [
                                f for f in os.listdir(temp_dir) 
                                if f.endswith('.txt')
                            ]
                            if existing_files:
                                target_file = os.path.join(temp_dir, random.choice(existing_files))
                                try:
                                    os.unlink(target_file)
                                except:
                                    pass  # Expected conflicts
                        
                        elif operation == "rename":
                            if os.path.exists(file_path):
                                new_name = f"renamed_{worker_id}_{op_id}.txt"
                                new_path = os.path.join(temp_dir, new_name)
                                try:
                                    os.rename(file_path, new_path)
                                except:
                                    pass  # Expected conflicts
                        
                        elif operation == "copy":
                            existing_files = [
                                f for f in os.listdir(temp_dir) 
                                if f.endswith('.txt')
                            ]
                            if existing_files:
                                source_file = os.path.join(temp_dir, random.choice(existing_files))
                                dest_file = os.path.join(temp_dir, f"copy_{worker_id}_{op_id}.txt")
                                try:
                                    shutil.copy2(source_file, dest_file)
                                except:
                                    pass  # Expected conflicts
                        
                        # Add small random delay to increase chance of conflicts
                        time.sleep(random.uniform(0.001, 0.005))
                    
                    except Exception as e:
                        issues.append(f"Worker {worker_id} operation {operation}: {str(e)}")
                
                return issues
            
            # Launch all workers concurrently
            print(f"   Launching {num_workers} file chaos workers...")
            start_time = time.time()
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = [
                    executor.submit(file_chaos_worker, i) 
                    for i in range(num_workers)
                ]
                
                results = []
                for future in concurrent.futures.as_completed(futures):
                    try:
                        result = future.result(timeout=10)
                        results.append(result)
                    except Exception as e:
                        self.log_issue(
                            "file_concurrency",
                            f"File worker crashed: {str(e)}",
                            "high"
                        )
                        results.append([f"Worker crashed: {str(e)}"])
            
            end_time = time.time()
            duration = end_time - start_time
            
            # Analyze results
            total_operations = num_workers * operations_per_worker
            total_issues = sum(len(result) for result in results)
            
            print(f"   Completed {total_operations} file operations in {duration:.2f}s")
            print(f"   Total file operation issues: {total_issues}")
            
            # Check final directory state
            try:
                final_files = os.listdir(temp_dir)
                print(f"   Final file count: {len(final_files)}")
                
                # Check for corrupted files
                corrupted_files = 0
                for file_name in final_files:
                    file_path = os.path.join(temp_dir, file_name)
                    try:
                        with open(file_path, 'r') as f:
                            content = f.read()
                        
                        # Check for signs of corruption
                        if len(content) == 0 and file_name.endswith('.txt'):
                            corrupted_files += 1
                    
                    except Exception:
                        corrupted_files += 1
                
                if corrupted_files > 0:
                    self.log_issue(
                        "file_concurrency",
                        f"Found {corrupted_files} corrupted files after concurrent operations",
                        "medium"
                    )
            
            except Exception as e:
                self.log_issue(
                    "file_concurrency",
                    f"Error checking final directory state: {str(e)}",
                    "medium"
                )
            
            # Log a sample of issues
            all_issues = [issue for result in results for issue in result]
            for issue in all_issues[:5]:  # Log first 5 issues
                self.log_issue(
                    "file_concurrency",
                    f"Concurrent file operation issue: {issue}",
                    "medium"
                )
            
            print(f"✅ Concurrent file bomb test completed")
            return total_issues < total_operations * 0.1  # Accept up to 10% error rate
            
        except Exception as e:
            self.log_issue(
                "file_concurrency",
                f"Concurrent file bomb test crashed: {str(e)}",
                "critical"
            )
            return False
    
    async def test_deadlock_scenarios(self) -> bool:
        """
        DISCOVERS: Deadlock and race condition vulnerabilities
        CHAOS: Create scenarios that can cause deadlocks
        """
        print("🔍 Testing deadlock scenarios...")
        
        try:
            # Test concurrent access to score manager
            num_workers = 10
            operations_per_worker = 20
            shared_score_ids = [f"shared_score_{i}" for i in range(5)]
            
            async def deadlock_worker(worker_id: int) -> List[str]:
                """Worker that performs operations that might cause deadlocks"""
                issues = []
                
                for op_id in range(operations_per_worker):
                    try:
                        # Randomly pick operations that might conflict
                        operations = [
                            ("add_shared", random.choice(shared_score_ids)),
                            ("get_shared", random.choice(shared_score_ids)),
                            ("remove_shared", random.choice(shared_score_ids)),
                            ("list_all", None),
                            ("add_unique", f"unique_{worker_id}_{op_id}"),
                        ]
                        
                        operation, score_id = random.choice(operations)
                        
                        start_time = time.time()
                        
                        if operation == "add_shared":
                            await self.score_manager.add_score(
                                score_id, 
                                {"worker": worker_id, "op": op_id, "type": "shared"}
                            )
                        
                        elif operation == "get_shared":
                            await self.score_manager.get_score(score_id)
                        
                        elif operation == "remove_shared":
                            await self.score_manager.remove_score(score_id)
                        
                        elif operation == "list_all":
                            await self.score_manager.list_scores()
                        
                        elif operation == "add_unique":
                            await self.score_manager.add_score(
                                score_id,
                                {"worker": worker_id, "op": op_id, "type": "unique"}
                            )
                        
                        operation_time = time.time() - start_time
                        
                        # Check for suspiciously long operations (potential deadlock)
                        if operation_time > 1.0:
                            issues.append(f"Slow operation {operation}: {operation_time:.2f}s")
                        
                        # Add small delay to increase chance of race conditions
                        await asyncio.sleep(random.uniform(0.001, 0.01))
                    
                    except Exception as e:
                        issues.append(f"Worker {worker_id} {operation}: {str(e)}")
                
                return issues
            
            # Launch all workers concurrently
            print(f"   Testing deadlock scenarios with {num_workers} workers...")
            start_time = time.time()
            
            tasks = [deadlock_worker(i) for i in range(num_workers)]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            end_time = time.time()
            duration = end_time - start_time
            
            # Analyze results
            total_operations = num_workers * operations_per_worker
            deadlock_issues = 0
            
            for result in results:
                if isinstance(result, Exception):
                    self.log_issue(
                        "deadlock",
                        f"Deadlock worker crashed: {str(result)}",
                        "high"
                    )
                    deadlock_issues += 1
                elif isinstance(result, list):
                    deadlock_issues += len(result)
                    for issue in result[:2]:  # Log first 2 issues per worker
                        self.log_issue(
                            "deadlock",
                            f"Potential deadlock/race condition: {issue}",
                            "medium"
                        )
            
            operations_per_second = total_operations / duration
            
            print(f"   Completed {total_operations} operations in {duration:.2f}s ({operations_per_second:.1f} ops/sec)")
            print(f"   Deadlock/race condition issues: {deadlock_issues}")
            
            # Check final state consistency
            try:
                final_scores = await self.score_manager.list_scores()
                print(f"   Final score count: {len(final_scores)}")
            except Exception as e:
                self.log_issue(
                    "deadlock",
                    f"Final state check failed: {str(e)}",
                    "high"
                )
                deadlock_issues += 1
            
            # Performance regression check
            if operations_per_second < 50:  # Should handle at least 50 ops/sec
                self.log_issue(
                    "deadlock",
                    f"Poor performance in deadlock test: {operations_per_second:.1f} ops/sec",
                    "medium"
                )
            
            print(f"✅ Deadlock scenario test completed")
            return deadlock_issues < total_operations * 0.05  # Accept up to 5% error rate
            
        except Exception as e:
            self.log_issue(
                "deadlock",
                f"Deadlock scenario test crashed: {str(e)}",
                "critical"
            )
            return False
    
    async def test_disk_space_chaos(self) -> bool:
        """
        DISCOVERS: Disk space handling vulnerabilities
        CHAOS: Simulate low disk space scenarios
        """
        print("🔍 Testing disk space chaos...")
        
        try:
            temp_dir = tempfile.mkdtemp()
            self.temp_dirs.append(temp_dir)
            
            # Check initial disk space
            disk_usage = psutil.disk_usage(temp_dir)
            initial_free_gb = disk_usage.free / (1024**3)
            
            print(f"   Initial free space: {initial_free_gb:.2f}GB")
            
            # Create progressively larger files to simulate disk pressure
            large_files = []
            total_created_mb = 0
            
            try:
                for i in range(10):
                    file_size_mb = min(50, initial_free_gb * 1024 // 20)  # Don't use more than 5% of disk
                    
                    if file_size_mb < 1:
                        break
                    
                    file_path = os.path.join(temp_dir, f"large_file_{i}.dat")
                    
                    print(f"   Creating {file_size_mb}MB file...")
                    
                    # Create large file
                    with open(file_path, 'wb') as f:
                        chunk_size = 1024 * 1024  # 1MB chunks
                        for chunk in range(int(file_size_mb)):
                            f.write(b'0' * chunk_size)
                    
                    large_files.append(file_path)
                    total_created_mb += file_size_mb
                    
                    # Check remaining space
                    disk_usage = psutil.disk_usage(temp_dir)
                    remaining_gb = disk_usage.free / (1024**3)
                    
                    print(f"   Remaining space: {remaining_gb:.2f}GB")
                    
                    # Test score manager under disk pressure
                    try:
                        score_id = f"disk_pressure_test_{i}"
                        large_score_data = {
                            "file_number": i,
                            "large_data": "X" * (1024 * 1024),  # 1MB of data
                            "disk_pressure_test": True
                        }
                        
                        start_time = time.time()
                        await self.score_manager.add_score(score_id, large_score_data)
                        add_time = time.time() - start_time
                        
                        if add_time > 2.0:
                            self.log_issue(
                                "disk_space",
                                f"Slow score add under disk pressure: {add_time:.2f}s",
                                "medium"
                            )
                        
                        # Try to retrieve it
                        retrieved = await self.score_manager.get_score(score_id)
                        if not retrieved:
                            self.log_issue(
                                "disk_space",
                                f"Failed to retrieve score under disk pressure",
                                "high"
                            )
                    
                    except Exception as e:
                        self.log_issue(
                            "disk_space",
                            f"Score manager error under disk pressure: {str(e)}",
                            "medium"
                        )
                    
                    if remaining_gb < 1.0:  # Stop if less than 1GB remaining
                        print(f"   Stopping at {remaining_gb:.2f}GB remaining")
                        break
            
            finally:
                # Clean up large files
                print(f"   Cleaning up {total_created_mb}MB of test files...")
                for file_path in large_files:
                    try:
                        os.unlink(file_path)
                    except:
                        pass
            
            print(f"✅ Disk space chaos test completed")
            return True
            
        except Exception as e:
            self.log_issue(
                "disk_space",
                f"Disk space chaos test crashed: {str(e)}",
                "critical"
            )
            return False

async def run_file_system_chaos_tests():
    """Run all file system chaos and concurrency bomb tests"""
    print("📁💥 FILE SYSTEM CHAOS & CONCURRENCY BOMB TESTS")
    print("=" * 60)
    print("Testing file system failures, I/O disasters, and concurrency bombs")
    print("Finding disasters hiding in file operations and parallel processing")
    print()
    
    tester = FileSystemChaosTester()
    
    try:
        chaos_tests = [
            ("Corrupted File Scenarios", tester.test_corrupted_file_scenarios),
            ("File Permission Chaos", tester.test_file_permission_chaos),
            ("Concurrent File Bombs", tester.test_concurrent_file_bombs),
            ("Deadlock Scenarios", tester.test_deadlock_scenarios),
            ("Disk Space Chaos", tester.test_disk_space_chaos),
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
        print("📊 FILE SYSTEM CHAOS SUMMARY:")
        print("=" * 60)
        print(f"Total chaos tests: {total}")
        print(f"Tests passed: {passed}")
        print(f"Tests failed: {total - passed}")
        print(f"File system survival rate: {passed/total*100:.1f}%")
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
            print("🔴 CRITICAL: File system has critical vulnerabilities - IMMEDIATE ACTION REQUIRED")
        elif high_issues > 0:
            print("🟡 HIGH RISK: File system has high-severity issues - Action needed soon")
        elif total_issues > 0:
            print("🟠 MEDIUM RISK: File system has some issues - Review and fix when possible")
        else:
            print("🟢 ROBUST: File system survived chaos testing - No critical issues found")
        
        print()
        print("🔥 FILE SYSTEM CHAOS INSIGHTS:")
        print("1. Corrupted files reveal parsing and validation weaknesses")
        print("2. Permission chaos exposes privilege handling bugs")
        print("3. Concurrent file bombs discover race conditions and corruption")
        print("4. Deadlock scenarios reveal locking and synchronization issues")
        print("5. Disk space chaos tests resource exhaustion handling")
        
        return {
            'total_tests': total,
            'passed_tests': passed,
            'total_issues': total_issues,
            'issues_by_severity': issues_by_severity,
            'issues_by_category': issues_by_category,
            'file_system_survival_rate': passed/total*100 if total > 0 else 0
        }
    
    finally:
        # Always clean up
        tester.cleanup()

def main():
    """Main entry point"""
    return asyncio.run(run_file_system_chaos_tests())

if __name__ == "__main__":
    results = main()
    print(f"\n📁 File System Survival Rate: {results['file_system_survival_rate']:.1f}%")
    print(f"💥 Total File System Disasters Discovered: {results['total_issues']}")