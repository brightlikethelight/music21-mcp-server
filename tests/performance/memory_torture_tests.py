#!/usr/bin/env python3
"""
🧠💥 MEMORY TORTURE & LEAK DETECTION TESTS
Tests memory leaks, corruption, buffer overflows, and memory exhaustion
These tests find memory-related disasters hiding in the system
"""

import asyncio
import gc
import json
import os
import random
import string
import sys
import threading
import time
import traceback
import weakref
from typing import Any, Dict, List, Optional, Set, Tuple
import psutil

# Add src to path for imports
sys.path.insert(0, '../../src')

from music21_mcp.server import ScoreManager, ServerConfig

class MemoryTortureTester:
    """Memory torture testing and leak detection"""
    
    def __init__(self):
        self.config = ServerConfig()
        self.score_manager = ScoreManager(max_scores=self.config.max_scores)
        self.discovered_issues = []
        self.memory_snapshots = []
        self.leak_references = []
        
    def log_issue(self, category: str, description: str, severity: str = "medium"):
        """Log discovered issues"""
        issue = {
            "category": category,
            "description": description,
            "severity": severity,
            "timestamp": time.time(),
            "memory_usage": self._get_memory_usage(),
            "gc_stats": self._get_gc_stats()
        }
        self.discovered_issues.append(issue)
        print(f"🔥 DISCOVERED {severity.upper()}: {description}")
    
    def _get_memory_usage(self) -> Dict[str, float]:
        """Get detailed memory usage information"""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_percent = process.memory_percent()
            
            return {
                "rss_mb": memory_info.rss / 1024 / 1024,
                "vms_mb": memory_info.vms / 1024 / 1024, 
                "percent": memory_percent,
                "available_mb": psutil.virtual_memory().available / 1024 / 1024
            }
        except:
            return {"rss_mb": 0, "vms_mb": 0, "percent": 0, "available_mb": 0}
    
    def _get_gc_stats(self) -> Dict[str, Any]:
        """Get garbage collection statistics"""
        try:
            return {
                "counts": gc.get_count(),
                "stats": gc.get_stats() if hasattr(gc, 'get_stats') else [],
                "threshold": gc.get_threshold(),
                "referrers": len(gc.get_referrers(*gc.get_objects()[:100])) if gc.get_objects() else 0
            }
        except:
            return {"counts": [0, 0, 0], "stats": [], "threshold": [0, 0, 0], "referrers": 0}
    
    def take_memory_snapshot(self, label: str):
        """Take a memory snapshot for leak detection"""
        snapshot = {
            "label": label,
            "timestamp": time.time(),
            "memory": self._get_memory_usage(),
            "gc_stats": self._get_gc_stats(),
            "object_count": len(gc.get_objects()) if hasattr(gc, 'get_objects') else 0
        }
        self.memory_snapshots.append(snapshot)
        print(f"📸 Memory snapshot '{label}': {snapshot['memory']['rss_mb']:.1f}MB RSS, {snapshot['object_count']} objects")
    
    async def test_massive_allocation_torture(self) -> bool:
        """
        DISCOVERS: Memory allocation and deallocation issues
        TORTURE: Rapidly allocate and deallocate massive amounts of memory
        """
        print("🔍 Testing massive allocation torture...")
        
        try:
            self.take_memory_snapshot("before_allocation_torture")
            
            allocation_sizes = [
                ("small", 1024),           # 1KB
                ("medium", 1024 * 1024),   # 1MB  
                ("large", 10 * 1024 * 1024), # 10MB
                ("huge", 50 * 1024 * 1024),  # 50MB
            ]
            
            leaked_objects = []
            allocation_failures = 0
            
            for size_name, size_bytes in allocation_sizes:
                print(f"   Testing {size_name} allocations ({size_bytes // 1024}KB)...")
                
                # Allocate many objects of this size
                objects = []
                start_memory = self._get_memory_usage()["rss_mb"]
                
                try:
                    for i in range(20):  # Create 20 objects of each size
                        # Create different types of large objects
                        if i % 4 == 0:
                            # Large string
                            obj = "X" * size_bytes
                        elif i % 4 == 1:
                            # Large list
                            obj = [random.randint(0, 255) for _ in range(size_bytes // 4)]
                        elif i % 4 == 2:
                            # Large dict
                            obj = {f"key_{j}": f"value_{j}" * (size_bytes // 10000) for j in range(min(1000, size_bytes // 100))}
                        else:
                            # Large bytes
                            obj = os.urandom(size_bytes)
                        
                        objects.append(obj)
                        
                        # Add to score manager (this might cause leaks)
                        score_id = f"allocation_test_{size_name}_{i}"
                        try:
                            await self.score_manager.add_score(score_id, {"large_object": obj})
                        except Exception as e:
                            self.log_issue(
                                "memory_allocation",
                                f"Failed to store {size_name} allocation: {str(e)}",
                                "medium"
                            )
                            allocation_failures += 1
                
                except MemoryError:
                    self.log_issue(
                        "memory_allocation", 
                        f"Memory exhaustion during {size_name} allocation test",
                        "high"
                    )
                    allocation_failures += 1
                except Exception as e:
                    self.log_issue(
                        "memory_allocation",
                        f"Unexpected error during {size_name} allocation: {str(e)}",
                        "medium"
                    )
                    allocation_failures += 1
                
                # Check memory usage
                after_memory = self._get_memory_usage()["rss_mb"]
                memory_increase = after_memory - start_memory
                
                print(f"     Memory increase: {memory_increase:.1f}MB")
                
                # Store weak references to detect leaks
                for obj in objects:
                    try:
                        weak_ref = weakref.ref(obj)
                        self.leak_references.append({
                            "ref": weak_ref,
                            "size_name": size_name,
                            "created": time.time()
                        })
                    except:
                        pass  # Some objects can't have weak references
                
                # Clear objects and force GC
                del objects
                gc.collect()
                
                # Check if memory was freed
                freed_memory = self._get_memory_usage()["rss_mb"]
                memory_freed = after_memory - freed_memory
                
                print(f"     Memory freed: {memory_freed:.1f}MB")
                
                if memory_freed < memory_increase * 0.5:  # Less than 50% freed
                    self.log_issue(
                        "memory_leak",
                        f"Poor memory recovery after {size_name} allocation: {memory_freed:.1f}MB freed of {memory_increase:.1f}MB allocated",
                        "medium"
                    )
                
                # Add delay to allow system to stabilize
                await asyncio.sleep(0.1)
            
            self.take_memory_snapshot("after_allocation_torture")
            
            print(f"✅ Allocation torture completed - {allocation_failures} allocation failures")
            return allocation_failures == 0
            
        except Exception as e:
            self.log_issue(
                "memory_allocation",
                f"Allocation torture test crashed: {str(e)}",
                "critical"
            )
            return False
    
    async def test_memory_fragmentation_bomb(self) -> bool:
        """
        DISCOVERS: Memory fragmentation issues
        TORTURE: Create fragmented memory patterns that stress the allocator
        """
        print("🔍 Testing memory fragmentation bomb...")
        
        try:
            self.take_memory_snapshot("before_fragmentation")
            
            # Create fragmentation pattern: allocate, keep some, free others
            allocations = []
            fragmentation_issues = 0
            
            print("   Creating fragmentation pattern...")
            
            # Phase 1: Create many small allocations
            for i in range(1000):
                try:
                    # Vary allocation sizes to create fragmentation
                    size = random.choice([100, 1000, 10000, 100000])
                    allocation = "F" * size
                    
                    # Store in score manager
                    score_id = f"frag_{i}"
                    await self.score_manager.add_score(score_id, {"frag_data": allocation})
                    
                    allocations.append((score_id, allocation, size))
                    
                except Exception as e:
                    self.log_issue(
                        "fragmentation",
                        f"Error during fragmentation creation: {str(e)}",
                        "medium"
                    )
                    fragmentation_issues += 1
            
            mid_memory = self._get_memory_usage()["rss_mb"]
            print(f"   Memory after fragmentation: {mid_memory:.1f}MB")
            
            # Phase 2: Free every other allocation to create holes
            print("   Creating memory holes...")
            for i, (score_id, allocation, size) in enumerate(allocations):
                if i % 2 == 0:  # Free every other allocation
                    try:
                        await self.score_manager.remove_score(score_id)
                    except Exception as e:
                        fragmentation_issues += 1
            
            # Force garbage collection
            gc.collect()
            
            hole_memory = self._get_memory_usage()["rss_mb"]
            print(f"   Memory after creating holes: {hole_memory:.1f}MB")
            
            # Phase 3: Try to allocate large blocks (should be harder with fragmentation)
            print("   Testing large allocations in fragmented memory...")
            large_allocation_failures = 0
            
            for i in range(10):
                try:
                    # Try to allocate a moderately large block
                    large_size = 1024 * 1024  # 1MB
                    large_allocation = "L" * large_size
                    
                    start_time = time.time()
                    score_id = f"large_frag_{i}"
                    await self.score_manager.add_score(score_id, {"large_data": large_allocation})
                    allocation_time = time.time() - start_time
                    
                    if allocation_time > 1.0:  # Slow allocation might indicate fragmentation issues
                        self.log_issue(
                            "fragmentation",
                            f"Slow large allocation in fragmented memory: {allocation_time:.2f}s",
                            "medium"
                        )
                        fragmentation_issues += 1
                
                except MemoryError:
                    large_allocation_failures += 1
                    self.log_issue(
                        "fragmentation",
                        f"Large allocation failed in fragmented memory",
                        "high"
                    )
                except Exception as e:
                    large_allocation_failures += 1
                    self.log_issue(
                        "fragmentation",
                        f"Error during large allocation in fragmented memory: {str(e)}",
                        "medium"
                    )
            
            final_memory = self._get_memory_usage()["rss_mb"]
            self.take_memory_snapshot("after_fragmentation")
            
            print(f"   Final memory: {final_memory:.1f}MB")
            print(f"   Large allocation failures: {large_allocation_failures}")
            
            # Analyze fragmentation impact
            if large_allocation_failures > 5:  # More than 50% failure rate
                self.log_issue(
                    "fragmentation",
                    f"High large allocation failure rate: {large_allocation_failures}/10",
                    "high"
                )
            
            print(f"✅ Fragmentation bomb completed - {fragmentation_issues} fragmentation issues")
            return fragmentation_issues < 10  # Accept some fragmentation issues
            
        except Exception as e:
            self.log_issue(
                "fragmentation",
                f"Fragmentation bomb test crashed: {str(e)}",
                "critical"
            )
            return False
    
    async def test_circular_reference_bomb(self) -> bool:
        """
        DISCOVERS: Circular reference and garbage collection issues
        TORTURE: Create complex circular reference patterns
        """
        print("🔍 Testing circular reference bomb...")
        
        try:
            self.take_memory_snapshot("before_circular_refs")
            
            circular_ref_issues = 0
            
            # Create various circular reference patterns
            circular_patterns = [
                "simple_cycle",
                "deep_cycle", 
                "complex_web",
                "self_reference",
                "mixed_types"
            ]
            
            for pattern in circular_patterns:
                print(f"   Testing {pattern} circular references...")
                
                try:
                    if pattern == "simple_cycle":
                        # A -> B -> A
                        obj_a = {"name": "A", "data": "x" * 10000}
                        obj_b = {"name": "B", "data": "y" * 10000}
                        obj_a["ref"] = obj_b
                        obj_b["ref"] = obj_a
                        
                        score_id = f"circular_simple"
                        await self.score_manager.add_score(score_id, {"cycle": obj_a})
                    
                    elif pattern == "deep_cycle":
                        # A -> B -> C -> D -> A
                        objects = []
                        for i in range(10):
                            obj = {"name": f"obj_{i}", "data": "z" * 5000}
                            objects.append(obj)
                        
                        # Create circular chain
                        for i in range(len(objects)):
                            next_idx = (i + 1) % len(objects)
                            objects[i]["next"] = objects[next_idx]
                        
                        score_id = f"circular_deep"
                        await self.score_manager.add_score(score_id, {"cycle": objects[0]})
                    
                    elif pattern == "complex_web":
                        # Multiple interconnected cycles
                        nodes = {}
                        for i in range(20):
                            nodes[f"node_{i}"] = {
                                "id": i,
                                "data": "w" * 2000,
                                "connections": []
                            }
                        
                        # Create complex interconnections
                        for node_id, node in nodes.items():
                            # Each node connects to 3 random others
                            connections = random.sample(list(nodes.keys()), min(3, len(nodes) - 1))
                            for conn_id in connections:
                                if conn_id != node_id:
                                    node["connections"].append(nodes[conn_id])
                        
                        score_id = f"circular_web"
                        await self.score_manager.add_score(score_id, {"web": nodes})
                    
                    elif pattern == "self_reference":
                        # Object that references itself
                        obj = {"name": "self_ref", "data": "s" * 15000}
                        obj["self"] = obj
                        obj["also_self"] = obj
                        
                        score_id = f"circular_self"
                        await self.score_manager.add_score(score_id, {"self_ref": obj})
                    
                    elif pattern == "mixed_types":
                        # Mix lists and dicts in cycles
                        list_obj = ["item1", "item2"]
                        dict_obj = {"type": "dict", "data": "m" * 8000}
                        
                        list_obj.append(dict_obj)
                        dict_obj["list_ref"] = list_obj
                        dict_obj["self_ref"] = dict_obj
                        
                        score_id = f"circular_mixed"
                        await self.score_manager.add_score(score_id, {"mixed": [list_obj, dict_obj]})
                
                except Exception as e:
                    self.log_issue(
                        "circular_refs",
                        f"Error creating {pattern} circular references: {str(e)}",
                        "medium"
                    )
                    circular_ref_issues += 1
            
            # Let the objects exist for a while
            await asyncio.sleep(0.5)
            
            # Check memory before GC
            before_gc_memory = self._get_memory_usage()["rss_mb"]
            
            # Force garbage collection multiple times
            print("   Forcing garbage collection...")
            for _ in range(5):
                collected = gc.collect()
                print(f"     GC collected {collected} objects")
            
            # Check memory after GC
            after_gc_memory = self._get_memory_usage()["rss_mb"]
            memory_freed = before_gc_memory - after_gc_memory
            
            print(f"   Memory freed by GC: {memory_freed:.1f}MB")
            
            if memory_freed < 1.0:  # Should free at least 1MB with all those references
                self.log_issue(
                    "circular_refs",
                    f"Poor garbage collection efficiency: only {memory_freed:.1f}MB freed",
                    "medium"
                )
                circular_ref_issues += 1
            
            self.take_memory_snapshot("after_circular_refs")
            
            print(f"✅ Circular reference bomb completed - {circular_ref_issues} issues")
            return circular_ref_issues == 0
            
        except Exception as e:
            self.log_issue(
                "circular_refs",
                f"Circular reference bomb test crashed: {str(e)}",
                "critical"
            )
            return False
    
    async def test_memory_leak_detection(self) -> bool:
        """
        DISCOVERS: Memory leaks in the system
        TORTURE: Perform operations that should not increase memory permanently
        """
        print("🔍 Testing memory leak detection...")
        
        try:
            # Baseline memory measurement
            gc.collect()  # Clean up first
            self.take_memory_snapshot("baseline_for_leak_test")
            baseline_memory = self._get_memory_usage()["rss_mb"]
            
            # Perform operations that should not leak memory
            leak_test_cycles = 5
            operations_per_cycle = 100
            
            memory_progression = [baseline_memory]
            
            for cycle in range(leak_test_cycles):
                print(f"   Leak test cycle {cycle + 1}/{leak_test_cycles}...")
                
                cycle_start_memory = self._get_memory_usage()["rss_mb"]
                
                # Perform many operations
                for op in range(operations_per_cycle):
                    try:
                        # Add and remove scores (should not leak)
                        score_id = f"leak_test_{cycle}_{op}"
                        score_data = {
                            "cycle": cycle,
                            "operation": op,
                            "data": "L" * random.randint(1000, 10000),
                            "metadata": {"temp": True}
                        }
                        
                        await self.score_manager.add_score(score_id, score_data)
                        retrieved = await self.score_manager.get_score(score_id)
                        await self.score_manager.remove_score(score_id)
                        
                        # Also test listing (this was causing issues)
                        if op % 10 == 0:
                            try:
                                # This might fail due to the metadata bug we found
                                await self.score_manager.list_scores()
                            except:
                                pass  # Known issue, don't let it stop the leak test
                    
                    except Exception as e:
                        # Don't let individual operation failures stop the leak test
                        pass
                
                # Force garbage collection
                gc.collect()
                
                cycle_end_memory = self._get_memory_usage()["rss_mb"]
                cycle_increase = cycle_end_memory - cycle_start_memory
                memory_progression.append(cycle_end_memory)
                
                print(f"     Cycle {cycle + 1} memory change: {cycle_increase:+.1f}MB")
                
                if cycle_increase > 10:  # More than 10MB increase per cycle
                    self.log_issue(
                        "memory_leak",
                        f"Significant memory increase in cycle {cycle + 1}: {cycle_increase:.1f}MB",
                        "medium"
                    )
            
            final_memory = self._get_memory_usage()["rss_mb"]
            total_increase = final_memory - baseline_memory
            
            self.take_memory_snapshot("after_leak_test")
            
            print(f"   Total memory increase: {total_increase:.1f}MB")
            print(f"   Memory progression: {[f'{m:.1f}' for m in memory_progression]}")
            
            # Analyze leak pattern
            if total_increase > 50:  # More than 50MB total increase
                self.log_issue(
                    "memory_leak",
                    f"Significant total memory increase: {total_increase:.1f}MB",
                    "high"
                )
                return False
            
            # Check for consistent increase (leak pattern)
            increases = [memory_progression[i+1] - memory_progression[i] for i in range(len(memory_progression)-1)]
            consistent_increases = sum(1 for inc in increases if inc > 5)  # Cycles with >5MB increase
            
            if consistent_increases >= 3:  # 3 or more cycles with significant increase
                self.log_issue(
                    "memory_leak",
                    f"Consistent memory increase pattern detected: {consistent_increases} cycles with >5MB increase",
                    "medium"
                )
            
            print(f"✅ Memory leak detection completed")
            return total_increase < 50
            
        except Exception as e:
            self.log_issue(
                "memory_leak",
                f"Memory leak detection test crashed: {str(e)}",
                "critical"
            )
            return False
    
    async def test_buffer_overflow_scenarios(self) -> bool:
        """
        DISCOVERS: Buffer overflow and bounds checking issues
        TORTURE: Test with extremely large inputs that might overflow buffers
        """
        print("🔍 Testing buffer overflow scenarios...")
        
        try:
            overflow_issues = 0
            
            # Test various overflow scenarios
            overflow_tests = [
                ("extremely_long_string", "X" * (1024 * 1024 * 10)),  # 10MB string
                ("deeply_nested_dict", self._create_deep_nested_dict(1000)),
                ("wide_dict", {f"key_{i}": f"value_{i}" for i in range(100000)}),  # 100k keys
                ("long_list", list(range(1000000))),  # 1M items
                ("unicode_bomb", "🎵" * (1024 * 1024)),  # 1MB of 4-byte unicode chars
                ("json_bomb", json.dumps({"key": "value" * 100000})),  # Large JSON
            ]
            
            for test_name, test_data in overflow_tests:
                print(f"   Testing {test_name}...")
                
                try:
                    start_time = time.time()
                    start_memory = self._get_memory_usage()["rss_mb"]
                    
                    # Try to store the large data
                    score_id = f"overflow_test_{test_name}"
                    await self.score_manager.add_score(score_id, {"overflow_data": test_data})
                    
                    # Try to retrieve it
                    retrieved = await self.score_manager.get_score(score_id)
                    
                    end_time = time.time()
                    end_memory = self._get_memory_usage()["rss_mb"]
                    
                    operation_time = end_time - start_time
                    memory_increase = end_memory - start_memory
                    
                    # Check for performance issues (potential overflow handling)
                    if operation_time > 5.0:
                        self.log_issue(
                            "buffer_overflow",
                            f"Slow operation with {test_name}: {operation_time:.2f}s",
                            "medium"
                        )
                        overflow_issues += 1
                    
                    # Check for excessive memory usage
                    if memory_increase > 500:  # More than 500MB increase
                        self.log_issue(
                            "buffer_overflow",
                            f"Excessive memory usage with {test_name}: {memory_increase:.1f}MB",
                            "medium"
                        )
                        overflow_issues += 1
                    
                    # Verify data integrity
                    if retrieved is None:
                        self.log_issue(
                            "buffer_overflow",
                            f"Data lost during {test_name} overflow test",
                            "high"
                        )
                        overflow_issues += 1
                    
                    # Clean up
                    await self.score_manager.remove_score(score_id)
                
                except MemoryError:
                    self.log_issue(
                        "buffer_overflow",
                        f"Memory exhaustion during {test_name} test",
                        "high"
                    )
                    overflow_issues += 1
                except Exception as e:
                    error_msg = str(e)
                    if "overflow" in error_msg.lower() or "buffer" in error_msg.lower():
                        self.log_issue(
                            "buffer_overflow",
                            f"Buffer overflow detected in {test_name}: {error_msg}",
                            "critical"
                        )
                        overflow_issues += 1
                    else:
                        self.log_issue(
                            "buffer_overflow",
                            f"Error during {test_name} overflow test: {error_msg}",
                            "medium"
                        )
                        overflow_issues += 1
            
            print(f"✅ Buffer overflow testing completed - {overflow_issues} issues found")
            return overflow_issues == 0
            
        except Exception as e:
            self.log_issue(
                "buffer_overflow",
                f"Buffer overflow testing crashed: {str(e)}",
                "critical"
            )
            return False
    
    def _create_deep_nested_dict(self, depth: int) -> Dict[str, Any]:
        """Create a deeply nested dictionary for testing"""
        if depth <= 0:
            return {"end": "value"}
        return {"level": depth, "nested": self._create_deep_nested_dict(depth - 1)}
    
    def analyze_memory_snapshots(self):
        """Analyze all memory snapshots for patterns"""
        print("\\n🔍 Analyzing memory snapshots...")
        
        if len(self.memory_snapshots) < 2:
            print("   Not enough snapshots for analysis")
            return
        
        print(f"   Total snapshots: {len(self.memory_snapshots)}")
        
        for i, snapshot in enumerate(self.memory_snapshots):
            memory = snapshot["memory"]
            print(f"   {i+1}. {snapshot['label']}: {memory['rss_mb']:.1f}MB RSS, {memory['percent']:.1f}%")
        
        # Check for overall memory growth
        initial = self.memory_snapshots[0]["memory"]["rss_mb"]
        final = self.memory_snapshots[-1]["memory"]["rss_mb"]
        total_growth = final - initial
        
        print(f"   Total memory growth: {total_growth:+.1f}MB")
        
        if total_growth > 100:
            self.log_issue(
                "memory_analysis",
                f"Significant overall memory growth: {total_growth:.1f}MB",
                "medium"
            )

async def run_memory_torture_tests():
    """Run all memory torture and leak detection tests"""
    print("🧠💥 MEMORY TORTURE & LEAK DETECTION TESTS")
    print("=" * 60)
    print("Testing memory leaks, corruption, buffer overflows, and exhaustion")
    print("Finding memory-related disasters hiding in the system")
    print()
    
    tester = MemoryTortureTester()
    
    torture_tests = [
        ("Massive Allocation Torture", tester.test_massive_allocation_torture),
        ("Memory Fragmentation Bomb", tester.test_memory_fragmentation_bomb),
        ("Circular Reference Bomb", tester.test_circular_reference_bomb),
        ("Memory Leak Detection", tester.test_memory_leak_detection),
        ("Buffer Overflow Scenarios", tester.test_buffer_overflow_scenarios),
    ]
    
    results = {}
    passed = 0
    total = len(torture_tests)
    
    for test_name, test_func in torture_tests:
        print(f"\\n💥 {test_name}:")
        try:
            result = await test_func()
            results[test_name] = result
            if result:
                passed += 1
                print(f"✅ {test_name} - No critical memory issues found")
            else:
                print(f"❌ {test_name} - Critical memory issues discovered")
        except Exception as e:
            print(f"💥 {test_name} - TEST CRASHED: {str(e)}")
            print(f"Traceback: {traceback.format_exc()}")
            results[test_name] = False
        print()
    
    # Analyze memory snapshots
    tester.analyze_memory_snapshots()
    
    # Analyze discovered issues
    print("\\n📊 MEMORY TORTURE SUMMARY:")
    print("=" * 60)
    print(f"Total memory tests: {total}")
    print(f"Tests passed: {passed}")
    print(f"Tests failed: {total - passed}")
    print(f"Memory survival rate: {passed/total*100:.1f}%")
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
        print("🔴 CRITICAL: Memory system has critical vulnerabilities - IMMEDIATE ACTION REQUIRED")
    elif high_issues > 0:
        print("🟡 HIGH RISK: Memory system has high-severity issues - Action needed soon")
    elif total_issues > 0:
        print("🟠 MEDIUM RISK: Memory system has some issues - Review and fix when possible")
    else:
        print("🟢 ROBUST: Memory system survived torture testing - No critical issues found")
    
    print()
    print("🔥 MEMORY TORTURE INSIGHTS:")
    print("1. Allocation torture reveals memory management efficiency")
    print("2. Fragmentation bombs expose allocator weaknesses")
    print("3. Circular references test garbage collection robustness")
    print("4. Leak detection finds persistent memory growth")
    print("5. Buffer overflow tests reveal bounds checking issues")
    
    return {
        'total_tests': total,
        'passed_tests': passed,
        'total_issues': total_issues,
        'issues_by_severity': issues_by_severity,
        'issues_by_category': issues_by_category,
        'memory_survival_rate': passed/total*100 if total > 0 else 0
    }

def main():
    """Main entry point"""
    return asyncio.run(run_memory_torture_tests())

if __name__ == "__main__":
    results = main()
    print(f"\\n🧠 Memory Survival Rate: {results['memory_survival_rate']:.1f}%")
    print(f"💥 Total Memory Disasters Discovered: {results['total_issues']}")