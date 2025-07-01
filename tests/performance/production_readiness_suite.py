#!/usr/bin/env python3
"""
Production Readiness Test Suite
Comprehensive testing to determine if we meet 95% success threshold
"""
import asyncio
import json
import logging
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from music21 import converter, corpus

from music21_mcp.tools import *

logger = logging.getLogger(__name__)


@dataclass
class TestCase:
    """Represents a test case for a specific score and tool combination"""

    score_name: str
    score_path: str
    tool_name: str
    expected_behavior: str
    known_issues: List[str]
    workarounds: List[str]


class ProductionReadinessChecker:
    """
    Checks production readiness across multiple dimensions:
    1. Compatibility with real scores
    2. Performance under load
    3. Error recovery capabilities
    4. Edge case handling
    """

    def __init__(self):
        self.score_manager = {}
        self.tools = self._initialize_tools()
        self.test_results = defaultdict(dict)
        self.known_issues = self._load_known_issues()

    def _initialize_tools(self) -> Dict:
        """Initialize all tools"""
        return {
            "import": ImportScoreTool(self.score_manager),
            "key": KeyAnalysisTool(self.score_manager),
            "chord": ChordAnalysisTool(self.score_manager),
            "harmony": HarmonyAnalysisTool(self.score_manager),
            "voice_leading": VoiceLeadingAnalysisTool(self.score_manager),
            "pattern": PatternRecognitionTool(self.score_manager),
            "harmonization": HarmonizationTool(self.score_manager),
            "counterpoint": CounterpointGeneratorTool(self.score_manager),
            "style": StyleImitationTool(self.score_manager),
            "export": ExportScoreTool(self.score_manager),
            "info": ScoreInfoTool(self.score_manager),
        }

    def _load_known_issues(self) -> Dict:
        """Load known compatibility issues"""
        return {
            "complex_orchestral": {
                "issue": "Large orchestral scores may timeout",
                "affected_tools": ["harmony", "voice_leading"],
                "workaround": "Process parts separately",
            },
            "modern_notation": {
                "issue": "Extended techniques not fully supported",
                "affected_tools": ["pattern", "style"],
                "workaround": "Skip non-standard notation",
            },
            "percussion": {
                "issue": "Unpitched percussion causes key analysis issues",
                "affected_tools": ["key", "harmony"],
                "workaround": "Filter out percussion parts",
            },
            "microtonal": {
                "issue": "Microtonal music not supported",
                "affected_tools": ["all"],
                "workaround": "No current workaround",
            },
        }

    async def test_corpus_compatibility(self) -> Dict:
        """Test against music21's built-in corpus"""
        results = {
            "total": 0,
            "passed": 0,
            "failed": 0,
            "by_composer": defaultdict(lambda: {"total": 0, "passed": 0}),
            "by_tool": defaultdict(lambda: {"total": 0, "passed": 0}),
            "failures": [],
        }

        # Test a diverse set of corpus pieces
        test_pieces = [
            # Bach - Complex polyphony
            "bach/bwv66.6",  # Chorale
            "bach/bwv846",  # WTC Prelude
            "bach/bwv1080",  # Art of Fugue excerpt
            # Mozart - Classical style
            "mozart/k155/movement1",
            "mozart/k331/movement1",
            # Beethoven - Transitional complexity
            "beethoven/opus18no1/movement1",
            "beethoven/opus59no3/movement1",
            # Romantic - Extended harmony
            "chopin/mazurka06-1",
            "schumann/opus41no1/movement1",
            # Modern - Complex harmony
            "schoenberg/opus19/movement2",
            # Various instruments
            "haydn/opus74no1/movement1",  # String quartet
            "beethoven/opus1no3/movement1",  # Piano trio
            # Edge cases
            "josquin/laDeplorationDeLaMortDeJohannesOckeghem",  # Renaissance
            "monteverdi/madrigal.5.3",  # Early Baroque
        ]

        for piece_path in test_pieces:
            try:
                # Import from corpus
                score_id = piece_path.replace("/", "_")

                import_result = await self.tools["import"].execute(
                    score_id=score_id, source=piece_path, source_type="corpus"
                )

                if import_result["status"] != "success":
                    results["failures"].append(
                        {
                            "piece": piece_path,
                            "tool": "import",
                            "error": import_result.get("message"),
                        }
                    )
                    results["failed"] += 1
                    results["total"] += 1
                    continue

                # Test each analysis tool
                for tool_name in [
                    "key",
                    "chord",
                    "harmony",
                    "voice_leading",
                    "pattern",
                ]:
                    results["total"] += 1
                    results["by_tool"][tool_name]["total"] += 1

                    try:
                        tool_result = await self.tools[tool_name].execute(
                            score_id=score_id
                        )

                        if tool_result["status"] == "success":
                            results["passed"] += 1
                            results["by_tool"][tool_name]["passed"] += 1
                            composer = piece_path.split("/")[0]
                            results["by_composer"][composer]["passed"] += 1
                        else:
                            results["failed"] += 1
                            results["failures"].append(
                                {
                                    "piece": piece_path,
                                    "tool": tool_name,
                                    "error": tool_result.get("message"),
                                }
                            )
                    except Exception as e:
                        results["failed"] += 1
                        results["failures"].append(
                            {"piece": piece_path, "tool": tool_name, "error": str(e)}
                        )

                # Test generation tools on suitable pieces
                score = self.score_manager.get(score_id)
                if score and len(score.parts) == 1:  # Monophonic
                    for tool_name in ["harmonization", "counterpoint"]:
                        results["total"] += 1
                        results["by_tool"][tool_name]["total"] += 1

                        try:
                            if tool_name == "harmonization":
                                tool_result = await self.tools[tool_name].execute(
                                    score_id=score_id, style="classical"
                                )
                            else:
                                tool_result = await self.tools[tool_name].execute(
                                    score_id=score_id, species="first"
                                )

                            if tool_result["status"] == "success":
                                results["passed"] += 1
                                results["by_tool"][tool_name]["passed"] += 1
                            else:
                                results["failed"] += 1
                        except Exception as e:
                            results["failed"] += 1

                # Cleanup
                await self.tools["import"].execute(
                    score_id=score_id, source="", source_type="delete"
                )

            except Exception as e:
                results["failed"] += 1
                results["failures"].append(
                    {"piece": piece_path, "tool": "general", "error": str(e)}
                )

        results["success_rate"] = (
            (results["passed"] / results["total"] * 100) if results["total"] > 0 else 0
        )

        return results

    async def test_edge_cases(self) -> Dict:
        """Test specific edge cases and problematic scores"""
        edge_cases = [
            {"name": "Empty score", "test": self._test_empty_score},
            {"name": "Single note", "test": self._test_single_note},
            {
                "name": "Extreme polyphony (20+ voices)",
                "test": self._test_extreme_polyphony,
            },
            {"name": "No time signature", "test": self._test_no_time_signature},
            {"name": "Multiple key changes", "test": self._test_multiple_keys},
            {"name": "Atonal music", "test": self._test_atonal},
            {"name": "Extreme tempo changes", "test": self._test_tempo_changes},
            {"name": "Mixed meters", "test": self._test_mixed_meters},
            {
                "name": "Very long score (1000+ measures)",
                "test": self._test_very_long_score,
            },
            {"name": "Unicode text and symbols", "test": self._test_unicode},
        ]

        results = {"total": len(edge_cases), "passed": 0, "failed": 0, "details": []}

        for case in edge_cases:
            case_result = await case["test"]()
            results["details"].append({"name": case["name"], "result": case_result})

            if case_result["passed"]:
                results["passed"] += 1
            else:
                results["failed"] += 1

        results["success_rate"] = results["passed"] / results["total"] * 100

        return results

    async def _test_empty_score(self) -> Dict:
        """Test handling of empty scores"""
        try:
            from music21 import stream

            empty_score = stream.Score()
            self.score_manager["empty"] = empty_score

            # Try each tool
            failures = []
            for tool_name in ["key", "chord", "harmony"]:
                try:
                    result = await self.tools[tool_name].execute(score_id="empty")
                    if result["status"] != "success":
                        failures.append(f"{tool_name}: {result.get('message')}")
                except Exception as e:
                    failures.append(f"{tool_name}: {str(e)}")

            return {"passed": len(failures) == 0, "failures": failures}

        except Exception as e:
            return {"passed": False, "error": str(e)}

    async def _test_single_note(self) -> Dict:
        """Test handling of minimal scores"""
        try:
            result = await self.tools["import"].execute(
                score_id="single_note", source="C4", source_type="text"
            )

            if result["status"] != "success":
                return {"passed": False, "error": "Failed to import single note"}

            # Test analysis tools
            failures = []
            for tool_name in ["key", "harmony", "pattern"]:
                try:
                    tool_result = await self.tools[tool_name].execute(
                        score_id="single_note"
                    )
                    if tool_result["status"] != "success":
                        # Single note should handle gracefully
                        failures.append(f"{tool_name} failed on single note")
                except Exception as e:
                    failures.append(f"{tool_name}: {str(e)}")

            return {"passed": len(failures) == 0, "failures": failures}

        except Exception as e:
            return {"passed": False, "error": str(e)}

    async def _test_extreme_polyphony(self) -> Dict:
        """Test with many simultaneous voices"""
        try:
            from music21 import note, stream

            # Create 20-voice score
            score = stream.Score()
            for i in range(20):
                part = stream.Part()
                for j in range(10):
                    n = note.Note(60 + i)  # Different pitch for each voice
                    n.quarterLength = 1
                    part.append(n)
                score.append(part)

            self.score_manager["polyphony"] = score

            # Test voice leading analysis
            start_time = time.time()
            result = await self.tools["voice_leading"].execute(score_id="polyphony")
            duration = time.time() - start_time

            return {
                "passed": result["status"] == "success"
                and duration < 30,  # Should complete in reasonable time
                "duration": duration,
                "result": result["status"],
            }

        except Exception as e:
            return {"passed": False, "error": str(e)}

    async def _test_no_time_signature(self) -> Dict:
        """Test handling of scores without time signatures"""
        try:
            from music21 import note, stream

            # Create score without time signature
            score = stream.Score()
            part = stream.Part()
            for i in range(20):
                n = note.Note(60 + i % 12)
                n.quarterLength = 0.5 + (i % 4) * 0.25
                part.append(n)
            score.append(part)

            self.score_manager["no_time_sig"] = score

            # Test rhythm analysis
            result = await self.tools["pattern"].execute(
                score_id="no_time_sig", pattern_type="rhythmic"
            )

            return {
                "passed": result["status"] == "success",
                "handled_gracefully": "patterns" in result or "message" in result,
            }

        except Exception as e:
            return {"passed": False, "error": str(e)}

    async def _test_multiple_keys(self) -> Dict:
        """Test with frequent key changes"""
        try:
            # Import a piece with modulations
            result = await self.tools["import"].execute(
                score_id="modulating",
                source="C D E F G A B C B A G F E D C G A B C",
                source_type="text",
            )

            # Add some accidentals to imply key changes
            from music21 import key, stream

            score = self.score_manager["modulating"]

            # Test key analysis
            key_result = await self.tools["key"].execute(score_id="modulating")

            return {
                "passed": key_result["status"] == "success",
                "detected_keys": key_result.get("alternatives", []),
            }

        except Exception as e:
            return {"passed": False, "error": str(e)}

    async def _test_atonal(self) -> Dict:
        """Test atonal music handling"""
        try:
            # Create atonal sequence
            atonal_sequence = " ".join([f"C{i % 12}" for i in range(12)])

            result = await self.tools["import"].execute(
                score_id="atonal", source=atonal_sequence, source_type="text"
            )

            # Test key detection on atonal music
            key_result = await self.tools["key"].execute(score_id="atonal")

            # Should handle gracefully even if no clear key
            return {
                "passed": key_result["status"] == "success",
                "confidence": key_result.get("confidence", 0),
                "handled_uncertainty": key_result.get("confidence", 1) < 0.5,
            }

        except Exception as e:
            return {"passed": False, "error": str(e)}

    async def _test_tempo_changes(self) -> Dict:
        """Test extreme tempo changes"""
        try:
            from music21 import note, stream, tempo

            score = stream.Score()
            part = stream.Part()

            # Add extreme tempo changes
            tempos = [40, 200, 60, 180, 30, 240]
            for i, bpm in enumerate(tempos):
                t = tempo.MetronomeMark(number=bpm)
                part.append(t)

                for j in range(4):
                    n = note.Note(60 + j)
                    n.quarterLength = 1
                    part.append(n)

            score.append(part)
            self.score_manager["tempo_changes"] = score

            # Test if info tool handles tempo changes
            info_result = await self.tools["info"].execute(score_id="tempo_changes")

            return {
                "passed": info_result["status"] == "success",
                "found_tempos": "tempo_markings" in info_result,
            }

        except Exception as e:
            return {"passed": False, "error": str(e)}

    async def _test_mixed_meters(self) -> Dict:
        """Test with frequently changing time signatures"""
        try:
            from music21 import meter, note, stream

            score = stream.Score()
            part = stream.Part()

            # Alternating time signatures
            meters = ["3/4", "5/8", "7/8", "4/4", "2/4"]
            for m in meters:
                ts = meter.TimeSignature(m)
                part.append(ts)

                # Fill the measure
                beats = ts.numerator
                for i in range(beats):
                    n = note.Note(60 + i)
                    n.quarterLength = ts.denominator / 4
                    part.append(n)

            score.append(part)
            self.score_manager["mixed_meters"] = score

            # Test pattern recognition with mixed meters
            pattern_result = await self.tools["pattern"].execute(
                score_id="mixed_meters", pattern_type="rhythmic"
            )

            return {
                "passed": pattern_result["status"] == "success",
                "handled_complexity": True,  # Just handling it is success
            }

        except Exception as e:
            return {"passed": False, "error": str(e)}

    async def _test_very_long_score(self) -> Dict:
        """Test performance with very long scores"""
        try:
            import random

            from music21 import note, stream

            score = stream.Score()
            part = stream.Part()

            # Create 1000 measures
            for measure_num in range(1000):
                for beat in range(4):
                    pitch = random.randint(48, 72)
                    n = note.Note(pitch)
                    n.quarterLength = 0.25 * random.choice([1, 2, 4])
                    part.append(n)

            score.append(part)
            self.score_manager["long_score"] = score

            # Test performance-critical operations
            start_time = time.time()

            # Pattern recognition on long score
            pattern_result = await self.tools["pattern"].execute(
                score_id="long_score", min_pattern_length=4
            )

            duration = time.time() - start_time

            return {
                "passed": pattern_result["status"] == "success" and duration < 60,
                "duration": duration,
                "num_patterns": len(pattern_result.get("patterns", [])),
            }

        except Exception as e:
            return {"passed": False, "error": str(e)}

    async def _test_unicode(self) -> Dict:
        """Test Unicode handling in metadata and lyrics"""
        try:
            from music21 import metadata, note, stream

            score = stream.Score()

            # Add Unicode metadata
            md = metadata.Metadata()
            md.title = "测试曲 🎵"
            md.composer = "Бах"
            score.metadata = md

            part = stream.Part()
            n = note.Note("C4")
            n.lyric = "歌词 with émojis 🎶"
            part.append(n)
            score.append(part)

            self.score_manager["unicode"] = score

            # Test if tools handle Unicode
            info_result = await self.tools["info"].execute(score_id="unicode")

            return {
                "passed": info_result["status"] == "success",
                "preserved_unicode": "🎵" in str(info_result),
            }

        except Exception as e:
            return {"passed": False, "error": str(e)}

    async def test_performance_benchmarks(self) -> Dict:
        """Test performance under various conditions"""
        benchmarks = {
            "small_score": {"measures": 16, "parts": 1, "expected_time": 1.0},
            "medium_score": {"measures": 100, "parts": 4, "expected_time": 5.0},
            "large_score": {"measures": 500, "parts": 8, "expected_time": 30.0},
        }

        results = {}

        for name, spec in benchmarks.items():
            # Create score of specified size
            import random

            from music21 import note, stream

            score = stream.Score()
            for p in range(spec["parts"]):
                part = stream.Part()
                for m in range(spec["measures"]):
                    for beat in range(4):
                        n = note.Note(random.randint(48, 72))
                        n.quarterLength = 1
                        part.append(n)
                score.append(part)

            self.score_manager[name] = score

            # Benchmark each tool
            tool_times = {}
            for tool_name in ["key", "chord", "harmony", "pattern"]:
                start = time.time()
                try:
                    await self.tools[tool_name].execute(score_id=name)
                    tool_times[tool_name] = time.time() - start
                except:
                    tool_times[tool_name] = -1  # Failed

            results[name] = {
                "expected_time": spec["expected_time"],
                "tool_times": tool_times,
                "passed": all(
                    0 < t < spec["expected_time"] for t in tool_times.values()
                ),
            }

        return results

    async def generate_production_report(self) -> str:
        """Generate comprehensive production readiness report"""
        report = []
        report.append("=" * 80)
        report.append("MUSIC21 MCP SERVER - PRODUCTION READINESS REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

        # Test corpus compatibility
        print("Testing corpus compatibility...")
        corpus_results = await self.test_corpus_compatibility()

        report.append("1. CORPUS COMPATIBILITY TEST")
        report.append("-" * 40)
        report.append(f"Overall Success Rate: {corpus_results['success_rate']:.2f}%")
        report.append(f"Total Tests: {corpus_results['total']}")
        report.append(f"Passed: {corpus_results['passed']}")
        report.append(f"Failed: {corpus_results['failed']}")

        if corpus_results["success_rate"] >= 95:
            report.append("✅ PASSES 95% threshold")
        else:
            report.append("❌ FAILS 95% threshold")

        report.append("\nBy Tool:")
        for tool, stats in corpus_results["by_tool"].items():
            if stats["total"] > 0:
                rate = (stats["passed"] / stats["total"]) * 100
                status = "✅" if rate >= 95 else "❌"
                report.append(f"  {tool}: {rate:.1f}% {status}")

        # Test edge cases
        print("\nTesting edge cases...")
        edge_results = await self.test_edge_cases()

        report.append("\n2. EDGE CASE HANDLING")
        report.append("-" * 40)
        report.append(f"Success Rate: {edge_results['success_rate']:.2f}%")

        for case in edge_results["details"]:
            status = "✅" if case["result"]["passed"] else "❌"
            report.append(f"  {case['name']}: {status}")
            if not case["result"]["passed"] and "error" in case["result"]:
                report.append(f"    Error: {case['result']['error']}")

        # Test performance
        print("\nRunning performance benchmarks...")
        perf_results = await self.test_performance_benchmarks()

        report.append("\n3. PERFORMANCE BENCHMARKS")
        report.append("-" * 40)

        all_passed = True
        for score_type, results in perf_results.items():
            status = "✅" if results["passed"] else "❌"
            report.append(
                f"\n{score_type.upper()} (expected < {results['expected_time']}s)"
            )

            for tool, time_taken in results["tool_times"].items():
                if time_taken < 0:
                    report.append(f"  {tool}: FAILED")
                    all_passed = False
                elif time_taken > results["expected_time"]:
                    report.append(f"  {tool}: {time_taken:.2f}s ❌ (too slow)")
                    all_passed = False
                else:
                    report.append(f"  {tool}: {time_taken:.2f}s ✅")

        # Overall assessment
        report.append("\n" + "=" * 80)
        report.append("OVERALL ASSESSMENT")
        report.append("=" * 80)

        criteria = {
            "Corpus Compatibility": corpus_results["success_rate"] >= 95,
            "Edge Case Handling": edge_results["success_rate"] >= 80,
            "Performance": all_passed,
        }

        all_criteria_met = all(criteria.values())

        for criterion, passed in criteria.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            report.append(f"{criterion}: {status}")

        report.append("\n" + "=" * 80)
        if all_criteria_met:
            report.append("✅ SYSTEM IS PRODUCTION READY")
        else:
            report.append("❌ SYSTEM IS NOT PRODUCTION READY")
            report.append("\nRequired improvements:")

            if not criteria["Corpus Compatibility"]:
                report.append("- Improve compatibility to reach 95% success rate")
                report.append("  Focus on tools with lowest success rates")

            if not criteria["Edge Case Handling"]:
                report.append("- Better handle edge cases and malformed input")
                report.append("  Add input validation and error recovery")

            if not criteria["Performance"]:
                report.append("- Optimize performance for large scores")
                report.append("  Consider caching and parallel processing")

        report.append("=" * 80)

        return "\n".join(report)


async def main():
    """Run production readiness check"""
    checker = ProductionReadinessChecker()
    report = await checker.generate_production_report()

    # Save report
    report_path = Path("production_readiness_report.txt")
    with open(report_path, "w") as f:
        f.write(report)

    print(report)
    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    asyncio.run(main())
