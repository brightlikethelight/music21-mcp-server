#!/usr/bin/env python3
"""
🎯 MCP TOOLS INTEGRATION TEST SUITE
Tests the ACTUAL MCP tool interface that users interact with
This is what really matters - not internal implementation details
"""

import asyncio
import sys
import time
import traceback
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

try:
    from music21 import chord, corpus, key, meter, note, stream

    MUSIC21_AVAILABLE = True
except ImportError:
    MUSIC21_AVAILABLE = False
    print("⚠️ Music21 not available - some tests will be skipped")

from music21_mcp.tools import (
    DeleteScoreTool,
    ExportScoreTool,
    HarmonyAnalysisTool,
    ImportScoreTool,
    KeyAnalysisTool,
    ListScoresTool,
    PatternRecognitionTool,
    ScoreInfoTool,
    VoiceLeadingAnalysisTool,
)


class MCPToolIntegrationTester:
    """Integration tests for the actual MCP tools"""

    def __init__(self):
        # Use simple dict for score storage - matching the new server architecture
        self.scores = {}
        self.test_results = []
        self.errors = []

        # Initialize tools with simple dict (as the new server does)
        self.tools = {
            "import": ImportScoreTool(self.scores),
            "list": ListScoresTool(self.scores),
            "key_analysis": KeyAnalysisTool(self.scores),
            "harmony_analysis": HarmonyAnalysisTool(self.scores),
            "voice_leading": VoiceLeadingAnalysisTool(self.scores),
            "pattern_recognition": PatternRecognitionTool(self.scores),
            "score_info": ScoreInfoTool(self.scores),
            "export": ExportScoreTool(self.scores),
            "delete": DeleteScoreTool(self.scores),
        }

    def log_error(self, test_name: str, error: str, severity: str = "medium"):
        """Log test errors"""
        self.errors.append(
            {
                "test": test_name,
                "error": error,
                "severity": severity,
                "timestamp": time.time(),
            }
        )
        print(f"❌ {severity.upper()}: {error}")

    async def test_import_score_tool(self) -> bool:
        """Test the ImportScoreTool with real music21 corpus"""
        print("🔍 Testing ImportScoreTool...")

        if not MUSIC21_AVAILABLE:
            print("⚠️ Skipping - Music21 not available")
            return True

        try:
            # Test 1: Import from corpus
            print("   Testing corpus import...")
            result = await self.tools["import"].execute(
                score_id="bach_test", source="bach/bwv66.6", source_type="corpus"
            )

            if result.get("status") == "success":
                print(f"   ✅ Successfully imported: {result.get('title', 'Unknown')}")
            else:
                self.log_error(
                    "import_corpus",
                    f"Failed to import corpus: {result.get('error')}",
                    "high",
                )
                return False

            # Test 2: Import from invalid source
            print("   Testing error handling...")
            result = await self.tools["import"].execute(
                score_id="invalid_test",
                source="nonexistent/piece",
                source_type="corpus",
            )

            if result.get("status") == "error":
                print("   ✅ Properly handled invalid import")
            else:
                self.log_error(
                    "import_error", "Failed to handle invalid import", "medium"
                )

            # Test 3: Import with text notation
            print("   Testing text notation import...")
            # Use ABC notation which is supported by music21
            abc_notation = """X:1
T:Custom Test
M:4/4
L:1/4
K:C
C E G c |]"""
            result = await self.tools["import"].execute(
                score_id="custom_test", source=abc_notation, source_type="text"
            )

            if result.get("status") == "success":
                print("   ✅ Successfully created custom score")
            else:
                self.log_error(
                    "import_custom",
                    f"Failed to create custom score: {result.get('error')}",
                    "medium",
                )

            return len([e for e in self.errors if e["severity"] == "high"]) == 0

        except Exception as e:
            self.log_error("import_tool", f"Import tool crashed: {str(e)}", "critical")
            return False

    async def test_key_analysis_tool(self) -> bool:
        """Test the KeyAnalysisTool with real scores"""
        print("🔍 Testing KeyAnalysisTool...")

        if not MUSIC21_AVAILABLE:
            print("⚠️ Skipping - Music21 not available")
            return True

        try:
            # First, ensure we have a score to analyze
            await self.tools["import"].execute(
                score_id="key_test", source="bach/bwv66.6", source_type="corpus"
            )

            # Test 1: Basic key analysis
            print("   Testing basic key analysis...")
            result = await self.tools["key_analysis"].execute(
                score_id="key_test", algorithm="krumhansl"
            )

            if result.get("status") == "success":
                key = result.get("key", "Unknown")
                confidence = result.get("confidence", 0)
                print(f"   ✅ Detected key: {key} (confidence: {confidence:.2f})")

                # Check if analysis is reasonable
                if confidence < 0.1:
                    self.log_error(
                        "key_confidence", f"Very low confidence: {confidence}", "medium"
                    )
            else:
                self.log_error(
                    "key_analysis",
                    f"Key analysis failed: {result.get('error')}",
                    "high",
                )
                return False

            # Test 2: All algorithms
            print("   Testing consensus analysis...")
            result = await self.tools["key_analysis"].execute(
                score_id="key_test", algorithm="all"
            )

            if result.get("status") == "success":
                algorithms = result.get("algorithm_results", {})
                print(f"   ✅ Ran {len(algorithms)} algorithms")

                # Check for consistency
                keys = [v.get("key") for v in algorithms.values() if v.get("key")]
                if len(set(keys)) > 3:  # Too much disagreement
                    self.log_error(
                        "key_consensus", "Algorithms strongly disagree on key", "medium"
                    )
            else:
                self.log_error(
                    "key_all",
                    f"Consensus analysis failed: {result.get('error')}",
                    "medium",
                )

            return len([e for e in self.errors if e["severity"] == "high"]) == 0

        except Exception as e:
            self.log_error(
                "key_tool", f"Key analysis tool crashed: {str(e)}", "critical"
            )
            return False

    async def test_harmony_analysis_tool(self) -> bool:
        """Test the HarmonyAnalysisTool"""
        print("🔍 Testing HarmonyAnalysisTool...")

        if not MUSIC21_AVAILABLE:
            print("⚠️ Skipping - Music21 not available")
            return True

        try:
            # Create a simple harmonic progression
            print("   Creating test progression...")
            from music21 import metadata

            score = stream.Score()
            # Add metadata so ListScoresTool doesn't crash
            score.metadata = metadata.Metadata()
            score.metadata.title = "Harmony Test Score"
            part = stream.Part()

            # Add time signature and key
            part.append(meter.TimeSignature("4/4"))
            part.append(key.Key("C"))

            # Create I-IV-V-I progression
            chords_to_add = [
                chord.Chord(["C4", "E4", "G4"]),  # I
                chord.Chord(["F4", "A4", "C5"]),  # IV
                chord.Chord(["G4", "B4", "D5"]),  # V
                chord.Chord(["C4", "E4", "G4"]),  # I
            ]

            for ch in chords_to_add:
                ch.duration.quarterLength = 1.0
                part.append(ch)

            score.append(part)

            # Store the score - tools use dict interface
            self.scores["harmony_test"] = score

            # Test harmony analysis
            print("   Testing harmony analysis...")
            result = await self.tools["harmony_analysis"].execute(
                score_id="harmony_test"
            )

            if result.get("status") == "success":
                data = result.get("data", {})
                roman_numerals = data.get("roman_numerals", [])
                print(f"   ✅ Found {len(roman_numerals)} Roman numerals")

                # Check if we detected the basic progression
                if len(roman_numerals) < 4:
                    self.log_error(
                        "harmony_detection",
                        f"Only found {len(roman_numerals)} chords in I-IV-V-I",
                        "medium",
                    )

                # Check for reasonable analysis
                if roman_numerals and all(rn == "?" for rn in roman_numerals):
                    self.log_error("harmony_quality", "All chords unidentified", "high")
            else:
                self.log_error(
                    "harmony_analysis",
                    f"Harmony analysis failed: {result.get('error')}",
                    "high",
                )
                return False

            return len([e for e in self.errors if e["severity"] == "high"]) == 0

        except Exception as e:
            self.log_error(
                "harmony_tool", f"Harmony analysis tool crashed: {str(e)}", "critical"
            )
            return False

    async def test_pattern_recognition_tool(self) -> bool:
        """Test the PatternRecognitionTool"""
        print("🔍 Testing PatternRecognitionTool...")

        if not MUSIC21_AVAILABLE:
            print("⚠️ Skipping - Music21 not available")
            return True

        try:
            # Create a score with obvious patterns
            print("   Creating test patterns...")
            from music21 import metadata

            score = stream.Score()
            # Add metadata so ListScoresTool doesn't crash
            score.metadata = metadata.Metadata()
            score.metadata.title = "Pattern Test Score"
            part = stream.Part()

            # Create a simple melodic sequence
            pattern = [
                note.Note("C4", quarterLength=0.5),
                note.Note("D4", quarterLength=0.5),
                note.Note("E4", quarterLength=0.5),
                note.Note("F4", quarterLength=0.5),
            ]

            # Repeat pattern at different pitch levels (sequence)
            for transposition in [0, 2, 4]:  # C, D, E starting notes
                for n in pattern:
                    new_note = note.Note()
                    new_note.pitch = n.pitch.transpose(transposition)
                    new_note.duration = n.duration
                    part.append(new_note)

            score.append(part)
            # Store the score - tools use dict interface
            self.scores["pattern_test"] = score

            # Test pattern recognition
            print("   Testing pattern detection...")
            result = await self.tools["pattern_recognition"].execute(
                score_id="pattern_test", pattern_type="both", min_pattern_length=2
            )

            if result.get("status") == "success":
                # Pattern results are returned directly, not under 'data'
                melodic_data = result.get("melodic_patterns", {})
                rhythmic_data = result.get("rhythmic_patterns", {})

                # Extract sequences and motifs from melodic patterns
                melodic = melodic_data.get("sequences", []) + melodic_data.get(
                    "motifs", []
                )
                rhythmic = rhythmic_data.get("rhythmic_motifs", [])

                print(f"   ✅ Found {len(melodic)} melodic patterns")
                print(f"   ✅ Found {len(rhythmic)} rhythmic patterns")

                # Should find at least some patterns in our sequence
                if len(melodic) == 0 and len(rhythmic) == 0:
                    self.log_error(
                        "pattern_detection",
                        "No patterns found in obvious sequence",
                        "high",
                    )
            else:
                self.log_error(
                    "pattern_analysis",
                    f"Pattern analysis failed: {result.get('error')}",
                    "high",
                )
                return False

            return len([e for e in self.errors if e["severity"] == "high"]) == 0

        except Exception as e:
            self.log_error(
                "pattern_tool",
                f"Pattern recognition tool crashed: {str(e)}",
                "critical",
            )
            return False

    async def test_tool_integration_workflow(self) -> bool:
        """Test a complete workflow using multiple tools"""
        print("🔍 Testing complete tool workflow...")

        if not MUSIC21_AVAILABLE:
            print("⚠️ Skipping - Music21 not available")
            return True

        try:
            workflow_score_id = "workflow_test"

            # Step 1: Import a score
            print("   Step 1: Importing score...")
            import_result = await self.tools["import"].execute(
                score_id=workflow_score_id, source="bach/bwv66.6", source_type="corpus"
            )

            if import_result.get("status") != "success":
                self.log_error("workflow_import", "Workflow failed at import", "high")
                return False

            # Step 2: Get score info
            print("   Step 2: Getting score info...")
            info_result = await self.tools["score_info"].execute(
                score_id=workflow_score_id
            )

            if info_result.get("status") != "success":
                self.log_error("workflow_info", "Workflow failed at info", "high")
                return False

            # Step 3: Analyze key
            print("   Step 3: Analyzing key...")
            key_result = await self.tools["key_analysis"].execute(
                score_id=workflow_score_id, algorithm="krumhansl"
            )

            if key_result.get("status") != "success":
                self.log_error(
                    "workflow_key", "Workflow failed at key analysis", "high"
                )
                return False

            # Step 4: List scores
            print("   Step 4: Listing scores...")
            list_result = await self.tools["list"].execute()

            if list_result.get("status") != "success":
                self.log_error("workflow_list", "Workflow failed at list", "high")
                return False

            scores = list_result.get("scores", [])
            if not any(s["id"] == workflow_score_id for s in scores):
                self.log_error(
                    "workflow_persistence", "Score not found in list", "high"
                )
                return False

            # Step 5: Export score
            print("   Step 5: Exporting score...")
            export_result = await self.tools["export"].execute(
                score_id=workflow_score_id, format="musicxml"
            )

            if export_result.get("status") != "success":
                self.log_error("workflow_export", "Workflow failed at export", "medium")

            # Step 6: Delete score
            print("   Step 6: Cleaning up...")
            delete_result = await self.tools["delete"].execute(
                score_id=workflow_score_id
            )

            if delete_result.get("status") != "success":
                self.log_error("workflow_delete", "Workflow failed at delete", "medium")

            print("   ✅ Complete workflow executed successfully")
            return len([e for e in self.errors if e["severity"] == "high"]) == 0

        except Exception as e:
            self.log_error("workflow", f"Workflow crashed: {str(e)}", "critical")
            return False


async def run_mcp_tools_integration_tests():
    """Run comprehensive MCP tools integration tests"""
    print("🎯 MCP TOOLS INTEGRATION TEST SUITE")
    print("=" * 50)
    print("Testing the ACTUAL MCP tool interface that users interact with")
    print("This is what really matters - not internal implementation details")
    print()

    if not MUSIC21_AVAILABLE:
        print("❌ CRITICAL: Music21 not available!")
        print("Install with: pip install music21")
        return {"total_tests": 0, "passed_tests": 0}

    tester = MCPToolIntegrationTester()

    tests = [
        ("Import Score Tool", tester.test_import_score_tool),
        ("Key Analysis Tool", tester.test_key_analysis_tool),
        ("Harmony Analysis Tool", tester.test_harmony_analysis_tool),
        ("Pattern Recognition Tool", tester.test_pattern_recognition_tool),
        ("Complete Tool Workflow", tester.test_tool_integration_workflow),
    ]

    results = {}
    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n🎼 {test_name}:")
        try:
            result = await test_func()
            results[test_name] = result
            if result:
                passed += 1
                print(f"✅ {test_name} - PASSED")
            else:
                print(f"❌ {test_name} - FAILED")
        except Exception as e:
            print(f"💥 {test_name} - CRASHED: {str(e)}")
            print(f"Traceback: {traceback.format_exc()}")
            results[test_name] = False
        print()

    # Summary
    print("📊 MCP TOOLS INTEGRATION SUMMARY:")
    print("=" * 50)
    print(f"Total tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    print(f"Success rate: {passed / total * 100:.1f}%")
    print(f"Total errors: {len(tester.errors)}")
    print()

    # Error analysis
    errors_by_severity = {"critical": 0, "high": 0, "medium": 0, "low": 0}
    for error in tester.errors:
        severity = error["severity"]
        if severity in errors_by_severity:
            errors_by_severity[severity] += 1

    if tester.errors:
        print("🚨 ERRORS BY SEVERITY:")
        for severity, count in errors_by_severity.items():
            if count > 0:
                print(f"   {severity.upper()}: {count}")
        print()

    # Production readiness assessment
    critical_errors = errors_by_severity["critical"]
    high_errors = errors_by_severity["high"]

    if critical_errors == 0 and high_errors == 0 and passed >= total * 0.8:
        print("🟢 PRODUCTION READY: MCP tools are working correctly")
        print("   - All critical functionality verified")
        print("   - Tools integrate properly with music21")
        print("   - Error handling is robust")
    elif critical_errors == 0 and passed >= total * 0.6:
        print("🟡 MOSTLY READY: MCP tools need some fixes")
        print("   - Core functionality works")
        print("   - Some features need improvement")
        print("   - Address high-priority issues before production")
    else:
        print("🔴 NOT READY: MCP tools have significant issues")
        print("   - Critical functionality is broken")
        print("   - Major fixes required before production")
        print("   - Do not deploy in current state")

    return {
        "total_tests": total,
        "passed_tests": passed,
        "errors": tester.errors,
        "errors_by_severity": errors_by_severity,
        "success_rate": passed / total * 100 if total > 0 else 0,
    }


def main():
    """Main entry point"""
    return asyncio.run(run_mcp_tools_integration_tests())


if __name__ == "__main__":
    results = main()
    print(f"\n🎯 MCP Tools Success Rate: {results['success_rate']:.1f}%")
