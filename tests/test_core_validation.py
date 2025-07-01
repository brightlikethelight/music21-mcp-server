#!/usr/bin/env python3
"""
Core validation test - Tests basic functionality without numpy dependencies
"""
import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Try to import corpus pieces from music21
from music21 import corpus, stream

from music21_mcp.tools.chord_analysis_tool import ChordAnalysisTool
from music21_mcp.tools.delete_tool import DeleteScoreTool
from music21_mcp.tools.export_tool import ExportScoreTool
# Import only core tools that don't require numpy
from music21_mcp.tools.import_tool import ImportScoreTool
from music21_mcp.tools.key_analysis_tool import KeyAnalysisTool
from music21_mcp.tools.list_tool import ListScoresTool
from music21_mcp.tools.score_info_tool import ScoreInfoTool


async def test_basic_functionality():
    """Test basic functionality with music21 corpus"""
    print("=" * 80)
    print("MUSIC21 MCP SERVER - CORE VALIDATION TEST")
    print("=" * 80)
    print("")

    # Initialize score manager
    score_manager = {}

    # Initialize tools
    tools = {
        "import": ImportScoreTool(score_manager),
        "list": ListScoresTool(score_manager),
        "key": KeyAnalysisTool(score_manager),
        "chord": ChordAnalysisTool(score_manager),
        "info": ScoreInfoTool(score_manager),
        "export": ExportScoreTool(score_manager),
        "delete": DeleteScoreTool(score_manager),
    }

    # Test pieces from corpus
    test_pieces = [
        ("bach/bwv66.6", "Bach Chorale"),
        ("mozart/k331/movement1", "Mozart Sonata"),
        ("beethoven/opus18no1/movement1", "Beethoven String Quartet"),
    ]

    results = {"total": 0, "passed": 0, "failed": 0, "tool_results": {}}

    for corpus_path, name in test_pieces:
        print(f"\nTesting: {name} ({corpus_path})")
        print("-" * 40)

        score_id = corpus_path.replace("/", "_")

        # Test import from corpus
        print("  Importing...", end=" ")
        try:
            import_result = await tools["import"].execute(
                score_id=score_id, source=corpus_path, source_type="corpus"
            )

            if import_result["status"] == "success":
                print("✓")
                results["passed"] += 1
            else:
                print(f"✗ ({import_result.get('message', 'Unknown error')})")
                results["failed"] += 1
                continue
        except Exception as e:
            print(f"✗ (Exception: {str(e)})")
            results["failed"] += 1
            continue

        results["total"] += 1

        # Test each analysis tool
        for tool_name in ["key", "chord", "info"]:
            print(f"  Testing {tool_name}...", end=" ")
            results["total"] += 1

            try:
                tool_result = await tools[tool_name].execute(score_id=score_id)

                if tool_result["status"] == "success":
                    print("✓")
                    results["passed"] += 1

                    # Print some results for verification
                    if tool_name == "key":
                        print(f"    Key: {tool_result.get('key', 'Unknown')}")
                    elif tool_name == "chord" and "chords" in tool_result:
                        total_chords = sum(
                            len(m["chords"]) for m in tool_result["measures"]
                        )
                        print(f"    Found {total_chords} chords")
                else:
                    print(f"✗ ({tool_result.get('message', 'Unknown error')})")
                    results["failed"] += 1
            except Exception as e:
                print(f"✗ (Exception: {str(e)})")
                results["failed"] += 1

        # Clean up
        try:
            await tools["delete"].execute(score_id=score_id)
        except:
            pass

    # Test creating a simple score from text
    print("\nTesting text import...")
    print("-" * 40)

    print("  Creating simple melody...", end=" ")
    try:
        text_result = await tools["import"].execute(
            score_id="test_melody", source="C4 D4 E4 F4 G4 A4 B4 C5", source_type="text"
        )

        if text_result["status"] == "success":
            print("✓")
            results["passed"] += 1

            # Test analysis on simple melody
            key_result = await tools["key"].execute(score_id="test_melody")
            if key_result["status"] == "success":
                print(f"    Key detected: {key_result.get('key', 'Unknown')}")
        else:
            print("✗")
            results["failed"] += 1
    except Exception as e:
        print(f"✗ (Exception: {str(e)})")
        results["failed"] += 1

    results["total"] += 1

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total tests: {results['total']}")
    print(f"Passed: {results['passed']}")
    print(f"Failed: {results['failed']}")

    if results["total"] > 0:
        success_rate = (results["passed"] / results["total"]) * 100
        print(f"Success rate: {success_rate:.2f}%")

        if success_rate >= 95:
            print("\n✅ CORE VALIDATION PASSED")
        else:
            print("\n❌ CORE VALIDATION FAILED")
    else:
        print("\n❌ NO TESTS RAN")


async def main():
    """Run core validation tests"""
    try:
        await test_basic_functionality()
    except Exception as e:
        print(f"\nFATAL ERROR: {str(e)}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
