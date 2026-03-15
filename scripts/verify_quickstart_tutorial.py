#!/usr/bin/env python3
"""
Test script to validate the quickstart tutorial notebook works correctly.

This ensures all code examples in the tutorial execute without errors
and provide meaningful output for users.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path for importing
sys.path.insert(0, str(Path(__file__).parent / "src"))

from music21_mcp.services import MusicAnalysisService


async def test_quickstart_tutorial():
    """Test all the code examples from the quickstart tutorial"""
    print("🧪 Testing Quickstart Tutorial")
    print("=" * 50)

    # Step 1: Initialize service (from notebook)
    print("\n📝 Step 1: Initialize Service")
    service = MusicAnalysisService()
    print(f"🎵 Music21 MCP Server initialized!")
    print(f"📊 Available tools: {len(service.get_available_tools())}")
    print(f"💾 Memory limit: {service.get_memory_usage()['max_scores']} scores")
    print(f"✨ Ready for analysis!")

    # Step 2: Import Bach chorale (from notebook)
    print("\n📝 Step 2: Import Bach Chorale")
    result = await service.import_score(
        score_id="my_first_bach", source="bach/bwv66.6", source_type="corpus"
    )

    if result["status"] == "success":
        print(f"🎼 Successfully imported Bach BWV 66.6!")
        print(f"🎵 Notes: {result['num_notes']}")
        print(f"🎭 Parts: {result['num_parts']} (Soprano, Alto, Tenor, Bass)")
        print(f"🎹 Pitch range: {result['pitch_range']} semitones")
    else:
        print(f"❌ Import failed: {result['message']}")
        return False

    # Step 3: Key analysis (from notebook)
    print("\n📝 Step 3: Key Analysis")
    key_result = await service.analyze_key("my_first_bach")

    if key_result["status"] == "success":
        print(f"🔑 Key Analysis Results:")
        print(f"   Primary key: {key_result['key']}")
        print(f"   Confidence: {key_result['confidence']:.1%}")
        print(f"   Algorithm: {key_result.get('algorithm', 'Consensus')}")

        if key_result.get("alternatives"):
            print(f"   🎯 Alternative keys: {len(key_result['alternatives'])}")
            for alt in key_result["alternatives"][:2]:
                print(f"     - {alt['key']} (confidence: {alt['confidence']:.1%})")
        else:
            print(f"   🎯 Strong key determination - no alternatives!")
    else:
        print(f"❌ Key analysis failed: {key_result['message']}")
        return False

    # Step 4: Chord analysis (from notebook)
    print("\n📝 Step 4: Chord Analysis")
    chord_result = await service.analyze_chords("my_first_bach")

    if chord_result["status"] == "success":
        print(f"🎼 Chord Analysis Results:")
        print(f"   Total chords: {chord_result['total_chords']}")

        summary = chord_result.get("summary", {})
        if summary:
            print(f"   Unique chords: {summary.get('unique_chords', 'Unknown')}")
            if summary.get("most_common_chords"):
                most_common = summary["most_common_chords"][0]
                print(
                    f"   Most common: {most_common['chord']} ({most_common['count']} times)"
                )

        # Show the first few chords
        print(f"\n🎵 First few chords:")
        for i, chord in enumerate(chord_result["chord_progression"][:3]):
            offset = chord.get("offset", i)
            symbol = chord.get("symbol", "Unknown")
            print(f"   Offset {offset}: {symbol}")
    else:
        print(f"❌ Chord analysis failed: {chord_result['message']}")
        return False

    # Step 5: Voice leading (from notebook)
    print("\n📝 Step 5: Voice Leading Analysis")
    voice_result = await service.analyze_voice_leading("my_first_bach")

    if voice_result["status"] == "success":
        print(f"🎭 Voice Leading Analysis:")
        smoothness_analysis = voice_result.get("smoothness_analysis", {})
        smoothness_score = (
            smoothness_analysis.get("smoothness_score", 0) / 10
        )  # Convert to 0-10 scale

        print(f"   Smoothness score: {smoothness_score:.2f}/10")
        print(f"   Parallel violations: {len(voice_result.get('parallel_issues', []))}")
        print(f"   Voice crossings: {len(voice_result.get('voice_crossings', []))}")
        print(f"   Large leaps: {smoothness_analysis.get('large_leap_motion', 0)}")

        # Overall assessment
        smoothness = smoothness_score
        if smoothness >= 8.0:
            print(f"   🌟 Exceptional voice leading! Bach at his finest.")
        elif smoothness >= 6.0:
            print(f"   ✨ Good voice leading with Bach's characteristic style.")
        else:
            print(f"   📝 Some challenges in voice leading - educational opportunity!")
    else:
        print(f"❌ Voice leading analysis failed: {voice_result['message']}")
        return False

    # Step 6: Score info (from notebook)
    print("\n📝 Step 6: Score Information")
    info_result = await service.get_score_info("my_first_bach")

    if info_result["status"] == "success":
        print(f"📊 Complete Score Analysis:")
        print(f"   Title: {info_result.get('title', 'BWV 66.6')}")
        print(f"   Composer: {info_result.get('composer', 'J.S. Bach')}")
        print(f"   Time signature: {info_result.get('time_signature', '4/4')}")
        print(f"   Tempo: {info_result.get('tempo', 'Moderate chorale')}")

        if "structure" in info_result:
            structure = info_result["structure"]
            print(
                f"   Duration: {structure.get('duration_quarters', 'Unknown')} quarter notes"
            )
            print(f"   Measures: {structure.get('measures', 'Unknown')}")
    else:
        print(f"❌ Score info failed: {info_result['message']}")
        return False

    # Step 7: System status (from notebook)
    print("\n📝 Step 7: System Status")
    status = service.get_service_status()

    print(f"🖥️  System Status Report:")
    print(f"   Service: {status['service']['name']} v{status['service']['version']}")
    print(f"   Health: {status['health']['status']} ✅")
    print(f"   Uptime: {status['service']['uptime_seconds']:.1f} seconds")

    # Resource usage
    resources = status["resources"]["storage"]
    print(f"\n💾 Resource Usage:")
    print(f"   Loaded scores: {resources['total_scores']}")
    print(f"   Memory usage: {resources['memory_usage_mb']:.1f}MB")
    print(f"   Cache hit rate: {resources['hit_rate_percent']:.1f}%")

    # Performance summary
    performance = status["performance"]
    operations = len(performance["operation_counts"])
    print(f"\n⚡ Performance:")
    print(f"   Operations completed: {operations}")
    print(
        f"   System healthy: {'✅' if status['health']['status'] == 'healthy' else '⚠️'}"
    )

    print(f"\n🎉 Tutorial Validation Complete! All steps working:")
    print(f"   ✅ Service initialization")
    print(f"   ✅ Bach chorale import")
    print(f"   ✅ Key signature analysis")
    print(f"   ✅ Chord progression analysis")
    print(f"   ✅ Voice leading analysis")
    print(f"   ✅ Score information retrieval")
    print(f"   ✅ System status monitoring")

    return True


if __name__ == "__main__":
    print("🎵 Music21 MCP Server - Quickstart Tutorial Validation")
    print("=" * 65)

    try:
        success = asyncio.run(test_quickstart_tutorial())
        if success:
            print("\n✅ All tutorial examples validated successfully!")
            print("🎓 The quickstart tutorial is ready for users!")
        else:
            print("\n❌ Some tutorial examples failed!")
            print("🔧 Please fix the issues before releasing.")
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()

    print("\n✨ Validation complete!")
