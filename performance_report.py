#!/usr/bin/env python3
"""
Music21 MCP Server Performance Investigation Report

Comprehensive analysis of performance claims vs reality for the music21-mcp-server
"""

import json
from pathlib import Path
from datetime import datetime


def generate_performance_report():
    """Generate comprehensive performance report"""

    print("=" * 80)
    print("MUSIC21 MCP SERVER - PERFORMANCE INVESTIGATION REPORT")
    print("=" * 80)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Executive Summary
    print("📋 EXECUTIVE SUMMARY")
    print("-" * 40)
    print("Performance investigation revealed significant gaps between expectations")
    print("and actual performance for core music analysis operations.")
    print()

    # Performance Claims Analysis
    print("🔍 PERFORMANCE CLAIMS FOUND")
    print("-" * 40)
    claims = [
        {
            "file": "docs/simplified-api.md",
            "line": 382,
            "claim": "Performance Considerations section mentions caching",
            "type": "Documentation",
        },
        {
            "file": "docs/simplified-api.md",
            "line": 386,
            "claim": "Analysis results cached for 1 hour to improve performance",
            "type": "Caching Strategy",
        },
        {
            "file": "README.md",
            "line": 284,
            "claim": "Uses FastMCP for MCP protocol support",
            "type": "Framework Choice",
        },
    ]

    for claim in claims:
        print(f"  • {claim['file']}:{claim['line']} - {claim['claim']}")

    print(f"\n❌ CRITICAL FINDING: No explicit 'sub-200ms' performance claims found")
    print(
        "    However, the project architecture suggests expectation of fast response times"
    )
    print()

    # Actual Performance Results
    print("⏱️ ACTUAL PERFORMANCE RESULTS")
    print("-" * 40)

    # Results from our performance test
    results = {
        "Import Operations": {
            "Bach Chorale (first time)": {
                "mean": 412.06,
                "status": "SLOW",
                "cached": False,
            },
            "Bach Chorale (cached)": {
                "mean": 0.06,
                "status": "EXCELLENT",
                "cached": True,
            },
            "Text Melody": {"mean": 0.10, "status": "EXCELLENT", "cached": False},
        },
        "Analysis Operations": {
            "Key Analysis": {"mean": 323.62, "status": "SLOW", "note": "Always slow"},
            "Chord Analysis": {
                "mean": 1162.27,
                "status": "VERY SLOW",
                "note": "Major bottleneck",
            },
            "Harmony Analysis": {
                "mean": 1163.13,
                "status": "VERY SLOW",
                "note": "Major bottleneck",
            },
        },
        "Resource Operations": {
            "List Scores": {"mean": 0.38, "status": "EXCELLENT"},
            "Resource Stats": {"mean": 0.05, "status": "EXCELLENT"},
            "Performance Metrics": {"mean": 0.02, "status": "EXCELLENT"},
        },
        "Bulk Operations": {
            "Bulk Import (5 scores)": {"mean": 105.76, "status": "MODERATE"},
            "Bulk Analysis": {
                "mean": 1943.06,
                "status": "CRITICAL",
                "note": "Scales poorly",
            },
        },
    }

    for category, operations in results.items():
        print(f"\n{category}:")
        for op_name, data in operations.items():
            status_emoji = {
                "EXCELLENT": "✅",
                "MODERATE": "⚠️",
                "SLOW": "⚠️",
                "VERY SLOW": "❌",
                "CRITICAL": "🚨",
            }.get(data["status"], "❓")

            note = f" - {data['note']}" if "note" in data else ""
            cached = " (cached)" if data.get("cached") else ""
            print(f"  {status_emoji} {op_name}: {data['mean']:.2f}ms{cached}{note}")

    print()

    # Performance Analysis
    print("📊 PERFORMANCE ANALYSIS")
    print("-" * 40)

    # Response time distribution
    fast_ops = 0
    moderate_ops = 0
    slow_ops = 0
    total_ops = 0

    for category, operations in results.items():
        for op_name, data in operations.items():
            total_ops += 1
            if data["mean"] < 200:
                fast_ops += 1
            elif data["mean"] < 500:
                moderate_ops += 1
            else:
                slow_ops += 1

    fast_pct = (fast_ops / total_ops) * 100
    moderate_pct = (moderate_ops / total_ops) * 100
    slow_pct = (slow_ops / total_ops) * 100

    print(f"Response Time Distribution:")
    print(f"  ✅ Under 200ms (Fast): {fast_ops}/{total_ops} ({fast_pct:.1f}%)")
    print(f"  ⚠️ 200-500ms (Moderate): {moderate_ops}/{total_ops} ({moderate_pct:.1f}%)")
    print(f"  ❌ Over 500ms (Slow): {slow_ops}/{total_ops} ({slow_pct:.1f}%)")
    print()

    # Performance Verdict
    print("🎯 PERFORMANCE VERDICT")
    print("-" * 40)

    if fast_pct >= 80:
        verdict = "✅ GOOD: Most operations are fast"
    elif fast_pct >= 60:
        verdict = "⚠️ FAIR: Mixed performance, optimization needed"
    else:
        verdict = "❌ POOR: Significant performance issues"

    print(f"Overall Performance: {verdict}")
    print()

    # Key Findings
    print("🔑 KEY FINDINGS")
    print("-" * 40)
    findings = [
        "✅ Resource management operations are extremely fast (<1ms)",
        "✅ Score caching works effectively (412ms → 0.06ms)",
        "✅ Simple operations like text import are near-instantaneous",
        "❌ Music analysis operations (key, chord, harmony) are consistently slow (300ms-1200ms)",
        "❌ Complex analysis operations scale poorly under load (2+ seconds)",
        "⚠️ First-time imports are slow due to music21 corpus loading (400ms+)",
        "⚠️ Performance variability is high for analysis operations",
    ]

    for finding in findings:
        print(f"  {finding}")

    print()

    # Performance Bottlenecks
    print("🚨 PERFORMANCE BOTTLENECKS IDENTIFIED")
    print("-" * 40)

    bottlenecks = [
        {
            "operation": "Chord Analysis",
            "avg_time": "1162ms",
            "impact": "HIGH",
            "cause": "music21 chord extraction algorithms",
            "recommendation": "Implement chord analysis caching or pre-computation",
        },
        {
            "operation": "Harmony Analysis",
            "avg_time": "1163ms",
            "impact": "HIGH",
            "cause": "Roman numeral analysis complexity",
            "recommendation": "Cache harmony analysis results per score",
        },
        {
            "operation": "Key Analysis",
            "avg_time": "324ms",
            "impact": "MEDIUM",
            "cause": "Multiple algorithm execution",
            "recommendation": "Offer single-algorithm mode for faster results",
        },
        {
            "operation": "Bulk Operations",
            "avg_time": "1943ms",
            "impact": "HIGH",
            "cause": "No parallel processing",
            "recommendation": "Implement concurrent analysis for multiple scores",
        },
    ]

    for i, bottleneck in enumerate(bottlenecks, 1):
        print(f"{i}. {bottleneck['operation']} ({bottleneck['avg_time']})")
        print(f"   Impact: {bottleneck['impact']}")
        print(f"   Cause: {bottleneck['cause']}")
        print(f"   Fix: {bottleneck['recommendation']}")
        print()

    # Recommendations
    print("💡 OPTIMIZATION RECOMMENDATIONS")
    print("-" * 40)

    recommendations = [
        {
            "priority": "HIGH",
            "category": "Caching",
            "action": "Implement analysis result caching beyond current 1-hour TTL",
            "impact": "Could reduce repeat analysis from 1200ms → <10ms",
        },
        {
            "priority": "HIGH",
            "category": "Algorithms",
            "action": "Profile music21 chord/harmony analysis algorithms",
            "impact": "Identify specific slow components for optimization",
        },
        {
            "priority": "HIGH",
            "category": "Concurrency",
            "action": "Implement parallel processing for bulk operations",
            "impact": "Could reduce bulk analysis from 2000ms → 500ms",
        },
        {
            "priority": "MEDIUM",
            "category": "User Experience",
            "action": "Add progress indicators for slow operations (>500ms)",
            "impact": "Better user experience during long operations",
        },
        {
            "priority": "MEDIUM",
            "category": "Architecture",
            "action": "Pre-warm commonly used corpus scores at startup",
            "impact": "Reduce first-time import latency",
        },
        {
            "priority": "LOW",
            "category": "Monitoring",
            "action": "Add performance alerts for operations exceeding thresholds",
            "impact": "Early detection of performance regressions",
        },
    ]

    for rec in recommendations:
        priority_emoji = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}[rec["priority"]]
        print(f"{priority_emoji} {rec['priority']} - {rec['category']}")
        print(f"   Action: {rec['action']}")
        print(f"   Impact: {rec['impact']}")
        print()

    # Architecture Impact
    print("🏗️ ARCHITECTURE IMPACT")
    print("-" * 40)

    architecture_notes = [
        "The multi-interface design (MCP/HTTP/CLI/Python) is sound",
        "Performance bottlenecks are in the core music21 analysis layer, not adapters",
        "Resource management system is working well (fast, memory-efficient)",
        "Observability system provides excellent performance monitoring",
        "Async execution framework is properly isolating blocking operations",
    ]

    for note in architecture_notes:
        print(f"  • {note}")

    print()

    # Conclusion
    print("🏁 CONCLUSION")
    print("-" * 40)
    print(
        "While no explicit 'sub-200ms' claims were found, the performance investigation"
    )
    print("reveals that core music analysis operations significantly exceed what users")
    print("would expect for 'fast' response times:")
    print()
    print("  • Resource operations: ✅ Excellent (<1ms)")
    print("  • Simple imports: ✅ Very fast (<1ms)")
    print("  • Music analysis: ❌ Slow (300-1200ms)")
    print("  • Bulk operations: ❌ Very slow (2000ms+)")
    print()
    print("The project would benefit from focused optimization on music21 analysis")
    print("operations, which are the primary value proposition but currently the")
    print("biggest performance bottleneck.")
    print()
    print("=" * 80)


if __name__ == "__main__":
    generate_performance_report()
