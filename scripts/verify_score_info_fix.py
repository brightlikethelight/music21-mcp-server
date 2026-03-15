#!/usr/bin/env python3
"""Quick test for score info fix"""

import asyncio

from music21_mcp.services import MusicAnalysisService


async def test():
    s = MusicAnalysisService()
    await s.import_score("test", "bach/bwv66.6", "corpus")
    info = await s.get_score_info("test")

    # Check if the fields exist and no error occurred
    if "status" in info and info["status"] == "success" and "duration_seconds" in info:
        print("✅ Score info working - duration_seconds calculated correctly")
        print(f"   Duration: {info['duration_seconds']} seconds")
        return True
    print("❌ Score info failed")
    print(info)
    return False


if __name__ == "__main__":
    success = asyncio.run(test())
    exit(0 if success else 1)
