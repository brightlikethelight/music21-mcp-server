#!/usr/bin/env python3
"""Debug test script to check core functionality"""

import asyncio
from src.music21_mcp.services import MusicAnalysisService

async def test_core():
    print("Testing core MusicAnalysisService...")
    
    # Initialize service
    service = MusicAnalysisService()
    print(f"Service initialized. Tools: {len(service.get_available_tools())}")
    print(f"Initial scores: {service.get_score_count()}")
    
    # Test import
    print("\nTesting import...")
    try:
        result = await service.import_score("test_chorale", "bach/bwv66.6", "corpus")
        print(f"Import result: {result}")
        print(f"Scores after import: {service.get_score_count()}")
        print(f"Score exists check: {service.is_score_loaded('test_chorale')}")
        
        # Check what's in the scores dict
        print(f"Scores dict keys: {list(service.scores.keys())}")
        if 'test_chorale' in service.scores:
            score = service.scores['test_chorale']
            print(f"Score type: {type(score)}")
            print(f"Score has parts: {hasattr(score, 'parts')}")
        
    except Exception as e:
        print(f"Import error: {e}")
        import traceback
        traceback.print_exc()
    
    # Test list
    print("\nTesting list...")
    try:
        result = await service.list_scores()
        print(f"List result: {result}")
    except Exception as e:
        print(f"List error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_core())