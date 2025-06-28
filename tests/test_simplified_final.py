#!/usr/bin/env python3
"""
Final test of simplified server - ensure all 5 core functions work
"""
import asyncio
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from music21_mcp.server_simple import (
    import_score, analyze_key, analyze_chords, 
    get_score_info, export_score, scores
)


async def test_all_functions():
    """Test all 5 core functions"""
    print("🧪 Testing Simplified Server - Final Validation")
    print("="*50)
    
    test_results = {}
    
    # Clear scores
    scores.clear()
    
    # Test 1: Import corpus file
    print("\n1. Testing corpus import (bach/bwv66.6)...")
    result = await import_score("bach_test", "bach/bwv66.6")
    test_results['import_corpus'] = result['status'] == 'success'
    print(f"   {'✅' if test_results['import_corpus'] else '❌'} Status: {result['status']}")
    if result['status'] == 'success':
        print(f"   Notes: {result['num_notes']}, Measures: {result['num_measures']}")
    
    # Test 2: Key analysis
    print("\n2. Testing key analysis...")
    result = await analyze_key("bach_test")
    test_results['key_analysis'] = (
        result['status'] == 'success' and 
        result['confidence'] > 0.3
    )
    print(f"   {'✅' if test_results['key_analysis'] else '❌'} Key: {result.get('key', 'N/A')}")
    print(f"   Confidence: {result.get('confidence', 0):.2f}")
    
    # Test 3: Chord analysis
    print("\n3. Testing chord analysis...")
    result = await analyze_chords("bach_test")
    test_results['chord_analysis'] = (
        result['status'] == 'success' and 
        result['chord_count'] > 10
    )
    print(f"   {'✅' if test_results['chord_analysis'] else '❌'} Chords: {result.get('chord_count', 0)}")
    
    # Test 4: Score info
    print("\n4. Testing score info...")
    result = await get_score_info("bach_test")
    test_results['score_info'] = result['status'] == 'success'
    print(f"   {'✅' if test_results['score_info'] else '❌'} Status: {result['status']}")
    if result['status'] == 'success':
        print(f"   Parts: {result['num_parts']}, Duration: {result['duration_quarters']:.1f} quarters")
    
    # Test 5: Export
    print("\n5. Testing export...")
    result = await export_score("bach_test", format="musicxml")
    test_results['export'] = (
        result['status'] == 'success' and 
        os.path.exists(result.get('file_path', ''))
    )
    print(f"   {'✅' if test_results['export'] else '❌'} Status: {result['status']}")
    if result['status'] == 'success':
        file_size = os.path.getsize(result['file_path'])
        print(f"   File: {result['file_path']} ({file_size} bytes)")
        os.unlink(result['file_path'])  # Clean up
    
    # Test 6: Text import (bonus)
    print("\n6. Testing text import...")
    result = await import_score("text_test", "C4 D4 E4 F4 G4")
    test_results['import_text'] = result['status'] == 'success'
    print(f"   {'✅' if test_results['import_text'] else '❌'} Status: {result['status']}")
    if result['status'] == 'success':
        print(f"   Notes: {result['num_notes']}")
    
    # Summary
    print("\n" + "="*50)
    print("SUMMARY:")
    print("="*50)
    
    core_tests = ['import_corpus', 'key_analysis', 'chord_analysis', 'score_info', 'export']
    core_passed = sum(test_results.get(t, False) for t in core_tests)
    total_passed = sum(test_results.values())
    
    print(f"\nCore Functions (5): {core_passed}/5 passed")
    print(f"All Tests (6): {total_passed}/6 passed")
    
    for test, passed in test_results.items():
        print(f"  {'✅' if passed else '❌'} {test}")
    
    # Decision
    if core_passed == 5:
        print("\n✅ ALL CORE FUNCTIONS WORKING!")
        print("The simplified server is ready for use.")
        return True
    else:
        print("\n❌ Some core functions still failing.")
        print("Fix these before proceeding.")
        return False


async def test_error_handling():
    """Test that errors are handled gracefully"""
    print("\n\n🛡️ Testing Error Handling")
    print("="*50)
    
    # Test with non-existent score
    print("\n1. Non-existent score...")
    result = await analyze_key("does_not_exist")
    print(f"   {'✅' if result['status'] == 'error' else '❌'} Correctly returned error")
    
    # Test with invalid file
    print("\n2. Invalid file import...")
    result = await import_score("bad", "/does/not/exist.xyz")
    print(f"   {'✅' if result['status'] == 'error' else '❌'} Correctly returned error")
    
    # Test with empty score
    print("\n3. Empty score...")
    from music21 import stream
    scores['empty'] = stream.Score()
    result = await analyze_chords('empty')
    print(f"   {'✅' if result['status'] == 'success' else '❌'} Handled empty score")
    print(f"   Chords: {result.get('chord_count', 'N/A')}")


async def main():
    """Run all tests"""
    # Main functionality test
    all_working = await test_all_functions()
    
    # Error handling test
    await test_error_handling()
    
    print("\n" + "="*50)
    if all_working:
        print("🎉 SUCCESS: Simplified server is stable and ready!")
        print("\nYou can now:")
        print("1. Replace the complex server with this simplified version")
        print("2. Add proper MCP integration")
        print("3. Create comprehensive tests")
        print("4. Document the simplified API")
        return 0
    else:
        print("❌ FAILURE: Core functionality still has issues.")
        print("\nFocus on fixing the failing functions above.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)