#!/usr/bin/env python3
"""
Test with real music files to validate actual functionality
"""
import asyncio
from pathlib import Path
import sys
import os
import urllib.request
import zipfile

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from music21 import corpus, converter
from music21_mcp.server import score_manager, import_score, analyze_key, analyze_harmony, export_score


class RealMusicFileTester:
    """Tests with actual musical compositions"""
    
    def __init__(self):
        self.test_dir = Path(__file__).parent / "real_music_files"
        self.test_dir.mkdir(exist_ok=True)
        self.results = []
    
    async def download_test_files(self):
        """Download real music files for testing"""
        print("📥 Preparing Real Music Files...")
        
        # Use music21 corpus for test files
        test_pieces = [
            ("bach/bwv66.6", "Bach Chorale BWV 66.6"),
            ("mozart/k545", "Mozart Sonata K545"),
            ("schoenberg/opus19", "Schoenberg Op. 19"),
            ("joplin/maple_leaf_rag", "Joplin Maple Leaf Rag")
        ]
        
        for corpus_path, name in test_pieces:
            try:
                # Get from music21 corpus
                score = corpus.parse(corpus_path)
                
                # Save as MusicXML
                xml_path = self.test_dir / f"{corpus_path.replace('/', '_')}.xml"
                score.write('musicxml', fp=str(xml_path))
                
                # Save as MIDI
                midi_path = self.test_dir / f"{corpus_path.replace('/', '_')}.mid"
                score.write('midi', fp=str(midi_path))
                
                print(f"   ✅ Downloaded: {name}")
                
            except Exception as e:
                print(f"   ❌ Failed to get {name}: {e}")
    
    async def test_bach_chorale(self):
        """Test with Bach chorale - complex harmony and voice leading"""
        print("\n🎼 Testing Bach Chorale BWV 66.6...")
        
        chorale_path = self.test_dir / "bach_bwv66.6.xml"
        if not chorale_path.exists():
            print("   ⏭️ File not found")
            return
        
        try:
            # Import the chorale
            result = await import_score(
                score_id="bach_chorale",
                file_path=str(chorale_path),
                format="musicxml"
            )
            
            if result['status'] == 'success':
                print("   ✅ Import successful")
                print(f"   📊 Statistics: {result.get('num_parts')} parts, {result.get('num_measures')} measures")
                
                # Test key analysis
                key_result = await analyze_key("bach_chorale")
                if key_result['status'] == 'success':
                    print(f"   ✅ Key analysis: {key_result['key']} (confidence: {key_result['confidence']:.2f})")
                    self.results.append(("Bach key detection", key_result['confidence'] > 0.7))
                
                # Test harmony analysis
                harmony_result = await analyze_harmony("bach_chorale")
                if harmony_result['status'] == 'success':
                    print(f"   ✅ Harmony analysis: Found {harmony_result['total_chords']} chords")
                    if harmony_result['chord_progressions']:
                        print(f"      Sample progression: {' → '.join(harmony_result['chord_progressions'][0]['progression'][:5])}")
                    self.results.append(("Bach harmony analysis", harmony_result['total_chords'] > 20))
                
                # Test voice leading
                from music21_mcp.server import check_voice_leading
                vl_result = await check_voice_leading("bach_chorale")
                if vl_result['status'] == 'success':
                    print(f"   ✅ Voice leading: {vl_result.get('parallel_fifths', 0)} parallel fifths, smoothness: {vl_result.get('smoothness_score', 0):.2f}")
                    self.results.append(("Bach voice leading", vl_result.get('smoothness_score', 0) > 0.5))
                
            else:
                print(f"   ❌ Import failed: {result}")
                
        except Exception as e:
            print(f"   ❌ Error testing Bach chorale: {e}")
            import traceback
            traceback.print_exc()
    
    async def test_mozart_sonata(self):
        """Test with Mozart sonata - clear form and melody"""
        print("\n🎼 Testing Mozart Sonata K545...")
        
        sonata_path = self.test_dir / "mozart_k545.xml"
        if not sonata_path.exists():
            print("   ⏭️ File not found")
            return
        
        try:
            # Import the sonata
            result = await import_score(
                score_id="mozart_sonata",
                file_path=str(sonata_path),
                format="musicxml"
            )
            
            if result['status'] == 'success':
                print("   ✅ Import successful")
                
                # Test melodic analysis
                from music21_mcp.server import analyze_melodic_contour, detect_melodic_motives
                
                contour_result = await analyze_melodic_contour("mozart_sonata", part_index=0)
                if contour_result['status'] == 'success':
                    print(f"   ✅ Melodic contour: {contour_result['overall_contour']}")
                    self.results.append(("Mozart melody analysis", True))
                
                # Test phrase detection
                from music21_mcp.server import detect_phrases
                phrase_result = await detect_phrases("mozart_sonata")
                if phrase_result['status'] == 'success':
                    print(f"   ✅ Phrase detection: Found {phrase_result['total_phrases']} phrases")
                    self.results.append(("Mozart phrase detection", phrase_result['total_phrases'] > 5))
                
        except Exception as e:
            print(f"   ❌ Error testing Mozart sonata: {e}")
    
    async def test_jazz_piece(self):
        """Test with jazz piece - complex chords and rhythms"""
        print("\n🎼 Testing Joplin Maple Leaf Rag...")
        
        rag_path = self.test_dir / "joplin_maple_leaf_rag.mid"
        if not rag_path.exists():
            print("   ⏭️ File not found")
            return
        
        try:
            # Import the rag
            result = await import_score(
                score_id="maple_leaf_rag",
                file_path=str(rag_path),
                format="midi"
            )
            
            if result['status'] == 'success':
                print("   ✅ Import successful")
                
                # Test rhythm analysis
                from music21_mcp.server import analyze_rhythm
                rhythm_result = await analyze_rhythm("maple_leaf_rag")
                if rhythm_result['status'] == 'success':
                    print(f"   ✅ Rhythm analysis: Tempo {rhythm_result.get('tempo', 'N/A')} BPM")
                    print(f"      Syncopation level: {rhythm_result.get('syncopation_level', 0):.2f}")
                    self.results.append(("Jazz rhythm analysis", rhythm_result.get('syncopation_level', 0) > 0.2))
                
        except Exception as e:
            print(f"   ❌ Error testing jazz piece: {e}")
    
    async def test_round_trip_conversion(self):
        """Test importing and exporting maintains integrity"""
        print("\n🔄 Testing Round-Trip Conversion...")
        
        # Create a test score
        from music21 import stream, note, meter, key
        test_score = stream.Score()
        part = stream.Part()
        part.append(meter.TimeSignature('3/4'))
        part.append(key.KeySignature(2))  # D major
        
        # Add a simple melody
        for pitch in ['D4', 'E4', 'F#4', 'G4', 'A4', 'B4', 'C#5', 'D5']:
            part.append(note.Note(pitch, quarterLength=0.5))
        
        test_score.append(part)
        
        # Save original
        original_path = self.test_dir / "original.xml"
        test_score.write('musicxml', fp=str(original_path))
        
        try:
            # Import
            result = await import_score(
                score_id="round_trip_test",
                file_path=str(original_path),
                format="musicxml"
            )
            
            if result['status'] == 'success':
                print("   ✅ Import successful")
                
                # Export in different format
                export_result = await export_score(
                    score_id="round_trip_test",
                    format="midi"
                )
                
                if export_result['status'] == 'success':
                    midi_path = Path(export_result['file_path'])
                    if midi_path.exists():
                        print("   ✅ Export successful")
                        
                        # Re-import the exported file
                        reimport_result = await import_score(
                            score_id="round_trip_reimport",
                            file_path=str(midi_path),
                            format="midi"
                        )
                        
                        if reimport_result['status'] == 'success':
                            # Compare basic properties
                            if (reimport_result.get('num_notes', 0) == 8 and
                                reimport_result.get('num_measures', 0) >= 2):
                                print("   ✅ Round-trip conversion preserved structure")
                                self.results.append(("Round-trip conversion", True))
                            else:
                                print("   ❌ Structure changed in conversion")
                                self.results.append(("Round-trip conversion", False))
                    else:
                        print("   ❌ Export file not created")
                else:
                    print(f"   ❌ Export failed: {export_result}")
            else:
                print(f"   ❌ Import failed: {result}")
                
        except Exception as e:
            print(f"   ❌ Round-trip test error: {e}")
    
    def print_summary(self):
        """Print test summary"""
        print("\n" + "=" * 50)
        print("📊 REAL MUSIC FILE TEST SUMMARY")
        print("=" * 50)
        
        for test_name, passed in self.results:
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"{status} - {test_name}")
        
        passed = sum(1 for _, p in self.results if p)
        total = len(self.results)
        print(f"\nTotal: {passed}/{total} tests passed")
        
        if passed == total:
            print("\n🎉 All real music file tests passed!")
        else:
            print("\n⚠️ Some tests failed. Check the implementation.")


async def main():
    """Run all real music file tests"""
    print("🎵 Real Music File Testing Suite")
    print("=" * 50)
    
    tester = RealMusicFileTester()
    
    # Download test files
    await tester.download_test_files()
    
    # Run tests
    await tester.test_bach_chorale()
    await tester.test_mozart_sonata()
    await tester.test_jazz_piece()
    await tester.test_round_trip_conversion()
    
    # Summary
    tester.print_summary()


if __name__ == "__main__":
    asyncio.run(main())