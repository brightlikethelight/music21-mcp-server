"""
Comprehensive unit tests for ListScoresTool
Tests listing functionality with various storage states
"""

import pytest
from music21 import stream, note, metadata, meter

from music21_mcp.tools.list_tool import ListScoresTool


class TestListScoresTool:
    """Test ListScoresTool with actual implementation"""

    @pytest.fixture
    def tool(self, clean_score_storage):
        """Create tool instance with clean storage"""
        return ListScoresTool(clean_score_storage)

    @pytest.fixture
    def populated_storage(self, clean_score_storage):
        """Create storage with multiple scores"""
        # Score 1: Simple melody with measures
        s1 = stream.Score()
        part1 = stream.Part()
        m1 = stream.Measure(number=1)
        m1.append(meter.TimeSignature('4/4'))
        for pitch in ['C4', 'D4', 'E4', 'F4']:
            m1.append(note.Note(pitch, quarterLength=1))
        part1.append(m1)
        s1.append(part1)
        s1.metadata = metadata.Metadata()
        s1.metadata.title = "Simple Melody"
        clean_score_storage['score1'] = s1
        
        # Score 2: Two-part piece
        s2 = stream.Score()
        part2a = stream.Part()
        part2b = stream.Part()
        m2 = stream.Measure(number=1)
        for pitch in ['G4', 'A4', 'B4', 'C5']:
            m2.append(note.Note(pitch, quarterLength=1))
        part2a.append(m2.copy())
        part2b.append(m2.copy())
        s2.append(part2a)
        s2.append(part2b)
        s2.metadata = metadata.Metadata()
        s2.metadata.title = "Two Part Invention"
        clean_score_storage['score2'] = s2
        
        # Score 3: No metadata, use movementName
        s3 = stream.Score()
        part3 = stream.Part()
        part3.append(note.Note('C4'))
        s3.append(part3)
        s3.metadata = metadata.Metadata()
        s3.metadata.movementName = "Movement 1"
        clean_score_storage['score3'] = s3
        
        return clean_score_storage

    @pytest.mark.asyncio
    async def test_instantiation(self, clean_score_storage):
        """Test tool can be instantiated with score storage"""
        tool = ListScoresTool(clean_score_storage)
        assert tool.score_manager is clean_score_storage
        assert hasattr(tool, 'execute')

    @pytest.mark.asyncio
    async def test_list_empty_storage(self, tool):
        """Test listing when no scores are stored"""
        result = await tool.execute()
        
        assert result['status'] == 'success'
        assert result['count'] == 0
        assert result['scores'] == []

    @pytest.mark.asyncio
    async def test_list_single_score(self, tool, clean_score_storage):
        """Test listing with one score"""
        # Add a single score
        s = stream.Score()
        s.metadata = metadata.Metadata()
        s.metadata.title = "Test Score"
        part = stream.Part()
        part.append(note.Note('C4'))
        s.append(part)
        clean_score_storage['test_score'] = s
        
        result = await tool.execute()
        
        assert result['status'] == 'success'
        assert result['count'] == 1
        assert len(result['scores']) == 1
        
        score_info = result['scores'][0]
        assert score_info['id'] == 'test_score'
        assert score_info['title'] == 'Test Score'
        assert score_info['notes'] == 1
        assert score_info['parts'] == 1

    @pytest.mark.asyncio
    async def test_list_multiple_scores(self, tool, populated_storage):
        """Test listing with multiple scores"""
        result = await tool.execute()
        
        assert result['status'] == 'success'
        assert result['count'] == 3
        assert len(result['scores']) == 3
        
        # Check that all scores are present
        score_ids = [s['id'] for s in result['scores']]
        assert 'score1' in score_ids
        assert 'score2' in score_ids
        assert 'score3' in score_ids

    @pytest.mark.asyncio
    async def test_score_info_with_metadata(self, tool, populated_storage):
        """Test that metadata is correctly extracted"""
        result = await tool.execute()
        
        # Find score1 in results
        score1_info = next(s for s in result['scores'] if s['id'] == 'score1')
        assert score1_info['title'] == 'Simple Melody'
        assert score1_info['parts'] == 1
        assert score1_info['measures'] == 1
        assert score1_info['notes'] == 4

    @pytest.mark.asyncio
    async def test_score_info_with_movement_name(self, tool, populated_storage):
        """Test using movementName when title is not available"""
        result = await tool.execute()
        
        # Find score3 in results
        score3_info = next(s for s in result['scores'] if s['id'] == 'score3')
        assert score3_info['title'] == 'Movement 1'  # Should use movementName
        assert score3_info['parts'] == 1
        assert score3_info['notes'] == 1

    @pytest.mark.asyncio
    async def test_score_without_title_or_movement(self, tool, clean_score_storage):
        """Test score without title or movementName"""
        s = stream.Score()
        s.metadata = metadata.Metadata()  # Empty metadata
        part = stream.Part()
        part.append(note.Note('C4'))
        s.append(part)
        clean_score_storage['no_title'] = s
        
        result = await tool.execute()
        
        score_info = result['scores'][0]
        assert score_info['id'] == 'no_title'
        assert 'title' not in score_info  # No title field when both are missing
        assert score_info['notes'] == 1

    @pytest.mark.asyncio
    async def test_sorting_by_id(self, tool, clean_score_storage):
        """Test that scores are sorted by ID"""
        # Add scores in non-alphabetical order
        for id in ['zebra', 'alpha', 'middle']:
            s = stream.Score()
            part = stream.Part()
            part.append(note.Note('C4'))
            s.append(part)
            clean_score_storage[id] = s
        
        result = await tool.execute()
        
        ids = [s['id'] for s in result['scores']]
        assert ids == ['alpha', 'middle', 'zebra']  # Sorted alphabetically

    @pytest.mark.asyncio
    async def test_error_handling_for_invalid_score(self, tool, clean_score_storage):
        """Test error handling when score processing fails"""
        # Add a valid score
        s = stream.Score()
        part = stream.Part()
        part.append(note.Note('C4'))
        s.append(part)
        clean_score_storage['valid'] = s
        
        # Add an object that will cause errors
        class BadScore:
            def flatten(self):
                raise Exception("Processing error")
        
        clean_score_storage['bad'] = BadScore()
        
        result = await tool.execute()
        
        # Should still succeed but mark the bad score
        assert result['status'] == 'success'
        assert result['count'] == 2
        
        # Find the bad score entry
        bad_info = next(s for s in result['scores'] if s['id'] == 'bad')
        assert 'error' in bad_info
        assert bad_info['error'] == 'Failed to extract metadata'

    @pytest.mark.asyncio
    async def test_score_with_no_parts_attribute(self, tool, clean_score_storage):
        """Test handling of scores without parts attribute"""
        # Create a stream.Part directly (not a Score)
        part = stream.Part()
        for pitch in ['C4', 'D4', 'E4']:
            part.append(note.Note(pitch, quarterLength=1))
        clean_score_storage['part_only'] = part
        
        result = await tool.execute()
        
        score_info = result['scores'][0]
        assert score_info['id'] == 'part_only'
        assert score_info['parts'] == 1  # Default when no parts attribute
        assert score_info['notes'] == 3

    @pytest.mark.asyncio
    async def test_unicode_handling(self, tool, clean_score_storage):
        """Test handling of unicode in titles"""
        s = stream.Score()
        s.metadata = metadata.Metadata()
        s.metadata.title = "Prélude à l'après-midi"
        part = stream.Part()
        part.append(note.Note('C4'))
        s.append(part)
        clean_score_storage['unicode_score'] = s
        
        result = await tool.execute()
        
        score_info = result['scores'][0]
        assert score_info['title'] == "Prélude à l'après-midi"

    @pytest.mark.asyncio
    async def test_large_storage_performance(self, tool, clean_score_storage):
        """Test performance with many scores"""
        # Add 50 scores (reduced for faster tests)
        for i in range(50):
            s = stream.Score()
            s.metadata = metadata.Metadata()
            s.metadata.title = f"Score {i}"
            part = stream.Part()
            part.append(note.Note('C4'))
            s.append(part)
            clean_score_storage[f'score_{i:03d}'] = s
        
        import time
        start = time.time()
        result = await tool.execute()
        duration = time.time() - start
        
        assert result['status'] == 'success'
        assert result['count'] == 50
        assert len(result['scores']) == 50
        # Should complete reasonably quickly
        assert duration < 2.0  # Less than 2 seconds for 50 scores

    @pytest.mark.asyncio
    async def test_empty_score_handling(self, tool, clean_score_storage):
        """Test handling of empty scores"""
        # Completely empty score
        s1 = stream.Score()
        clean_score_storage['empty'] = s1
        
        # Score with empty part
        s2 = stream.Score()
        s2.append(stream.Part())
        clean_score_storage['empty_part'] = s2
        
        result = await tool.execute()
        
        assert result['count'] == 2
        # Both should be listed even if empty
        for score_info in result['scores']:
            assert score_info['id'] in ['empty', 'empty_part']
            assert score_info['notes'] == 0
            assert score_info['measures'] == 0

    @pytest.mark.asyncio
    async def test_validate_inputs_returns_none(self, tool):
        """Test that validate_inputs returns None (no validation needed)"""
        # ListScoresTool doesn't need input validation
        result = tool.validate_inputs()
        assert result is None

    @pytest.mark.asyncio
    async def test_special_characters_in_ids(self, tool, clean_score_storage):
        """Test handling of special characters in score IDs"""
        s = stream.Score()
        part = stream.Part()
        part.append(note.Note('C4'))
        s.append(part)
        
        # Test various special characters
        special_ids = ['score-with-dash', 'score_with_underscore', 'score.with.dot']
        
        for score_id in special_ids:
            clean_score_storage[score_id] = s
        
        result = await tool.execute()
        
        assert result['count'] == len(special_ids)
        returned_ids = [score['id'] for score in result['scores']]
        for special_id in special_ids:
            assert special_id in returned_ids

    @pytest.mark.asyncio
    async def test_measure_counting(self, tool, clean_score_storage):
        """Test accurate measure counting"""
        s = stream.Score()
        part = stream.Part()
        
        # Add multiple measures
        for i in range(4):
            m = stream.Measure(number=i+1)
            m.append(note.Note('C4', quarterLength=1))
            part.append(m)
        
        s.append(part)
        clean_score_storage['measured'] = s
        
        result = await tool.execute()
        
        score_info = result['scores'][0]
        assert score_info['measures'] == 4
        assert score_info['notes'] == 4