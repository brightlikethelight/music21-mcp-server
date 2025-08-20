"""
Cache Pre-warming Module for Music21 MCP Server

Automatically pre-computes and caches common musical progressions
and analyses to improve performance for typical use cases.
"""

import asyncio
import logging
import time
from typing import Any

from music21 import chord, corpus, key, roman

from .performance_optimizations import PerformanceOptimizer

logger = logging.getLogger(__name__)


class CacheWarmer:
    """Pre-warms caches with common musical patterns and analyses"""

    def __init__(self, optimizer: PerformanceOptimizer):
        self.optimizer = optimizer
        self.stats = {
            "progressions_cached": 0,
            "chords_cached": 0,
            "keys_processed": 0,
            "time_taken_ms": 0,
        }

    async def warm_all_caches(self) -> dict[str, Any]:
        """Warm all caches with common patterns"""
        start_time = time.time()
        logger.info("Starting comprehensive cache warming...")

        # Run warming tasks in parallel
        tasks = [
            self._warm_common_progressions(),
            self._warm_standard_chords(),
            self._warm_bach_chorales(),
        ]

        await asyncio.gather(*tasks)

        self.stats["time_taken_ms"] = (time.time() - start_time) * 1000

        logger.info(
            f"Cache warming completed in {self.stats['time_taken_ms']:.1f}ms. "
            f"Cached {self.stats['progressions_cached']} progressions, "
            f"{self.stats['chords_cached']} chords across "
            f"{self.stats['keys_processed']} keys"
        )

        return self.stats

    async def _warm_common_progressions(self):
        """Pre-compute common chord progressions"""
        # Extended list of common progressions
        progressions = {
            # Pop/Rock progressions
            "I-V-vi-IV": ["I", "V", "vi", "IV"],  # Most common pop progression
            "vi-IV-I-V": ["vi", "IV", "I", "V"],  # Alternative pop
            "I-vi-IV-V": ["I", "vi", "IV", "V"],  # 50s progression
            "I-IV-V-I": ["I", "IV", "V", "I"],  # Basic blues
            # Jazz progressions
            "ii-V-I": ["ii", "V", "I"],  # Most fundamental jazz
            "I-vi-ii-V": ["I", "vi", "ii", "V"],  # Rhythm changes A
            "iii-vi-ii-V": ["iii", "vi", "ii", "V"],  # Turnaround
            "IMaj7-ii7-V7-IMaj7": ["IMaj7", "ii7", "V7", "IMaj7"],
            # Classical progressions
            "I-IV-V7-I": ["I", "IV", "V7", "I"],  # Authentic cadence
            "I-ii6-V-I": ["I", "ii6", "V", "I"],  # With first inversion
            "I-V-V7-I": ["I", "V", "V7", "I"],
            "I-vi-IV-ii-V": ["I", "vi", "IV", "ii", "V"],  # Extended progression
            # Modal progressions
            "i-VII-VI-V": ["i", "VII", "VI", "V"],  # Andalusian cadence
            "i-iv-v-i": ["i", "iv", "v", "i"],  # Natural minor
            "i-iv-V-i": ["i", "iv", "V", "i"],  # Harmonic minor
        }

        # Common keys to process
        major_keys = ["C", "G", "D", "A", "E", "F", "Bb", "Eb"]
        minor_keys = ["a", "e", "d", "g", "c", "f"]

        for key_str in major_keys + minor_keys:
            try:
                k = key.Key(key_str)
                self.stats["keys_processed"] += 1

                for prog_name, prog_symbols in progressions.items():
                    for symbol in prog_symbols:
                        try:
                            # Create Roman numeral and chord
                            rn = roman.RomanNumeral(symbol, k)
                            ch = chord.Chord(rn.pitches)

                            # Cache the analysis
                            self.optimizer.get_cached_roman_numeral(ch, k)
                            self.stats["progressions_cached"] += 1

                        except Exception as e:
                            logger.debug(f"Failed to cache {symbol} in {key_str}: {e}")

            except Exception as e:
                logger.warning(f"Failed to process key {key_str}: {e}")

    async def _warm_standard_chords(self):
        """Pre-compute standard chord types in all keys"""
        # All 12 pitch classes
        roots = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]

        # Common chord types
        chord_types = [
            ("major", ["", "3", "5"]),  # Major triad
            ("minor", ["", "b3", "5"]),  # Minor triad
            ("diminished", ["", "b3", "b5"]),  # Diminished triad
            ("augmented", ["", "3", "#5"]),  # Augmented triad
            ("major7", ["", "3", "5", "7"]),  # Major seventh
            ("dominant7", ["", "3", "5", "b7"]),  # Dominant seventh
            ("minor7", ["", "b3", "5", "b7"]),  # Minor seventh
            ("half-diminished7", ["", "b3", "b5", "b7"]),  # Half-diminished
            ("diminished7", ["", "b3", "b5", "bb7"]),  # Fully diminished
        ]

        for root_note in roots:
            for chord_type, intervals in chord_types:
                try:
                    # Build chord from intervals
                    ch = chord.Chord()
                    ch.add(root_note + "4")  # Add root

                    # Map interval symbols to semitones
                    interval_map = {
                        "b3": 3,
                        "3": 4,
                        "b5": 6,
                        "5": 7,
                        "#5": 8,
                        "b7": 10,
                        "7": 11,
                        "bb7": 9,
                    }

                    for interval in intervals[1:]:  # Skip root
                        if interval in interval_map:
                            ch.add(ch.root().transpose(interval_map[interval]))

                    # Cache in multiple keys where this chord might appear
                    # For example, C major appears in keys of C, F, and G
                    related_keys = self._get_related_keys(root_note, chord_type)

                    for k in related_keys:
                        self.optimizer.get_cached_roman_numeral(ch, k)
                        self.stats["chords_cached"] += 1

                except Exception as e:
                    logger.debug(f"Failed to cache {root_note} {chord_type}: {e}")

    async def _warm_bach_chorales(self):
        """Pre-analyze common patterns from Bach chorales"""
        try:
            # Load a sample Bach chorale for analysis patterns
            bach = corpus.parse("bach/bwv66.6")

            if bach:
                # Analyze key
                k = bach.analyze("key")

                # Get all chords
                chords = bach.chordify().flatten().getElementsByClass(chord.Chord)

                # Cache first 20 chords (most common progression patterns)
                for ch in list(chords)[:20]:
                    try:
                        self.optimizer.get_cached_roman_numeral(ch, k)
                        self.stats["chords_cached"] += 1
                    except Exception:
                        pass  # Some chords might not analyze well

        except Exception as e:
            logger.debug(f"Failed to warm Bach chorale cache: {e}")

    def _get_related_keys(self, root: str, chord_type: str) -> list[key.Key]:
        """Get keys where a chord commonly appears"""
        related = []

        try:
            # For major chords
            if "major" in chord_type or chord_type == "dominant7":
                # Appears as I in its own key
                related.append(key.Key(root))
                # Appears as V in key a fifth below
                related.append(key.Key(root).transpose(-7))
                # Appears as IV in key a fifth above
                related.append(key.Key(root).transpose(7))

            # For minor chords
            elif "minor" in chord_type:
                # Appears as i in its own minor key
                related.append(key.Key(root.lower()))
                # Appears as vi in relative major
                related.append(key.Key(root).transpose(3))
                # Appears as ii in key a whole step below
                related.append(key.Key(root).transpose(-2))

        except Exception:
            pass  # Some transpositions might fail

        return related


async def warm_caches_on_startup(optimizer: PerformanceOptimizer) -> dict[str, Any]:
    """Convenience function to warm caches on server startup"""
    warmer = CacheWarmer(optimizer)
    return await warmer.warm_all_caches()
