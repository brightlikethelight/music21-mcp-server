#!/usr/bin/env python3
"""
Real-World Validation Testing Framework
Tests all tools against actual MusicXML files from IMSLP
Tracks success rates, failure patterns, and generates compatibility matrix
"""
import asyncio
import json
import logging
import os
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from music21_mcp.tools import (ChordAnalysisTool, CounterpointGeneratorTool,
                               DeleteScoreTool, ExportScoreTool,
                               HarmonizationTool, HarmonyAnalysisTool,
                               ImportScoreTool, KeyAnalysisTool,
                               ListScoresTool, PatternRecognitionTool,
                               ScoreInfoTool, StyleImitationTool,
                               VoiceLeadingAnalysisTool)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("real_world_validation.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


class RealWorldValidator:
    """Comprehensive validation against real musical scores"""

    def __init__(self, test_corpus_dir: str = "test_corpus/imslp"):
        self.test_corpus_dir = Path(test_corpus_dir)
        self.score_manager = {}
        self.results = {
            "test_date": datetime.now().isoformat(),
            "total_files": 0,
            "total_tests": 0,
            "passed_tests": 0,
            "failed_tests": 0,
            "success_rate": 0.0,
            "tool_results": defaultdict(
                lambda: {
                    "total": 0,
                    "passed": 0,
                    "failed": 0,
                    "errors": [],
                    "success_rate": 0.0,
                }
            ),
            "file_results": defaultdict(
                lambda: {"tools_passed": [], "tools_failed": [], "errors": {}}
            ),
            "failure_patterns": defaultdict(int),
            "performance_metrics": defaultdict(list),
        }

        # Initialize all tools
        self.tools = {
            "import": ImportScoreTool(self.score_manager),
            "list": ListScoresTool(self.score_manager),
            "key_analysis": KeyAnalysisTool(self.score_manager),
            "chord_analysis": ChordAnalysisTool(self.score_manager),
            "score_info": ScoreInfoTool(self.score_manager),
            "export": ExportScoreTool(self.score_manager),
            "harmony_analysis": HarmonyAnalysisTool(self.score_manager),
            "voice_leading": VoiceLeadingAnalysisTool(self.score_manager),
            "pattern_recognition": PatternRecognitionTool(self.score_manager),
            "harmonization": HarmonizationTool(self.score_manager),
            "counterpoint": CounterpointGeneratorTool(self.score_manager),
            "style_imitation": StyleImitationTool(self.score_manager),
            "delete": DeleteScoreTool(self.score_manager),
        }

        # Define test suites for different file types
        self.test_suites = {
            "basic_analysis": [
                "import",
                "key_analysis",
                "chord_analysis",
                "score_info",
            ],
            "advanced_analysis": [
                "harmony_analysis",
                "voice_leading",
                "pattern_recognition",
            ],
            "generation": ["harmonization", "counterpoint", "style_imitation"],
            "io_operations": ["export", "list", "delete"],
        }

    async def download_test_corpus(self):
        """
        Guide for downloading test corpus from IMSLP
        Since IMSLP has anti-scraping measures, we provide manual instructions
        """
        instructions = """
        === IMSLP Test Corpus Download Instructions ===
        
        To create a comprehensive test corpus, download these MusicXML files from IMSLP:
        
        1. BAROQUE (10 files)
           - Bach: Invention No. 1 BWV 772
           - Bach: Fugue in C minor BWV 847 (WTC I)
           - Bach: Chorale "Aus meines Herzens Grunde" BWV 269
           - Handel: "Hallelujah" from Messiah
           - Vivaldi: "Spring" from Four Seasons (1st movement)
           - Pachelbel: Canon in D
           - Corelli: Concerto Grosso Op. 6 No. 8 (excerpt)
           - Telemann: Fantasia No. 1 for Solo Flute
           - Purcell: "Dido's Lament"
           - Scarlatti: Sonata K. 1 in D minor
        
        2. CLASSICAL (10 files)
           - Mozart: Piano Sonata K. 331, 1st movement
           - Mozart: "Eine kleine Nachtmusik" K. 525 (1st movement)
           - Haydn: Symphony No. 94 "Surprise" (2nd movement theme)
           - Beethoven: "Für Elise" WoO 59
           - Beethoven: Symphony No. 5 (opening)
           - Clementi: Sonatina Op. 36 No. 1
           - C.P.E. Bach: Solfeggietto
           - Haydn: Piano Sonata Hob. XVI:35 (1st movement)
           - Mozart: Clarinet Concerto K. 622 (excerpt)
           - Beethoven: "Moonlight" Sonata (1st movement)
        
        3. ROMANTIC (10 files)
           - Chopin: Prelude Op. 28 No. 4 in E minor
           - Chopin: Waltz Op. 64 No. 2 in C# minor
           - Schumann: "Träumerei" from Kinderszenen
           - Brahms: Intermezzo Op. 117 No. 1
           - Liszt: "Liebestraum" No. 3
           - Mendelssohn: "Spring Song" Op. 62 No. 6
           - Schubert: "Ave Maria" D. 839
           - Grieg: "Morning Mood" from Peer Gynt
           - Tchaikovsky: "Swan Lake" theme
           - Rachmaninoff: Prelude in C# minor Op. 3 No. 2
        
        4. MODERN/CONTEMPORARY (10 files)
           - Debussy: "Clair de Lune"
           - Satie: "Gymnopédie No. 1"
           - Ravel: "Boléro" (excerpt)
           - Bartók: Mikrokosmos No. 140 "Free Variations"
           - Prokofiev: "Peter and the Wolf" (theme)
           - Shostakovich: Prelude Op. 34 No. 1
           - Gershwin: "Summertime"
           - Joplin: "The Entertainer"
           - Piazzolla: "Libertango" (excerpt)
           - Glass: "Metamorphosis One" (excerpt)
        
        5. VARIOUS INSTRUMENTS & ENSEMBLES (10 files)
           - String Quartet: Beethoven Op. 18 No. 1 (excerpt)
           - Wind Quintet: Nielsen Wind Quintet (excerpt)
           - Solo Violin: Bach Partita No. 2 (Allemande)
           - Solo Cello: Bach Suite No. 1 (Prelude)
           - Brass: Gabrieli Canzon per sonar primi toni
           - Choir: Palestrina "Sicut cervus"
           - Orchestra: Rossini "William Tell Overture"
           - Piano Trio: Schubert "Trout" Quintet (theme)
           - Guitar: Tárrega "Recuerdos de la Alhambra"
           - Voice & Piano: Schubert "Gretchen am Spinnrade"
        
        Save all files in: test_corpus/imslp/
        Organize by period: baroque/, classical/, romantic/, modern/, various/
        """

        print(instructions)

        # Create directory structure
        periods = ["baroque", "classical", "romantic", "modern", "various"]
        for period in periods:
            (self.test_corpus_dir / period).mkdir(parents=True, exist_ok=True)

        # Create a sample file list for testing
        sample_files = self.test_corpus_dir / "file_list.json"
        sample_data = {
            "baroque": [
                "bach_invention_1.xml",
                "bach_fugue_c_minor.xml",
                "bach_chorale_269.xml",
                "handel_hallelujah.xml",
                "vivaldi_spring.xml",
            ],
            "classical": [
                "mozart_k331_1st.xml",
                "mozart_kleine_nachtmusik.xml",
                "haydn_surprise.xml",
                "beethoven_fur_elise.xml",
                "beethoven_symphony_5.xml",
            ],
            "romantic": [
                "chopin_prelude_e_minor.xml",
                "chopin_waltz_c_sharp.xml",
                "schumann_traumerei.xml",
                "brahms_intermezzo.xml",
                "liszt_liebestraum.xml",
            ],
            "modern": [
                "debussy_clair_de_lune.xml",
                "satie_gymnopedie.xml",
                "ravel_bolero.xml",
                "bartok_mikrokosmos.xml",
                "prokofiev_peter_wolf.xml",
            ],
            "various": [
                "beethoven_quartet.xml",
                "nielsen_wind_quintet.xml",
                "bach_violin_partita.xml",
                "bach_cello_suite.xml",
                "gabrieli_canzon.xml",
            ],
        }

        with open(sample_files, "w") as f:
            json.dump(sample_data, f, indent=2)

        return sample_files

    async def validate_file(self, file_path: Path, file_id: str) -> Dict[str, Any]:
        """Validate a single file with all tools"""
        file_result = {
            "file": str(file_path),
            "file_id": file_id,
            "tools_passed": [],
            "tools_failed": [],
            "errors": {},
            "performance": {},
        }

        # First, try to import the file
        logger.info(f"Testing file: {file_path.name}")

        start_time = time.time()
        try:
            import_result = await self.tools["import"].execute(
                score_id=file_id, source=str(file_path), source_type="file"
            )

            if import_result["status"] != "success":
                file_result["tools_failed"].append("import")
                file_result["errors"]["import"] = import_result.get(
                    "message", "Import failed"
                )
                self.results["failure_patterns"]["import_failed"] += 1
                return file_result

            file_result["tools_passed"].append("import")
            file_result["performance"]["import"] = time.time() - start_time

        except Exception as e:
            file_result["tools_failed"].append("import")
            file_result["errors"]["import"] = f"Exception: {str(e)}"
            self.results["failure_patterns"]["import_exception"] += 1
            return file_result

        # Now test all other tools
        for suite_name, tool_names in self.test_suites.items():
            for tool_name in tool_names:
                if tool_name == "import":  # Already tested
                    continue

                start_time = time.time()
                try:
                    tool = self.tools[tool_name]

                    # Prepare appropriate arguments for each tool
                    if tool_name == "export":
                        result = await tool.execute(score_id=file_id, format="musicxml")
                    elif tool_name == "harmonization":
                        # Only for monophonic pieces
                        score = self.score_manager.get(file_id)
                        if score and len(score.parts) == 1:
                            result = await tool.execute(
                                score_id=file_id, style="classical"
                            )
                        else:
                            result = {"status": "skipped", "reason": "Not monophonic"}
                    elif tool_name == "counterpoint":
                        # Only for appropriate pieces
                        score = self.score_manager.get(file_id)
                        if score and len(score.parts) == 1:
                            result = await tool.execute(
                                score_id=file_id, species="first"
                            )
                        else:
                            result = {
                                "status": "skipped",
                                "reason": "Not suitable for counterpoint",
                            }
                    elif tool_name == "style_imitation":
                        result = await tool.analyze_style(
                            score_id=file_id, detailed=False
                        )
                    elif tool_name in ["list", "delete"]:
                        # These don't need score_id
                        result = await tool.execute()
                    else:
                        # Most tools just need score_id
                        result = await tool.execute(score_id=file_id)

                    # Check result
                    if result.get("status") == "success" or tool_name == "list":
                        file_result["tools_passed"].append(tool_name)
                        file_result["performance"][tool_name] = time.time() - start_time
                    elif result.get("status") == "skipped":
                        # Don't count skipped as failure
                        continue
                    else:
                        file_result["tools_failed"].append(tool_name)
                        file_result["errors"][tool_name] = result.get(
                            "message", "Unknown error"
                        )
                        self.results["failure_patterns"][f"{tool_name}_failed"] += 1

                except Exception as e:
                    file_result["tools_failed"].append(tool_name)
                    file_result["errors"][tool_name] = f"Exception: {str(e)}"
                    self.results["failure_patterns"][f"{tool_name}_exception"] += 1
                    logger.error(
                        f"Tool {tool_name} failed on {file_path.name}: {str(e)}"
                    )
                    logger.debug(traceback.format_exc())

        # Clean up
        try:
            await self.tools["delete"].execute(score_id=file_id)
        except:
            pass

        return file_result

    async def run_validation(self, limit: Optional[int] = None) -> Dict[str, Any]:
        """Run validation on all files in test corpus"""
        logger.info("Starting real-world validation testing")

        # Find all XML files
        xml_files = []
        for period_dir in self.test_corpus_dir.iterdir():
            if period_dir.is_dir():
                xml_files.extend(period_dir.glob("*.xml"))
                xml_files.extend(period_dir.glob("*.mxl"))
                xml_files.extend(period_dir.glob("*.musicxml"))

        if limit:
            xml_files = xml_files[:limit]

        self.results["total_files"] = len(xml_files)
        logger.info(f"Found {len(xml_files)} files to test")

        # Test each file
        for i, file_path in enumerate(xml_files):
            file_id = f"test_{i}_{file_path.stem}"
            file_result = await self.validate_file(file_path, file_id)

            # Update results
            self.results["file_results"][str(file_path)] = file_result

            # Update tool statistics
            for tool_name in file_result["tools_passed"]:
                self.results["tool_results"][tool_name]["total"] += 1
                self.results["tool_results"][tool_name]["passed"] += 1
                self.results["total_tests"] += 1
                self.results["passed_tests"] += 1

            for tool_name in file_result["tools_failed"]:
                self.results["tool_results"][tool_name]["total"] += 1
                self.results["tool_results"][tool_name]["failed"] += 1
                self.results["total_tests"] += 1
                self.results["failed_tests"] += 1

            # Progress update
            if (i + 1) % 5 == 0:
                logger.info(f"Progress: {i + 1}/{len(xml_files)} files tested")

        # Calculate success rates
        if self.results["total_tests"] > 0:
            self.results["success_rate"] = (
                self.results["passed_tests"] / self.results["total_tests"]
            ) * 100

        for tool_name, tool_stats in self.results["tool_results"].items():
            if tool_stats["total"] > 0:
                tool_stats["success_rate"] = (
                    tool_stats["passed"] / tool_stats["total"]
                ) * 100

        return self.results

    def generate_compatibility_matrix(self) -> str:
        """Generate a compatibility matrix report"""
        report = []
        report.append("=" * 80)
        report.append("MUSIC21 MCP SERVER - REAL WORLD COMPATIBILITY MATRIX")
        report.append("=" * 80)
        report.append(f"Test Date: {self.results['test_date']}")
        report.append(f"Total Files Tested: {self.results['total_files']}")
        report.append(f"Total Tests Run: {self.results['total_tests']}")
        report.append(f"Overall Success Rate: {self.results['success_rate']:.2f}%")
        report.append("")

        # Production readiness check
        if self.results["success_rate"] >= 95:
            report.append("✅ PRODUCTION READY - Success rate exceeds 95% threshold")
        else:
            report.append("❌ NOT PRODUCTION READY - Success rate below 95% threshold")

        report.append("\n" + "=" * 80)
        report.append("TOOL-BY-TOOL ANALYSIS")
        report.append("=" * 80)

        # Sort tools by success rate
        sorted_tools = sorted(
            self.results["tool_results"].items(),
            key=lambda x: x[1]["success_rate"],
            reverse=True,
        )

        for tool_name, stats in sorted_tools:
            status = "✅" if stats["success_rate"] >= 95 else "❌"
            report.append(f"\n{tool_name.upper()}")
            report.append(f"  Status: {status}")
            report.append(f"  Success Rate: {stats['success_rate']:.2f}%")
            report.append(f"  Passed: {stats['passed']}/{stats['total']}")

            if stats["failed"] > 0:
                report.append(f"  Common Errors:")
                # Analyze error patterns for this tool
                error_counts = defaultdict(int)
                for file_path, file_result in self.results["file_results"].items():
                    if tool_name in file_result["errors"]:
                        error_msg = file_result["errors"][tool_name]
                        # Simplify error message for grouping
                        if "Exception:" in error_msg:
                            error_type = (
                                error_msg.split("Exception:")[1].split(":")[0].strip()
                            )
                        else:
                            error_type = error_msg[:50]
                        error_counts[error_type] += 1

                for error_type, count in sorted(
                    error_counts.items(), key=lambda x: x[1], reverse=True
                )[:3]:
                    report.append(f"    - {error_type}: {count} occurrences")

        report.append("\n" + "=" * 80)
        report.append("FILE COMPLEXITY ANALYSIS")
        report.append("=" * 80)

        # Analyze which types of files fail most
        period_stats = defaultdict(lambda: {"total": 0, "failed": 0})

        for file_path, file_result in self.results["file_results"].items():
            path = Path(file_path)
            period = path.parent.name
            period_stats[period]["total"] += 1
            if file_result["tools_failed"]:
                period_stats[period]["failed"] += 1

        report.append("\nFailure Rate by Period:")
        for period, stats in sorted(period_stats.items()):
            if stats["total"] > 0:
                failure_rate = (stats["failed"] / stats["total"]) * 100
                report.append(
                    f"  {period.capitalize()}: {failure_rate:.1f}% failure rate ({stats['failed']}/{stats['total']})"
                )

        report.append("\n" + "=" * 80)
        report.append("MOST PROBLEMATIC FILES")
        report.append("=" * 80)

        # Find files with most failures
        problem_files = []
        for file_path, file_result in self.results["file_results"].items():
            if len(file_result["tools_failed"]) > 3:
                problem_files.append(
                    (
                        Path(file_path).name,
                        len(file_result["tools_failed"]),
                        file_result["tools_failed"],
                    )
                )

        problem_files.sort(key=lambda x: x[1], reverse=True)

        for file_name, fail_count, failed_tools in problem_files[:10]:
            report.append(f"\n{file_name}")
            report.append(f"  Failed Tools ({fail_count}): {', '.join(failed_tools)}")

        report.append("\n" + "=" * 80)
        report.append("PERFORMANCE METRICS")
        report.append("=" * 80)

        # Calculate average performance for each tool
        tool_performance = defaultdict(list)
        for file_result in self.results["file_results"].values():
            for tool_name, duration in file_result.get("performance", {}).items():
                tool_performance[tool_name].append(duration)

        report.append("\nAverage Execution Time by Tool:")
        for tool_name, durations in sorted(tool_performance.items()):
            if durations:
                avg_time = sum(durations) / len(durations)
                max_time = max(durations)
                report.append(
                    f"  {tool_name}: {avg_time:.3f}s avg, {max_time:.3f}s max"
                )

        report.append("\n" + "=" * 80)
        report.append("RECOMMENDATIONS")
        report.append("=" * 80)

        # Generate recommendations based on results
        recommendations = []

        if self.results["success_rate"] < 95:
            recommendations.append(
                "1. Focus on improving tools with lowest success rates"
            )

            # Find most problematic tools
            for tool_name, stats in sorted_tools:
                if stats["success_rate"] < 80:
                    recommendations.append(
                        f"   - {tool_name}: Needs major improvements ({stats['success_rate']:.1f}% success)"
                    )

        if self.results["failure_patterns"].get("import_failed", 0) > 5:
            recommendations.append("2. Improve MusicXML parser compatibility")
            recommendations.append("   - Consider adding error recovery mechanisms")
            recommendations.append(
                "   - Add support for non-standard MusicXML elements"
            )

        if any(
            stats["success_rate"] < 90
            for _, stats in self.results["tool_results"].items()
        ):
            recommendations.append("3. Add input validation and error handling")
            recommendations.append("   - Validate score structure before processing")
            recommendations.append(
                "   - Add graceful degradation for unsupported features"
            )

        for rec in recommendations:
            report.append(rec)

        report.append("\n" + "=" * 80)

        return "\n".join(report)

    def save_results(self, output_dir: Path = Path("validation_results")):
        """Save all results and reports"""
        output_dir.mkdir(exist_ok=True)

        # Save raw results
        with open(
            output_dir / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", "w"
        ) as f:
            json.dump(self.results, f, indent=2, default=str)

        # Save compatibility matrix
        matrix_report = self.generate_compatibility_matrix()
        with open(
            output_dir
            / f"compatibility_matrix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            "w",
        ) as f:
            f.write(matrix_report)

        # Generate and save detailed error report
        error_report = self.generate_error_report()
        with open(
            output_dir
            / f"error_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            "w",
        ) as f:
            f.write(error_report)

        print(f"\nResults saved to {output_dir}")
        print(matrix_report)

        return output_dir

    def generate_error_report(self) -> str:
        """Generate detailed error analysis"""
        report = []
        report.append("=" * 80)
        report.append("DETAILED ERROR ANALYSIS")
        report.append("=" * 80)

        # Group errors by type
        error_categories = defaultdict(list)

        for file_path, file_result in self.results["file_results"].items():
            for tool_name, error_msg in file_result.get("errors", {}).items():
                error_categories[tool_name].append(
                    {"file": Path(file_path).name, "error": error_msg}
                )

        for tool_name, errors in sorted(error_categories.items()):
            report.append(f"\n{tool_name.upper()} ERRORS ({len(errors)} total)")
            report.append("-" * 40)

            # Show first 5 errors as examples
            for error_info in errors[:5]:
                report.append(f"\nFile: {error_info['file']}")
                report.append(f"Error: {error_info['error'][:200]}")

            if len(errors) > 5:
                report.append(f"\n... and {len(errors) - 5} more errors")

        return "\n".join(report)


async def main():
    """Run the real-world validation"""
    validator = RealWorldValidator()

    # First, provide download instructions
    await validator.download_test_corpus()

    print("\n" + "=" * 80)
    print("After downloading files, press Enter to start validation...")
    input()

    # Run validation
    results = await validator.run_validation()

    # Save results
    validator.save_results()

    # Print summary
    if results["success_rate"] >= 95:
        print("\n✅ VALIDATION PASSED - The system is production ready!")
    else:
        print(f"\n❌ VALIDATION FAILED - Success rate: {results['success_rate']:.2f}%")
        print("See compatibility matrix for details.")


if __name__ == "__main__":
    asyncio.run(main())
