#!/usr/bin/env python3
"""
IMSLP Test Corpus Manager
Manages real-world MusicXML files for comprehensive testing
"""
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class IMSLPScore:
    """Represents a score from IMSLP"""
    title: str
    composer: str
    period: str
    instrumentation: str
    imslp_url: str
    filename: str
    complexity: str  # simple, moderate, complex, extreme
    known_issues: List[str]
    expected_tools: List[str]  # Tools that should work on this score
    
    
class IMSLPTestCorpus:
    """
    Manages a curated test corpus of real IMSLP scores
    Provides metadata about each score for targeted testing
    """
    
    def __init__(self, corpus_dir: Path = Path("test_corpus/imslp")):
        self.corpus_dir = corpus_dir
        self.corpus_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.corpus_dir / "corpus_metadata.json"
        self.scores = self._load_metadata()
        
    def _load_metadata(self) -> Dict[str, IMSLPScore]:
        """Load or create corpus metadata"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                data = json.load(f)
                return {k: IMSLPScore(**v) for k, v in data.items()}
        else:
            return self._create_default_corpus()
    
    def _create_default_corpus(self) -> Dict[str, IMSLPScore]:
        """Create default corpus metadata"""
        corpus = {
            # BAROQUE SCORES
            "bach_invention_1": IMSLPScore(
                title="Invention No. 1 in C major, BWV 772",
                composer="Johann Sebastian Bach",
                period="baroque",
                instrumentation="keyboard",
                imslp_url="https://imslp.org/wiki/Inventions_and_Sinfonias,_BWV_772-801_(Bach,_Johann_Sebastian)",
                filename="bach_invention_1_bwv772.xml",
                complexity="moderate",
                known_issues=[],
                expected_tools=["import", "key", "chord", "harmony", "pattern", "counterpoint", "style"]
            ),
            
            "bach_fugue_c_minor": IMSLPScore(
                title="Fugue in C minor, BWV 847",
                composer="Johann Sebastian Bach", 
                period="baroque",
                instrumentation="keyboard",
                imslp_url="https://imslp.org/wiki/The_Well-Tempered_Clavier_I,_BWV_846-869_(Bach,_Johann_Sebastian)",
                filename="bach_fugue_c_minor_bwv847.xml",
                complexity="complex",
                known_issues=["Complex voice leading"],
                expected_tools=["import", "key", "chord", "harmony", "voice_leading", "pattern", "style"]
            ),
            
            "bach_chorale": IMSLPScore(
                title="Chorale: Aus meines Herzens Grunde, BWV 269",
                composer="Johann Sebastian Bach",
                period="baroque",
                instrumentation="satb",
                imslp_url="https://imslp.org/wiki/Chorales,_BWV_250-438_(Bach,_Johann_Sebastian)",
                filename="bach_chorale_bwv269.xml",
                complexity="moderate",
                known_issues=[],
                expected_tools=["import", "key", "chord", "harmony", "voice_leading", "pattern", "style"]
            ),
            
            "handel_hallelujah": IMSLPScore(
                title="Hallelujah Chorus from Messiah",
                composer="George Frideric Handel",
                period="baroque",
                instrumentation="orchestra_choir",
                imslp_url="https://imslp.org/wiki/Messiah,_HWV_56_(Handel,_George_Frideric)",
                filename="handel_messiah_hallelujah.xml",
                complexity="complex",
                known_issues=["Large ensemble", "Multiple parts"],
                expected_tools=["import", "key", "chord", "harmony", "pattern"]
            ),
            
            "vivaldi_spring": IMSLPScore(
                title="Spring from The Four Seasons, RV 269",
                composer="Antonio Vivaldi",
                period="baroque",
                instrumentation="strings",
                imslp_url="https://imslp.org/wiki/The_Four_Seasons_(Vivaldi,_Antonio)",
                filename="vivaldi_spring_1st_movement.xml",
                complexity="moderate",
                known_issues=["Multiple string parts"],
                expected_tools=["import", "key", "chord", "harmony", "pattern", "style"]
            ),
            
            # CLASSICAL SCORES
            "mozart_sonata_331": IMSLPScore(
                title="Piano Sonata No. 11, K. 331, 1st movement",
                composer="Wolfgang Amadeus Mozart",
                period="classical",
                instrumentation="piano",
                imslp_url="https://imslp.org/wiki/Piano_Sonata_No.11,_K.331_(Mozart,_Wolfgang_Amadeus)",
                filename="mozart_k331_1st_movement.xml",
                complexity="moderate",
                known_issues=[],
                expected_tools=["import", "key", "chord", "harmony", "pattern", "harmonization", "style"]
            ),
            
            "beethoven_fur_elise": IMSLPScore(
                title="Für Elise, WoO 59",
                composer="Ludwig van Beethoven",
                period="classical",
                instrumentation="piano",
                imslp_url="https://imslp.org/wiki/Für_Elise,_WoO_59_(Beethoven,_Ludwig_van)",
                filename="beethoven_fur_elise.xml",
                complexity="simple",
                known_issues=[],
                expected_tools=["import", "key", "chord", "harmony", "pattern", "style", "harmonization"]
            ),
            
            "haydn_surprise": IMSLPScore(
                title="Symphony No. 94 'Surprise', 2nd movement",
                composer="Joseph Haydn",
                period="classical",
                instrumentation="orchestra",
                imslp_url="https://imslp.org/wiki/Symphony_No.94_(Haydn,_Joseph)",
                filename="haydn_surprise_2nd_movement.xml",
                complexity="moderate",
                known_issues=["Orchestral texture"],
                expected_tools=["import", "key", "chord", "harmony", "pattern"]
            ),
            
            # ROMANTIC SCORES
            "chopin_prelude_e_minor": IMSLPScore(
                title="Prelude Op. 28 No. 4 in E minor",
                composer="Frédéric Chopin",
                period="romantic",
                instrumentation="piano",
                imslp_url="https://imslp.org/wiki/Preludes,_Op.28_(Chopin,_Frédéric)",
                filename="chopin_prelude_op28_no4.xml",
                complexity="moderate",
                known_issues=["Chromatic harmony"],
                expected_tools=["import", "key", "chord", "harmony", "pattern", "style"]
            ),
            
            "schumann_traumerei": IMSLPScore(
                title="Träumerei from Kinderszenen, Op. 15",
                composer="Robert Schumann",
                period="romantic",
                instrumentation="piano",
                imslp_url="https://imslp.org/wiki/Kinderszenen,_Op.15_(Schumann,_Robert)",
                filename="schumann_traumerei.xml",
                complexity="moderate",
                known_issues=["Complex voicing"],
                expected_tools=["import", "key", "chord", "harmony", "voice_leading", "style"]
            ),
            
            "brahms_intermezzo": IMSLPScore(
                title="Intermezzo Op. 117 No. 1",
                composer="Johannes Brahms",
                period="romantic",
                instrumentation="piano",
                imslp_url="https://imslp.org/wiki/3_Intermezzi,_Op.117_(Brahms,_Johannes)",
                filename="brahms_intermezzo_op117_no1.xml",
                complexity="complex",
                known_issues=["Dense texture", "Cross-rhythms"],
                expected_tools=["import", "key", "chord", "harmony", "pattern"]
            ),
            
            # MODERN SCORES
            "debussy_clair_de_lune": IMSLPScore(
                title="Clair de Lune from Suite Bergamasque",
                composer="Claude Debussy",
                period="modern",
                instrumentation="piano",
                imslp_url="https://imslp.org/wiki/Suite_bergamasque_(Debussy,_Claude)",
                filename="debussy_clair_de_lune.xml",
                complexity="complex",
                known_issues=["Impressionist harmony", "Non-functional progressions"],
                expected_tools=["import", "key", "chord", "pattern", "style"]
            ),
            
            "satie_gymnopedie": IMSLPScore(
                title="Gymnopédie No. 1",
                composer="Erik Satie",
                period="modern",
                instrumentation="piano",
                imslp_url="https://imslp.org/wiki/3_Gymnopédies_(Satie,_Erik)",
                filename="satie_gymnopedie_1.xml",
                complexity="simple",
                known_issues=["Modal harmony"],
                expected_tools=["import", "key", "chord", "harmony", "pattern", "harmonization"]
            ),
            
            "bartok_mikrokosmos": IMSLPScore(
                title="Free Variations from Mikrokosmos, Sz. 107",
                composer="Béla Bartók",
                period="modern",
                instrumentation="piano",
                imslp_url="https://imslp.org/wiki/Mikrokosmos,_Sz.107_(Bartók,_Béla)",
                filename="bartok_mikrokosmos_140.xml",
                complexity="complex",
                known_issues=["Dissonance", "Complex meters"],
                expected_tools=["import", "pattern"]
            ),
            
            # VARIOUS INSTRUMENTATIONS
            "beethoven_string_quartet": IMSLPScore(
                title="String Quartet Op. 18 No. 1, 1st movement",
                composer="Ludwig van Beethoven",
                period="classical",
                instrumentation="string_quartet",
                imslp_url="https://imslp.org/wiki/String_Quartet_No.1,_Op.18_No.1_(Beethoven,_Ludwig_van)",
                filename="beethoven_op18_no1_1st.xml",
                complexity="complex",
                known_issues=["4-part texture"],
                expected_tools=["import", "key", "chord", "harmony", "voice_leading", "pattern"]
            ),
            
            "bach_cello_suite": IMSLPScore(
                title="Cello Suite No. 1, BWV 1007 - Prelude",
                composer="Johann Sebastian Bach",
                period="baroque",
                instrumentation="solo_cello",
                imslp_url="https://imslp.org/wiki/Cello_Suite_No.1,_BWV_1007_(Bach,_Johann_Sebastian)",
                filename="bach_cello_suite_1_prelude.xml",
                complexity="moderate",
                known_issues=["Monophonic", "Implied harmony"],
                expected_tools=["import", "key", "chord", "pattern", "harmonization", "counterpoint"]
            ),
            
            "palestrina_sicut_cervus": IMSLPScore(
                title="Sicut cervus",
                composer="Giovanni Pierluigi da Palestrina",
                period="renaissance",
                instrumentation="satb",
                imslp_url="https://imslp.org/wiki/Sicut_cervus_(Palestrina,_Giovanni_Pierluigi_da)",
                filename="palestrina_sicut_cervus.xml",
                complexity="moderate",
                known_issues=["Modal", "Renaissance counterpoint"],
                expected_tools=["import", "key", "harmony", "voice_leading", "counterpoint"]
            ),
            
            # EDGE CASES
            "schoenberg_op19": IMSLPScore(
                title="Six Little Piano Pieces, Op. 19, No. 2",
                composer="Arnold Schoenberg",
                period="modern",
                instrumentation="piano",
                imslp_url="https://imslp.org/wiki/6_Little_Piano_Pieces,_Op.19_(Schoenberg,_Arnold)",
                filename="schoenberg_op19_no2.xml",
                complexity="extreme",
                known_issues=["Atonal", "No clear key"],
                expected_tools=["import", "pattern"]
            ),
            
            "cage_prepared_piano": IMSLPScore(
                title="Sonatas and Interludes - Sonata I",
                composer="John Cage",
                period="modern",
                instrumentation="prepared_piano",
                imslp_url="https://imslp.org/wiki/Sonatas_and_Interludes_(Cage,_John)",
                filename="cage_sonata_1.xml",
                complexity="extreme",
                known_issues=["Extended techniques", "Non-traditional notation"],
                expected_tools=["import"]
            ),
            
            "xenakis_metastasis": IMSLPScore(
                title="Metastaseis (excerpt)",
                composer="Iannis Xenakis",
                period="modern",
                instrumentation="orchestra",
                imslp_url="https://imslp.org/wiki/Metastaseis_(Xenakis,_Iannis)",
                filename="xenakis_metastasis_excerpt.xml",
                complexity="extreme",
                known_issues=["Graphic notation", "Extreme complexity"],
                expected_tools=["import"]
            )
        }
        
        # Save metadata
        self.save_metadata(corpus)
        return corpus
    
    def save_metadata(self, corpus: Optional[Dict[str, IMSLPScore]] = None):
        """Save corpus metadata to disk"""
        if corpus is None:
            corpus = self.scores
            
        data = {k: asdict(v) for k, v in corpus.items()}
        with open(self.metadata_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def get_download_instructions(self) -> str:
        """Generate download instructions for the corpus"""
        instructions = [
            "=" * 80,
            "IMSLP TEST CORPUS DOWNLOAD INSTRUCTIONS",
            "=" * 80,
            "",
            "To create a comprehensive test corpus, download these MusicXML files from IMSLP:",
            "",
            "IMPORTANT: Always choose MusicXML format when available!",
            ""
        ]
        
        # Group by period
        by_period = {}
        for score_id, score in self.scores.items():
            if score.period not in by_period:
                by_period[score.period] = []
            by_period[score.period].append((score_id, score))
        
        for period in ['baroque', 'classical', 'romantic', 'modern', 'renaissance']:
            if period in by_period:
                instructions.append(f"\n{period.upper()} PERIOD:")
                instructions.append("-" * 40)
                
                for score_id, score in by_period[period]:
                    instructions.append(f"\n{score.composer} - {score.title}")
                    instructions.append(f"  URL: {score.imslp_url}")
                    instructions.append(f"  Save as: {self.corpus_dir}/{score.period}/{score.filename}")
                    instructions.append(f"  Instrumentation: {score.instrumentation}")
                    instructions.append(f"  Complexity: {score.complexity}")
                    if score.known_issues:
                        instructions.append(f"  Known issues: {', '.join(score.known_issues)}")
        
        instructions.extend([
            "",
            "=" * 80,
            "DIRECTORY STRUCTURE:",
            "",
            "test_corpus/imslp/",
            "├── baroque/",
            "├── classical/",
            "├── romantic/",
            "├── modern/",
            "└── renaissance/",
            "",
            "Place each file in the appropriate period directory.",
            "=" * 80
        ])
        
        return "\n".join(instructions)
    
    def get_test_matrix(self) -> Dict[str, Dict[str, List[str]]]:
        """Get a matrix of which tools should be tested on which scores"""
        matrix = {}
        
        for score_id, score in self.scores.items():
            matrix[score_id] = {
                'expected_tools': score.expected_tools,
                'complexity': score.complexity,
                'instrumentation': score.instrumentation
            }
        
        return matrix
    
    def get_scores_by_complexity(self, complexity: str) -> List[Tuple[str, IMSLPScore]]:
        """Get all scores of a given complexity level"""
        return [(sid, s) for sid, s in self.scores.items() if s.complexity == complexity]
    
    def get_scores_by_instrumentation(self, instrumentation: str) -> List[Tuple[str, IMSLPScore]]:
        """Get all scores with given instrumentation"""
        return [(sid, s) for sid, s in self.scores.items() if s.instrumentation == instrumentation]
    
    def verify_corpus(self) -> Dict[str, bool]:
        """Verify which files are actually present"""
        verification = {}
        
        for score_id, score in self.scores.items():
            file_path = self.corpus_dir / score.period / score.filename
            verification[score_id] = file_path.exists()
        
        return verification


def main():
    """Generate download instructions and verify corpus"""
    corpus = IMSLPTestCorpus()
    
    # Generate download instructions
    instructions = corpus.get_download_instructions()
    
    # Save to file
    with open("IMSLP_DOWNLOAD_INSTRUCTIONS.txt", "w") as f:
        f.write(instructions)
    
    print(instructions)
    print("\nInstructions saved to: IMSLP_DOWNLOAD_INSTRUCTIONS.txt")
    
    # Verify what's already downloaded
    verification = corpus.verify_corpus()
    downloaded = sum(1 for v in verification.values() if v)
    total = len(verification)
    
    print(f"\nCorpus Status: {downloaded}/{total} files downloaded")
    
    if downloaded < total:
        print("\nMissing files:")
        for score_id, present in verification.items():
            if not present:
                score = corpus.scores[score_id]
                print(f"  - {score.filename} ({score.composer} - {score.title})")


if __name__ == "__main__":
    main()