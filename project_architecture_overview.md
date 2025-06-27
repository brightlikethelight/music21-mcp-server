music21-mcp-server/
├── src/
│   ├── music21_mcp/
│   │   ├── __init__.py
│   │   ├── server.py                    # Main MCP server
│   │   ├── core/
│   │   │   ├── score_parser.py          # Phase 1: Score I/O
│   │   │   ├── theory_analyzer.py       # Phase 1: Music theory
│   │   │   ├── rhythm_analyzer.py       # Phase 1: Rhythm analysis
│   │   │   ├── harmonic_analyzer.py     # Phase 2: Harmony
│   │   │   ├── melodic_analyzer.py      # Phase 2: Melody
│   │   │   ├── voice_analyzer.py        # Phase 2: Voice leading
│   │   │   ├── composer.py              # Phase 3: Composition
│   │   │   └── orchestrator.py          # Phase 3: Orchestration
│   │   ├── analysis/
│   │   │   ├── pattern_recognition.py
│   │   │   ├── style_classification.py
│   │   │   └── statistical_analysis.py
│   │   ├── visualization/
│   │   │   ├── score_renderer.py
│   │   │   ├── analysis_plots.py
│   │   │   └── interactive_displays.py
│   │   ├── utils/
│   │   │   ├── music_validators.py
│   │   │   ├── format_converters.py
│   │   │   └── cache_manager.py
│   │   └── schemas/
│   │       └── music_schemas.py
├── tests/
│   ├── test_data/
│   │   ├── midi_files/
│   │   ├── musicxml_files/
│   │   └── abc_notation/
│   ├── unit/
│   ├── integration/
│   └── performance/
├── examples/
│   ├── analysis_workflows/
│   ├── composition_examples/
│   └── educational_materials/
├── docs/
└── pyproject.toml