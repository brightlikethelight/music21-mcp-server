# Git Workflow and Branching Strategy

## Overview

This project uses a simplified git workflow optimized for stability and clear development progression. We prioritize simple, maintainable processes that support our core goal of 100% reliability.

## Branching Strategy

### Main Branches

```
main (protected)
├── develop (future)
└── feature/* (short-lived)
```

#### `main` Branch
- **Purpose**: Production-ready code only
- **Protection**: Requires PR review and CI passing
- **Quality**: 100% test success rate required
- **Releases**: All releases from this branch
- **Direct Commits**: Only emergency hotfixes (rare)

#### `develop` Branch (Future)
- **Purpose**: Integration branch for upcoming features
- **When**: Will be created when we start Phase 2
- **Current**: Not needed for simplified server
- **Merges**: Feature branches merge here first

### Feature Branches

#### Naming Convention
```bash
# Bug fixes
git checkout -b fix/issue-123-chord-analysis-bug

# New features
git checkout -b feature/add-tempo-detection

# Documentation
git checkout -b docs/improve-api-examples

# Tests
git checkout -b test/add-edge-case-coverage

# CI/CD
git checkout -b ci/add-performance-tests

# Refactoring
git checkout -b refactor/simplify-import-logic
```

#### Branch Lifecycle
1. **Create**: Branch from `main` for all work
2. **Develop**: Make changes with frequent commits
3. **Test**: Ensure all tests pass locally
4. **PR**: Create pull request with template
5. **Review**: Maintainer review and feedback
6. **Merge**: Squash merge to `main`
7. **Cleanup**: Delete feature branch

## Commit Message Convention

We use [Conventional Commits](https://www.conventionalcommits.org/) with our own flavor:

### Format
```
<type>(<scope>): <description>

<body>

<footer>
```

### Types
- `feat`: New feature or tool
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Test additions or changes
- `ci`: CI/CD pipeline changes
- `chore`: Maintenance tasks
- `refactor`: Code restructuring without feature changes
- `perf`: Performance improvements

### Examples

#### Good Commit Messages
```bash
feat(server): Add tempo detection tool

Implement tempo analysis using music21's tempo detection
algorithms. Supports automatic BPM detection and tempo
change analysis throughout a score.

- Add analyze_tempo() tool to server.py
- Include confidence scoring for tempo detection
- Support tempo change timeline analysis
- Add comprehensive tests with classical music examples
- Update API documentation with examples

Closes #45

```

```bash
fix(import): Handle malformed MIDI files gracefully

Fix crash when importing corrupted MIDI files by adding
proper validation and error handling.

Before: UnhandledError when parsing invalid MIDI
After: Returns {"status": "error", "message": "Invalid MIDI file"}

- Add MIDI validation before parsing
- Implement graceful error handling
- Maintain 100% success rate on error conditions
- Add tests for various malformed file types

Fixes #67
```

#### Bad Commit Messages
```bash
# Too vague
fix: fix bug

# No description
feat: add new feature

# Not conventional format
Updated the documentation and fixed some issues
```

## Release Process

### Version Numbering
We follow [Semantic Versioning](https://semver.org/):

- **MAJOR** (X.0.0): Breaking API changes, major rewrites
- **MINOR** (0.X.0): New features, backward compatible
- **PATCH** (0.0.X): Bug fixes, documentation updates

### Release Types

#### Major Release (2.0.0 → 3.0.0)
- Breaking API changes
- Major architecture changes
- Complete feature rewrites
- **Example**: v2.0.0 "The Great Simplification"

#### Minor Release (2.0.0 → 2.1.0)
- New tools or features
- API additions (non-breaking)
- Significant improvements
- **Example**: Adding voice leading analysis

#### Patch Release (2.0.0 → 2.0.1)
- Bug fixes
- Documentation improvements
- Performance optimizations
- Security updates

### Release Workflow

1. **Prepare Release**
   ```bash
   # Update version in pyproject.toml
   # Update CHANGELOG.md
   # Ensure all tests pass
   # Update documentation
   ```

2. **Create Release Commit**
   ```bash
   git commit -m "chore: Prepare release v2.1.0 [release]
   
   Update version to 2.1.0 and prepare release notes.
   
   New features:
   - Add tempo detection tool
   - Improve key detection accuracy
   - Add batch processing examples
   
   Bug fixes:
   - Fix MIDI import edge cases
   - Improve error messages
   
   Documentation:
   - Update API reference
   - Add tempo detection examples"
   ```

3. **Automated Release**
   - GitHub Actions detects `[release]` in commit message
   - Runs full test suite
   - Creates GitHub release
   - Builds and validates package
   - Future: Publishes to PyPI

4. **Post-Release**
   - Monitor for issues
   - Update documentation sites
   - Announce in discussions
   - Plan next release

## Pull Request Process

### PR Template
Every PR should include:

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that breaks existing functionality)
- [ ] Documentation update

## Testing
- [ ] All existing tests pass
- [ ] Added tests for new functionality
- [ ] Tested with real music files
- [ ] Examples work correctly

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No new warnings introduced
```

### Review Requirements

#### Automated Checks (Must Pass)
- ✅ All tests pass across Python 3.8-3.11
- ✅ Code style (Black, Flake8)
- ✅ Type checking (MyPy) [warnings OK]
- ✅ Security scan (Bandit) [warnings OK]
- ✅ Documentation structure validation

#### Manual Review (Required)
- **Code Quality**: Readable, maintainable, follows patterns
- **Testing**: Appropriate test coverage for changes
- **Documentation**: Updated docs and examples
- **API Consistency**: Follows existing patterns
- **Simplicity**: Maintains simplified architecture philosophy

### Review Process
1. **Submission**: Developer creates PR with template
2. **Automated**: CI runs all quality checks
3. **Review**: Maintainer reviews within 1 week
4. **Feedback**: Request changes or approve
5. **Updates**: Developer addresses feedback
6. **Approval**: Maintainer approves when ready
7. **Merge**: Squash merge to maintain clean history

## Git History Management

### Clean History Strategy
- **Squash Merge**: Feature branches squashed to single commit
- **Descriptive Commits**: Each commit tells a complete story
- **Linear History**: Avoid merge commits when possible
- **Atomic Changes**: Each commit is a complete, working change

### Example Clean History
```
2c77219 chore: Update configuration and preserve core analyzers for future use
666d0ec docs: Add comprehensive validation and status reports  
e2d51d5 feat: Add comprehensive examples and tutorials for easy adoption
a0596c5 ci: Add comprehensive CI/CD pipeline with quality gates
329038b test: Add comprehensive test suite achieving 100% success rate
abe41e5 docs: Complete documentation overhaul for simplified server
512fe11 feat: Emergency simplification - Replace complex server with stable version
```

### What to Avoid
```bash
# Messy history
fix typo
fix another typo
forgot to add file
fix tests
fix tests again
Merge branch 'feature' into main
```

## Emergency Procedures

### Hotfix Process
For critical production issues:

1. **Create Hotfix Branch**
   ```bash
   git checkout main
   git pull origin main
   git checkout -b hotfix/critical-security-fix
   ```

2. **Make Minimal Fix**
   - Fix only the critical issue
   - Add test to prevent regression
   - Update documentation if needed

3. **Emergency Review**
   - Create PR immediately
   - Request urgent review
   - Skip normal process if needed for security

4. **Deploy**
   - Merge to main after review
   - Create emergency release
   - Monitor deployment closely

### Rollback Process
If a release causes issues:

1. **Assess Impact**
   - Determine severity and scope
   - Check if hotfix is possible
   - Decide on rollback vs forward fix

2. **Create Rollback**
   ```bash
   git revert <problematic-commit>
   git commit -m "revert: Emergency rollback of v2.1.0
   
   Rolling back due to critical issue with tempo detection.
   Issue: Memory leak in large file processing
   Plan: Fix locally and re-release as v2.1.1"
   ```

3. **Emergency Release**
   - Push rollback commit
   - Create new release immediately
   - Communicate issue to users

## Repository Settings

### Branch Protection (main)
- ✅ Require pull request reviews
- ✅ Require status checks to pass
- ✅ Require branches to be up to date
- ✅ Include administrators
- ❌ Allow force pushes
- ❌ Allow deletions

### Required Status Checks
- `test (3.8, 3.9, 3.10, 3.11)`
- `security`
- `docs`

### Repository Rules
- No direct commits to main
- All changes through PRs
- Maintain linear history when possible
- Comprehensive commit messages required

## Tools and Automation

### Git Hooks
```bash
# Install pre-commit hooks
pre-commit install

# Hooks automatically run:
# - Black code formatting
# - Flake8 linting  
# - Basic test validation
# - Commit message validation
```

### GitHub Integrations
- **GitHub Actions**: Full CI/CD pipeline
- **Dependabot**: Automated dependency updates
- **CodeQL**: Security analysis
- **Branch Protection**: Enforce workflow rules

This git workflow ensures our simplified server maintains its 100% reliability promise through disciplined development practices and comprehensive quality gates.