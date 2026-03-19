# CI/CD Pipeline Guide

## Overview

The Music21 MCP Server uses GitHub Actions for continuous integration and deployment. Our pipeline ensures code quality, security, and reliability through comprehensive automated testing and deployment processes.

## Current Status

- **Coverage**: 84%+ (Required: 82%)
- **Python Versions**: 3.10, 3.11, 3.12, 3.13
- **Protected Branch**: main
- **Auto-deployment**: PyPI on tag push

## GitHub Actions Workflows

### 1. CI/CD Pipeline (`ci.yml`)

**Trigger**: Pull requests and pushes to main

**Jobs**:

#### Lint and Type Check
- **Tools**: Ruff (linting & formatting), MyPy (type checking)
- **Purpose**: Ensure code style consistency and type safety
- **Required**: Yes (blocks merge)

#### Run Tests
- **Matrix**: Python 3.10, 3.11, 3.12, 3.13
- **Coverage**: Must maintain >82% coverage
- **Tools**: pytest with coverage reporting
- **Reports**: Coverage badge updated automatically

#### Security Scan
- **Tools**: 
  - Bandit (Python security linter)
  - pip-audit (dependency vulnerability scanner)
- **Purpose**: Identify security vulnerabilities
- **Required**: Yes (blocks merge)

#### Build Documentation
- **Tools**: Sphinx or similar
- **Purpose**: Ensure documentation builds correctly
- **Required**: Yes (blocks merge)

#### Performance Benchmarks
- **Tools**: Custom benchmarking suite
- **Purpose**: Track performance metrics
- **Required**: No (informational)

#### Integration Tests
- **Purpose**: Test multi-component interactions
- **Environment**: Isolated test environment
- **Required**: No (informational)

#### Release Readiness Check
- **Checks**:
  - Version consistency
  - Changelog updates
  - Documentation completeness
- **Required**: No (informational)

#### Build and Test Distribution
- **Purpose**: Verify package builds correctly
- **Tests**: Installation in clean environment
- **Required**: No (informational)

### 2. Release Workflow (`release.yml`)

**Trigger**: Manual or tag push

**Process**:
1. Bump version based on conventional commits
2. Update CHANGELOG.md
3. Create GitHub release
4. Tag repository
5. Trigger publish workflow

**Semantic Versioning**:
- `feat:` → Minor version bump
- `fix:` → Patch version bump
- `BREAKING CHANGE:` → Major version bump

### 3. Publish Workflow (`publish.yml`)

**Trigger**: Release creation

**Process**:
1. Build source distribution
2. Build wheel distribution
3. Upload to PyPI
4. Verify installation

**Security**: Uses PyPI trusted publishing

### 4. Test PyPI Workflow (`test-pypi.yml`)

**Trigger**: Manual or pre-release

**Purpose**: Test package deployment to TestPyPI before production release

## Branch Protection Rules

### Main Branch Protection

**Required Status Checks**:
- ✅ Lint and Type Check
- ✅ Run Tests
- ✅ Security Scan
- ✅ Build Documentation

**Pull Request Requirements**:
- Minimum 1 approval
- Dismiss stale reviews on new commits
- Up-to-date with base branch

**Restrictions**:
- ❌ No force pushes
- ❌ No branch deletion
- ✅ Include administrators

## Deployment Process

### Development Workflow

1. **Create Feature Branch**
   ```bash
   git checkout -b feature/your-feature
   ```

2. **Develop and Test Locally**
   ```bash
   # Run tests
   pytest tests/ --cov=src/music21_mcp
   
   # Check linting
   uv run ruff check .
   
   # Format code
   uv run ruff format .
   
   # Type check
   uv run mypy .
   ```

3. **Push and Create PR**
   ```bash
   git push origin feature/your-feature
   gh pr create
   ```

4. **CI Pipeline Runs**
   - All checks must pass
   - Coverage must stay above 82%
   - Security scans must pass

5. **Code Review**
   - Get approval from maintainer
   - Address feedback
   - Update if needed

6. **Merge to Main**
   - Squash and merge recommended
   - Delete feature branch

### Release Process

1. **Prepare Release**
   ```bash
   # Ensure main is up to date
   git checkout main
   git pull origin main
   
   # Check release readiness
   python scripts/check_release.py
   ```

2. **Create Release Tag**
   ```bash
   # For patch release (bug fixes)
   git tag v1.0.1
   
   # For minor release (new features)
   git tag v1.1.0
   
   # For major release (breaking changes)
   git tag v2.0.0
   
   # Push tag
   git push origin v1.0.1
   ```

3. **Automatic Deployment**
   - Release workflow creates GitHub release
   - Publish workflow deploys to PyPI
   - Discord webhook notifies of release

4. **Verify Release**
   ```bash
   # Install from PyPI
   pip install music21-mcp-server==1.0.1
   
   # Test installation
   music21-mcp-server --version
   ```

## Monitoring and Notifications

### Discord Webhooks

**Events Monitored**:
- CI/CD pipeline status
- Pull request activity
- Release notifications
- Security alerts

**Setup**: See [Webhook Configuration](.github/webhook-config.md)

### GitHub Status Checks

View pipeline status:
- Pull request checks tab
- Actions tab in repository
- Branch protection status

### Debugging Failed Builds

1. **Check Action Logs**
   ```bash
   gh run list --limit 5
   gh run view RUN_ID --log-failed
   ```

2. **Common Issues**:
   - **Import errors**: Check module dependencies
   - **Coverage drop**: Add tests for new code
   - **Linting failures**: Run `ruff format`
   - **Type errors**: Fix type hints

3. **Re-run Failed Jobs**
   ```bash
   gh run rerun RUN_ID --failed
   ```

## Security Considerations

### Secrets Management

**Repository Secrets**:
- `PYPI_API_TOKEN`: PyPI publishing
- `DISCORD_WEBHOOK_URL`: Notifications
- `CODECOV_TOKEN`: Coverage reporting

**Add Secret**:
```bash
gh secret set SECRET_NAME
```

### Dependency Updates

- Automated security updates via Dependabot
- Weekly vulnerability scans
- Pin major versions in production

### Access Control

- Protected main branch
- Required reviews for changes
- Limited deployment permissions

## Best Practices

### Commit Messages

Follow conventional commits:
```bash
feat: Add new analysis tool
fix: Resolve memory leak in processor
docs: Update API documentation
test: Add integration tests
chore: Update dependencies
```

### Testing

- Write tests before code (TDD)
- Maintain >82% coverage
- Test edge cases
- Mock external dependencies

### Performance

- Run benchmarks before major changes
- Profile memory usage
- Optimize critical paths
- Document performance characteristics

## Troubleshooting

### Pipeline Failures

**Symptom**: CI fails but works locally
**Solution**: 
- Check Python version differences
- Verify all dependencies installed
- Check environment variables

**Symptom**: Coverage drops below threshold
**Solution**:
- Add tests for new code
- Check for dead code removal
- Verify test discovery

**Symptom**: Security scan failures
**Solution**:
- Update vulnerable dependencies
- Review security warnings
- Add security exemptions if false positive

### Deployment Issues

**Symptom**: PyPI upload fails
**Solution**:
- Verify PyPI token is valid
- Check package version unique
- Ensure build artifacts clean

**Symptom**: Installation fails
**Solution**:
- Test in clean environment
- Check dependency conflicts
- Verify Python version compatibility

## Resources

- [GitHub Actions Documentation](https://docs.github.com/actions)
- [PyPI Publishing Guide](https://packaging.python.org/guides/publishing-package-distribution-releases/)
- [Conventional Commits](https://www.conventionalcommits.org/)
- [Semantic Versioning](https://semver.org/)

## Support

For CI/CD issues:
1. Check [Actions tab](https://github.com/brightlikethelight/music21-mcp-server/actions)
2. Review this guide
3. Contact: brightliu@college.harvard.edu