# Scripts Directory

This directory contains utility scripts for development, testing, and deployment.

## Available Scripts

### Development Scripts
- `validate_core.sh` - Validate core functionality
- `fix_numpy_env.sh` - Fix NumPy environment issues

### Testing Scripts
- `setup_production_test.sh` - Set up production testing environment
- `setup_production_test_no_numpy.sh` - Set up production testing without NumPy

### Maintenance Scripts
- `clean_git_history.sh` - Clean up git history
- `git_cleanup_commands.sh` - Git cleanup commands
- `remove_claude_references.py` - Remove Claude-specific references

## Usage

Make scripts executable before running:
```bash
chmod +x scripts/script_name.sh
./scripts/script_name.sh
```

For Python scripts:
```bash
python scripts/script_name.py
```