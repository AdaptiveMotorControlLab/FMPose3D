# GitHub Actions Workflows

This directory contains GitHub Actions workflows for automated checks and builds.

## Available Workflows

### Build (`build.yaml`)
Builds and tests the package on multiple platforms (Ubuntu and Windows) with Python 3.10 and PyTorch 2.4.0.

**Triggers:** Push and Pull Requests to `main` branch

### Codespell (`codespell.yml`)
Checks for spelling errors in the codebase using codespell.

**Triggers:** Push and Pull Requests to `main` branch

### Check File Headers (`check-headers.yml`)
Ensures all Python files have the proper FMPose3D header with Apache 2.0 license information.

**Triggers:** Push and Pull Requests to `main` branch

**How it works:**
- Runs `scripts/update_headers.py --check` to verify headers
- Fails if any Python files are missing proper headers
- To fix locally, run: `python scripts/update_headers.py` and commit the changes

## Running Header Checks Locally

Before submitting a pull request, you can check and fix headers locally:

```bash
# Check if all files have proper headers
python scripts/update_headers.py --check

# Add/update headers to files that need them
python scripts/update_headers.py
```

The script will:
- Add the standard FMPose3D header to Python files that don't have one
- Replace old header formats with the current standard
- Preserve shebangs (#!) and `from __future__` imports at the top of files
- Skip `__init__.py` files that are very short
