# Publishing SublimeX to PyPI

This guide walks you through publishing the `sublimex` package to PyPI.

## Prerequisites

- PyPI account set up ✅
- Package tested and working ✅

## Quick Publishing Steps

### 1. Install Build Tools

```bash
pip install build twine
```

### 2. Update Version (if needed)

Edit `pyproject.toml` and increment the version:

```toml
version = "0.1.0"  # Change to "0.1.1", "0.2.0", etc.
```

### 3. Clean Previous Builds (optional)

```bash
rm -rf dist/ build/ *.egg-info
```

### 4. Build the Package

From the project root directory:

```bash
cd /Users/jwolber/Documents/sublimex
python -m build
```

This creates:
- `dist/sublimex-X.X.X.tar.gz` (source distribution)
- `dist/sublimex-X.X.X-py3-none-any.whl` (wheel distribution)

### 5. Upload to PyPI

```bash
python -m twine upload dist/*
```

You'll be prompted for:
- **Username**: `__token__`
- **Password**: Your PyPI API token (starts with `pypi-`)

### 6. Verify Installation

```bash
pip install sublimex
python -c "import sublimex; print(sublimex.__version__)"
```

## Using API Tokens (Recommended)

### Create a PyPI API Token

1. Go to https://pypi.org/manage/account/token/
2. Click "Add API token"
3. Name it (e.g., "sublimex-upload")
4. Set scope to "Entire account" or specific to "sublimex" project
5. Copy the token (starts with `pypi-`)

### Store Token in ~/.pypirc (Optional)

Create or edit `~/.pypirc`:

```ini
[pypi]
username = __token__
password = pypi-YOUR_TOKEN_HERE

[testpypi]
username = __token__
password = pypi-YOUR_TEST_TOKEN_HERE
```

Then you can upload without entering credentials:

```bash
python -m twine upload dist/*
```

## Version Numbering

Follow semantic versioning (MAJOR.MINOR.PATCH):

- **PATCH** (0.1.0 → 0.1.1): Bug fixes, minor changes
- **MINOR** (0.1.0 → 0.2.0): New features, backward compatible
- **MAJOR** (0.1.0 → 1.0.0): Breaking changes

## Pre-Release Checklist

Before publishing:

- [ ] All tests pass: `python tests/test_sublimex.py`
- [ ] Version number updated in `pyproject.toml`
- [ ] README.md is up to date
- [ ] CHANGELOG updated (if you have one)
- [ ] Git committed and tagged:
  ```bash
  git add .
  git commit -m "Release v0.1.0"
  git tag v0.1.0
  git push origin main --tags
  ```

## Troubleshooting

### "File already exists" error

You cannot upload the same version twice. Increment the version number in `pyproject.toml`.

### Import errors after installation

Make sure all dependencies are listed in `pyproject.toml` under `dependencies`.

### Missing files in package

Check `MANIFEST.in` to ensure all necessary files are included.

## Current Build Status

✅ Package built successfully:
- `dist/sublimex-0.1.0-py3-none-any.whl`
- `dist/sublimex-0.1.0.tar.gz`

Ready to upload to PyPI!
