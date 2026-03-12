# Troubleshooting Guide

This document provides solutions for common issues you might encounter when using the Therapy Session Analysis Pipeline.

## Installation and Setup Issues

### Entry Point Warning: `'therapy-nlp' is not installed as a script`

**Problem:**

```
Warning: 'therapy-nlp' is an entry point defined in pyproject.toml, but it's not installed as a script.
You may get improper `sys.argv[0]`.
```

**Solution:** This happens when trying to run the script without properly installing the package. Fix it by:

1. Make sure you're in the project root directory
2. Run a full installation with Poetry:
   ```bash
   poetry install
   ```
3. Then run the command:
   ```bash
   poetry run therapy-nlp --input data/raw --output reports
   ```

### Package Not Found Error

**Problem:**

```
No file/folder found for package python-therapy-nlp-pipeline
```

**Solution:** This means Poetry can't find the package structure. Make sure:

1. You've installed the package with `poetry install`
2. The project structure follows the expected format with src/ as the main package
3. If the error persists, try reinstalling:
   ```bash
   poetry lock --no-update
   poetry install
   ```

## Import Errors

### ModuleNotFoundError: No module named 'X'

**Problem:**

```
ModuleNotFoundError: No module named 'preprocessor'
ModuleNotFoundError: No module named 'analyzers'
```

**Solution:** This usually means Python can't find the package modules. Fix by:

1. Make sure you've installed the package with `poetry install`
2. Always use the full import path in your code:

   ```python
   # Instead of
   from preprocessor.text_processor import TextProcessor

   # Use
   from src.preprocessor.text_processor import TextProcessor
   ```

3. If running a script directly, add the project root to your Python path:
   ```python
   import sys
   from pathlib import Path
   root_dir = Path(__file__).resolve().parent.parent  # Adjust as needed
   sys.path.insert(0, str(root_dir))
   ```

## Environment Variables and Configuration

### API Keys Not Found

**Problem:** LLM features don't work because API keys are not found.

**Solution:**

1. Make sure you've created a `.env` file by copying `.env.example`:
   ```bash
   cp .env.example .env
   ```
2. Edit the `.env` file to add your actual API keys
3. Verify the environment is loaded in your code:
   ```python
   from dotenv import load_dotenv
   load_dotenv()  # This should be called before accessing env vars
   ```

## Running Example Scripts

If you encounter issues running example scripts:

1. Make sure to install all dependencies first:

   ```bash
   poetry install
   ```

2. Run from the project root:

   ```bash
   python src/examples/dotenv_example.py
   ```

3. If modules can't be found, check that you're using the correct import paths:
   ```python
   from src.analyzers.sentiment_analyzer import SentimentAnalyzer
   ```

## Still Having Issues?

If you continue experiencing problems:

1. Run `poetry debug:info` to check your Poetry configuration
2. Verify your Python version matches the requirements (Python 3.8.1+)
3. Try running with the verbose flag: `poetry --verbose run therapy-nlp --help`
4. Check the logs directory for detailed error logs
