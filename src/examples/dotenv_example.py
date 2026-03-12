"""
Example script demonstrating how to use python-dotenv with the therapy NLP pipeline.
This shows how to load environment variables and use them in your analysis.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
import matplotlib.pyplot as plt

# Add the project root to the path to ensure imports work
root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root_dir))

# Load environment variables from .env file
load_dotenv()

# Access environment variables with fallback defaults
DEBUG = os.getenv("DEBUG", "false").lower() == "true"
NUM_TOPICS = int(os.getenv("NUM_TOPICS", "5"))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
REPORT_FORMAT = os.getenv("REPORT_FORMAT", "markdown")


def main():
    """Run a sample pipeline using environment variables for configuration."""
    print("=== Therapy NLP Pipeline with dotenv ===")
    print(f"Debug mode: {DEBUG}")
    print(f"Number of topics: {NUM_TOPICS}")
    print(f"Report format: {REPORT_FORMAT}")

    # Check if API keys are available for LLM features
    if OPENAI_API_KEY:
        print("OpenAI API key is configured. LLM features are available.")
    else:
        print("OpenAI API key is not configured. LLM features will be disabled.")

    # Import pipeline components
    try:
        # Try to import using the package structure
        from src.analyzers.sentiment_analyzer import SentimentAnalyzer
        from src.analyzers.topic_modeler import TopicModeler

        print("Successfully imported pipeline modules.")

        # Load a sample transcript
        print("\nLoading sample transcript...")
        transcript_path = root_dir / "data" / "raw" / "session_2023-05-15_001.mdx"

        if transcript_path.exists():
            print(f"Found transcript at {transcript_path}")
            # Your processing code would go here
        else:
            print(f"Sample transcript not found at {transcript_path}")
            print("Please ensure you have sample data in the data/raw directory.")

    except ImportError as e:
        print(f"Error importing pipeline modules: {e}")
        print("Make sure to install the project with: poetry install")

    print("\n=== Configuration from .env file ===")
    # Display all environment variables that start with specific prefixes
    for key, value in os.environ.items():
        if key in [
            v.split("=")[0] for v in open(".env.example").readlines() if "=" in v
        ]:
            # Mask API keys for security
            if "API_KEY" in key:
                masked_value = (
                    value[:4] + "..."
                    if value and value != "your_openai_api_key_here"
                    else "Not set"
                )
                print(f"{key}: {masked_value}")
            else:
                print(f"{key}: {value}")


if __name__ == "__main__":
    main()
