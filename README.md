# Therapy Session Analysis Pipeline

A modular Python-based pipeline for extracting meaningful insights from therapy session transcripts. This tool analyzes the emotional tone, discovers underlying themes, tracks progress over time, and generates actionable insights from therapy sessions.

## Features

- **Sentiment Analysis**: Gauge the emotional tone of each session using VADER sentiment analysis.
- **Topic Modeling**: Discover underlying themes using Latent Dirichlet Allocation (LDA).
- **Progress Tracking**: Visualize changes in sentiment and topic prevalence over time.
- **Actionable Insights**: Generate summaries to highlight key takeaways from each session.
- **LLM Integration**: Use large language models (like GPT-3 or Claude) to generate enhanced summaries, contextual insights, and therapeutic recommendations.

## Installation

This project requires Python 3.8.1 or later. To install the required dependencies:

```bash
# Using Poetry (recommended)
poetry install

# With LLM support
poetry install -E llm

# Using pip
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
python src/pipeline.py --input data/raw --output reports
```

### Command Line Arguments

- `--config`: Path to the configuration file (default: `src/config/config.toml`)
- `--input`: Directory containing input transcripts
- `--output`: Directory for output reports and visualizations
- `--skip-llm`: Skip LLM-based analysis even if enabled in config

### Configuration

The pipeline is configured using a TOML file at `src/config/config.toml`. Key configuration options include:

- Input/output paths
- Text preprocessing settings
- Sentiment analysis parameters
- Topic modeling parameters
- Visualization settings
- Report generation options
- LLM integration settings

#### LLM Configuration

To use the LLM integration features, you need to configure the `[llm]` section in the config file:

```toml
[llm]
enabled = true     # Set to false to disable LLM features
model = "openai"   # Options: "openai", "anthropic", or "local"
api_key = ""       # Your API key (or use environment variables)
temperature = 0.3  # Controls randomness (0.0 = deterministic, 1.0 = creative)
max_tokens = 1000  # Maximum length of the LLM response
```

You can set the API key in the config file or use environment variables:

- For OpenAI: `OPENAI_API_KEY`
- For Anthropic: `ANTHROPIC_API_KEY`

## Input Data Format

The pipeline supports therapy transcript files in the following formats:

- **MDX**: Markdown files with optional YAML frontmatter for metadata
- **JSON**: Structured JSON files with dialogue and metadata
- **TXT**: Plain text files with basic formatting

### Expected Transcript Format

Transcripts should follow a basic dialogue format with speaker identifiers:

```
Therapist: How are you feeling today?
Client: I'm doing better than last week, but still having some anxiety.
Therapist: Can you tell me more about that anxiety?
```

Transcript filenames should include the session date in the format `YYYY-MM-DD` for automatic date extraction (e.g., `session_2023-05-15_001.mdx`).

## Output

The pipeline generates several types of output:

1. **Session Reports**: Individual reports for each therapy session with sentiment analysis, topic modeling, and actionable insights.
2. **Overall Progress Report**: A comprehensive report showing progress across all sessions.
3. **Visualizations**: Charts and graphs showing sentiment trends, topic distributions, and other insights.
4. **LLM Insights** (if enabled): Enhanced summaries, contextual analysis, and therapeutic recommendations.

## Project Structure

```
python-therapy-nlp-pipeline/
├── data/
│   ├── raw/           # Raw transcript files
│   └── processed/     # Processed transcript data
├── reports/           # Generated reports and visualizations
├── logs/              # Log files
├── src/
│   ├── analyzers/     # Sentiment and topic analysis modules
│   ├── config/        # Configuration files
│   ├── data/          # Data loading and processing modules
│   ├── llm/           # LLM integration modules
│   ├── preprocessor/  # Text preprocessing modules
│   ├── reports/       # Report generation modules
│   ├── visualizers/   # Visualization modules
│   └── pipeline.py    # Main pipeline script
├── pyproject.toml     # Project dependencies (Poetry)
├── requirements.txt   # Dependencies for pip
└── README.md          # This file
```

## Dependencies

This project relies on the following main libraries:

- pandas (~2.0.3) - Data manipulation
- nltk (~3.8.1) - Natural language processing
- gensim (~4.3.2) - Topic modeling
- scikit-learn (~1.3.2) - Machine learning
- matplotlib (~3.7.3) & seaborn (~0.12.2) - Visualization
- openai (~0.28.1) & anthropic (~0.8.1) - LLM integration (optional)

## License

This project is licensed under the MIT License - see the LICENSE file for details.
