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

## Pipeline Architecture

The therapy session analysis pipeline consists of several modular components that work together to process therapy transcripts and extract meaningful insights. Below is a detailed explanation of each step in the pipeline and the technologies used:

### 1. Data Loading and Preprocessing

**Component**: `src/data/loader.py` and `src/preprocessor/text_processor.py`

**Description**: This step involves loading therapy session transcripts from various file formats and preparing the text for analysis.

**Technologies Used**:

- **pandas**: For data manipulation and creating structured DataFrames from raw transcripts
- **regular expressions**: For pattern matching to identify speakers and dialogue
- **YAML parser**: For extracting metadata from frontmatter in MDX files
- **JSON parser**: For processing structured JSON transcripts

**Process**:

1. Identify file type (MDX, JSON, TXT) and extract session metadata
2. Parse dialogue, identifying therapist and client utterances
3. Extract session date from filename or metadata
4. Validate transcript structure and data integrity
5. Organize data into standardized DataFrame format

### 2. Text Preprocessing

**Component**: `src/preprocessor/text_processor.py`

**Description**: Raw text is cleaned and normalized to improve the quality of downstream analysis.

**Technologies Used**:

- **NLTK**: For tokenization, stop word removal, and stemming/lemmatization
- **string manipulation**: For text normalization and cleaning

**Process**:

1. Tokenization: Split text into individual words or tokens
2. Normalization: Convert text to lowercase, remove punctuation
3. Stop word removal: Filter out common words with little semantic value
4. Lemmatization: Reduce words to their base forms
5. Special character handling: Process domain-specific symbols and abbreviations

### 3. Sentiment Analysis

**Component**: `src/analyzers/sentiment_analyzer.py`

**Description**: Evaluates the emotional tone of the dialogue, identifying positive, negative, and neutral sentiments.

**Technologies Used**:

- **NLTK VADER**: For rule-based sentiment analysis optimized for social media and conversational text
- **pandas**: For organizing sentiment scores and metadata

**Process**:

1. Process each utterance with VADER to get compound, positive, negative, and neutral scores
2. Calculate session-level sentiment metrics (averages, trends within session)
3. Track sentiment changes between client and therapist
4. Identify emotional peaks and transitions during the session
5. Generate summary statistics for the session's emotional tone

### 4. Topic Modeling

**Component**: `src/analyzers/topic_modeler.py`

**Description**: Discovers latent topics or themes in the dialogue using unsupervised learning.

**Technologies Used**:

- **Gensim**: For Latent Dirichlet Allocation (LDA) implementation
- **scikit-learn**: For TF-IDF vectorization and NMF alternative topic modeling
- **NLTK**: For additional text processing specific to topic modeling

**Process**:

1. Create document-term matrix using TF-IDF vectorization
2. Train LDA model with optimal hyperparameters (determined through coherence scores)
3. Extract top keywords for each topic
4. Assign topic probability distributions to each utterance
5. Track topic prevalence throughout the session
6. Create human-readable topic labels based on keyword clusters

### 5. Progress Tracking

**Component**: `src/analyzers/progress_tracker.py`

**Description**: Analyzes changes in sentiment and topics across multiple sessions to track therapeutic progress.

**Technologies Used**:

- **pandas**: For time-series analysis and data manipulation
- **numpy**: For numerical calculations and trend analysis
- **scikit-learn**: For regression analysis to identify trends

**Process**:

1. Aggregate sentiment and topic data across sessions
2. Calculate moving averages and trends
3. Identify significant changes in emotional tone or topic prevalence
4. Compare client language patterns across sessions
5. Generate progress metrics and quantifiable indicators of therapeutic change

### 6. Visualization

**Component**: `src/visualizers/`

**Description**: Creates visual representations of the analysis results to facilitate understanding and interpretation.

**Technologies Used**:

- **matplotlib**: For creating base plots and charts
- **seaborn**: For enhanced statistical visualizations
- **plotly** (optional): For interactive visualizations

**Process**:

1. Generate sentiment trend charts showing emotional changes over time
2. Create topic distribution visualizations (bar charts, heatmaps)
3. Produce session comparison visualizations to track progress
4. Generate word clouds for topic keywords
5. Create interactive timeline visualizations of therapy progression

### 7. Report Generation

**Component**: `src/reports/report_generator.py`

**Description**: Compiles analysis results into comprehensive, human-readable reports.

**Technologies Used**:

- **Markdown**: For formatting report text
- **Jinja2**: For templating report structures
- **pandas**: For organizing data for reports

**Process**:

1. Compile analysis results from all pipeline components
2. Generate individual session reports with detailed insights
3. Create progress reports comparing multiple sessions
4. Highlight key findings and actionable insights
5. Format reports with appropriate headings, sections, and visualizations

### 8. LLM Integration (Optional)

**Component**: `src/llm/`

**Description**: Leverages large language models to generate enhanced insights, summaries, and recommendations.

**Technologies Used**:

- **OpenAI API**: For accessing GPT models (requires API key)
- **Anthropic API**: For accessing Claude models (requires API key)
- **requests**: For API communication

**Process**:

1. Prepare prompts based on session content and analysis results
2. Send contextual data to the LLM API with appropriate parameters
3. Process and validate LLM responses
4. Integrate LLM-generated insights into reports
5. Generate therapeutic recommendations based on session content
6. Create natural language summaries of technical analysis results

Each component in the pipeline is designed to be modular, allowing users to customize or replace specific parts based on their needs while maintaining the overall workflow.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
