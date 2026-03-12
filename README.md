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

## Getting Started

This section will guide you through setting up and running the therapy transcript analysis pipeline with sample data.

### Prerequisites

Before you begin, ensure you have:

1. Python 3.8.1 or later installed
2. Poetry (recommended) or pip for package management
3. Basic familiarity with command line tools
4. (Optional) API keys for OpenAI or Anthropic if using LLM features

### Quick Start Guide

#### 1. Clone the Repository and Install

```bash
# Clone the repository
git clone https://github.com/username/python-therapy-nlp-pipeline
cd python-therapy-nlp-pipeline

# Install dependencies
poetry install  # or 'pip install -r requirements.txt'
```

#### 2. Set Up Sample Data

The repository includes sample therapy transcript data in the `data/raw` directory. If you want to use your own data, format it according to the [Input Data Format](#input-data-format) section and place it in this directory.

#### 3. Run Basic Analysis

```bash
# Using Poetry
poetry run therapy-nlp --input data/raw --output reports

# Using Python directly
python src/pipeline.py --input data/raw --output reports
```

This will:

- Load transcripts from `data/raw`
- Process and analyze the text
- Generate sentiment analysis and topic modeling
- Create visualizations and reports in the `reports` directory

#### 4. Explore the Results

After running the pipeline, check the `reports` directory for:

- **Individual session reports**: Details about each transcript's analysis
- **Progress report**: Analysis of changes across multiple sessions
- **Visualizations**: Charts showing sentiment trends and topic distributions

#### 5. Enable LLM Features (Optional)

To use LLM-enhanced analysis:

```bash
# Set your API key as an environment variable
export OPENAI_API_KEY="your-api-key-here"  # For OpenAI
# OR
export ANTHROPIC_API_KEY="your-api-key-here"  # For Anthropic

# Run the pipeline with LLM features
poetry run therapy-nlp --input data/raw --output reports
```

### Walkthrough with Sample Data

Let's analyze the sample therapy transcripts step by step:

1. **Examine the input**: The sample transcripts in `data/raw` show therapy sessions for a client over multiple dates.

2. **Run the pipeline**:

   ```bash
   poetry run therapy-nlp --input data/raw --output reports
   ```

3. **Understand the output**:
   - Open the `reports/overall_progress.md` file to see the client's emotional progress
   - Examine the topic evolution chart to identify recurring themes
   - Review individual session reports to understand the client's journey

### Customizing the Pipeline

To adjust pipeline behavior, modify the configuration file at `src/config/config.toml`:

```bash
# Use a custom config file
poetry run therapy-nlp --config my_custom_config.toml --input data/raw --output reports
```

Common customizations include:

- Adjusting the number of topics for topic modeling
- Changing visualization styles and colors
- Modifying text preprocessing parameters
- Configuring LLM settings for different types of insights

### Troubleshooting

**Missing NLTK resources**: If you encounter errors about missing NLTK data, run:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')
```

**LLM API issues**:

- Verify your API key is set correctly
- Check your internet connection
- Ensure your API key has sufficient credits/quota

**Performance concerns**: For large datasets, consider processing files in smaller batches or adjusting the topic modeling parameters for faster execution.

**Installation or script issues**: If you encounter problems with package installation or running the scripts, see our detailed [Troubleshooting Guide](TROUBLESHOOTING.md).

### Simple Example: Creating and Analyzing a Transcript

Let's create a simple therapy transcript and analyze it:

1. **Create a new transcript file**:

Create a file named `session_2023-07-15_new.mdx` in the `data/raw` directory with this content:

```mdx
---
title: 'Weekly Therapy Session'
patient: 'John Doe'
therapist: 'Dr. Smith'
session_type: 'Individual'
---

Therapist: How have you been feeling since our last session?

Client: I've been having ups and downs. Work has been stressful, but I tried those breathing exercises you suggested when I feel overwhelmed.

Therapist: That's good to hear. How effective were the breathing exercises for you?

Client: They helped a lot actually. I was surprised. The first time I tried, I felt silly, but then I noticed my heart rate slowing down.

Therapist: That's excellent progress. What about your sleep patterns we discussed last time?

Client: Still struggling with that. I'm getting maybe 5-6 hours, and I wake up a few times during the night.

Therapist: Let's explore that more. What do you think is keeping you awake?

Client: Mostly worrying about work deadlines and replaying conversations in my head.

Therapist: I see. Would you be open to trying a new evening routine to help with sleep?

Client: Yes, definitely. I'm willing to try anything at this point.
```

2. **Run the pipeline on your new transcript**:

```bash
poetry run therapy-nlp --input data/raw --output reports
```

3. **Examine the results**:

Open `reports/session_2023-07-15_new.md` to see:

- Sentiment analysis showing emotional tone throughout the session
- Identified topics including "sleep issues," "stress management," and "coping techniques"
- Visualization of sentiment flow during the conversation
- (If LLM is enabled) Enhanced insights about John's progress with breathing techniques and areas that need work

This example demonstrates how the pipeline identifies key therapeutic elements like:

- Client's progress with coping strategies
- Ongoing challenges with sleep
- The therapist's approach to building on successful interventions
- Potential areas for future sessions

### Minimal Example Script

For those who want to understand the core functionality or integrate parts of the pipeline into their own code, here's a minimal example script that demonstrates the key components:

```python
# minimal_example.py
import os
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from gensim.models import LdaModel
from gensim.corpora import Dictionary

# Ensure NLTK resources are available
nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# 1. Load a sample transcript
transcript_path = 'data/raw/session_2023-07-15_new.mdx'
with open(transcript_path, 'r') as file:
    content = file.read()

# 2. Extract dialogue (simplified)
dialogue = []
for line in content.split('\n'):
    if 'Therapist:' in line or 'Client:' in line:
        parts = line.split(':', 1)
        if len(parts) == 2:
            speaker, text = parts[0], parts[1].strip()
            dialogue.append({'speaker': speaker, 'text': text})

# 3. Create a DataFrame
df = pd.DataFrame(dialogue)
print(f"Loaded {len(df)} dialogue turns")

# 4. Sentiment Analysis
sia = SentimentIntensityAnalyzer()
df['sentiment'] = df['text'].apply(lambda x: sia.polarity_scores(x)['compound'])

# 5. Prepare text for topic modeling
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words('english'))
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [t for t in tokens if t.isalpha() and t not in stop_words]
    return tokens

df['tokens'] = df['text'].apply(preprocess_text)

# 6. Topic Modeling
# Create dictionary and corpus
dictionary = Dictionary(df['tokens'])
corpus = [dictionary.doc2bow(text) for text in df['tokens']]

# Train LDA model
lda_model = LdaModel(
    corpus=corpus,
    id2word=dictionary,
    num_topics=3,
    passes=10
)

# 7. Display Results
print("\nSentiment Analysis:")
print(f"Average sentiment: {df['sentiment'].mean():.2f}")
print(f"Client sentiment: {df[df['speaker'] == 'Client']['sentiment'].mean():.2f}")
print(f"Therapist sentiment: {df[df['speaker'] == 'Therapist']['sentiment'].mean():.2f}")

print("\nDiscovered Topics:")
for topic_id, topic in lda_model.print_topics(num_words=5):
    print(f"Topic {topic_id+1}: {topic}")

# 8. Plot sentiment flow
plt.figure(figsize=(10, 5))
plt.plot(df.index, df['sentiment'], marker='o', linestyle='-')
plt.axhline(y=0, color='r', linestyle='--', alpha=0.3)
plt.xlabel('Dialogue Turn')
plt.ylabel('Sentiment Score')
plt.title('Emotional Flow During Therapy Session')
plt.tight_layout()
plt.savefig('sentiment_flow.png')
print("\nSentiment flow chart saved as 'sentiment_flow.png'")
```

Run this script with:

```bash
python minimal_example.py
```

This simplified script demonstrates:

- Basic transcript parsing
- Sentiment analysis with VADER
- Topic modeling with LDA
- Simple visualization
- Core NLP concepts used in the full pipeline

For advanced features like progress tracking across multiple sessions, LLM integration, or customized report generation, use the full pipeline as described above.

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

#### Environment Variables with dotenv

For sensitive information like API keys and environment-specific settings, the pipeline uses `python-dotenv` to load variables from a `.env` file. To use this feature:

1. Copy the example file to create your own `.env`:

   ```bash
   cp .env.example .env
   ```

2. Edit the `.env` file and fill in your values:

   ```
   OPENAI_API_KEY=sk-your-actual-key
   ANTHROPIC_API_KEY=sk-ant-your-actual-key
   ```

3. The application will automatically load these variables at startup. You can access them in your code using:

   ```python
   import os
   from dotenv import load_dotenv

   # Load environment variables from .env file
   load_dotenv()

   # Access variables
   openai_key = os.getenv("OPENAI_API_KEY")
   ```

This approach is safer than hardcoding keys in your configuration files or scripts, especially if you're sharing your code or using version control.

##### Example Script

The repository includes an example script that demonstrates how to use dotenv in your code:

```bash
# Run the dotenv example script
python src/examples/dotenv_example.py
```

This script shows how to:

- Load environment variables from a .env file
- Access variables with fallback values
- Configure different aspects of the pipeline based on environment settings
- Securely handle API keys

##### Available Environment Variables

| Variable                    | Description                                 | Default  |
| --------------------------- | ------------------------------------------- | -------- |
| `OPENAI_API_KEY`            | Your OpenAI API key                         | None     |
| `ANTHROPIC_API_KEY`         | Your Anthropic API key                      | None     |
| `DEBUG`                     | Enable debug mode                           | false    |
| `LOG_LEVEL`                 | Logging level                               | INFO     |
| `NUM_TOPICS`                | Number of topics for LDA modeling           | 5        |
| `MIN_TOPIC_COHERENCE`       | Minimum coherence score for topics          | 0.3      |
| `SENTIMENT_THRESHOLD`       | Threshold for significant sentiment changes | 0.05     |
| `GENERATE_VISUALIZATIONS`   | Whether to generate visualization files     | true     |
| `SAVE_INTERMEDIATE_RESULTS` | Save intermediate processing results        | false    |
| `REPORT_FORMAT`             | Format for generated reports                | markdown |

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
