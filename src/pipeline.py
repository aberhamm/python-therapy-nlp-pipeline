#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Therapy Session Analysis Pipeline

This module implements a pipeline to analyze therapy session transcripts,
performing sentiment analysis, topic modeling, and generating actionable insights.
"""

import os
import sys
import argparse
import logging
from datetime import datetime
import toml

# Local imports
from preprocessor.text_processor import TextProcessor
from analyzers.sentiment_analyzer import SentimentAnalyzer
from analyzers.topic_modeler import TopicModeler
from visualizers.visualization import Visualizer
from reports.report_generator import ReportGenerator
from data.data_loader import DataLoader
from llm.llm_insight_generator import LLMInsightGenerator


class Pipeline:
    """Main pipeline class that orchestrates the analysis process."""

    def __init__(self, config_path="src/config/config.toml"):
        """Initialize the pipeline with configuration settings.

        Args:
            config_path (str): Path to the TOML configuration file
        """
        self.logger = self._setup_logging()
        self.config = self._load_config(config_path)
        self.logger.info("Pipeline initialized with configuration from %s", config_path)

    def _setup_logging(self):
        """Set up logging configuration."""
        logger = logging.getLogger("therapy_pipeline")
        logger.setLevel(logging.INFO)

        # Create handlers
        console_handler = logging.StreamHandler(sys.stdout)
        file_handler = logging.FileHandler(
            f"logs/pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )

        # Create formatters
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)

        # Add handlers to logger
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

        return logger

    def _load_config(self, config_path):
        """Load configuration from TOML file.

        Args:
            config_path (str): Path to the configuration file

        Returns:
            dict: Configuration dictionary
        """
        try:
            return toml.load(config_path)
        except (FileNotFoundError, toml.TomlDecodeError) as e:
            self.logger.error("Failed to load configuration: %s", str(e))
            sys.exit(1)

    def run(self, input_dir=None, output_dir=None):
        """Run the complete analysis pipeline.

        Args:
            input_dir (str, optional): Directory containing input transcripts
            output_dir (str, optional): Directory for output reports and visualizations
        """
        # Override config with provided arguments if available
        if input_dir:
            self.config["data"]["raw_data_dir"] = input_dir
        if output_dir:
            self.config["reports"]["report_dir"] = output_dir

        self.logger.info("Starting pipeline processing")

        try:
            # 1. Load and preprocess data
            data_loader = DataLoader(self.config)
            transcripts = data_loader.load_transcripts()
            self.logger.info("Loaded %d transcripts", len(transcripts))

            # 2. Preprocess text
            processor = TextProcessor(self.config)
            processed_data = processor.process(transcripts)
            self.logger.info("Processed %d transcripts", len(processed_data))

            # 2a. Generate LLM summaries for raw transcripts if enabled
            llm_insights = {}
            if self.config.get("llm", {}).get("enabled", False):
                self.logger.info("Generating LLM summaries for transcripts")
                llm_generator = LLMInsightGenerator(self.config)

                # Process each transcript with LLM
                for i, transcript in enumerate(processed_data):
                    transcript_insights = llm_generator.generate_transcript_summary(
                        transcript
                    )
                    llm_insights[i] = transcript_insights

                self.logger.info("Completed LLM transcript summarization")

            # 3. Perform sentiment analysis
            sentiment_analyzer = SentimentAnalyzer(self.config)
            sentiment_results = sentiment_analyzer.analyze(processed_data)
            self.logger.info("Completed sentiment analysis")

            # 4. Perform topic modeling
            topic_modeler = TopicModeler(self.config)
            topic_results = topic_modeler.model_topics(processed_data)
            self.logger.info("Completed topic modeling")

            # 4a. Generate LLM analysis insights combining sentiment and topic results if enabled
            if self.config.get("llm", {}).get("enabled", False):
                self.logger.info("Generating LLM analysis insights")

                # Create analysis insights for each transcript
                for i, transcript in enumerate(processed_data):
                    analysis_insights = llm_generator.generate_analysis_insights(
                        transcript, sentiment_results, topic_results
                    )
                    if i in llm_insights:
                        llm_insights[i]["analysis_insights"] = analysis_insights
                    else:
                        llm_insights[i] = {"analysis_insights": analysis_insights}

                # Generate progress insights across all sessions
                if len(processed_data) >= 2:
                    progress_insights = llm_generator.generate_progress_insights(
                        processed_data, sentiment_results, topic_results
                    )
                    llm_insights["progress"] = progress_insights

                self.logger.info("Completed LLM analysis insights generation")

            # 5. Visualize results
            visualizer = Visualizer(self.config)
            visualizer.create_visualizations(
                transcripts=transcripts,
                processed_data=processed_data,
                sentiment_results=sentiment_results,
                topic_results=topic_results,
            )
            self.logger.info("Generated visualizations")

            # 6. Generate insights and reports
            report_generator = ReportGenerator(self.config)
            reports = report_generator.generate_reports(
                transcripts=transcripts,
                processed_data=processed_data,
                sentiment_results=sentiment_results,
                topic_results=topic_results,
                llm_insights=(
                    llm_insights
                    if self.config.get("llm", {}).get("enabled", False)
                    else None
                ),
            )
            self.logger.info("Generated reports and insights")

            self.logger.info("Pipeline completed successfully")
            return {
                "transcripts": transcripts,
                "processed_data": processed_data,
                "sentiment_results": sentiment_results,
                "topic_results": topic_results,
                "llm_insights": (
                    llm_insights
                    if self.config.get("llm", {}).get("enabled", False)
                    else None
                ),
                "reports": reports,
            }

        except Exception as e:
            self.logger.error("Pipeline failed: %s", str(e), exc_info=True)
            raise


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Analyze therapy session transcripts")
    parser.add_argument(
        "--config",
        default="src/config/config.toml",
        help="Path to the configuration file",
    )
    parser.add_argument("--input", help="Directory containing input transcripts")
    parser.add_argument(
        "--output", help="Directory for output reports and visualizations"
    )
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Skip LLM-based analysis even if enabled in config",
    )
    return parser.parse_args()


if __name__ == "__main__":
    # Ensure logs directory exists
    os.makedirs("logs", exist_ok=True)

    # Parse arguments
    args = parse_arguments()

    # Initialize and run pipeline
    pipeline = Pipeline(config_path=args.config)

    # Override LLM settings if --skip-llm is specified
    if args.skip_llm and "llm" in pipeline.config:
        pipeline.config["llm"]["enabled"] = False

    results = pipeline.run(input_dir=args.input, output_dir=args.output)

    print("Analysis completed successfully!")
