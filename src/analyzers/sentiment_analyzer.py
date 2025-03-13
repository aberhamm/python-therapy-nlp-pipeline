#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Sentiment Analyzer Module

This module analyzes the sentiment of therapy session transcripts
using VADER sentiment analysis from NLTK.
"""

import logging
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer


class SentimentAnalyzer:
    """Class for analyzing sentiment in therapy session transcripts."""

    def __init__(self, config):
        """Initialize the sentiment analyzer with configuration.

        Args:
            config (dict): Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger("therapy_pipeline.sentiment_analyzer")

        # Initialize sentiment model based on config
        model_name = config["sentiment_analysis"].get("model", "vader").lower()

        if model_name == "vader":
            self.sentiment_analyzer = SentimentIntensityAnalyzer()
        else:
            self.logger.warning(
                f"Unsupported sentiment model: {model_name}. Using VADER instead."
            )
            self.sentiment_analyzer = SentimentIntensityAnalyzer()

        # Set thresholds for sentiment classification
        self.positive_threshold = config["sentiment_analysis"].get(
            "compound_threshold_positive", 0.05
        )
        self.negative_threshold = config["sentiment_analysis"].get(
            "compound_threshold_negative", -0.05
        )

        self.logger.info(f"Initialized sentiment analyzer with model: {model_name}")

    def analyze(self, transcripts):
        """Analyze sentiment in a list of transcript dictionaries.

        Args:
            transcripts (list): List of processed transcript dictionaries

        Returns:
            dict: Dictionary containing sentiment analysis results
        """
        self.logger.info("Analyzing sentiment in transcripts")

        sentiment_results = {
            "dialogue_sentiments": [],  # Sentiment of each dialogue entry
            "transcript_sentiments": [],  # Overall sentiment of each transcript
            "speaker_sentiments": {},  # Sentiment by speaker
            "session_progression": [],  # How sentiment changes within each session
        }

        for transcript_idx, transcript in enumerate(transcripts):
            try:
                # Analyze dialogue entries
                if "dialogue" in transcript or "processed_dialogue" in transcript:
                    dialogue = transcript.get(
                        "processed_dialogue", transcript.get("dialogue", [])
                    )
                    dialogue_sentiments = self._analyze_dialogue(dialogue)

                    # Store the dialogue sentiments with transcript metadata
                    for sentiment in dialogue_sentiments:
                        sentiment_entry = sentiment.copy()
                        sentiment_entry.update(
                            {
                                "transcript_idx": transcript_idx,
                                "file_name": transcript.get("file_name"),
                                "session_date": transcript.get("session_date"),
                                "session_number": transcript.get("session_number"),
                            }
                        )
                        sentiment_results["dialogue_sentiments"].append(sentiment_entry)

                    # Calculate overall transcript sentiment
                    transcript_sentiment = self._calculate_transcript_sentiment(
                        dialogue_sentiments
                    )
                    transcript_sentiment.update(
                        {
                            "transcript_idx": transcript_idx,
                            "file_name": transcript.get("file_name"),
                            "session_date": transcript.get("session_date"),
                            "session_number": transcript.get("session_number"),
                        }
                    )
                    sentiment_results["transcript_sentiments"].append(
                        transcript_sentiment
                    )

                    # Calculate sentiment by speaker
                    self._update_speaker_sentiments(
                        dialogue_sentiments, sentiment_results["speaker_sentiments"]
                    )

                    # Calculate sentiment progression within the session
                    session_progression = self._calculate_session_progression(
                        dialogue_sentiments
                    )
                    session_progression.update(
                        {
                            "transcript_idx": transcript_idx,
                            "file_name": transcript.get("file_name"),
                            "session_date": transcript.get("session_date"),
                            "session_number": transcript.get("session_number"),
                        }
                    )
                    sentiment_results["session_progression"].append(session_progression)

            except Exception as e:
                self.logger.error(
                    f"Error analyzing sentiment for transcript {transcript_idx}: {str(e)}"
                )

        # Convert results to DataFrames for easier analysis
        sentiment_results["dialogue_sentiments_df"] = pd.DataFrame(
            sentiment_results["dialogue_sentiments"]
        )

        sentiment_results["transcript_sentiments_df"] = pd.DataFrame(
            sentiment_results["transcript_sentiments"]
        )

        # Convert speaker sentiments to DataFrame
        speaker_sentiment_rows = []
        for speaker, sentiments in sentiment_results["speaker_sentiments"].items():
            row = {"speaker": speaker}
            row.update(sentiments)
            speaker_sentiment_rows.append(row)

        sentiment_results["speaker_sentiments_df"] = pd.DataFrame(
            speaker_sentiment_rows
        )

        # Convert session progression to DataFrame
        sentiment_results["session_progression_df"] = pd.DataFrame(
            sentiment_results["session_progression"]
        )

        self.logger.info("Completed sentiment analysis")
        return sentiment_results

    def _analyze_dialogue(self, dialogue):
        """Analyze sentiment for each entry in a dialogue.

        Args:
            dialogue (list): List of dialogue entries

        Returns:
            list: List of sentiment analysis results for each dialogue entry
        """
        dialogue_sentiments = []

        for i, entry in enumerate(dialogue):
            # Skip if no text
            if not entry.get("text"):
                continue

            # Get the text to analyze
            text = entry.get("processed_text", entry.get("text", ""))

            # Analyze with VADER
            sentiment_scores = self.sentiment_analyzer.polarity_scores(text)

            # Determine sentiment category
            compound_score = sentiment_scores["compound"]
            if compound_score >= self.positive_threshold:
                sentiment_category = "positive"
            elif compound_score <= self.negative_threshold:
                sentiment_category = "negative"
            else:
                sentiment_category = "neutral"

            # Create result entry
            result = {
                "dialogue_idx": i,
                "speaker": entry.get("speaker"),
                "text": entry.get("text"),
                "pos_score": sentiment_scores["pos"],
                "neg_score": sentiment_scores["neg"],
                "neu_score": sentiment_scores["neu"],
                "compound_score": compound_score,
                "sentiment_category": sentiment_category,
            }

            dialogue_sentiments.append(result)

        return dialogue_sentiments

    def _calculate_transcript_sentiment(self, dialogue_sentiments):
        """Calculate overall sentiment for a transcript.

        Args:
            dialogue_sentiments (list): List of sentiment results for dialogue entries

        Returns:
            dict: Overall sentiment results for the transcript
        """
        if not dialogue_sentiments:
            return {
                "avg_compound": 0,
                "avg_positive": 0,
                "avg_negative": 0,
                "avg_neutral": 0,
                "overall_sentiment": "neutral",
                "sentiment_counts": {"positive": 0, "neutral": 0, "negative": 0},
            }

        # Calculate averages
        avg_compound = sum(
            entry["compound_score"] for entry in dialogue_sentiments
        ) / len(dialogue_sentiments)
        avg_positive = sum(entry["pos_score"] for entry in dialogue_sentiments) / len(
            dialogue_sentiments
        )
        avg_negative = sum(entry["neg_score"] for entry in dialogue_sentiments) / len(
            dialogue_sentiments
        )
        avg_neutral = sum(entry["neu_score"] for entry in dialogue_sentiments) / len(
            dialogue_sentiments
        )

        # Count sentiment categories
        sentiment_counts = {"positive": 0, "neutral": 0, "negative": 0}
        for entry in dialogue_sentiments:
            sentiment_counts[entry["sentiment_category"]] += 1

        # Determine overall sentiment
        if avg_compound >= self.positive_threshold:
            overall_sentiment = "positive"
        elif avg_compound <= self.negative_threshold:
            overall_sentiment = "negative"
        else:
            overall_sentiment = "neutral"

        return {
            "avg_compound": avg_compound,
            "avg_positive": avg_positive,
            "avg_negative": avg_negative,
            "avg_neutral": avg_neutral,
            "overall_sentiment": overall_sentiment,
            "sentiment_counts": sentiment_counts,
        }

    def _update_speaker_sentiments(self, dialogue_sentiments, speaker_sentiments):
        """Update speaker sentiment dictionary with new dialogue entries.

        Args:
            dialogue_sentiments (list): List of sentiment results for dialogue entries
            speaker_sentiments (dict): Dictionary to update with speaker sentiments
        """
        # Group by speaker
        for entry in dialogue_sentiments:
            speaker = entry.get("speaker")
            if not speaker:
                continue

            # Initialize speaker entry if not exists
            if speaker not in speaker_sentiments:
                speaker_sentiments[speaker] = {
                    "total_entries": 0,
                    "compound_sum": 0,
                    "pos_sum": 0,
                    "neg_sum": 0,
                    "neu_sum": 0,
                    "sentiment_counts": {"positive": 0, "neutral": 0, "negative": 0},
                }

            # Update speaker sentiments
            speaker_sentiments[speaker]["total_entries"] += 1
            speaker_sentiments[speaker]["compound_sum"] += entry["compound_score"]
            speaker_sentiments[speaker]["pos_sum"] += entry["pos_score"]
            speaker_sentiments[speaker]["neg_sum"] += entry["neg_score"]
            speaker_sentiments[speaker]["neu_sum"] += entry["neu_score"]
            speaker_sentiments[speaker]["sentiment_counts"][
                entry["sentiment_category"]
            ] += 1

            # Calculate averages
            total_entries = speaker_sentiments[speaker]["total_entries"]
            speaker_sentiments[speaker]["avg_compound"] = (
                speaker_sentiments[speaker]["compound_sum"] / total_entries
            )
            speaker_sentiments[speaker]["avg_positive"] = (
                speaker_sentiments[speaker]["pos_sum"] / total_entries
            )
            speaker_sentiments[speaker]["avg_negative"] = (
                speaker_sentiments[speaker]["neg_sum"] / total_entries
            )
            speaker_sentiments[speaker]["avg_neutral"] = (
                speaker_sentiments[speaker]["neu_sum"] / total_entries
            )

            # Determine overall sentiment
            avg_compound = speaker_sentiments[speaker]["avg_compound"]
            if avg_compound >= self.positive_threshold:
                speaker_sentiments[speaker]["overall_sentiment"] = "positive"
            elif avg_compound <= self.negative_threshold:
                speaker_sentiments[speaker]["overall_sentiment"] = "negative"
            else:
                speaker_sentiments[speaker]["overall_sentiment"] = "neutral"

    def _calculate_session_progression(self, dialogue_sentiments):
        """Calculate how sentiment changes throughout a session.

        Args:
            dialogue_sentiments (list): List of sentiment results for dialogue entries

        Returns:
            dict: Session progression metrics
        """
        if not dialogue_sentiments:
            return {
                "start_sentiment": None,
                "end_sentiment": None,
                "sentiment_shift": 0,
                "sentiment_volatility": 0,
                "sentiment_trend": "none",
            }

        # Split the session into beginning, middle, and end
        session_length = len(dialogue_sentiments)
        beginning = dialogue_sentiments[: max(1, session_length // 3)]
        end = dialogue_sentiments[-max(1, session_length // 3) :]

        # Calculate sentiment at beginning and end
        beginning_compound = sum(entry["compound_score"] for entry in beginning) / len(
            beginning
        )
        end_compound = sum(entry["compound_score"] for entry in end) / len(end)

        # Calculate sentiment shift
        sentiment_shift = end_compound - beginning_compound

        # Calculate sentiment volatility (standard deviation)
        compounds = [entry["compound_score"] for entry in dialogue_sentiments]
        mean_compound = sum(compounds) / len(compounds)
        variance = sum((score - mean_compound) ** 2 for score in compounds) / len(
            compounds
        )
        sentiment_volatility = variance**0.5  # Standard deviation

        # Determine trend
        if sentiment_shift > 0.1:
            sentiment_trend = "improving"
        elif sentiment_shift < -0.1:
            sentiment_trend = "deteriorating"
        else:
            sentiment_trend = "stable"

        return {
            "start_sentiment": beginning_compound,
            "end_sentiment": end_compound,
            "sentiment_shift": sentiment_shift,
            "sentiment_volatility": sentiment_volatility,
            "sentiment_trend": sentiment_trend,
        }
