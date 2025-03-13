#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Text Processor Module

This module handles text preprocessing for therapy session transcripts,
including tokenization, stopword removal, and other NLP tasks.
"""

import re
import logging
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


class TextProcessor:
    """Class for preprocessing text data from therapy session transcripts."""

    def __init__(self, config):
        """Initialize the text processor with configuration.

        Args:
            config (dict): Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger("therapy_pipeline.text_processor")

        # Download required NLTK resources if not already downloaded
        self._download_nltk_resources()

        # Initialize preprocessing settings from config
        self.remove_stopwords = config["preprocessing"].get("remove_stopwords", True)
        self.lemmatize = config["preprocessing"].get("lemmatize", True)
        self.min_word_length = config["preprocessing"].get("min_word_length", 3)
        self.remove_numbers = config["preprocessing"].get("remove_numbers", True)

        # Set up custom stopwords
        self.stopwords = set(stopwords.words("english"))
        custom_stopwords = config["preprocessing"].get("custom_stopwords", [])
        self.stopwords.update(custom_stopwords)

        # Initialize lemmatizer if needed
        self.lemmatizer = WordNetLemmatizer() if self.lemmatize else None

    def _download_nltk_resources(self):
        """Download necessary NLTK resources."""
        resources = [
            "punkt",
            "stopwords",
            "wordnet",
            "vader_lexicon",  # For sentiment analysis
        ]

        for resource in resources:
            try:
                nltk.data.find(
                    f"tokenizers/{resource}" if resource == "punkt" else resource
                )
            except LookupError:
                self.logger.info(f"Downloading NLTK resource: {resource}")
                nltk.download(resource, quiet=True)

    def process(self, transcripts):
        """Process a list of transcript dictionaries.

        Args:
            transcripts (list): List of transcript dictionaries

        Returns:
            list: List of processed transcript dictionaries
        """
        self.logger.info("Processing transcripts")
        processed_transcripts = []

        for transcript in transcripts:
            try:
                processed_transcript = self._process_transcript(transcript)
                processed_transcripts.append(processed_transcript)
            except Exception as e:
                self.logger.error(f"Error processing transcript: {str(e)}")
                # Add the original transcript to avoid data loss
                processed_transcripts.append(transcript)

        self.logger.info(f"Processed {len(processed_transcripts)} transcripts")
        return processed_transcripts

    def _process_transcript(self, transcript):
        """Process a single transcript dictionary.

        Args:
            transcript (dict): Transcript dictionary

        Returns:
            dict: Processed transcript dictionary
        """
        processed_transcript = transcript.copy()

        # Process dialogue text
        if "dialogue" in transcript:
            processed_dialogue = []

            for entry in transcript["dialogue"]:
                processed_entry = entry.copy()

                if "text" in entry and entry["text"]:
                    # Process the text
                    processed_text = self._preprocess_text(entry["text"])
                    processed_tokens = self._tokenize_and_filter(processed_text)

                    # Update the entry
                    processed_entry["processed_text"] = processed_text
                    processed_entry["tokens"] = processed_tokens

                    # Calculate some basic statistics
                    processed_entry["token_count"] = len(processed_tokens)
                    processed_entry["word_count"] = len(entry["text"].split())

                processed_dialogue.append(processed_entry)

            processed_transcript["processed_dialogue"] = processed_dialogue

        # Generate combined text for the entire transcript
        combined_text = " ".join(
            [entry.get("text", "") for entry in transcript.get("dialogue", [])]
        )
        processed_transcript["combined_text"] = combined_text

        # Process the combined text
        processed_text = self._preprocess_text(combined_text)
        processed_tokens = self._tokenize_and_filter(processed_text)
        processed_transcript["processed_combined_text"] = processed_text
        processed_transcript["combined_tokens"] = processed_tokens

        return processed_transcript

    def _preprocess_text(self, text):
        """Preprocess text by removing punctuation, extra whitespace, etc.

        Args:
            text (str): Raw text

        Returns:
            str: Preprocessed text
        """
        # Convert to lowercase
        text = text.lower()

        # Remove URLs
        text = re.sub(r"http\S+|www\S+|https\S+", "", text)

        # Remove punctuation
        text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)

        # Remove numbers if configured
        if self.remove_numbers:
            text = re.sub(r"\d+", "", text)

        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def _tokenize_and_filter(self, text):
        """Tokenize and filter text.

        Args:
            text (str): Preprocessed text

        Returns:
            list: List of filtered tokens
        """
        # Tokenize
        tokens = word_tokenize(text)

        # Filter stopwords if configured
        if self.remove_stopwords:
            tokens = [t for t in tokens if t not in self.stopwords]

        # Filter by minimum word length
        tokens = [t for t in tokens if len(t) >= self.min_word_length]

        # Lemmatize if configured
        if self.lemmatize:
            tokens = [self.lemmatizer.lemmatize(t) for t in tokens]

        return tokens
