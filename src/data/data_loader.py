#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Data Loader Module

This module is responsible for loading therapy session transcripts from various file formats.
"""

import os
import re
import json
import logging
from datetime import datetime
import pandas as pd


class DataLoader:
    """Class to load therapy session transcripts from files."""

    def __init__(self, config):
        """Initialize the data loader with configuration.

        Args:
            config (dict): Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger("therapy_pipeline.data_loader")
        self.raw_data_dir = config["data"]["raw_data_dir"]
        self.processed_data_dir = config["data"]["processed_data_dir"]
        self.input_format = config["data"]["input_format"]

        # Create processed data directory if it doesn't exist
        os.makedirs(self.processed_data_dir, exist_ok=True)

    def load_transcripts(self):
        """Load all therapy session transcripts from the raw data directory.

        Returns:
            list: List of transcript dictionaries with metadata and content
        """
        self.logger.info(f"Loading transcripts from {self.raw_data_dir}")

        # Check if directory exists
        if not os.path.exists(self.raw_data_dir):
            self.logger.error(f"Raw data directory {self.raw_data_dir} does not exist")
            return []

        transcripts = []

        # List all transcript files
        files = [
            f
            for f in os.listdir(self.raw_data_dir)
            if os.path.isfile(os.path.join(self.raw_data_dir, f))
            and f.endswith(f".{self.input_format}")
        ]

        if not files:
            self.logger.warning(
                f"No {self.input_format} files found in {self.raw_data_dir}"
            )
            return []

        for file_name in files:
            try:
                file_path = os.path.join(self.raw_data_dir, file_name)
                # Parse the file name to extract date and session number
                # Assumes format like: session_YYYY-MM-DD_001.mdx or YYYY-MM-DD_session_001.mdx
                date_match = re.search(r"(\d{4}-\d{2}-\d{2})", file_name)
                session_match = re.search(r"_(\d+)\.\w+$", file_name)

                session_date = None
                if date_match:
                    date_str = date_match.group(1)
                    try:
                        session_date = datetime.strptime(date_str, "%Y-%m-%d")
                    except ValueError:
                        self.logger.warning(
                            f"Could not parse date from filename: {file_name}"
                        )

                session_number = None
                if session_match:
                    session_number = int(session_match.group(1))

                # Load transcript based on file format
                if self.input_format.lower() == "mdx":
                    transcript = self._parse_mdx_file(file_path)
                elif self.input_format.lower() == "json":
                    transcript = self._parse_json_file(file_path)
                elif self.input_format.lower() in ["txt", "text"]:
                    transcript = self._parse_text_file(file_path)
                else:
                    self.logger.warning(f"Unsupported file format: {self.input_format}")
                    continue

                # Add metadata
                transcript["file_name"] = file_name
                transcript["file_path"] = file_path
                transcript["session_date"] = session_date
                transcript["session_number"] = session_number

                transcripts.append(transcript)
                self.logger.debug(f"Loaded transcript: {file_name}")

            except Exception as e:
                self.logger.error(f"Error loading transcript {file_name}: {str(e)}")

        # Sort transcripts by date if available
        transcripts = sorted(
            transcripts,
            key=lambda x: (
                x["session_date"] or datetime.max,
                x["session_number"] or float("inf"),
            ),
        )

        self.logger.info(f"Loaded {len(transcripts)} transcripts")

        # Save to processed format for faster loading next time
        self._save_to_processed(transcripts)

        return transcripts

    def _parse_mdx_file(self, file_path):
        """Parse an MDX (Markdown Extended) file containing a therapy session transcript.

        Args:
            file_path (str): Path to the MDX file

        Returns:
            dict: Parsed transcript with content and metadata
        """
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()

        # Extract frontmatter (metadata) if present
        metadata = {}
        frontmatter_match = re.match(r"^---\s+(.*?)\s+---\s*", content, re.DOTALL)
        if frontmatter_match:
            frontmatter = frontmatter_match.group(1)
            content = content[frontmatter_match.end() :]

            # Parse frontmatter lines
            for line in frontmatter.split("\n"):
                if ":" in line:
                    key, value = line.split(":", 1)
                    metadata[key.strip()] = value.strip()

        # Parse dialogue segments
        dialogue = []
        current_speaker = None
        current_text = []

        for line in content.split("\n"):
            # Speaker identification (e.g. "Therapist:", "Client:", etc.)
            speaker_match = re.match(r"^([\w\s]+):\s*(.*)", line)

            if speaker_match:
                # If we have accumulated text for a previous speaker, add it
                if current_speaker and current_text:
                    dialogue.append(
                        {"speaker": current_speaker, "text": " ".join(current_text)}
                    )

                # Start new speaker segment
                current_speaker = speaker_match.group(1)
                current_text = (
                    [speaker_match.group(2)] if speaker_match.group(2) else []
                )
            elif line.strip() and current_speaker:
                # Continue with current speaker
                current_text.append(line)

        # Add the last segment
        if current_speaker and current_text:
            dialogue.append(
                {"speaker": current_speaker, "text": " ".join(current_text)}
            )

        return {"metadata": metadata, "dialogue": dialogue, "raw_content": content}

    def _parse_json_file(self, file_path):
        """Parse a JSON file containing a therapy session transcript.

        Args:
            file_path (str): Path to the JSON file

        Returns:
            dict: Parsed transcript with content and metadata
        """
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        # Standardize structure
        if "metadata" not in data:
            data["metadata"] = {}

        if "dialogue" not in data:
            # Try to infer dialogue structure
            if "transcript" in data:
                data["dialogue"] = data["transcript"]
            else:
                data["dialogue"] = []

        return data

    def _parse_text_file(self, file_path):
        """Parse a plain text file containing a therapy session transcript.

        Args:
            file_path (str): Path to the text file

        Returns:
            dict: Parsed transcript with content and metadata
        """
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()

        # Parse dialogue segments (basic speaker:text format)
        dialogue = []
        current_speaker = None
        current_text = []

        for line in content.split("\n"):
            # Speaker identification (e.g. "Therapist:", "Client:", etc.)
            speaker_match = re.match(r"^([\w\s]+):\s*(.*)", line)

            if speaker_match:
                # If we have accumulated text for a previous speaker, add it
                if current_speaker and current_text:
                    dialogue.append(
                        {"speaker": current_speaker, "text": " ".join(current_text)}
                    )

                # Start new speaker segment
                current_speaker = speaker_match.group(1)
                current_text = (
                    [speaker_match.group(2)] if speaker_match.group(2) else []
                )
            elif line.strip() and current_speaker:
                # Continue with current speaker
                current_text.append(line)

        # Add the last segment
        if current_speaker and current_text:
            dialogue.append(
                {"speaker": current_speaker, "text": " ".join(current_text)}
            )

        return {"metadata": {}, "dialogue": dialogue, "raw_content": content}

    def _save_to_processed(self, transcripts):
        """Save loaded transcripts to processed data directory.

        Args:
            transcripts (list): List of transcript dictionaries
        """
        # Save as processed data for faster loading
        try:
            # Convert to DataFrame
            df_rows = []
            for t in transcripts:
                # Create a row for each dialogue entry
                for d in t.get("dialogue", []):
                    row = {
                        "file_name": t.get("file_name"),
                        "session_date": t.get("session_date"),
                        "session_number": t.get("session_number"),
                        "speaker": d.get("speaker"),
                        "text": d.get("text"),
                    }

                    # Add metadata fields
                    for k, v in t.get("metadata", {}).items():
                        row[f"metadata_{k}"] = v

                    df_rows.append(row)

            # Create DataFrame
            df = pd.DataFrame(df_rows)

            # Save to CSV
            processed_file = os.path.join(
                self.processed_data_dir, "processed_transcripts.csv"
            )
            df.to_csv(processed_file, index=False)
            self.logger.info(f"Saved processed transcripts to {processed_file}")

            # Save raw data as JSON
            raw_file = os.path.join(self.processed_data_dir, "raw_transcripts.json")
            with open(raw_file, "w", encoding="utf-8") as f:
                # Convert datetime objects to strings for JSON serialization
                processed_transcripts = []
                for t in transcripts:
                    t_copy = t.copy()
                    if t_copy.get("session_date"):
                        t_copy["session_date"] = t_copy["session_date"].strftime(
                            "%Y-%m-%d"
                        )
                    processed_transcripts.append(t_copy)

                json.dump(processed_transcripts, f, indent=2)

            self.logger.info(f"Saved raw transcripts to {raw_file}")

        except Exception as e:
            self.logger.error(f"Error saving processed data: {str(e)}")
