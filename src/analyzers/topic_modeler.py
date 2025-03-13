#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Topic Modeler Module

This module implements topic modeling using Latent Dirichlet Allocation (LDA)
to discover underlying themes in therapy session transcripts.
"""

import logging
import pandas as pd
import numpy as np
from collections import defaultdict
from gensim import corpora, models
from gensim.models.coherencemodel import CoherenceModel


class TopicModeler:
    """Class for topic modeling of therapy session transcripts."""

    def __init__(self, config):
        """Initialize the topic modeler with configuration.

        Args:
            config (dict): Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger("therapy_pipeline.topic_modeler")

        # Extract topic modeling parameters from config
        self.algorithm = config["topic_modeling"].get("algorithm", "lda").lower()
        self.num_topics = config["topic_modeling"].get("num_topics", 5)
        self.passes = config["topic_modeling"].get("passes", 10)
        self.min_df = config["topic_modeling"].get("min_df", 5)
        self.max_df = config["topic_modeling"].get("max_df", 0.9)

        self.logger.info(
            f"Initialized topic modeler with algorithm: {self.algorithm}, num_topics: {self.num_topics}"
        )

    def model_topics(self, transcripts):
        """Perform topic modeling on the transcripts.

        Args:
            transcripts (list): List of processed transcript dictionaries

        Returns:
            dict: Dictionary containing topic modeling results
        """
        self.logger.info("Starting topic modeling")

        # Extract tokens from all transcripts
        documents = []
        document_metadata = []

        for transcript_idx, transcript in enumerate(transcripts):
            # Add combined tokens from the entire transcript
            if "combined_tokens" in transcript and transcript["combined_tokens"]:
                documents.append(transcript["combined_tokens"])
                document_metadata.append(
                    {
                        "transcript_idx": transcript_idx,
                        "file_name": transcript.get("file_name"),
                        "session_date": transcript.get("session_date"),
                        "session_number": transcript.get("session_number"),
                        "type": "transcript",
                    }
                )

            # Add tokens from individual dialogue entries
            if "processed_dialogue" in transcript:
                for i, entry in enumerate(transcript["processed_dialogue"]):
                    if "tokens" in entry and entry["tokens"]:
                        documents.append(entry["tokens"])
                        document_metadata.append(
                            {
                                "transcript_idx": transcript_idx,
                                "dialogue_idx": i,
                                "file_name": transcript.get("file_name"),
                                "session_date": transcript.get("session_date"),
                                "session_number": transcript.get("session_number"),
                                "speaker": entry.get("speaker"),
                                "type": "dialogue",
                            }
                        )

        if not documents:
            self.logger.warning("No documents found for topic modeling")
            return {"topics": [], "document_topics": [], "topic_terms": []}

        # Create dictionary
        self.logger.info(f"Creating dictionary from {len(documents)} documents")
        dictionary = corpora.Dictionary(documents)

        # Filter extremes (remove very rare and very common words)
        dictionary.filter_extremes(no_below=self.min_df, no_above=self.max_df)

        if len(dictionary) < 10:
            self.logger.warning(
                f"Dictionary contains only {len(dictionary)} terms after filtering. Consider adjusting min_df/max_df parameters."
            )
            if len(dictionary) == 0:
                return {"topics": [], "document_topics": [], "topic_terms": []}

        # Convert to bag-of-words
        corpus = [dictionary.doc2bow(doc) for doc in documents]

        # Build LDA model
        if self.algorithm == "lda":
            self.logger.info(f"Building LDA model with {self.num_topics} topics")
            lda_model = models.LdaModel(
                corpus=corpus,
                id2word=dictionary,
                num_topics=self.num_topics,
                passes=self.passes,
                alpha="auto",
                per_word_topics=True,
            )
            model = lda_model
        else:
            self.logger.warning(
                f"Unsupported topic modeling algorithm: {self.algorithm}. Using LDA instead."
            )
            lda_model = models.LdaModel(
                corpus=corpus,
                id2word=dictionary,
                num_topics=self.num_topics,
                passes=self.passes,
                alpha="auto",
                per_word_topics=True,
            )
            model = lda_model

        # Calculate model coherence
        try:
            coherence_model = CoherenceModel(
                model=model, texts=documents, dictionary=dictionary, coherence="c_v"
            )
            coherence = coherence_model.get_coherence()
            self.logger.info(f"Model coherence: {coherence}")
        except Exception as e:
            self.logger.error(f"Error calculating model coherence: {str(e)}")
            coherence = None

        # Extract topics
        topics = []
        for topic_id in range(model.num_topics):
            topic_terms = model.show_topic(topic_id, topn=20)
            topics.append({"topic_id": topic_id, "terms": topic_terms})

        # Get document-topic distributions
        document_topics = []
        for i, (doc, meta) in enumerate(zip(corpus, document_metadata)):
            # Get topic distribution for this document
            doc_topics = model.get_document_topics(doc)

            # Sort by probability
            doc_topics = sorted(doc_topics, key=lambda x: x[1], reverse=True)

            # Add to results with metadata
            entry = {
                "doc_id": i,
                "topic_distribution": doc_topics,
                "dominant_topic": doc_topics[0][0] if doc_topics else None,
                "dominant_topic_prob": doc_topics[0][1] if doc_topics else 0,
            }
            entry.update(meta)
            document_topics.append(entry)

        # Calculate additional topic insights
        topic_insights = self._calculate_topic_insights(document_topics, transcripts)

        # Prepare results
        results = {
            "topics": topics,
            "document_topics": document_topics,
            "coherence": coherence,
            "topic_insights": topic_insights,
            "dictionary": dictionary,
            "corpus": corpus,
            "model": model,
        }

        # Convert to DataFrames for easier analysis
        results["topics_df"] = self._create_topics_dataframe(topics)
        results["document_topics_df"] = pd.DataFrame(document_topics)
        results["topic_insights_df"] = pd.DataFrame(topic_insights)

        self.logger.info("Completed topic modeling")
        return results

    def _create_topics_dataframe(self, topics):
        """Convert topics to a DataFrame with term weights.

        Args:
            topics (list): List of topic dictionaries

        Returns:
            pandas.DataFrame: DataFrame with topics and term weights
        """
        rows = []
        for topic in topics:
            topic_id = topic["topic_id"]
            for term, weight in topic["terms"]:
                rows.append({"topic_id": topic_id, "term": term, "weight": weight})

        return pd.DataFrame(rows)

    def _calculate_topic_insights(self, document_topics, transcripts):
        """Calculate additional insights about topics.

        Args:
            document_topics (list): List of document-topic assignments
            transcripts (list): List of transcript dictionaries

        Returns:
            list: List of topic insight dictionaries
        """
        # Group document topics by transcript
        transcript_topics = defaultdict(list)
        for doc in document_topics:
            if doc["type"] == "dialogue":
                transcript_idx = doc["transcript_idx"]
                transcript_topics[transcript_idx].append(doc)

        # Calculate topic prevalence over time
        topic_prevalence = defaultdict(list)

        for transcript_idx, docs in sorted(transcript_topics.items()):
            # Get transcript metadata
            transcript = (
                transcripts[transcript_idx] if transcript_idx < len(transcripts) else {}
            )

            # Count topic occurrences
            topic_counts = defaultdict(float)
            for doc in docs:
                for topic_id, prob in doc["topic_distribution"]:
                    topic_counts[topic_id] += prob

            # Normalize by number of documents
            for topic_id, count in topic_counts.items():
                topic_prevalence[topic_id].append(
                    {
                        "transcript_idx": transcript_idx,
                        "file_name": transcript.get("file_name"),
                        "session_date": transcript.get("session_date"),
                        "session_number": transcript.get("session_number"),
                        "topic_id": topic_id,
                        "prevalence": count / len(docs) if docs else 0,
                    }
                )

        # Flatten results
        results = []
        for topic_id, prevalences in topic_prevalence.items():
            for prevalence in prevalences:
                results.append(prevalence)

        return results
