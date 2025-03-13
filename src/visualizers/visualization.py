#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Visualization Module

This module generates visualizations from therapy session analysis results,
including sentiment trends, topic distributions, and other insights.
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime
from matplotlib.ticker import MaxNLocator


class Visualizer:
    """Class for generating visualizations from therapy session analysis."""

    def __init__(self, config):
        """Initialize the visualizer with configuration.

        Args:
            config (dict): Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger("therapy_pipeline.visualizer")

        # Extract visualization parameters from config
        self.fig_width = config["visualization"].get("fig_width", 12)
        self.fig_height = config["visualization"].get("fig_height", 8)
        self.dpi = config["visualization"].get("dpi", 100)
        self.style = config["visualization"].get("style", "seaborn-v0_8-whitegrid")
        self.palette = config["visualization"].get("color_palette", "viridis")
        self.date_format = config["visualization"].get("date_format", "%Y-%m-%d")

        # Set up visualization style
        plt.style.use(self.style)

        # Create reports directory if it doesn't exist
        self.reports_dir = config["reports"].get("report_dir", "reports")
        self.viz_dir = os.path.join(self.reports_dir, "visualizations")
        os.makedirs(self.viz_dir, exist_ok=True)

        self.logger.info(f"Initialized visualizer with style: {self.style}")

    def create_visualizations(
        self, transcripts, processed_data, sentiment_results, topic_results
    ):
        """Create visualizations from analysis results.

        Args:
            transcripts (list): Original transcript dictionaries
            processed_data (list): Processed transcript dictionaries
            sentiment_results (dict): Sentiment analysis results
            topic_results (dict): Topic modeling results

        Returns:
            dict: Dictionary of generated visualization file paths
        """
        self.logger.info("Creating visualizations")

        visualization_paths = {}

        # Make sure we have data to visualize
        if not transcripts or not processed_data:
            self.logger.warning("No data available for visualization")
            return visualization_paths

        # 1. Sentiment Visualizations
        if sentiment_results:
            sentiment_paths = self._create_sentiment_visualizations(sentiment_results)
            visualization_paths.update(sentiment_paths)

        # 2. Topic Visualizations
        if topic_results:
            topic_paths = self._create_topic_visualizations(topic_results)
            visualization_paths.update(topic_paths)

        # 3. Integrated Visualizations
        if sentiment_results and topic_results:
            integrated_paths = self._create_integrated_visualizations(
                sentiment_results, topic_results, transcripts
            )
            visualization_paths.update(integrated_paths)

        self.logger.info(f"Created {len(visualization_paths)} visualizations")
        return visualization_paths

    def _create_sentiment_visualizations(self, sentiment_results):
        """Create visualizations from sentiment analysis results.

        Args:
            sentiment_results (dict): Sentiment analysis results

        Returns:
            dict: Dictionary of generated visualization file paths
        """
        self.logger.info("Creating sentiment visualizations")
        paths = {}

        # 1. Sentiment over time
        try:
            if "transcript_sentiments_df" in sentiment_results:
                df = sentiment_results["transcript_sentiments_df"]

                if not df.empty and "session_date" in df.columns:
                    # Filter out rows with missing dates
                    df = df[df["session_date"].notna()].copy()

                    if not df.empty:
                        # Sort by date
                        df.sort_values("session_date", inplace=True)

                        # Create figure
                        fig, ax = plt.subplots(
                            figsize=(self.fig_width, self.fig_height), dpi=self.dpi
                        )

                        # Plot sentiment over time
                        ax.plot(
                            df["session_date"],
                            df["avg_compound"],
                            marker="o",
                            linestyle="-",
                            linewidth=2,
                            markersize=8,
                        )

                        # Add horizontal lines for sentiment thresholds
                        ax.axhline(
                            y=0.05,
                            color="green",
                            linestyle="--",
                            alpha=0.5,
                            label="Positive Threshold",
                        )
                        ax.axhline(
                            y=-0.05,
                            color="red",
                            linestyle="--",
                            alpha=0.5,
                            label="Negative Threshold",
                        )
                        ax.axhline(y=0, color="gray", linestyle="-", alpha=0.2)

                        # Format date axis
                        ax.xaxis.set_major_formatter(
                            mdates.DateFormatter(self.date_format)
                        )
                        plt.xticks(rotation=45)

                        # Add labels and title
                        ax.set_xlabel("Session Date")
                        ax.set_ylabel("Sentiment Score (Compound)")
                        ax.set_title("Sentiment Progression Across Therapy Sessions")

                        # Set y-axis limits with some padding
                        y_min = min(df["avg_compound"].min(), -0.1)
                        y_max = max(df["avg_compound"].max(), 0.1)
                        padding = (y_max - y_min) * 0.1
                        ax.set_ylim(y_min - padding, y_max + padding)

                        # Add legend
                        ax.legend()

                        # Add grid
                        ax.grid(True, alpha=0.3)

                        # Tight layout
                        plt.tight_layout()

                        # Save figure
                        path = os.path.join(self.viz_dir, "sentiment_over_time.png")
                        plt.savefig(path)
                        plt.close()

                        paths["sentiment_over_time"] = path
                        self.logger.info(
                            f"Created sentiment over time visualization: {path}"
                        )
        except Exception as e:
            self.logger.error(
                f"Error creating sentiment over time visualization: {str(e)}"
            )

        # 2. Sentiment distribution
        try:
            if "dialogue_sentiments_df" in sentiment_results:
                df = sentiment_results["dialogue_sentiments_df"]

                if not df.empty:
                    # Create figure
                    fig, ax = plt.subplots(
                        figsize=(self.fig_width, self.fig_height), dpi=self.dpi
                    )

                    # Create histogram
                    sns.histplot(df["compound_score"], bins=30, kde=True, ax=ax)

                    # Add vertical lines for sentiment thresholds
                    ax.axvline(
                        x=0.05,
                        color="green",
                        linestyle="--",
                        alpha=0.5,
                        label="Positive Threshold",
                    )
                    ax.axvline(
                        x=-0.05,
                        color="red",
                        linestyle="--",
                        alpha=0.5,
                        label="Negative Threshold",
                    )
                    ax.axvline(x=0, color="gray", linestyle="-", alpha=0.2)

                    # Add labels and title
                    ax.set_xlabel("Sentiment Score (Compound)")
                    ax.set_ylabel("Frequency")
                    ax.set_title("Distribution of Sentiment Scores Across All Dialogue")

                    # Add legend
                    ax.legend()

                    # Tight layout
                    plt.tight_layout()

                    # Save figure
                    path = os.path.join(self.viz_dir, "sentiment_distribution.png")
                    plt.savefig(path)
                    plt.close()

                    paths["sentiment_distribution"] = path
                    self.logger.info(
                        f"Created sentiment distribution visualization: {path}"
                    )
        except Exception as e:
            self.logger.error(
                f"Error creating sentiment distribution visualization: {str(e)}"
            )

        # 3. Sentiment by speaker
        try:
            if "speaker_sentiments_df" in sentiment_results:
                df = sentiment_results["speaker_sentiments_df"]

                if not df.empty and "speaker" in df.columns:
                    # Create figure
                    fig, ax = plt.subplots(
                        figsize=(self.fig_width, self.fig_height), dpi=self.dpi
                    )

                    # Create horizontal bar chart
                    colors = []
                    for score in df["avg_compound"]:
                        if score >= 0.05:
                            colors.append("green")
                        elif score <= -0.05:
                            colors.append("red")
                        else:
                            colors.append("gray")

                    # Sort by compound score
                    df = df.sort_values("avg_compound")

                    # Plot bars
                    bars = ax.barh(df["speaker"], df["avg_compound"], color=colors)

                    # Add labels and title
                    ax.set_xlabel("Average Sentiment Score (Compound)")
                    ax.set_ylabel("Speaker")
                    ax.set_title("Average Sentiment by Speaker")

                    # Add vertical lines for sentiment thresholds
                    ax.axvline(
                        x=0.05,
                        color="green",
                        linestyle="--",
                        alpha=0.5,
                        label="Positive Threshold",
                    )
                    ax.axvline(
                        x=-0.05,
                        color="red",
                        linestyle="--",
                        alpha=0.5,
                        label="Negative Threshold",
                    )
                    ax.axvline(x=0, color="gray", linestyle="-", alpha=0.2)

                    # Add values to the end of each bar
                    for bar in bars:
                        width = bar.get_width()
                        label_x_pos = width if width >= 0 else width - 0.05
                        ax.text(
                            label_x_pos,
                            bar.get_y() + bar.get_height() / 2,
                            f"{width:.2f}",
                            va="center",
                            ha="left" if width >= 0 else "right",
                        )

                    # Add legend
                    ax.legend()

                    # Tight layout
                    plt.tight_layout()

                    # Save figure
                    path = os.path.join(self.viz_dir, "sentiment_by_speaker.png")
                    plt.savefig(path)
                    plt.close()

                    paths["sentiment_by_speaker"] = path
                    self.logger.info(
                        f"Created sentiment by speaker visualization: {path}"
                    )
        except Exception as e:
            self.logger.error(
                f"Error creating sentiment by speaker visualization: {str(e)}"
            )

        # 4. Sentiment progression within sessions
        try:
            if "session_progression_df" in sentiment_results:
                df = sentiment_results["session_progression_df"]

                if not df.empty and "session_date" in df.columns:
                    # Filter out rows with missing dates
                    df = df[df["session_date"].notna()].copy()

                    if not df.empty:
                        # Sort by date
                        df.sort_values("session_date", inplace=True)

                        # Create figure
                        fig, ax = plt.subplots(
                            figsize=(self.fig_width, self.fig_height), dpi=self.dpi
                        )

                        # Plot start and end sentiment
                        ax.plot(
                            df["session_date"],
                            df["start_sentiment"],
                            marker="o",
                            linestyle="-",
                            label="Start of Session",
                            color="blue",
                            alpha=0.7,
                        )
                        ax.plot(
                            df["session_date"],
                            df["end_sentiment"],
                            marker="s",
                            linestyle="-",
                            label="End of Session",
                            color="orange",
                            alpha=0.7,
                        )

                        # Add arrows to show sentiment shift
                        for i, row in df.iterrows():
                            plt.arrow(
                                row["session_date"],
                                row["start_sentiment"],
                                0,
                                row["sentiment_shift"],
                                color="green" if row["sentiment_shift"] > 0 else "red",
                                width=0.3,
                                head_width=2,
                                head_length=0.02,
                                length_includes_head=True,
                                alpha=0.4,
                            )

                        # Format date axis
                        ax.xaxis.set_major_formatter(
                            mdates.DateFormatter(self.date_format)
                        )
                        plt.xticks(rotation=45)

                        # Add labels and title
                        ax.set_xlabel("Session Date")
                        ax.set_ylabel("Sentiment Score")
                        ax.set_title("Sentiment Change Within Sessions")

                        # Add horizontal line at neutral sentiment
                        ax.axhline(y=0, color="gray", linestyle="-", alpha=0.2)

                        # Add legend
                        ax.legend()

                        # Add grid
                        ax.grid(True, alpha=0.3)

                        # Tight layout
                        plt.tight_layout()

                        # Save figure
                        path = os.path.join(
                            self.viz_dir, "sentiment_within_sessions.png"
                        )
                        plt.savefig(path)
                        plt.close()

                        paths["sentiment_within_sessions"] = path
                        self.logger.info(
                            f"Created sentiment within sessions visualization: {path}"
                        )
        except Exception as e:
            self.logger.error(
                f"Error creating sentiment within sessions visualization: {str(e)}"
            )

        return paths

    def _create_topic_visualizations(self, topic_results):
        """Create visualizations from topic modeling results.

        Args:
            topic_results (dict): Topic modeling results

        Returns:
            dict: Dictionary of generated visualization file paths
        """
        self.logger.info("Creating topic visualizations")
        paths = {}

        # 1. Topic word cloud for each topic
        try:
            if "topics_df" in topic_results:
                df = topic_results["topics_df"]

                if not df.empty and "topic_id" in df.columns:
                    # Create word clouds for top N topics (limit to 6 for space)
                    top_topics = sorted(df["topic_id"].unique())[:6]

                    for topic_id in top_topics:
                        # Filter for this topic
                        topic_df = df[df["topic_id"] == topic_id]

                        # Sort by weight
                        topic_df = topic_df.sort_values("weight", ascending=False)

                        # Take top terms
                        top_terms = topic_df.head(15)

                        # Create horizontal bar chart
                        fig, ax = plt.subplots(
                            figsize=(self.fig_width, self.fig_height / 2), dpi=self.dpi
                        )

                        # Plot bars
                        bars = ax.barh(
                            top_terms["term"],
                            top_terms["weight"],
                            color=plt.cm.get_cmap(self.palette)(
                                topic_id / len(top_topics)
                            ),
                        )

                        # Add labels and title
                        ax.set_xlabel("Weight")
                        ax.set_ylabel("Term")
                        ax.set_title(f"Top Terms in Topic {topic_id}")

                        # Add grid
                        ax.grid(True, alpha=0.3)

                        # Add values to the end of each bar
                        for bar in bars:
                            width = bar.get_width()
                            ax.text(
                                width + 0.01,
                                bar.get_y() + bar.get_height() / 2,
                                f"{width:.3f}",
                                va="center",
                                ha="left",
                            )

                        # Tight layout
                        plt.tight_layout()

                        # Save figure
                        path = os.path.join(self.viz_dir, f"topic_{topic_id}_terms.png")
                        plt.savefig(path)
                        plt.close()

                        paths[f"topic_{topic_id}_terms"] = path

                    self.logger.info(
                        f"Created topic term visualizations for {len(top_topics)} topics"
                    )
        except Exception as e:
            self.logger.error(f"Error creating topic term visualizations: {str(e)}")

        # 2. Topic prevalence over time
        try:
            if "topic_insights_df" in topic_results:
                df = topic_results["topic_insights_df"]

                if (
                    not df.empty
                    and "session_date" in df.columns
                    and "topic_id" in df.columns
                ):
                    # Filter out rows with missing dates
                    df = df[df["session_date"].notna()].copy()

                    if not df.empty:
                        # Pivot to get a column for each topic
                        pivot_df = df.pivot_table(
                            index="session_date",
                            columns="topic_id",
                            values="prevalence",
                            aggfunc="mean",
                        ).reset_index()

                        # Sort by date
                        pivot_df.sort_values("session_date", inplace=True)

                        # Create figure
                        fig, ax = plt.subplots(
                            figsize=(self.fig_width, self.fig_height), dpi=self.dpi
                        )

                        # Get topic columns
                        topic_cols = [
                            col for col in pivot_df.columns if col != "session_date"
                        ]

                        # Plot lines for each topic
                        for topic_id in topic_cols:
                            ax.plot(
                                pivot_df["session_date"],
                                pivot_df[topic_id],
                                marker="o",
                                linestyle="-",
                                label=f"Topic {topic_id}",
                            )

                        # Format date axis
                        ax.xaxis.set_major_formatter(
                            mdates.DateFormatter(self.date_format)
                        )
                        plt.xticks(rotation=45)

                        # Add labels and title
                        ax.set_xlabel("Session Date")
                        ax.set_ylabel("Topic Prevalence")
                        ax.set_title("Topic Prevalence Over Time")

                        # Add legend
                        ax.legend()

                        # Add grid
                        ax.grid(True, alpha=0.3)

                        # Tight layout
                        plt.tight_layout()

                        # Save figure
                        path = os.path.join(
                            self.viz_dir, "topic_prevalence_over_time.png"
                        )
                        plt.savefig(path)
                        plt.close()

                        paths["topic_prevalence_over_time"] = path
                        self.logger.info(
                            f"Created topic prevalence over time visualization: {path}"
                        )
        except Exception as e:
            self.logger.error(
                f"Error creating topic prevalence over time visualization: {str(e)}"
            )

        # 3. Topic distribution across sessions
        try:
            if "document_topics_df" in topic_results:
                df = topic_results["document_topics_df"]

                if (
                    not df.empty
                    and "transcript_idx" in df.columns
                    and "dominant_topic" in df.columns
                ):
                    # Filter for transcript-level documents
                    df = df[df["type"] == "transcript"].copy()

                    if not df.empty:
                        # Create figure
                        fig, ax = plt.subplots(
                            figsize=(self.fig_width, self.fig_height), dpi=self.dpi
                        )

                        # Count dominant topics
                        topic_counts = df["dominant_topic"].value_counts().sort_index()

                        # Plot bars
                        bars = ax.bar(
                            [f"Topic {i}" for i in topic_counts.index],
                            topic_counts.values,
                            color=[
                                plt.cm.get_cmap(self.palette)(i / len(topic_counts))
                                for i in range(len(topic_counts))
                            ],
                        )

                        # Add labels and title
                        ax.set_xlabel("Topic")
                        ax.set_ylabel("Number of Sessions")
                        ax.set_title("Dominant Topic Distribution Across Sessions")

                        # Add grid
                        ax.grid(True, alpha=0.3, axis="y")

                        # Add values above bars
                        for bar in bars:
                            height = bar.get_height()
                            ax.text(
                                bar.get_x() + bar.get_width() / 2.0,
                                height + 0.1,
                                f"{height:.0f}",
                                ha="center",
                                va="bottom",
                            )

                        # Tight layout
                        plt.tight_layout()

                        # Save figure
                        path = os.path.join(self.viz_dir, "topic_distribution.png")
                        plt.savefig(path)
                        plt.close()

                        paths["topic_distribution"] = path
                        self.logger.info(
                            f"Created topic distribution visualization: {path}"
                        )
        except Exception as e:
            self.logger.error(
                f"Error creating topic distribution visualization: {str(e)}"
            )

        return paths

    def _create_integrated_visualizations(
        self, sentiment_results, topic_results, transcripts
    ):
        """Create visualizations that integrate sentiment and topic analysis.

        Args:
            sentiment_results (dict): Sentiment analysis results
            topic_results (dict): Topic modeling results
            transcripts (list): Original transcript dictionaries

        Returns:
            dict: Dictionary of generated visualization file paths
        """
        self.logger.info("Creating integrated visualizations")
        paths = {}

        # 1. Sentiment by topic
        try:
            if (
                "document_topics_df" in topic_results
                and "sentiment_results" in sentiment_results
            ):

                # Get document topics
                doc_topics_df = topic_results["document_topics_df"]

                # Get sentiment results
                sent_df = sentiment_results["dialogue_sentiments_df"]

                if not doc_topics_df.empty and not sent_df.empty:
                    # Filter for dialogue-level documents
                    doc_topics_df = doc_topics_df[
                        doc_topics_df["type"] == "dialogue"
                    ].copy()

                    # Merge sentiment and topic data
                    merged_df = pd.merge(
                        doc_topics_df,
                        sent_df,
                        on=["transcript_idx", "dialogue_idx"],
                        how="inner",
                    )

                    if not merged_df.empty:
                        # Create figure
                        fig, ax = plt.subplots(
                            figsize=(self.fig_width, self.fig_height), dpi=self.dpi
                        )

                        # Group by dominant topic and calculate average sentiment
                        topic_sentiment = merged_df.groupby("dominant_topic")[
                            "compound_score"
                        ].mean()

                        # Sort by topic index
                        topic_sentiment = topic_sentiment.sort_index()

                        # Plot bars
                        bars = ax.bar(
                            [f"Topic {i}" for i in topic_sentiment.index],
                            topic_sentiment.values,
                            color=[
                                (
                                    "green"
                                    if score >= 0.05
                                    else "red" if score <= -0.05 else "gray"
                                )
                                for score in topic_sentiment.values
                            ],
                        )

                        # Add labels and title
                        ax.set_xlabel("Topic")
                        ax.set_ylabel("Average Sentiment Score")
                        ax.set_title("Average Sentiment by Topic")

                        # Add horizontal line at neutral sentiment
                        ax.axhline(y=0, color="gray", linestyle="-", alpha=0.2)

                        # Add horizontal lines for sentiment thresholds
                        ax.axhline(
                            y=0.05,
                            color="green",
                            linestyle="--",
                            alpha=0.5,
                            label="Positive Threshold",
                        )
                        ax.axhline(
                            y=-0.05,
                            color="red",
                            linestyle="--",
                            alpha=0.5,
                            label="Negative Threshold",
                        )

                        # Add grid
                        ax.grid(True, alpha=0.3, axis="y")

                        # Add values above/below bars
                        for bar in bars:
                            height = bar.get_height()
                            va = "bottom" if height >= 0 else "top"
                            offset = 0.02 if height >= 0 else -0.02
                            ax.text(
                                bar.get_x() + bar.get_width() / 2.0,
                                height + offset,
                                f"{height:.2f}",
                                ha="center",
                                va=va,
                            )

                        # Add legend
                        ax.legend()

                        # Tight layout
                        plt.tight_layout()

                        # Save figure
                        path = os.path.join(self.viz_dir, "sentiment_by_topic.png")
                        plt.savefig(path)
                        plt.close()

                        paths["sentiment_by_topic"] = path
                        self.logger.info(
                            f"Created sentiment by topic visualization: {path}"
                        )
        except Exception as e:
            self.logger.error(
                f"Error creating sentiment by topic visualization: {str(e)}"
            )

        # 2. Topics and sentiment progression
        try:
            if (
                "topic_insights_df" in topic_results
                and "transcript_sentiments_df" in sentiment_results
            ):

                # Get topic insights
                topic_df = topic_results["topic_insights_df"]

                # Get transcript sentiments
                sent_df = sentiment_results["transcript_sentiments_df"]

                if (
                    not topic_df.empty
                    and not sent_df.empty
                    and "session_date" in topic_df.columns
                ):
                    # Filter out rows with missing dates
                    topic_df = topic_df[topic_df["session_date"].notna()].copy()
                    sent_df = sent_df[sent_df["session_date"].notna()].copy()

                    if not topic_df.empty and not sent_df.empty:
                        # Pivot to get a column for each topic
                        pivot_df = topic_df.pivot_table(
                            index="session_date",
                            columns="topic_id",
                            values="prevalence",
                            aggfunc="mean",
                        ).reset_index()

                        # Merge with sentiment data
                        merged_df = pd.merge(
                            pivot_df,
                            sent_df[["session_date", "avg_compound"]],
                            on="session_date",
                            how="inner",
                        )

                        # Sort by date
                        merged_df.sort_values("session_date", inplace=True)

                        if not merged_df.empty:
                            # Create figure with two subplots sharing x-axis
                            fig, (ax1, ax2) = plt.subplots(
                                2,
                                1,
                                figsize=(self.fig_width, self.fig_height * 1.2),
                                dpi=self.dpi,
                                sharex=True,
                                gridspec_kw={"height_ratios": [2, 1]},
                            )

                            # Plot topic prevalence on top subplot
                            topic_cols = [
                                col
                                for col in merged_df.columns
                                if isinstance(col, int)
                                or (isinstance(col, str) and col.isdigit())
                            ]

                            for topic_id in topic_cols:
                                ax1.plot(
                                    merged_df["session_date"],
                                    merged_df[topic_id],
                                    marker="o",
                                    linestyle="-",
                                    label=f"Topic {topic_id}",
                                )

                            # Add labels and title for top subplot
                            ax1.set_ylabel("Topic Prevalence")
                            ax1.set_title("Topic Prevalence and Sentiment Over Time")
                            ax1.legend()
                            ax1.grid(True, alpha=0.3)

                            # Plot sentiment on bottom subplot
                            ax2.plot(
                                merged_df["session_date"],
                                merged_df["avg_compound"],
                                marker="s",
                                linestyle="-",
                                color="purple",
                                label="Sentiment",
                            )

                            # Add horizontal lines for sentiment thresholds
                            ax2.axhline(
                                y=0.05,
                                color="green",
                                linestyle="--",
                                alpha=0.5,
                                label="Positive Threshold",
                            )
                            ax2.axhline(
                                y=-0.05,
                                color="red",
                                linestyle="--",
                                alpha=0.5,
                                label="Negative Threshold",
                            )
                            ax2.axhline(y=0, color="gray", linestyle="-", alpha=0.2)

                            # Add labels for bottom subplot
                            ax2.set_xlabel("Session Date")
                            ax2.set_ylabel("Sentiment Score")
                            ax2.legend()
                            ax2.grid(True, alpha=0.3)

                            # Format date axis
                            ax2.xaxis.set_major_formatter(
                                mdates.DateFormatter(self.date_format)
                            )
                            plt.xticks(rotation=45)

                            # Tight layout
                            plt.tight_layout()

                            # Save figure
                            path = os.path.join(
                                self.viz_dir, "topics_and_sentiment_over_time.png"
                            )
                            plt.savefig(path)
                            plt.close()

                            paths["topics_and_sentiment_over_time"] = path
                            self.logger.info(
                                f"Created topics and sentiment over time visualization: {path}"
                            )
        except Exception as e:
            self.logger.error(
                f"Error creating topics and sentiment over time visualization: {str(e)}"
            )

        return paths
