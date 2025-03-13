#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Report Generator Module

This module generates summary reports and actionable insights from therapy session analysis results.
"""

import os
import logging
from datetime import datetime
import pandas as pd


class ReportGenerator:
    """Class for generating reports from therapy session analysis."""

    def __init__(self, config):
        """Initialize the report generator with configuration.

        Args:
            config (dict): Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger("therapy_pipeline.report_generator")

        # Extract report parameters from config
        self.summary_length = config["reports"].get("summary_length", 3)
        self.highlight_count = config["reports"].get("highlight_count", 5)
        self.save_format = config["reports"].get("save_format", "md")

        # Create reports directory if it doesn't exist
        self.reports_dir = config["reports"].get("report_dir", "reports")
        os.makedirs(self.reports_dir, exist_ok=True)

        self.logger.info(
            f"Initialized report generator with format: {self.save_format}"
        )

    def generate_reports(
        self,
        transcripts,
        processed_data,
        sentiment_results,
        topic_results,
        llm_insights=None,
    ):
        """Generate reports from analysis results.

        Args:
            transcripts (list): Original transcript dictionaries
            processed_data (list): Processed transcript dictionaries
            sentiment_results (dict): Sentiment analysis results
            topic_results (dict): Topic modeling results
            llm_insights (dict, optional): LLM-generated insights

        Returns:
            dict: Dictionary of generated report file paths and content
        """
        self.logger.info("Generating reports")

        reports = {"session_reports": [], "overall_report": {}, "report_paths": {}}

        # Make sure we have data to report on
        if not transcripts or not processed_data:
            self.logger.warning("No data available for report generation")
            return reports

        # 1. Generate individual session reports
        for i, (transcript, processed) in enumerate(zip(transcripts, processed_data)):
            try:
                session_report = self._generate_session_report(
                    transcript,
                    processed,
                    sentiment_results,
                    topic_results,
                    index=i,
                    llm_insights=llm_insights,
                )
                reports["session_reports"].append(session_report)

                # Save the report
                report_path = self._save_report(
                    session_report["content"],
                    f"session_{session_report['session_number'] or i + 1}_report",
                )
                reports["report_paths"][
                    f"session_{session_report['session_number'] or i + 1}"
                ] = report_path
            except Exception as e:
                self.logger.error(f"Error generating report for session {i}: {str(e)}")

        # 2. Generate overall progress report
        try:
            overall_report = self._generate_overall_report(
                transcripts,
                processed_data,
                sentiment_results,
                topic_results,
                reports["session_reports"],
                llm_insights=llm_insights,
            )
            reports["overall_report"] = overall_report

            # Save the report
            report_path = self._save_report(
                overall_report["content"], "overall_progress_report"
            )
            reports["report_paths"]["overall"] = report_path
        except Exception as e:
            self.logger.error(f"Error generating overall report: {str(e)}")

        self.logger.info(
            f"Generated {len(reports['session_reports'])} session reports and 1 overall report"
        )
        return reports

    def _generate_session_report(
        self,
        transcript,
        processed_data,
        sentiment_results,
        topic_results,
        index,
        llm_insights=None,
    ):
        """Generate a report for a single therapy session.

        Args:
            transcript (dict): Original transcript dictionary
            processed_data (dict): Processed transcript dictionary
            sentiment_results (dict): Sentiment analysis results
            topic_results (dict): Topic modeling results
            index (int): Index of the transcript in the list
            llm_insights (dict, optional): LLM-generated insights for this transcript

        Returns:
            dict: Session report dictionary
        """
        # Extract session metadata
        session_date = transcript.get("session_date")
        session_number = transcript.get("session_number") or index + 1
        file_name = transcript.get("file_name")

        date_str = session_date.strftime("%Y-%m-%d") if session_date else "Unknown Date"
        title = f"Session {session_number} Report ({date_str})"

        # Extract relevant sentiment data for this session
        sentiment_data = None
        if "transcript_sentiments_df" in sentiment_results:
            df = sentiment_results["transcript_sentiments_df"]
            if not df.empty:
                session_sentiments = df[df["transcript_idx"] == index]
                if not session_sentiments.empty:
                    sentiment_data = session_sentiments.iloc[0].to_dict()

        # Extract relevant topic data for this session
        topic_data = None
        topic_terms = {}
        if "document_topics_df" in topic_results:
            doc_df = topic_results["document_topics_df"]
            if not doc_df.empty:
                session_topics = doc_df[
                    (doc_df["transcript_idx"] == index)
                    & (doc_df["type"] == "transcript")
                ]
                if not session_topics.empty:
                    topic_data = session_topics.iloc[0].to_dict()

                    # Get terms for the dominant topic
                    if "topics_df" in topic_results and "dominant_topic" in topic_data:
                        topics_df = topic_results["topics_df"]
                        if not topics_df.empty:
                            dominant_topic = topic_data["dominant_topic"]
                            if dominant_topic is not None:
                                topic_terms_df = topics_df[
                                    topics_df["topic_id"] == dominant_topic
                                ]
                                if not topic_terms_df.empty:
                                    topic_terms_df = topic_terms_df.sort_values(
                                        "weight", ascending=False
                                    )
                                    topic_terms = topic_terms_df.head(10).to_dict(
                                        "records"
                                    )

        # Generate report content
        content = []

        # Title
        content.append(f"# {title}")
        content.append("")

        # Metadata section
        content.append("## Session Information")
        content.append("")
        content.append(f"- **Date**: {date_str}")
        content.append(f"- **Session Number**: {session_number}")
        content.append(f"- **File**: {file_name}")
        content.append("")

        # Summary section
        content.append("## Summary")
        content.append("")

        if sentiment_data:
            overall_sentiment = sentiment_data.get("overall_sentiment", "neutral")
            sentiment_emoji = (
                "😊"
                if overall_sentiment == "positive"
                else "😐" if overall_sentiment == "neutral" else "😔"
            )

            avg_compound = sentiment_data.get("avg_compound", 0)
            sentiment_description = (
                "very positive"
                if avg_compound > 0.5
                else (
                    "positive"
                    if avg_compound > 0.05
                    else (
                        "slightly positive"
                        if avg_compound > 0
                        else (
                            "neutral"
                            if avg_compound == 0
                            else (
                                "slightly negative"
                                if avg_compound > -0.05
                                else (
                                    "negative"
                                    if avg_compound > -0.5
                                    else "very negative"
                                )
                            )
                        )
                    )
                )
            )

            content.append(
                f"This session had an overall {sentiment_description} tone ({sentiment_emoji} {avg_compound:.2f})."
            )

        if (
            topic_data
            and "dominant_topic" in topic_data
            and topic_data["dominant_topic"] is not None
        ):
            dominant_topic = topic_data["dominant_topic"]
            dominant_topic_prob = topic_data.get("dominant_topic_prob", 0)

            content.append(
                f"The main theme (Topic {dominant_topic}) was present with a strength of {dominant_topic_prob:.2f}."
            )

            if topic_terms:
                term_list = ", ".join([t["term"] for t in topic_terms[:5]])
                content.append(f"Key terms: {term_list}.")

        content.append("")

        # Sentiment analysis section
        content.append("## Emotional Tone Analysis")
        content.append("")

        if sentiment_data:
            # Overall sentiment
            content.append(
                f"- **Overall Emotional Tone**: {sentiment_data.get('overall_sentiment', 'neutral').capitalize()}"
            )
            content.append(
                f"  - Compound Score: {sentiment_data.get('avg_compound', 0):.3f}"
            )
            content.append(
                f"  - Positive Aspects: {sentiment_data.get('avg_positive', 0):.3f}"
            )
            content.append(
                f"  - Negative Aspects: {sentiment_data.get('avg_negative', 0):.3f}"
            )
            content.append(
                f"  - Neutral Aspects: {sentiment_data.get('avg_neutral', 0):.3f}"
            )
            content.append("")

            # Sentiment breakdown
            if "sentiment_counts" in sentiment_data:
                counts = sentiment_data["sentiment_counts"]
                total = sum(counts.values())
                if total > 0:
                    content.append("- **Emotional Tone Breakdown**:")
                    for category, count in counts.items():
                        percentage = (count / total) * 100
                        content.append(
                            f"  - {category.capitalize()}: {count} statements ({percentage:.1f}%)"
                        )
                    content.append("")

            # Session progression
            if "session_progression_df" in sentiment_results:
                prog_df = sentiment_results["session_progression_df"]
                if not prog_df.empty:
                    session_prog = prog_df[prog_df["transcript_idx"] == index]
                    if not session_prog.empty:
                        prog_data = session_prog.iloc[0]

                        start_sentiment = prog_data.get("start_sentiment")
                        end_sentiment = prog_data.get("end_sentiment")
                        sentiment_shift = prog_data.get("sentiment_shift")
                        sentiment_trend = prog_data.get("sentiment_trend")

                        if None not in [
                            start_sentiment,
                            end_sentiment,
                            sentiment_shift,
                            sentiment_trend,
                        ]:
                            content.append("- **Session Progression**:")
                            content.append(f"  - Started: {start_sentiment:.3f}")
                            content.append(f"  - Ended: {end_sentiment:.3f}")
                            content.append(
                                f"  - Shift: {sentiment_shift:.3f} ({sentiment_trend})"
                            )
                            content.append("")
        else:
            content.append("No sentiment analysis data available for this session.")
            content.append("")

        # Topic analysis section
        content.append("## Theme Analysis")
        content.append("")

        if (
            topic_data
            and "dominant_topic" in topic_data
            and topic_data["dominant_topic"] is not None
        ):
            dominant_topic = topic_data["dominant_topic"]
            dominant_topic_prob = topic_data.get("dominant_topic_prob", 0)

            content.append(f"- **Primary Theme**: Topic {dominant_topic}")
            content.append(f"  - Strength: {dominant_topic_prob:.3f}")
            content.append("")

            if topic_terms:
                content.append("- **Key Terms**:")
                for term in topic_terms[:10]:
                    content.append(f"  - {term['term']} ({term['weight']:.3f})")
                content.append("")

            # Other topics if available
            if "topic_distribution" in topic_data and topic_data["topic_distribution"]:
                topic_dist = topic_data["topic_distribution"]
                if len(topic_dist) > 1:  # If there are secondary topics
                    content.append("- **Secondary Themes**:")
                    for topic_id, prob in topic_dist[1:4]:  # Show next 3 topics
                        content.append(f"  - Topic {topic_id}: {prob:.3f}")
                    content.append("")
        else:
            content.append("No topic analysis data available for this session.")
            content.append("")

        # Key insights section
        content.append("## Key Insights")
        content.append("")

        insights = []

        # Add sentiment-based insights
        if sentiment_data:
            overall_sentiment = sentiment_data.get("overall_sentiment", "neutral")

            if overall_sentiment == "positive":
                insights.append(
                    "The session had a positive emotional tone, suggesting constructive engagement."
                )
            elif overall_sentiment == "negative":
                insights.append(
                    "The session had a negative emotional tone, which may indicate challenging content or distress."
                )

            # Session progression insights
            if "session_progression_df" in sentiment_results:
                prog_df = sentiment_results["session_progression_df"]
                if not prog_df.empty:
                    session_prog = prog_df[prog_df["transcript_idx"] == index]
                    if not session_prog.empty:
                        prog_data = session_prog.iloc[0]
                        sentiment_trend = prog_data.get("sentiment_trend")

                        if sentiment_trend == "improving":
                            insights.append(
                                "The emotional tone improved during the session, suggesting progress or relief."
                            )
                        elif sentiment_trend == "deteriorating":
                            insights.append(
                                "The emotional tone declined during the session, which may indicate increasing distress or uncovering difficult topics."
                            )
                        elif sentiment_trend == "stable":
                            insights.append(
                                "The emotional tone remained stable throughout the session."
                            )

        # Add topic-based insights
        if topic_data and topic_terms:
            # Use the top terms to formulate an insight
            top_terms = [t["term"] for t in topic_terms[:5]]
            term_str = ", ".join(top_terms)

            insights.append(f"The session focused on themes related to {term_str}.")

            # Add dominant topic insight
            if (
                "dominant_topic" in topic_data
                and topic_data["dominant_topic"] is not None
            ):
                dominant_topic = topic_data["dominant_topic"]
                dominant_topic_prob = topic_data.get("dominant_topic_prob", 0)

                if dominant_topic_prob > 0.7:
                    insights.append(
                        f"The session was strongly focused on a single theme (Topic {dominant_topic})."
                    )
                elif dominant_topic_prob < 0.3:
                    insights.append(
                        "The session covered a diverse range of themes without a strong focus on any one topic."
                    )

        # Add insights to report
        if insights:
            for i, insight in enumerate(insights[: self.highlight_count], 1):
                content.append(f"{i}. {insight}")
        else:
            content.append("No specific insights available for this session.")

        content.append("")

        # Recommendations section
        content.append("## Recommendations")
        content.append("")
        content.append(
            "Based on the analysis, consider the following for future sessions:"
        )
        content.append("")

        recommendations = []

        # Generate recommendations based on sentiment
        if sentiment_data:
            overall_sentiment = sentiment_data.get("overall_sentiment", "neutral")

            if overall_sentiment == "negative":
                recommendations.append(
                    "Follow up on topics that evoked negative emotions to address underlying concerns."
                )

            # Session progression recommendations
            if "session_progression_df" in sentiment_results:
                prog_df = sentiment_results["session_progression_df"]
                if not prog_df.empty:
                    session_prog = prog_df[prog_df["transcript_idx"] == index]
                    if not session_prog.empty:
                        prog_data = session_prog.iloc[0]
                        sentiment_trend = prog_data.get("sentiment_trend")

                        if sentiment_trend == "deteriorating":
                            recommendations.append(
                                "Begin future sessions by checking in on the client's emotional state, as this session ended on a more negative note."
                            )

        # Generate topic-based recommendations
        if topic_data and topic_terms:
            # Recommend exploration of the main topic
            if (
                "dominant_topic" in topic_data
                and topic_data["dominant_topic"] is not None
            ):
                dominant_topic = topic_data["dominant_topic"]
                top_terms = [t["term"] for t in topic_terms[:3]]
                term_str = ", ".join(top_terms)

                recommendations.append(
                    f"Continue exploring themes related to {term_str} in future sessions."
                )

            # Recommend checking on secondary topics
            if "topic_distribution" in topic_data and topic_data["topic_distribution"]:
                topic_dist = topic_data["topic_distribution"]
                if len(topic_dist) > 1:
                    secondary_topic = topic_dist[1][0]
                    recommendations.append(
                        f"Consider exploring Topic {secondary_topic} in more depth as it appeared as a secondary theme."
                    )

        # Add generic recommendations if needed
        if not recommendations:
            recommendations = [
                "Review the session transcript for any topics that warrant follow-up.",
                "Consider exploring the emotional context of the main themes in greater detail.",
                "Track changes in sentiment on these topics in future sessions.",
            ]

        # Add recommendations to report
        for i, recommendation in enumerate(recommendations[: self.highlight_count], 1):
            content.append(f"{i}. {recommendation}")

        content.append("")
        content.append("---")
        content.append("")
        content.append(
            f"*Report generated at {datetime.now().strftime('%Y-%m-%d %H:%M')}*"
        )

        # Add LLM-generated insights if available
        if (
            llm_insights
            and index in llm_insights
            and self.config.get("reports", {}).get("include_llm_insights", True)
        ):
            content += "\n\n## LLM-Generated Insights\n\n"

            # Add transcript summary insights
            transcript_insights = llm_insights[index]

            if "summary" in transcript_insights and transcript_insights["summary"]:
                content += "### Summary\n\n"
                content += transcript_insights["summary"] + "\n\n"

            # Add key insights
            if "insights" in transcript_insights and transcript_insights["insights"]:
                content += "### Key Insights\n\n"
                for insight in transcript_insights["insights"]:
                    content += f"- {insight.strip()}\n"
                content += "\n"

            # Add emotional themes
            if (
                "emotional_themes" in transcript_insights
                and transcript_insights["emotional_themes"]
            ):
                content += "### Emotional Themes\n\n"
                for theme in transcript_insights["emotional_themes"]:
                    content += f"- {theme.strip()}\n"
                content += "\n"

            # Add therapeutic focus areas
            if (
                "therapeutic_focus" in transcript_insights
                and transcript_insights["therapeutic_focus"]
            ):
                content += "### Therapeutic Focus Areas\n\n"
                for focus in transcript_insights["therapeutic_focus"]:
                    content += f"- {focus.strip()}\n"
                content += "\n"

            # Add contextual analysis insights if available
            if "analysis_insights" in transcript_insights:
                analysis = transcript_insights["analysis_insights"]
                if (
                    "contextual_insights" in analysis
                    and analysis["contextual_insights"]
                ):
                    content += "### Contextual Analysis\n\n"
                    for insight in analysis["contextual_insights"]:
                        content += f"{insight.strip()}\n\n"
                    content += "\n"

        # Compile the report
        report = {
            "session_number": session_number,
            "session_date": session_date,
            "file_name": file_name,
            "sentiment_data": sentiment_data,
            "topic_data": topic_data,
            "insights": insights,
            "recommendations": recommendations,
            "content": "\n".join(content),
        }

        return report

    def _generate_overall_report(
        self,
        transcripts,
        processed_data,
        sentiment_results,
        topic_results,
        session_reports,
        llm_insights=None,
    ):
        """Generate an overall report summarizing all therapy sessions.

        Args:
            transcripts (list): List of transcript dictionaries
            processed_data (list): List of processed transcript dictionaries
            sentiment_results (dict): Sentiment analysis results
            topic_results (dict): Topic modeling results
            session_reports (list): List of generated session reports
            llm_insights (dict, optional): LLM-generated insights

        Returns:
            dict: Overall report dictionary
        """
        title = "Therapy Progress Report"

        # Get session date range
        session_dates = [
            t.get("session_date") for t in transcripts if t.get("session_date")
        ]
        if session_dates:
            min_date = min(session_dates).strftime("%Y-%m-%d")
            max_date = max(session_dates).strftime("%Y-%m-%d")
            date_range = f"{min_date} to {max_date}"
        else:
            date_range = "Unknown date range"

        # Generate report content
        content = []

        # Title
        content.append(f"# {title}")
        content.append("")
        content.append(f"**Period**: {date_range}")
        content.append(f"**Sessions Analyzed**: {len(transcripts)}")
        content.append("")

        # Executive summary
        content.append("## Executive Summary")
        content.append("")

        # Overall sentiment trends
        if "transcript_sentiments_df" in sentiment_results:
            df = sentiment_results["transcript_sentiments_df"]
            if not df.empty and "session_date" in df.columns:
                # Filter out rows with missing dates
                df = df[df["session_date"].notna()].copy()

                if not df.empty:
                    # Sort by date
                    df.sort_values("session_date", inplace=True)

                    # Calculate trend
                    first_sentiment = df.iloc[0]["avg_compound"]
                    last_sentiment = df.iloc[-1]["avg_compound"]
                    sentiment_change = last_sentiment - first_sentiment

                    # Describe trend
                    if sentiment_change > 0.2:
                        trend_description = "significant improvement"
                    elif sentiment_change > 0.05:
                        trend_description = "moderate improvement"
                    elif sentiment_change < -0.2:
                        trend_description = "significant decline"
                    elif sentiment_change < -0.05:
                        trend_description = "moderate decline"
                    else:
                        trend_description = "relatively stable"

                    # Count positive/negative sessions
                    positive_count = len(df[df["overall_sentiment"] == "positive"])
                    negative_count = len(df[df["overall_sentiment"] == "negative"])
                    neutral_count = len(df[df["overall_sentiment"] == "neutral"])

                    content.append(
                        f"Over the course of therapy, the emotional tone has shown {trend_description}."
                    )
                    content.append(
                        f"Out of {len(df)} sessions, {positive_count} were predominantly positive, "
                        + f"{negative_count} were negative, and {neutral_count} were neutral."
                    )

        # Main topics
        if "document_topics_df" in topic_results:
            doc_df = topic_results["document_topics_df"]
            # Filter for transcript-level documents
            doc_df = doc_df[doc_df["type"] == "transcript"].copy()

            if not doc_df.empty:
                # Count dominant topics
                topic_counts = doc_df["dominant_topic"].value_counts()
                most_common_topic = (
                    topic_counts.index[0] if not topic_counts.empty else None
                )

                if most_common_topic is not None:
                    # Get terms for the most common topic
                    if "topics_df" in topic_results:
                        topics_df = topic_results["topics_df"]
                        if not topics_df.empty:
                            topic_terms_df = topics_df[
                                topics_df["topic_id"] == most_common_topic
                            ]
                            if not topic_terms_df.empty:
                                topic_terms_df = topic_terms_df.sort_values(
                                    "weight", ascending=False
                                )
                                top_terms = ", ".join(
                                    topic_terms_df.head(5)["term"].tolist()
                                )

                                content.append(
                                    f"The most recurrent theme across sessions was Topic {most_common_topic}, "
                                    + f"which includes terms like {top_terms}."
                                )

        content.append("")

        # Progress Tracking
        content.append("## Progress Tracking")
        content.append("")

        # Sentiment over time
        if "transcript_sentiments_df" in sentiment_results:
            df = sentiment_results["transcript_sentiments_df"]
            if not df.empty and "session_date" in df.columns:
                # Filter out rows with missing dates
                df = df[df["session_date"].notna()].copy()

                if not df.empty:
                    # Sort by date
                    df.sort_values("session_date", inplace=True)

                    content.append("### Emotional Tone Progression")
                    content.append("")
                    content.append("| Session | Date | Sentiment | Score |")
                    content.append("|---------|------|-----------|-------|")

                    for i, row in df.iterrows():
                        session_num = row.get("session_number", i + 1)
                        date = row["session_date"].strftime("%Y-%m-%d")
                        sentiment = row["overall_sentiment"].capitalize()
                        score = f"{row['avg_compound']:.2f}"

                        content.append(
                            f"| {session_num} | {date} | {sentiment} | {score} |"
                        )

                    content.append("")

        # Topic evolution
        if "topic_insights_df" in topic_results:
            topic_df = topic_results["topic_insights_df"]
            if not topic_df.empty and "session_date" in topic_df.columns:
                # Filter out rows with missing dates
                topic_df = topic_df[topic_df["session_date"].notna()].copy()

                if not topic_df.empty:
                    content.append("### Theme Evolution")
                    content.append("")
                    content.append(
                        "The following themes have shown significant changes in prevalence over time:"
                    )
                    content.append("")

                    # Group by topic and calculate trend
                    topic_trends = {}

                    for topic_id in topic_df["topic_id"].unique():
                        topic_sessions = topic_df[
                            topic_df["topic_id"] == topic_id
                        ].sort_values("session_date")

                        if len(topic_sessions) > 1:
                            first_prev = topic_sessions.iloc[0]["prevalence"]
                            last_prev = topic_sessions.iloc[-1]["prevalence"]
                            change = last_prev - first_prev

                            # Only include topics with significant changes
                            if abs(change) > 0.1:
                                trend = "increasing" if change > 0 else "decreasing"
                                topic_trends[topic_id] = {
                                    "change": change,
                                    "trend": trend,
                                    "first": first_prev,
                                    "last": last_prev,
                                }

                    if topic_trends:
                        # Get terms for these topics
                        if "topics_df" in topic_results:
                            topics_df = topic_results["topics_df"]

                            for topic_id, trend in sorted(
                                topic_trends.items(),
                                key=lambda x: abs(x[1]["change"]),
                                reverse=True,
                            ):
                                topic_terms_df = topics_df[
                                    topics_df["topic_id"] == topic_id
                                ]

                                if not topic_terms_df.empty:
                                    topic_terms_df = topic_terms_df.sort_values(
                                        "weight", ascending=False
                                    )
                                    top_terms = ", ".join(
                                        topic_terms_df.head(3)["term"].tolist()
                                    )

                                    change_pct = abs(trend["change"]) * 100
                                    content.append(
                                        f"- **Topic {topic_id}** ({top_terms}): {trend['trend']} by {change_pct:.1f}% "
                                        + f"(from {trend['first']:.2f} to {trend['last']:.2f})"
                                    )
                    else:
                        content.append(
                            "No significant changes in theme prevalence were observed."
                        )

                    content.append("")

        # Key Insights
        content.append("## Key Insights")
        content.append("")

        insights = []

        # Overall sentiment insights
        if "transcript_sentiments_df" in sentiment_results:
            df = sentiment_results["transcript_sentiments_df"]
            if not df.empty and "session_date" in df.columns:
                # Filter out rows with missing dates
                df = df[df["session_date"].notna()].copy()

                if not df.empty:
                    # Sort by date
                    df.sort_values("session_date", inplace=True)

                    # Calculate trend
                    first_sentiment = df.iloc[0]["avg_compound"]
                    last_sentiment = df.iloc[-1]["avg_compound"]
                    sentiment_change = last_sentiment - first_sentiment

                    if sentiment_change > 0.05:
                        insights.append(
                            "The overall emotional tone has improved over the course of therapy, suggesting positive progress."
                        )
                    elif sentiment_change < -0.05:
                        insights.append(
                            "The overall emotional tone has declined, which may indicate uncovering of challenging issues or increasing distress."
                        )
                    else:
                        insights.append(
                            "The emotional tone has remained relatively stable throughout the therapy period."
                        )

                    # Analyze volatility
                    sentiment_values = df["avg_compound"].tolist()
                    diffs = [
                        abs(sentiment_values[i] - sentiment_values[i - 1])
                        for i in range(1, len(sentiment_values))
                    ]
                    avg_volatility = sum(diffs) / len(diffs) if diffs else 0

                    if avg_volatility > 0.2:
                        insights.append(
                            "There has been high emotional volatility between sessions, suggesting significant processing of emotions."
                        )
                    elif avg_volatility < 0.05:
                        insights.append(
                            "Emotional tone has been very consistent between sessions, suggesting stability or potential plateauing."
                        )

        # Topic insights
        if "topic_insights_df" in topic_results and "topics_df" in topic_results:
            topic_df = topic_results["topic_insights_df"]
            topics_df = topic_results["topics_df"]

            if not topic_df.empty and not topics_df.empty:
                # Identify most consistent topic
                topic_consistency = {}

                for topic_id in topic_df["topic_id"].unique():
                    topic_sessions = topic_df[topic_df["topic_id"] == topic_id]
                    if len(topic_sessions) > 1:
                        topic_consistency[topic_id] = topic_sessions[
                            "prevalence"
                        ].mean()

                if topic_consistency:
                    most_consistent_topic = max(
                        topic_consistency.items(), key=lambda x: x[1]
                    )[0]

                    # Get terms for this topic
                    topic_terms_df = topics_df[
                        topics_df["topic_id"] == most_consistent_topic
                    ]
                    if not topic_terms_df.empty:
                        topic_terms_df = topic_terms_df.sort_values(
                            "weight", ascending=False
                        )
                        top_terms = ", ".join(topic_terms_df.head(3)["term"].tolist())

                        insights.append(
                            f"Theme {most_consistent_topic} ({top_terms}) has been consistently present throughout therapy, "
                            + "suggesting an ongoing area of focus or concern."
                        )

        # Add insights about speaker patterns if available
        if "speaker_sentiments_df" in sentiment_results:
            df = sentiment_results["speaker_sentiments_df"]

            if not df.empty and "speaker" in df.columns:
                # Look for client vs therapist patterns
                client_rows = df[df["speaker"].str.lower().str.contains("client")]
                therapist_rows = df[df["speaker"].str.lower().str.contains("therapist")]

                if not client_rows.empty and not therapist_rows.empty:
                    client_sentiment = client_rows.iloc[0]["avg_compound"]
                    therapist_sentiment = therapist_rows.iloc[0]["avg_compound"]

                    sentiment_gap = client_sentiment - therapist_sentiment

                    if sentiment_gap < -0.2:
                        insights.append(
                            "The client consistently expresses more negative emotions compared to the therapist, "
                            + "which may indicate ongoing distress or challenging issues."
                        )
                    elif sentiment_gap > 0.2:
                        insights.append(
                            "The client consistently expresses more positive emotions compared to the therapist, "
                            + "suggesting either good progress or potential avoidance of difficult topics."
                        )

        # Add insights to report
        if insights:
            for i, insight in enumerate(insights, 1):
                content.append(f"{i}. {insight}")
        else:
            content.append("No specific overall insights available.")

        content.append("")

        # Recommendations
        content.append("## Recommendations")
        content.append("")

        recommendations = []

        # Generate overall recommendations
        if "transcript_sentiments_df" in sentiment_results:
            df = sentiment_results["transcript_sentiments_df"]
            if not df.empty and len(df) > 1:
                # Get last session sentiment
                last_sentiment = df.iloc[-1]["avg_compound"]
                last_category = df.iloc[-1]["overall_sentiment"]

                if last_category == "negative":
                    recommendations.append(
                        "The most recent session had a negative emotional tone. Consider following up on recent distressing topics."
                    )

                # Check for progress stagnation
                if len(df) >= 3:
                    recent_sentiments = df.iloc[-3:]["avg_compound"].tolist()
                    if max(recent_sentiments) - min(recent_sentiments) < 0.1:
                        recommendations.append(
                            "Recent sessions show minimal emotional variation. Consider introducing new therapeutic approaches or exploring different areas."
                        )

        # Topic-based recommendations
        if "topic_insights_df" in topic_results and "topics_df" in topic_results:
            topic_df = topic_results["topic_insights_df"]
            topics_df = topic_results["topics_df"]

            if not topic_df.empty and not topics_df.empty:
                # Find emerging topics
                topic_trends = {}

                for topic_id in topic_df["topic_id"].unique():
                    topic_sessions = topic_df[
                        topic_df["topic_id"] == topic_id
                    ].sort_values("session_date")

                    if len(topic_sessions) > 1:
                        first_prev = topic_sessions.iloc[0]["prevalence"]
                        last_prev = topic_sessions.iloc[-1]["prevalence"]
                        change = last_prev - first_prev

                        topic_trends[topic_id] = change

                # Find most increasing topic
                if topic_trends:
                    increasing_topics = [
                        (topic_id, change)
                        for topic_id, change in topic_trends.items()
                        if change > 0.1
                    ]

                    if increasing_topics:
                        most_increasing = max(increasing_topics, key=lambda x: x[1])[0]

                        # Get terms for this topic
                        topic_terms_df = topics_df[
                            topics_df["topic_id"] == most_increasing
                        ]
                        if not topic_terms_df.empty:
                            topic_terms_df = topic_terms_df.sort_values(
                                "weight", ascending=False
                            )
                            top_terms = ", ".join(
                                topic_terms_df.head(3)["term"].tolist()
                            )

                            recommendations.append(
                                f"Theme {most_increasing} ({top_terms}) is becoming increasingly prominent. "
                                + "Consider dedicating more focus to this emerging area."
                            )

        # Add generic recommendations if needed
        if not recommendations:
            recommendations = [
                "Continue to monitor emotional progression across sessions to identify patterns.",
                "Periodically revisit key themes to track progress and development.",
                "Consider supplementing with quantitative assessments to validate these qualitative findings.",
            ]

        # Add recommendations to report
        for i, recommendation in enumerate(recommendations, 1):
            content.append(f"{i}. {recommendation}")

        content.append("")

        # Session summaries
        content.append("## Session Summaries")
        content.append("")

        for i, report in enumerate(session_reports):
            session_num = report.get("session_number", i + 1)
            session_date = report.get("session_date")
            date_str = session_date.strftime("%Y-%m-%d") if session_date else "Unknown"

            sentiment_data = report.get("sentiment_data", {})
            sentiment_str = sentiment_data.get(
                "overall_sentiment", "neutral"
            ).capitalize()
            compound_score = sentiment_data.get("avg_compound", 0)

            topic_data = report.get("topic_data", {})
            dominant_topic = topic_data.get("dominant_topic", "N/A")

            insights = report.get("insights", [])
            key_insight = insights[0] if insights else "No specific insights."

            content.append(f"### Session {session_num} ({date_str})")
            content.append("")
            content.append(
                f"- **Emotional Tone**: {sentiment_str} ({compound_score:.2f})"
            )
            content.append(f"- **Primary Theme**: Topic {dominant_topic}")
            content.append(f"- **Key Insight**: {key_insight}")
            content.append("")

        content.append("")
        content.append("---")
        content.append("")
        content.append(
            f"*Report generated at {datetime.now().strftime('%Y-%m-%d %H:%M')}*"
        )

        # Add LLM-generated progress insights if available
        if (
            llm_insights
            and "progress" in llm_insights
            and self.config.get("reports", {}).get("include_llm_insights", True)
        ):
            content += "\n\n## LLM-Generated Progress Insights\n\n"

            progress_insights = llm_insights["progress"]

            if (
                "progress_insights" in progress_insights
                and progress_insights["progress_insights"]
            ):
                for insight in progress_insights["progress_insights"]:
                    content += f"{insight.strip()}\n\n"

            # If there's no structured format, just use the full response
            elif (
                "full_response" in progress_insights
                and progress_insights["full_response"]
            ):
                content += progress_insights["full_response"] + "\n\n"

        # Compile the report
        report = {
            "date_range": date_range,
            "session_count": len(transcripts),
            "insights": insights,
            "recommendations": recommendations,
            "content": "\n".join(content),
        }

        return report

    def _save_report(self, content, filename):
        """Save a report to a file.

        Args:
            content (str): Report content
            filename (str): Base filename without extension

        Returns:
            str: Path to the saved report file
        """
        ext = self.save_format.lower()

        if ext not in ["md", "txt", "html"]:
            self.logger.warning(f"Unsupported save format: {ext}. Using md instead.")
            ext = "md"

        # Ensure filename doesn't contain spaces
        safe_filename = filename.replace(" ", "_")

        # Add timestamp to avoid overwriting
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(self.reports_dir, f"{safe_filename}_{timestamp}.{ext}")

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)

            self.logger.info(f"Saved report to {filepath}")
            return filepath

        except Exception as e:
            self.logger.error(f"Error saving report to {filepath}: {str(e)}")
            return None
