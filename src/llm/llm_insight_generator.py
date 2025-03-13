#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LLM Insight Generator Module

This module integrates large language models (LLMs) to generate enhanced summaries
and contextual insights from therapy session transcripts.
"""

import os
import logging
import json
import requests
from typing import Dict, List, Any, Optional, Union


class LLMInsightGenerator:
    """Class for generating enhanced insights from therapy session transcripts using LLMs."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize the LLM insight generator with configuration.

        Args:
            config (dict): Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger("therapy_pipeline.llm_insight_generator")

        # Extract LLM-related settings from config
        self.llm_config = config.get("llm", {})
        self.model = self.llm_config.get("model", "openai")
        self.api_key = self.llm_config.get("api_key") or os.environ.get(
            f"{self.model.upper()}_API_KEY"
        )
        self.temperature = self.llm_config.get("temperature", 0.3)
        self.max_tokens = self.llm_config.get("max_tokens", 1000)
        self.system_prompt = self.llm_config.get(
            "system_prompt",
            "You are an expert therapist analyzing therapy session transcripts.",
        )

        if not self.api_key:
            self.logger.warning(
                f"No API key found for {self.model}. LLM features will be unavailable."
            )

        self.logger.info(f"Initialized LLM insight generator with model: {self.model}")

    def generate_transcript_summary(self, transcript: Dict[str, Any]) -> Dict[str, Any]:
        """Generate an enhanced summary of a single transcript.

        Args:
            transcript (dict): Transcript dictionary

        Returns:
            dict: Dictionary containing enhanced summary insights
        """
        if not self.api_key:
            self.logger.warning("Cannot generate LLM insights without an API key")
            return {"error": "API key not configured", "summary": "", "insights": []}

        try:
            # Extract the raw content or dialogue from the transcript
            content = transcript.get("raw_content", "")

            if not content and "dialogue" in transcript:
                # Reconstruct content from dialogue entries
                content = "\n\n".join(
                    [
                        f"{entry.get('speaker', 'Unknown')}: {entry.get('text', '')}"
                        for entry in transcript.get("dialogue", [])
                    ]
                )

            # If we still don't have content, return an error
            if not content:
                self.logger.warning("No content found in transcript")
                return {
                    "error": "No content found in transcript",
                    "summary": "",
                    "insights": [],
                }

            # Metadata for context
            metadata = transcript.get("metadata", {})
            session_number = transcript.get("session_number", "Unknown")
            session_date = transcript.get("session_date", "Unknown date")

            # Construct prompt
            user_prompt = (
                f"Please analyze this therapy session transcript and provide:\n\n"
                f"1. A concise summary (3-5 sentences)\n"
                f"2. 3-5 key insights or patterns\n"
                f"3. Notable emotional themes\n"
                f"4. Potential areas for therapeutic focus\n\n"
                f"Session metadata: Date: {session_date}, Session number: {session_number}\n"
                f"Patient: {metadata.get('patient', 'Unknown')}\n\n"
                f"TRANSCRIPT:\n{content}"
            )

            # Call the appropriate LLM API
            if self.model.lower() == "openai":
                response = self._call_openai_api(user_prompt)
            elif self.model.lower() == "anthropic":
                response = self._call_anthropic_api(user_prompt)
            elif self.model.lower() == "local":
                response = self._call_local_llm(user_prompt)
            else:
                self.logger.error(f"Unsupported LLM model: {self.model}")
                return {
                    "error": f"Unsupported model: {self.model}",
                    "summary": "",
                    "insights": [],
                }

            # Process the response
            if response and isinstance(response, str):
                result = self._parse_llm_response(response)
                result["source"] = "llm"
                result["model"] = self.model
                return result
            else:
                return {
                    "error": "Failed to get LLM response",
                    "summary": "",
                    "insights": [],
                }

        except Exception as e:
            self.logger.error(f"Error generating LLM insights: {str(e)}")
            return {"error": str(e), "summary": "", "insights": []}

    def generate_analysis_insights(
        self,
        transcript: Dict[str, Any],
        sentiment_results: Dict[str, Any],
        topic_results: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate insights that combine transcript content with sentiment and topic analysis results.

        Args:
            transcript (dict): Transcript dictionary
            sentiment_results (dict): Sentiment analysis results
            topic_results (dict): Topic modeling results

        Returns:
            dict: Dictionary containing contextual insights
        """
        if not self.api_key:
            self.logger.warning("Cannot generate LLM insights without an API key")
            return {"error": "API key not configured", "contextual_insights": []}

        try:
            # Extract the raw content or dialogue from the transcript
            content = transcript.get("raw_content", "")

            if not content and "dialogue" in transcript:
                # Reconstruct content from dialogue entries
                content = "\n\n".join(
                    [
                        f"{entry.get('speaker', 'Unknown')}: {entry.get('text', '')}"
                        for entry in transcript.get("dialogue", [])
                    ]
                )

            # Extract key sentiment information
            sentiment_info = self._extract_sentiment_highlights(
                transcript, sentiment_results
            )

            # Extract key topic information
            topic_info = self._extract_topic_highlights(transcript, topic_results)

            # Metadata for context
            metadata = transcript.get("metadata", {})
            session_number = transcript.get("session_number", "Unknown")
            session_date = transcript.get("session_date", "Unknown date")

            # Construct prompt
            user_prompt = (
                f"Please analyze this therapy session transcript along with the provided sentiment and topic analysis results. Provide:\n\n"
                f"1. A contextual interpretation integrating the sentiment and topic analysis with the content\n"
                f"2. 3-5 evidence-based insights that draw from both the quantitative analysis and the transcript content\n"
                f"3. Suggested therapeutic approaches based on the combined analysis\n\n"
                f"Session metadata: Date: {session_date}, Session number: {session_number}\n"
                f"Patient: {metadata.get('patient', 'Unknown')}\n\n"
                f"SENTIMENT ANALYSIS:\n{sentiment_info}\n\n"
                f"TOPIC ANALYSIS:\n{topic_info}\n\n"
                f"TRANSCRIPT:\n{content[:4000]}..."
                if len(content) > 4000
                else content
            )

            # Call the appropriate LLM API
            if self.model.lower() == "openai":
                response = self._call_openai_api(user_prompt)
            elif self.model.lower() == "anthropic":
                response = self._call_anthropic_api(user_prompt)
            elif self.model.lower() == "local":
                response = self._call_local_llm(user_prompt)
            else:
                self.logger.error(f"Unsupported LLM model: {self.model}")
                return {
                    "error": f"Unsupported model: {self.model}",
                    "contextual_insights": [],
                }

            # Process the response
            if response and isinstance(response, str):
                # For combined analysis, we'll format a bit differently
                result = {
                    "source": "llm",
                    "model": self.model,
                    "contextual_insights": response.strip().split("\n\n"),
                    "full_response": response,
                }
                return result
            else:
                return {
                    "error": "Failed to get LLM response",
                    "contextual_insights": [],
                }

        except Exception as e:
            self.logger.error(f"Error generating LLM analysis insights: {str(e)}")
            return {"error": str(e), "contextual_insights": []}

    def generate_progress_insights(
        self,
        transcripts: List[Dict[str, Any]],
        sentiment_results: Dict[str, Any],
        topic_results: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate insights about client progress across multiple sessions.

        Args:
            transcripts (list): List of transcript dictionaries
            sentiment_results (dict): Sentiment analysis results
            topic_results (dict): Topic modeling results

        Returns:
            dict: Dictionary containing progress insights
        """
        if not self.api_key:
            self.logger.warning("Cannot generate LLM insights without an API key")
            return {"error": "API key not configured", "progress_insights": []}

        if not transcripts or len(transcripts) < 2:
            self.logger.warning("Need at least 2 transcripts to analyze progress")
            return {
                "error": "Need at least 2 transcripts to analyze progress",
                "progress_insights": [],
            }

        try:
            # Extract key information from all transcripts
            transcripts_info = []
            for i, transcript in enumerate(transcripts):
                # Get date and session number
                session_date = transcript.get("session_date", f"Session {i+1}")
                session_number = transcript.get("session_number", i + 1)

                # Extract a brief summary of the session content
                dialogue = transcript.get("dialogue", [])
                summary = "\n".join(
                    [
                        f"{entry.get('speaker', 'Unknown')}: {entry.get('text', '')[:100]}..."
                        for entry in dialogue[:5]
                    ]
                )

                # Get sentiment information
                sentiment_info = self._extract_sentiment_highlights(
                    transcript, sentiment_results
                )

                # Get topic information
                topic_info = self._extract_topic_highlights(transcript, topic_results)

                # Add to list
                transcripts_info.append(
                    f"SESSION {session_number} ({session_date}):\n"
                    f"Content preview: {summary}\n"
                    f"Sentiment: {sentiment_info}\n"
                    f"Topics: {topic_info}\n"
                )

            # Get patient name from first transcript metadata
            patient_name = (
                transcripts[0].get("metadata", {}).get("patient", "the client")
            )

            # Construct prompt
            user_prompt = (
                f"Please analyze these therapy session transcripts from {patient_name} and provide insights about the client's progress across sessions:\n\n"
                f"1. Key changes in emotional patterns and sentiment\n"
                f"2. Evolution of discussion topics and themes\n"
                f"3. Observable therapeutic progress\n"
                f"4. Potential areas for future focus\n"
                f"5. Evidence-based recommendations for continued treatment\n\n"
                f"SESSION DATA:\n\n" + "\n\n".join(transcripts_info)
            )

            # Call the appropriate LLM API
            if self.model.lower() == "openai":
                response = self._call_openai_api(user_prompt)
            elif self.model.lower() == "anthropic":
                response = self._call_anthropic_api(user_prompt)
            elif self.model.lower() == "local":
                response = self._call_local_llm(user_prompt)
            else:
                self.logger.error(f"Unsupported LLM model: {self.model}")
                return {
                    "error": f"Unsupported model: {self.model}",
                    "progress_insights": [],
                }

            # Process the response
            if response and isinstance(response, str):
                result = {
                    "source": "llm",
                    "model": self.model,
                    "progress_insights": response.strip().split("\n\n"),
                    "full_response": response,
                }
                return result
            else:
                return {"error": "Failed to get LLM response", "progress_insights": []}

        except Exception as e:
            self.logger.error(f"Error generating LLM progress insights: {str(e)}")
            return {"error": str(e), "progress_insights": []}

    def _extract_sentiment_highlights(
        self, transcript: Dict[str, Any], sentiment_results: Dict[str, Any]
    ) -> str:
        """Extract key sentiment information for a specific transcript.

        Args:
            transcript (dict): Transcript dictionary
            sentiment_results (dict): Sentiment analysis results

        Returns:
            str: Formatted string with key sentiment information
        """
        transcript_idx = transcript.get("transcript_idx", None)
        filename = transcript.get("file_name", "")

        # If we don't have the transcript index, try to find it by filename
        if (
            transcript_idx is None
            and filename
            and "transcript_sentiments_df" in sentiment_results
        ):
            df = sentiment_results["transcript_sentiments_df"]
            matching_rows = df[df["file_name"] == filename]
            if not matching_rows.empty:
                transcript_idx = matching_rows.iloc[0].get("transcript_idx")

        # If we still don't have an index, return a generic message
        if transcript_idx is None:
            return "Sentiment data not available for this specific transcript."

        # Extract transcript-level sentiment
        if "transcript_sentiments_df" in sentiment_results:
            df = sentiment_results["transcript_sentiments_df"]
            matching_rows = df[df["transcript_idx"] == transcript_idx]

            if not matching_rows.empty:
                row = matching_rows.iloc[0]
                sentiment_info = (
                    f"Overall sentiment: {row.get('overall_sentiment', 'unknown')}\n"
                    f"Compound score: {row.get('avg_compound', 0):.2f}\n"
                    f"Positive: {row.get('avg_positive', 0):.2f}, "
                    f"Negative: {row.get('avg_negative', 0):.2f}, "
                    f"Neutral: {row.get('avg_neutral', 0):.2f}\n"
                )

                # Add sentiment progression if available
                if "session_progression_df" in sentiment_results:
                    prog_df = sentiment_results["session_progression_df"]
                    prog_rows = prog_df[prog_df["transcript_idx"] == transcript_idx]

                    if not prog_rows.empty:
                        prog_row = prog_rows.iloc[0]
                        sentiment_info += (
                            f"Sentiment trend: {prog_row.get('sentiment_trend', 'unknown')}\n"
                            f"Start sentiment: {prog_row.get('start_sentiment', 0):.2f}, "
                            f"End sentiment: {prog_row.get('end_sentiment', 0):.2f}\n"
                            f"Sentiment shift: {prog_row.get('sentiment_shift', 0):.2f}\n"
                        )

                return sentiment_info

        return "Detailed sentiment data not available."

    def _extract_topic_highlights(
        self, transcript: Dict[str, Any], topic_results: Dict[str, Any]
    ) -> str:
        """Extract key topic information for a specific transcript.

        Args:
            transcript (dict): Transcript dictionary
            topic_results (dict): Topic modeling results

        Returns:
            str: Formatted string with key topic information
        """
        transcript_idx = transcript.get("transcript_idx", None)
        filename = transcript.get("file_name", "")

        # If we don't have the transcript index, try to find it by filename
        if (
            transcript_idx is None
            and filename
            and "document_topics_df" in topic_results
        ):
            df = topic_results["document_topics_df"]
            matching_rows = df[
                (df["file_name"] == filename) & (df["type"] == "transcript")
            ]
            if not matching_rows.empty:
                transcript_idx = matching_rows.iloc[0].get("transcript_idx")

        # If we still don't have an index, return a generic message
        if transcript_idx is None:
            return "Topic data not available for this specific transcript."

        # Extract document topics
        if "document_topics_df" in topic_results:
            df = topic_results["document_topics_df"]
            matching_rows = df[
                (df["transcript_idx"] == transcript_idx) & (df["type"] == "transcript")
            ]

            if not matching_rows.empty:
                row = matching_rows.iloc[0]
                dominant_topic = row.get("dominant_topic")
                topic_info = f"Dominant topic: Topic {dominant_topic} (probability: {row.get('dominant_topic_prob', 0):.2f})\n"

                # Get topic terms
                if "topics_df" in topic_results and dominant_topic is not None:
                    topics_df = topic_results["topics_df"]
                    topic_terms = topics_df[
                        topics_df["topic_id"] == dominant_topic
                    ].sort_values("weight", ascending=False)

                    if not topic_terms.empty:
                        top_terms = topic_terms.head(10)
                        terms_str = ", ".join(
                            [f"{row['term']}" for _, row in top_terms.iterrows()]
                        )
                        topic_info += f"Top terms: {terms_str}\n"

                return topic_info

        return "Detailed topic data not available."

    def _call_openai_api(self, user_prompt: str) -> Optional[str]:
        """Call the OpenAI API.

        Args:
            user_prompt (str): User prompt

        Returns:
            str or None: API response text or None if failed
        """
        try:
            import openai

            openai.api_key = self.api_key

            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            response = openai.ChatCompletion.create(
                model=self.llm_config.get("openai_model", "gpt-4"),
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            return response.choices[0].message["content"]

        except Exception as e:
            self.logger.error(f"Error calling OpenAI API: {str(e)}")
            return None

    def _call_anthropic_api(self, user_prompt: str) -> Optional[str]:
        """Call the Anthropic Claude API.

        Args:
            user_prompt (str): User prompt

        Returns:
            str or None: API response text or None if failed
        """
        try:
            import anthropic

            client = anthropic.Anthropic(api_key=self.api_key)

            response = client.messages.create(
                model=self.llm_config.get("anthropic_model", "claude-3-opus-20240229"),
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=self.system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )

            return response.content[0].text

        except Exception as e:
            self.logger.error(f"Error calling Anthropic API: {str(e)}")
            return None

    def _call_local_llm(self, user_prompt: str) -> Optional[str]:
        """Call a local LLM server (e.g., LM Studio, Ollama).

        Args:
            user_prompt (str): User prompt

        Returns:
            str or None: API response text or None if failed
        """
        try:
            # Get configuration for local LLM
            local_url = self.llm_config.get(
                "local_url", "http://localhost:8000/v1/chat/completions"
            )
            local_model = self.llm_config.get("local_model", "local-model")

            # Prepare request
            headers = {
                "Content-Type": "application/json",
            }

            payload = {
                "model": local_model,
                "messages": [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }

            # Make request
            response = requests.post(local_url, headers=headers, json=payload)
            response.raise_for_status()

            # Parse response
            result = response.json()
            return result.get("choices", [{}])[0].get("message", {}).get("content", "")

        except Exception as e:
            self.logger.error(f"Error calling local LLM: {str(e)}")
            return None

    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse the LLM response into a structured format.

        Args:
            response (str): Raw LLM response text

        Returns:
            dict: Structured response with summary and insights
        """
        result = {
            "summary": "",
            "insights": [],
            "emotional_themes": [],
            "therapeutic_focus": [],
            "full_response": response,
        }

        # Very basic parsing - actual implementation might use more sophisticated techniques
        # based on the LLM response structure
        sections = response.split("\n\n")

        for i, section in enumerate(sections):
            lower_section = section.lower()

            if "summary" in lower_section and i < len(sections) - 1:
                result["summary"] = sections[i + 1]
            elif "insight" in lower_section or "pattern" in lower_section:
                result["insights"].append(section)
            elif "emotion" in lower_section or "theme" in lower_section:
                result["emotional_themes"].append(section)
            elif (
                "therapeutic" in lower_section
                or "focus" in lower_section
                or "intervention" in lower_section
            ):
                result["therapeutic_focus"].append(section)

        # If we couldn't parse it properly, just return the full response as summary
        if not result["summary"] and not result["insights"]:
            result["summary"] = response

        return result
