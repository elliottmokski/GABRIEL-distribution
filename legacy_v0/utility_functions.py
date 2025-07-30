# ──────────────────────────────────────────────────────────────────────
# utility_functions.py  •  OpenAI Responses helpers  (2025-05-30)
# ──────────────────────────────────────────────────────────────────────
from __future__ import annotations

import ast
import asyncio
import concurrent.futures
import csv
import datetime
import hashlib
import json
import math
import multiprocessing
import os
import pickle
import queue
import random
import re
import signal
import textwrap
import threading
import time
import urllib
from collections import Counter, OrderedDict, defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, TimeoutError, as_completed
from datetime import datetime, timedelta
from itertools import groupby
from math import ceil
from typing import Any, Dict, List, Optional, Tuple, Union

# from serpapi import GoogleSearch
import aiohttp
import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import nest_asyncio
import numpy as np
import openai

# from youtube_transcript_api import YouTubeTranscriptApi
import pandas as pd

# import wikipedia
# import wikipediaapi
import requests
import seaborn as sns
import serpapi
from aiolimiter import AsyncLimiter  # Import the aiolimiter
from bs4 import BeautifulSoup, Comment, SoupStrainer
from jinja2 import Environment, FileSystemLoader, select_autoescape
from markupsafe import Markup
from matplotlib import rcParams, ticker
from matplotlib.patches import Patch
from openai import AsyncOpenAI, OpenAI

# import statsmodels.api as sm
from scipy import stats
from scipy.ndimage import uniform_filter1d
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from timeout_decorator import TimeoutError, timeout
from tqdm.auto import tqdm  # progress bars that work in notebooks and terminals


class APIError(Exception):
    pass


class SearchError(Exception):
    pass


class StopExecution(Exception):
    pass


class AlarmException(Exception):
    pass


class TimeoutException(Exception):
    pass


PATH = os.path.expanduser("~/Documents/emergency_data")

prompt_path = "prompts/emergency_thesis"

import functools

api_key = "..."
client = OpenAI(api_key=api_key)
client_async = openai.AsyncClient(api_key=api_key)


class SchemaManager:
    """
    Simple registry of reusable schema templates for use with json_mode prompts.
    These methods return Python representations of JSON schemas or type hints that can
    be used both to instruct the API (when set as expected_schema) and to generate
    dummy responses when testing.
    """

    @staticmethod
    def classification_schema() -> Dict[str, Any]:
        """
        Schema for classification prompts where keys are dynamic entity names
        and values are class labels.
        """
        return {"__dynamic__": {"type": str}}

    @staticmethod
    def differentiator_schema() -> Dict[str, Any]:
        """
        Schema for Commondant differentiator prompt outputs.
        """
        return {
            "major axes of difference": [str],
            "minor axes of difference": [str],
            "entity list square common ground": {
                "complete common ground": [str],
                "partial common ground": [str],
            },
            "entity list circle common ground": {
                "complete common ground": [str],
                "partial common ground": [str],
            },
        }


def generate_dummy_from_schema(schema: Any) -> Any:
    """
    Generate a dummy Python object conforming to a simplified schema description.
    This is useful in dummy mode when testing JSON responses.

    The schema can be expressed in several ways:
    - As a JSON Schema style dict with "type", "properties", etc.
    - As a dict mapping field names to Python type objects or nested schemas.
    - As a dict with a "__dynamic__" entry to indicate an object whose keys are dynamic
      and values conform to the given type.
    - As a list with a single element type to indicate arrays of that type.
    For primitive values, either Python type objects (str, int, float, bool) or strings
    ("string", "integer", "number", "boolean") are accepted.
    """
    # Handle None schema gracefully
    if schema is None:
        return {}
    # If already a Python type object
    if isinstance(schema, type):
        if schema is str:
            return "dummy text"
        if schema is int:
            return random.randint(0, 10)
        if schema is float:
            return round(random.uniform(0.0, 10.0), 3)
        if schema is bool:
            return random.choice([True, False])
        # fallback for unknown type
        return f"dummy {schema.__name__}"
    # If schema is a primitive string description
    if isinstance(schema, str):
        if schema == "string":
            return "dummy text"
        if schema == "integer":
            return random.randint(0, 10)
        if schema == "number":
            return round(random.uniform(0.0, 10.0), 3)
        if schema == "boolean":
            return random.choice([True, False])
        # fallback to using the string as dummy
        return f"dummy {schema}"
    # If schema is a list, assume homogeneous items based on first element
    if isinstance(schema, list):
        item_schema = schema[0] if schema else "string"
        # Generate between 1 and 3 items
        count = random.randint(1, 3)
        return [generate_dummy_from_schema(item_schema) for _ in range(count)]
    if isinstance(schema, dict):
        # Handle special dynamic schema
        if "__dynamic__" in schema:
            dyn_spec = schema["__dynamic__"]
            dyn_type = dyn_spec.get("type", "string")
            # generate a handful of entries
            num_keys = random.randint(1, 3)
            return {f"key{i + 1}": generate_dummy_from_schema(dyn_type) for i in range(num_keys)}
        # Handle JSON Schema-like dicts
        if "type" in schema:
            t = schema.get("type")
            if t == "object":
                props = schema.get("properties", {})
                return {k: generate_dummy_from_schema(v) for k, v in props.items()}
            if t == "array":
                items_schema = schema.get("items", "string")
                count = random.randint(1, 3)
                return [generate_dummy_from_schema(items_schema) for _ in range(count)]
            # primitive types handled above
            return generate_dummy_from_schema(t)
        # Otherwise treat as mapping of properties to schemas
        return {k: generate_dummy_from_schema(v) for k, v in schema.items()}
    # Fallback for unexpected types
    return "dummy"


class Teleprompter:
    def __init__(self, prompt_path):
        # Set up Jinja environment
        self.env = Environment(loader=FileSystemLoader(prompt_path))

    def clean_json_prompt(self, dirty_json_output, format_template_name):
        format_template = self.env.get_template(format_template_name).render()
        template = self.env.get_template("clean_json_prompt.jinja2")
        prompt = template.render(
            dirty_json_output=dirty_json_output, format_template=format_template
        )
        return prompt

    def merger_format_template(self):
        template = self.env.get_template("merger_format.jinja2")
        return template.render()

    def merger_examples(self):
        template = self.env.get_template("merger_examples.jinja2")
        return template.render()

    def merger_prompt(self, short_list, long_list):
        format_template = self.merger_format_template()
        examples = self.merger_examples()
        template = self.env.get_template("merger_prompt.jinja2")
        prompt = template.render(
            format_template=format_template,
            examples=examples,
            short_list=short_list,
            long_list=long_list,
        )
        return prompt

    def create_environment_prompt(
        self, population_description: str, demographic_description: str
    ) -> str:
        template = self.env.get_template("create_environment_prompt.jinja2")
        return template.render(
            population_description=population_description,
            demographic_description=demographic_description,
        )

    def generate_demographics_prompt(self, population_description: str, n_personas: int) -> str:
        template = self.env.get_template("generate_demographics_prompt.jinja2")
        return template.render(
            population_description=population_description,
            n_personas=n_personas,
        )

    def polish_demographics_prompt(
        self, demographic_description: str, population_description: str
    ) -> str:
        template = self.env.get_template("polish_demographics_prompt.jinja2")
        return template.render(
            demographic_description=demographic_description,
            population_description=population_description,
        )

    def persona_task_prompt(
        self,
        demographic_description: str,
        persona_biography: str,
        environment_text: str,
        task_text: str,
        english_only: bool,
    ) -> str:
        template = self.env.get_template("persona_task_prompt.jinja2")
        return template.render(
            demographic_description=demographic_description,
            persona_biography=persona_biography,
            environment_text=environment_text,
            task_text=task_text,
            english_only=english_only,
        )

    def create_persona_prompt(
        self,
        demographic_description: str,
        environment_text: str,
        extra_instructions: Optional[str] = None,
    ) -> str:
        template = self.env.get_template("create_persona_prompt.jinja2")
        return template.render(
            demographic_description=demographic_description,
            environment_text=environment_text,
            extra_instructions=extra_instructions or "",
        )

    def ai_task_prompt(self, task_text: str) -> str:
        template = self.env.get_template("ai_task_prompt.jinja2")
        return template.render(task_text=task_text)

    def basic_classifier_prompt(
        self,
        text: str,
        labels: dict[str, str],
        additional_instructions: str = "",
    ) -> str:
        template = self.env.get_template("basic_classifier_prompt.jinja2")
        return template.render(
            text=text,
            labels=labels,
            additional_instructions=additional_instructions,
        )

    def valuation_prompt(
        self,
        task_description: str,
        examples: Optional[Dict[str, float]] = None,
        additional_instructions: str = "",
    ) -> str:
        template = self.env.get_template("valuation_prompt.jinja2")
        return template.render(
            task_description=task_description,
            examples=examples,
            additional_instructions=additional_instructions,
        )

    def clique_extraction_prompt(
        self,
        text: str,
        research_question: str,
        instructions: str = "",
        bucket_context: Optional[str] = None,
    ) -> str:
        """
        Render the prompt for extracting mutually exclusive short terms (keys) and their
        textual justifications (values) from a single document in response to a research question.
        """
        template = self.env.get_template("clique_extraction_prompt.jinja2")
        return template.render(
            text=text,
            research_question=research_question,
            instructions=instructions,
            bucket_context=bucket_context,
        )

    def clique_bucket_prompt(
        self,
        terms: list[dict[str, str]],
        n_cliques: int,
        research_question: str,
        instructions: str = "",
        bucket_context: Optional[str] = None,
        hierarchy: Optional[dict] = None,
        second_column_name: Optional[str] = None,
    ) -> str:
        """
        Render the prompt for proposing a fixed number of higher-level bucket terms
        that could group many of the extracted terms.
        """
        template = self.env.get_template("clique_bucket_prompt.jinja2")
        return template.render(
            terms=terms,
            n_cliques=n_cliques,
            research_question=research_question,
            instructions=instructions,
            bucket_context=bucket_context,
            hierarchy=hierarchy,
            second_column_name=second_column_name,
        )

    def clique_vote_prompt(
        self,
        bucket_terms: list[str],
        regular_terms: list[dict[str, str]],
        num_to_select: int,
        research_question: str,
        instructions: str = "",
        bucket_context: Optional[str] = None,
        selected_buckets: Optional[dict[str, str]] = None,
        hierarchy: Optional[dict] = None,
        bucket_definitions: Optional[dict] = None,
        second_column_name: Optional[str] = None,
    ) -> str:
        """
        Render the prompt to vote on a subset of candidate bucket terms that best capture
        the underlying regular terms. Optionally pass previously selected buckets to avoid.
        """

        template = self.env.get_template("clique_vote_prompt.jinja2")
        return template.render(
            bucket_terms=bucket_terms,
            regular_terms=regular_terms,
            num_to_select=num_to_select,
            selected_buckets=selected_buckets,
            research_question=research_question,
            instructions=instructions,
            bucket_context=bucket_context,
            hierarchy=hierarchy,
            bucket_definitions=bucket_definitions,
            second_column_name=second_column_name,
        )

    def clique_apply_prompt(
        self,
        text: str,
        bucket_terms: list[str],
        research_question: str,
        instructions: str = "",
        bucket_context: Optional[str] = None,
        text_square: Optional[str] = None,
        circle_first: bool = True,
        bucket_definitions: Optional[dict] = None,
    ) -> str:
        """
        Render the prompt to classify a single document into zero or more final buckets.
        """
        template = self.env.get_template("clique_apply_prompt.jinja2")
        return template.render(
            text=text,
            bucket_terms=bucket_terms,
            research_question=research_question,
            instructions=instructions,
            bucket_context=bucket_context,
            text_square=text_square,
            circle_first=circle_first,
            bucket_definitions=bucket_definitions or {},
        )

    def clique_differentiation_prompt(
        self,
        text_circle: str,
        text_square: str,
        instructions: str = "",
        bucket_context: Optional[str] = None,
        circle_first: bool = True,
    ) -> str:
        template = self.env.get_template("clique_differentiator_prompt.jinja2")
        return template.render(
            text_circle=text_circle,
            text_square=text_square,
            instructions=instructions,
            bucket_context=bucket_context,
            circle_first=circle_first,
        )

    def clique_diff_apply_prompt(
        self,
        text_circle: str,
        text_square: str,
        bucket_terms: list[str],
        instructions: str = "",
        bucket_context: Optional[str] = None,
        circle_first: bool = True,
        bucket_definitions: Optional[dict] = None,
    ) -> str:
        template = self.env.get_template("clique_apply_diff_prompt.jinja2")
        return template.render(
            text_circle=text_circle,
            text_square=text_square,
            bucket_terms=bucket_terms,
            instructions=instructions,
            bucket_context=bucket_context,
            circle_first=circle_first,
            bucket_definitions=bucket_definitions or {},
        )

    def clique_elo_prompt(
        self,
        text_circle: str,
        text_square: str,
        bucket_terms: list[str],
        research_question: str,
        instructions: str = "",
        bucket_context: Optional[str] = None,
        circle_first: bool = True,
        bucket_definitions: Optional[dict] = None,
    ) -> str:
        template = self.env.get_template("clique_elo_prompt.jinja2")
        return template.render(
            text_circle=text_circle,
            text_square=text_square,
            bucket_terms=bucket_terms,
            research_question=research_question,
            instructions=instructions,
            bucket_context=bucket_context,
            circle_first=circle_first,
            bucket_definitions=bucket_definitions or {},
        )

    def clique_directional_prompt(
        self,
        bucket_term: str,
        bucket_definition: str,
    ) -> str:
        """Prompt to rewrite bucket terms as short directional attributes and matching definitions."""
        template = self.env.get_template("clique_directional_prompt.jinja2")
        return template.render(
            bucket_term=bucket_term,
            bucket_definition=bucket_definition,
        )

    def generic_elo_prompt(
        self,
        text_circle: str,
        text_square: str,
        attributes: Union[Dict[str, str], List[str]],
        instructions: str = "",
    ) -> str:
        """Prompt to compare two passages across a list of attributes."""
        if isinstance(attributes, list):
            attributes = {a: a for a in attributes}
        template = self.env.get_template("generic_elo_prompt.jinja2")
        return template.render(
            text_circle=text_circle,
            text_square=text_square,
            attributes=attributes,
            instructions=instructions,
        )

    def faceless_prompt(self, text: str, current_map: dict, guidelines: str = "") -> str:
        """Render the prompt for anonymizing personal information within a text block."""
        current_map_json = json.dumps(current_map, indent=4) if current_map else ""
        template = self.env.get_template("faceless_prompt.jinja2")
        return template.render(text=text, current_map=current_map_json, guidelines=guidelines)

    def deconfliction_format_template(self):
        template = self.env.get_template("deconfliction_format.jinja2")
        return template.render()

    def deconfliction_examples(self):
        template = self.env.get_template("deconfliction_examples.jinja2")
        return template.render()

    def deconfliction_prompt(self, terms, context={}, looseness=""):
        format_template = self.deconfliction_format_template()
        examples = self.deconfliction_examples()
        template = self.env.get_template("deconfliction_prompt.jinja2")
        prompt = template.render(
            format_template=format_template,
            examples=examples,
            terms=terms,
            context=context,
            looseness=looseness,
        )
        return prompt

    def deconfliction_best_map_format_template(self):
        template = self.env.get_template("deconfliction_best_map_format.jinja2")
        return template.render()

    def deconfliction_best_map_prompt(self, terms, group_candidates):
        format_template = self.deconfliction_best_map_format_template()
        template = self.env.get_template("deconfliction_best_map_prompt.jinja2")
        prompt = template.render(
            format_template=format_template, terms=terms, group_candidates=group_candidates
        )
        return prompt

    def common_ground_format_template(self):
        template = self.env.get_template("common_ground_format.jinja2")
        return template.render()

    def common_ground_examples(self):
        template = self.env.get_template("common_ground_examples.jinja2")
        return template.render()

    def common_ground_prompt(self, entity_list, entity_name, common_grounding):
        format_template = self.common_ground_format_template()
        examples = self.common_ground_examples()
        template = self.env.get_template("common_ground_prompt.jinja2")
        prompt = template.render(
            format_template=format_template,
            examples=examples,
            entity_list=entity_list,
            entity_name=entity_name,
            common_grounding=common_grounding,
        )
        return prompt

    def rate_attribute_prompt(self, conversation, attributes, attribute_descriptions):
        template = self.env.get_template("rate_attribute_prompt.jinja2")
        prompt = template.render(
            conversation=conversation,
            attributes=attributes,
            attribute_descriptions=attribute_descriptions,
        )
        return prompt

    def task_question_prompt(self, occupation, task, n_queries):
        template = self.env.get_template("task_question_prompt.jinja2")
        prompt = template.render(occupation=occupation, task=task, n_queries=n_queries)
        return prompt

    def task_collaboration_prompt(self, occupation, task, n_queries):
        template = self.env.get_template("task_collaboration_prompt.jinja2")
        prompt = template.render(occupation=occupation, task=task, n_queries=n_queries)
        return prompt

    def task_directive_prompt(self, occupation, task, n_queries):
        template = self.env.get_template("task_directive_prompt.jinja2")
        prompt = template.render(occupation=occupation, task=task, n_queries=n_queries)
        return prompt

    def job_in_history_prompt(self, occupation, year):
        template = self.env.get_template("job_in_history_prompt.jinja2")
        prompt = template.render(occupation=occupation, year=year)
        return prompt

    def wiki_tech_filtering_prompt(self, page_titles):
        examples_template = self.env.get_template("lightweight_wikipedia_tech_format.jinja2")
        examples = examples_template.render()
        template = self.env.get_template("lightweight_wikipedia_tech_prompt.jinja2")
        prompt = template.render(output_format=examples, page_titles=page_titles)
        return prompt

    def tech_not_tech_prompt(self, tech_list):
        examples_template = self.env.get_template("simple_tech_check_format.jinja2")
        examples = examples_template.render()
        template = self.env.get_template("simple_tech_check_prompt.jinja2")
        prompt = template.render(output_format=examples, techs=tech_list)
        return prompt

    def tech_classification_prompt(self, tech_list):
        # Uses the type_of_tech_prompt described in your instructions
        format_template = self.env.get_template("type_of_tech_format.jinja2")
        format_str = format_template.render()
        template = self.env.get_template("type_of_tech_prompt.jinja2")
        prompt = template.render(output_format=format_str, tech_list=tech_list)
        return prompt

    def generic_classification_prompt(
        self,
        entity_list,
        possible_classes,
        class_definitions,
        entity_category,
        output_format="json",
    ):
        # Load the output_format from 'type_of_tech_format.jinja2'
        # since this is the same JSON format pattern we rely on
        format_template = self.env.get_template("type_of_tech_format.jinja2").render()

        # Now load the generic classification prompt
        template = self.env.get_template("generic_classification_prompt.jinja2")
        prompt = template.render(
            entity_list=entity_list,
            possible_classes=possible_classes,
            class_definitions=class_definitions,
            entity_category=entity_category,
            output_format=format_template,  # use the loaded format as the output_format
        )
        return prompt

    def ancestral_tree_of_tech_prompt(self, technology):
        prompt = self.env.get_template("ancestral_tree_of_tech_prompt.jinja2").render(
            technology=technology
        )
        return prompt

    def descendant_tree_of_tech_prompt(self, technology):
        prompt = self.env.get_template("descendant_tree_of_tech_prompt.jinja2").render(
            technology=technology
        )
        return prompt

    def stockname_prompt(self):
        template = self.env.get_template("stockname_prompt.jinja2")
        prompt = template.render()
        return prompt

    def stock_bio_format(self):
        template = self.env.get_template("stock_bio_format.jinja2")
        prompt = template.render()
        return prompt

    def stock_bio_prompt(self, num_stocks):
        format_template = self.stock_bio_format()
        template = self.env.get_template("stock_bio_prompt.jinja2")
        prompt = template.render(num_stocks=num_stocks, format_template=format_template)
        return prompt

    def news_noise_format(self):
        template = self.env.get_template("news_noise_format.jinja2")
        prompt = template.render()
        return prompt

    def news_noise_prompt(self, ticker, stock_bio, prior_news_noise, num_news_noise):
        format_template = self.news_noise_format()
        template = self.env.get_template("news_noise_prompt.jinja2")
        prompt = template.render(
            ticker=ticker,
            stock_bio=stock_bio,
            prior_news_noise=prior_news_noise,
            num_news_noise=num_news_noise,
            format_template=format_template,
        )
        return prompt

    def username_prompt(self):
        template = self.env.get_template("username_prompt.jinja2")
        prompt = template.render()
        return prompt

    def write_posts_format(self):
        template = self.env.get_template("write_posts_format.jinja2")
        prompt = template.render()
        return prompt

    def write_posts_examples(self):
        template = self.env.get_template("write_posts_examples.jinja2")
        prompt = template.render()
        return prompt

    def write_posts_prompt(
        self,
        username,
        salary,
        social_media_following,
        social_media_following_percentile,
        investment_portfolio,
        investment_portfolio_dollars,
        investment_portfolio_percent,
        total_investment_portfolio_value,
        current_prices,
        percent_change_in_price,
        percent_change_in_price_lifetime,
        dollar_growth_in_portfolio,
        dollar_growth_in_portfolio_lifetime,
        percent_growth_in_portfolio,
        percent_growth_in_portfolio_lifetime,
        total_dollar_growth_in_portfolio,
        total_dollar_growth_in_portfolio_lifetime,
        total_percent_growth_in_portfolio,
        total_percent_growth_in_portfolio_lifetime,
        financial_math_problems,
        reddit_post_history,
        reddit_feed,
        num_posts,
        stock_bios,
        news_noise,
    ):
        format_template = self.write_posts_format()
        examples = self.write_posts_examples()
        template = self.env.get_template("write_posts_prompt.jinja2")
        prompt = template.render(
            username=username,
            salary=salary,
            social_media_following=social_media_following,
            social_media_following_percentile=social_media_following_percentile,
            stock_bios=stock_bios,
            news_noise=news_noise,
            investment_portfolio=investment_portfolio,
            investment_portfolio_dollars=investment_portfolio_dollars,
            investment_portfolio_percent=investment_portfolio_percent,
            total_investment_portfolio_value=total_investment_portfolio_value,
            current_prices=current_prices,
            percent_change_in_price=percent_change_in_price,
            percent_change_in_price_lifetime=percent_change_in_price_lifetime,
            dollar_growth_in_portfolio=dollar_growth_in_portfolio,
            dollar_growth_in_portfolio_lifetime=dollar_growth_in_portfolio_lifetime,
            percent_growth_in_portfolio=percent_growth_in_portfolio,
            percent_growth_in_portfolio_lifetime=percent_growth_in_portfolio_lifetime,
            total_dollar_growth_in_portfolio=total_dollar_growth_in_portfolio,
            total_dollar_growth_in_portfolio_lifetime=total_dollar_growth_in_portfolio_lifetime,
            total_percent_growth_in_portfolio=total_percent_growth_in_portfolio,
            total_percent_growth_in_portfolio_lifetime=total_percent_growth_in_portfolio_lifetime,
            financial_math_problems=financial_math_problems,
            reddit_post_history=reddit_post_history,
            reddit_feed=reddit_feed,
            num_posts=num_posts,
            format_template=format_template,
            examples=examples,
        )
        return prompt

    def make_investments_format(self):
        template = self.env.get_template("make_investments_format.jinja2")
        prompt = template.render()
        return prompt

    def make_investments_prompt(
        self,
        username,
        salary,
        social_media_following,
        social_media_following_percentile,
        investment_portfolio,
        investment_portfolio_dollars,
        investment_portfolio_percent,
        total_investment_portfolio_value,
        current_prices,
        percent_change_in_price,
        percent_change_in_price_lifetime,
        dollar_growth_in_portfolio,
        dollar_growth_in_portfolio_lifetime,
        percent_growth_in_portfolio,
        percent_growth_in_portfolio_lifetime,
        total_dollar_growth_in_portfolio,
        total_dollar_growth_in_portfolio_lifetime,
        total_percent_growth_in_portfolio,
        total_percent_growth_in_portfolio_lifetime,
        financial_math_problems,
        reddit_post_history,
        reddit_posts,
        reddit_feed,
        stock_bios,
        news_noise,
    ):
        format_template = self.make_investments_format()
        template = self.env.get_template("make_investments_prompt.jinja2")
        prompt = template.render(
            username=username,
            salary=salary,
            social_media_following=social_media_following,
            social_media_following_percentile=social_media_following_percentile,
            stock_bios=stock_bios,
            news_noise=news_noise,
            investment_portfolio=investment_portfolio,
            investment_portfolio_dollars=investment_portfolio_dollars,
            investment_portfolio_percent=investment_portfolio_percent,
            total_investment_portfolio_value=total_investment_portfolio_value,
            current_prices=current_prices,
            percent_change_in_price=percent_change_in_price,
            percent_change_in_price_lifetime=percent_change_in_price_lifetime,
            dollar_growth_in_portfolio=dollar_growth_in_portfolio,
            dollar_growth_in_portfolio_lifetime=dollar_growth_in_portfolio_lifetime,
            percent_growth_in_portfolio=percent_growth_in_portfolio,
            percent_growth_in_portfolio_lifetime=percent_growth_in_portfolio_lifetime,
            total_dollar_growth_in_portfolio=total_dollar_growth_in_portfolio,
            total_dollar_growth_in_portfolio_lifetime=total_dollar_growth_in_portfolio_lifetime,
            total_percent_growth_in_portfolio=total_percent_growth_in_portfolio,
            total_percent_growth_in_portfolio_lifetime=total_percent_growth_in_portfolio_lifetime,
            financial_math_problems=financial_math_problems,
            reddit_post_history=reddit_post_history,
            reddit_posts=reddit_posts,
            reddit_feed=reddit_feed,
            format_template=format_template,
        )
        return prompt

    def regional_analysis_prompt(
        self, region: str, topic: str, additional_instructions: str = ""
    ) -> str:
        """Prompt for deep-dive regional analysis of a topic."""
        template = self.env.get_template("regional_analysis_prompt.jinja2")
        return template.render(
            region=region, topic=topic, additional_instructions=additional_instructions
        )


teleprompter = Teleprompter(prompt_path="prompts")


client_async = AsyncOpenAI()  # single connection pool per process


# ── internal builder ──────────────────────────────────────────────────
def _build_params(
    *,
    model: str,
    input_data,
    max_tokens: int,
    system_instruction: str,
    temperature: float,
    tools,
    tool_choice,
    web_search: bool,
    search_context_size: str,
    json_mode: bool,
    expected_schema,
    reasoning_effort: str,
    **extra,
):
    params = {
        "model": model,
        "input": input_data,
        "max_output_tokens": max_tokens,
        "truncation": "auto",  # let the API crop if ctx is too long
    }

    # JSON output flavours
    if json_mode:
        params["text"] = (
            {"format": {"type": "json_schema", "schema": expected_schema}}
            if expected_schema
            else {"format": {"type": "json_object"}}
        )

    # Tools / web search
    all_tools = list(tools) if tools else []
    if web_search:
        all_tools.append({"type": "web_search_preview", "search_context_size": search_context_size})
    if all_tools:
        params["tools"] = all_tools
    if tool_choice is not None:
        params["tool_choice"] = tool_choice

    # Reasoning models (“o*”) vs. sampling models
    if model.startswith("o"):
        params["reasoning"] = {
            "effort": reasoning_effort,
        }
    else:
        params["temperature"] = temperature

    params.update(extra)
    return params


# ── get_response ──────────────────────────────────────────────────────
async def get_response(
    prompt: str,
    model: str = "gpt-4.1-mini",
    system_instruction: str = (
        "Please provide a helpful response to this inquiry for purposes of academic research."
    ),
    n: int = 1,
    max_tokens: int = 25_000,  #  ⬅ default raised
    temperature: float = 0.9,
    json_mode: bool = False,
    expected_schema: Optional[Dict[str, Any]] = None,
    timeout: float = 90.0,
    tools: Optional[List[dict]] = None,
    tool_choice: Optional[dict] = None,
    web_search: bool = False,
    search_context_size: str = "medium",
    reasoning_effort: str = "medium",
    **kwargs,
) -> Tuple[List[str], float]:
    """
    Async wrapper around /responses. Returns (list_of_outputs, elapsed_sec).
    """

    if json_mode:
        system_instruction += " Output the response in JSON format."

    # minimal roles for reasoning models
    input_data = (
        [{"role": "user", "content": prompt}]
        if model.startswith("o")
        else [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": prompt},
        ]
    )

    if reasoning_effort == "high":
        timeout = timeout * 3

    params = _build_params(
        model=model,
        input_data=input_data,
        max_tokens=max_tokens,
        system_instruction=system_instruction,
        temperature=temperature,
        tools=tools,
        tool_choice=tool_choice,
        web_search=web_search,
        search_context_size=search_context_size,
        json_mode=json_mode,
        expected_schema=expected_schema,
        reasoning_effort=reasoning_effort,
        **kwargs,
    )

    start = time.time()
    tasks = [client_async.responses.create(**params, timeout=timeout) for _ in range(max(n, 1))]

    try:
        raw = await asyncio.gather(*tasks)
    except asyncio.TimeoutError:
        raise Exception(f"API call timed out after {timeout} s")
    except Exception as e:
        raise Exception(f"API call resulted in exception: {e}")

    return [r.output_text for r in raw], time.time() - start


import json


def _ser(x):
    """Any Python obj → *deterministic* JSON string (or None)."""
    return None if x is None else json.dumps(x, ensure_ascii=False)


def _de(x):
    """JSON string → original Python value (or None)."""
    if pd.isna(x):
        return None
    try:
        return json.loads(x)
    except Exception:
        # legacy rows: try ast.literal_eval
        import ast

        try:
            return ast.literal_eval(x)
        except Exception:
            return None


# ── get_all_responses  (progress bar + retries) ───────────────────────
async def get_all_responses(
    prompts: List[str],
    identifiers: Optional[List[str]] = None,
    n_parallels: int = 100,
    save_path: str = "temp.csv",
    reset_files: bool = False,
    n: int = 1,
    max_tokens: int = 25_000,
    requests_per_minute: int = 40_000,
    tokens_per_minute: int = 15_000_000_000,
    rate_limit_factor: float = 0.8,
    timeout: int = 90,
    max_retries: int = 7,
    save_every_x_responses: int = 1_000,
    save_every_x_seconds: Optional[int] = None,
    use_dummy: bool = False,
    print_example_prompt: bool = False,
    use_web_search: bool = False,
    search_context_size: str = "medium",
    tools: Optional[List[dict]] = None,
    tool_choice: Optional[dict] = None,
    **get_response_kwargs,
) -> pd.DataFrame:
    if identifiers is None:
        identifiers = prompts

    model = get_response_kwargs.get("model", "o4-mini")
    print(f"Model used: {model}")

    # propagate flags
    get_response_kwargs.setdefault("web_search", use_web_search)
    get_response_kwargs.setdefault("search_context_size", search_context_size)
    get_response_kwargs.setdefault("tools", tools)
    get_response_kwargs.setdefault("tool_choice", tool_choice)

    # resume file
    if os.path.exists(save_path) and not reset_files:
        df = pd.read_csv(save_path)
        df["Response"] = df["Response"].apply(_de)
        done = set(df["Identifier"])
    else:
        df = pd.DataFrame(columns=["Identifier", "Response", "Time Taken"])
        done = set()

    todo = [(p, i) for p, i in zip(prompts, identifiers) if i not in done]
    total = len(todo)
    if total == 0:
        print("No new prompts to process.")
        return df

    if print_example_prompt:
        print(f"Example prompt: {todo[0][0]}\n")

    # dummy shortcut
    if use_dummy:
        rows = [
            {"Identifier": i, "Response": _ser([f"DUMMY {i}"] * n), "Time Taken": 0.0}
            for _, i in todo
        ]
        df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
        df.to_csv(save_path, index=False)
        return df

    # rate limiters
    req_lim = AsyncLimiter(int(requests_per_minute * rate_limit_factor), 60)
    tok_lim = AsyncLimiter(int(tokens_per_minute * rate_limit_factor), 60)

    queue = asyncio.Queue()
    for item in todo:
        queue.put_nowait(item)

    results, processed = [], 0
    pbar = tqdm(total=total, desc="Processing prompts", leave=True)

    async def flush():
        nonlocal results
        if results:
            batch = pd.DataFrame(results)
            batch["Response"] = batch["Response"].apply(_ser)
            batch.to_csv(
                save_path,
                mode="a",
                header=not os.path.exists(save_path),
                index=False,
                quoting=csv.QUOTE_MINIMAL,
            )
            results = []

    async def worker():
        nonlocal processed
        while True:
            try:
                prompt, ident = await queue.get()
            except asyncio.CancelledError:
                break

            attempt = 1
            while attempt <= max_retries:
                try:
                    # rough token estimate; API auto-truncates anyway
                    approx = int(len(prompt.split()) * 1.5)
                    await req_lim.acquire()
                    await tok_lim.acquire((approx + max_tokens) * n)

                    resps, t = await asyncio.wait_for(
                        get_response(
                            prompt,
                            n=n,
                            max_tokens=max_tokens,
                            timeout=timeout,
                            **get_response_kwargs,
                        ),
                        timeout=timeout,
                    )

                    results.append({"Identifier": ident, "Response": _ser(resps), "Time Taken": t})
                    processed += 1
                    pbar.update(1)
                    if processed % save_every_x_responses == 0:
                        await flush()
                    break
                except Exception:
                    if attempt >= max_retries:
                        results.append({"Identifier": ident, "Response": None, "Time Taken": None})
                        processed += 1
                        pbar.update(1)
                        await flush()
                        break
                    await asyncio.sleep(5 * attempt)
                    attempt += 1
            queue.task_done()

    workers = [asyncio.create_task(worker()) for _ in range(n_parallels)]
    if save_every_x_seconds:

        async def periodic():
            while True:
                await asyncio.sleep(save_every_x_seconds)
                await flush()

        ticker = asyncio.create_task(periodic())

    await queue.join()
    for w in workers:
        w.cancel()
    await flush()
    if save_every_x_seconds:
        ticker.cancel()
    pbar.close()

    print(f"All done – saved to {save_path}")
    return pd.read_csv(save_path).assign(Response=lambda d: d.Response.apply(_de))


async def get_embedding(text, model="text-embedding-3-small", api_key=api_key, timeout=15):
    """
    Asynchronously fetch an embedding for the given text using the specified model.
    """
    url = "https://api.openai.com/v1/embeddings"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": model, "input": text}

    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=payload, timeout=timeout) as response:
            if response.status != 200:
                raise Exception(f"Failed to get embedding: {await response.text()}")
            data = await response.json()
            return data["data"][0]["embedding"]  # Assuming single embedding returned


async def get_all_embeddings(
    texts,
    identifiers=None,
    model="text-embedding-3-small",
    n_parallels=400,
    save_path="embeddings.pkl",
    timeout=15,
    save_every_x=10000,
):
    """
    Asynchronously fetch embeddings for a list of texts, associating each with an identifier,
    and saving results to a file periodically.
    """
    embeddings = {}
    # Load existing embeddings if available
    if os.path.exists(save_path):
        with open(save_path, "rb") as f:
            embeddings = pickle.load(f)

    # If identifiers not provided, use texts as identifiers
    if identifiers is None:
        identifiers = texts

    # Pair up identifiers and texts, filter out those already processed
    items_to_process = [(id, text) for id, text in zip(identifiers, texts) if id not in embeddings]
    total_items = len(items_to_process)
    if total_items == 0:
        print("All embeddings already obtained.")
        return embeddings

    queue = asyncio.Queue()
    for item in items_to_process:
        queue.put_nowait(item)

    processed = 0

    async def worker():
        nonlocal processed
        while not queue.empty():
            identifier, text = await queue.get()
            try:
                embed = await get_embedding(text, model=model, timeout=timeout)
                embeddings[identifier] = embed
            except Exception as e:
                print(f"Error embedding text for {identifier}: {e}")
            processed += 1
            if processed % save_every_x == 0:
                with open(save_path, "wb") as f:
                    pickle.dump(embeddings, f)
            queue.task_done()

    tasks = [asyncio.create_task(worker()) for _ in range(n_parallels)]
    pbar = tqdm(total=total_items, desc="Getting embeddings", leave=True)

    # Periodically update progress bar until workers are done
    while any(not t.done() for t in tasks):
        pbar.update(processed - pbar.n)
        await asyncio.sleep(1)
    pbar.update(processed - pbar.n)
    pbar.close()

    await queue.join()
    for t in tasks:
        t.cancel()

    # Final save
    with open(save_path, "wb") as f:
        pickle.dump(embeddings, f)

    return embeddings


import textwrap

import matplotlib.pyplot as plt
import numpy as np


def create_bar_chart(
    categories,
    values,
    title="Classification of ChatGPT Conversations",
    x_label="Classification",
    y_label="Number of Conversations",
    is_percentage=False,
    gradient_cmap="Reds",
    gradient_start=0.3,
    gradient_end=1.0,
    background_color="#ffffff",  # soft sepia "#f4ecd8"
    font_family="monospace",
    figsize=(16, 7),
    dpi=400,
    label_font_size=12,
    title_font_size=14,
    wrap_width=16,
    rotate_xlabels=False,
    x_label_font_size=12,
    annotation_font_size=10,
    annotation_fontweight="bold",
    precision=3,  # New parameter: number of significant digits to keep
):
    """
    Creates a bar chart with options for:
      - Replacing "none" with "other"
      - Percentage vs. raw count annotation
      - Custom color gradient, background color, fonts, etc.
      - Wrapping and optionally rotating x labels
      - Rounding numbers to a specified precision (default: 3 significant digits)
    """

    # 1. Replace literal "none" with "other"
    updated_categories = []
    for cat in categories:
        if cat.strip().lower() == "none":
            updated_categories.append("other")
        else:
            updated_categories.append(cat)

    # 2. Format function for bar annotations using the specified precision
    def format_value(val):
        if is_percentage:
            # e.g., 98.23 -> "98.2%"
            return f"{val:.{precision}g}%"
        else:
            if val >= 1_000_000:
                # e.g., 1085794 -> "1.09M"
                return f"{val / 1_000_000:.{precision}g}M"
            elif val >= 1_000:
                # e.g., 124048 -> "124K"
                return f"{val / 1_000:.{precision}g}K"
            else:
                # e.g., 5.438 -> "5.44" or 500 -> "500"
                return f"{val:.{precision}g}"

    # 3. Wrap category labels
    wrapped_labels = [textwrap.fill(cat, width=wrap_width) for cat in updated_categories]

    # 4. Prepare color gradient
    colors = plt.cm.get_cmap(gradient_cmap)(np.linspace(gradient_start, gradient_end, len(values)))

    # 5. Set up figure and style
    plt.style.use("default")
    plt.rcParams["font.family"] = font_family
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_facecolor(background_color)
    fig.patch.set_facecolor(background_color)

    # 6. Create bars
    bars = ax.bar(wrapped_labels, values, color=colors, edgecolor="black")

    # 7. Annotate bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.annotate(
            format_value(val),
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=annotation_font_size,
            fontweight=annotation_fontweight,
        )

    # 8. Axis labels and title
    ax.set_title(textwrap.fill(title, width=100), fontsize=title_font_size, fontweight="bold")
    ax.set_xlabel(x_label, fontsize=label_font_size, fontweight="bold")
    ax.set_ylabel(y_label, fontsize=label_font_size, fontweight="bold")

    # 9. X-axis label rotation
    if rotate_xlabels:
        plt.xticks(rotation=45, ha="right")
    else:
        plt.xticks(rotation=0, ha="center")

    ax.tick_params(axis="x", labelsize=x_label_font_size)

    # 10. Adjust layout
    plt.tight_layout()
    plt.show()


def create_county_choropleth(
    df: pd.DataFrame,
    fips_col: str,
    value_col: str,
    title: str = "County Ratings",
    color_scale: str = "RdBu",
    font_family: str = "monospace",
    save_path: Optional[str] = None,
    county_col: str = None,  # Optionally pass county name column
    z_score: bool = False,   # If True, plot z-scores instead of raw values
):
    """
    Create a county-level choropleth map using Plotly.
    - Ensures FIPS codes are zero-padded to 5 digits (string type).
    - Optionally converts values to z-scores (if z_score=True).
    - Uses red-to-blue color scale (or best diverging scale for z-scores).
    - Adds county name and FIPS to hover/HTML map render.

    Args:
        df: DataFrame with county data.
        fips_col: Column for FIPS code (must be 5-digit string).
        value_col: Column to plot.
        title: Title for the map.
        color_scale: Plotly color scale to use (default 'RdBu').
        font_family: Font family for the map.
        save_path: If provided, saves map to this path.
        county_col: Optionally, column for county name.
        z_score: If True, plot z-scores instead of raw values (default: False).
    """
    import json
    import plotly.express as px
    import requests
    import numpy as np
    from scipy.stats import zscore

    geojson_url = (
        "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"
    )
    cache_path = os.path.join(os.path.expanduser("~"), ".cache", "county_geo.json")
    if not os.path.exists(cache_path):
        try:
            r = requests.get(geojson_url, timeout=30)
            r.raise_for_status()
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, "w") as f:
                f.write(r.text)
        except Exception as e:
            raise RuntimeError(f"Failed to download county geojson: {e}")
    with open(cache_path) as f:
        counties = json.load(f)

    # --- FIPS HANDLING ---
    # Ensure FIPS is string and zero-padded to 5 digits
    df = df.copy()
    df[fips_col] = df[fips_col].astype(str).str.zfill(5)

    # Try to get county name column
    if county_col is None:
        # Try common names
        for name_col in ["county", "County", "region", "Region"]:
            if name_col in df.columns:
                county_col = name_col
                break
    # If not found, just use blank
    if county_col is None:
        df["_county_name"] = ""
        county_col = "_county_name"

    # --- Z-SCORE NORMALIZATION ---
    plot_col = value_col
    if z_score:
        # Compute z-scores, handle nan/constant cases robustly
        vals = df[value_col].values.astype(float)
        # Only compute zscore if >1 value and not all the same
        if len(vals) > 1 and np.nanstd(vals) > 0:
            zs = zscore(vals, nan_policy="omit")
        else:
            zs = np.zeros_like(vals)
        plot_col = f"_zscore_{value_col}"
        df[plot_col] = zs
        # Use a diverging color scale suitable for z-scores
        color_scale = "RdBu" if color_scale == "RdBu" else "PuOr"
    # --- HOVER DATA ---
    hover_data = {county_col: True, fips_col: True, value_col: True}

    fig = px.choropleth(
        df,
        geojson=counties,
        locations=fips_col,
        color=plot_col,
        color_continuous_scale=color_scale,
        scope="usa",
        hover_data=hover_data,
    )
    # Center colorbar at zero for z-scores
    if z_score:
        fig.update_coloraxes(cmid=0)

    import textwrap
    # --- CLEAN COLORBAR LABEL ---
    # Use the original topic/column name, prettified
    def prettify_label(label, zscore=False):
        label = label.replace('_', ' ')
        label = label.strip()
        if zscore:
            label += ' (z score)'
        # Wrap after ~40 chars
        return '<br>'.join(textwrap.wrap(label, width=40))

    colorbar_label = prettify_label(value_col, zscore=z_score)

    # --- CENTERED, WRAPPED TITLE ---
    pretty_title = textwrap.fill(title, width=70)
    fig.update_layout(
        title={
            'text': pretty_title,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 20, 'family': font_family}
        },
        font_family=font_family,
        margin=dict(l=20, r=20, t=60, b=20),
        coloraxis_colorbar={
            'title': colorbar_label,
            'title_side': 'right',
            'tickfont': {'size': 12, 'family': font_family},
            
            'lenmode': 'pixels',
            'len': 320
        },
        plot_bgcolor='white',
        paper_bgcolor='white',
    )
    fig.update_traces(
        hovertemplate=f"County: %{{customdata[0]}}<br>FIPS: %{{location}}<br>{prettify_label(value_col, zscore=z_score)}: %{{z}}<extra></extra>"
    )
    if save_path:
        ext = os.path.splitext(save_path)[1].lower()
        try:
            if ext in {".png", ".jpg", ".jpeg"}:
                fig.write_image(save_path, scale=3)
            else:
                fig.write_html(save_path)
        except Exception:
            fig.write_html(save_path if ext.endswith(".html") else save_path + ".html")
    return fig

