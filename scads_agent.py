"""
SCADS AI Agent - Uses ScaDS.AI OpenAI-compatible API for LLM inference.

ScaDS.AI (Center for Scalable Data Analytics and Artificial Intelligence)
provides an OpenAI-compatible endpoint.

Set the environment variable:
    SCADS_API_KEY=<your_api_key>

Optionally override the base URL:
    SCADS_API_BASE=https://llm.scads.ai/v1   (default)
"""

import os
import time
import math
import logging

from openai import OpenAI

logger = logging.getLogger(__name__)

# Default values – can be overridden via environment variables or constructor args
_DEFAULT_BASE_URL = "https://llm.scads.ai/v1"
_DEFAULT_MODEL = "meta-llama/Llama-4-Scout-17B-16E-Instruct"


class ScadsAgent:
    """
    LLM agent that uses the ScaDS.AI OpenAI-compatible API.

    Provides the same interface as LLMAgent (local_agent.py) and FalconAgent
    (api_agent.py), so it can be used as a drop-in replacement.
    """

    def __init__(
        self,
        api_key: str = None,
        model: str = None,
        base_url: str = None,
        max_retries: int = 3,
    ):
        """
        Initialize the ScaDS.AI agent.

        Args:
            api_key:     SCADS AI API key. Falls back to SCADS_API_KEY env var.
            model:       Model identifier (default: meta-llama/Llama-3.3-70B-Instruct).
            base_url:    API base URL. Falls back to SCADS_API_BASE env var,
                         then to https://llm.scads.ai/v1.
            max_retries: Number of retry attempts on transient errors.
        """
        self.api_key = api_key or os.environ.get("SCADS_API_KEY")
        if not self.api_key:
            raise ValueError(
                "SCADS AI API key not provided. "
                "Pass api_key= or set the SCADS_API_KEY environment variable."
            )

        self.base_url = (
            base_url
            or os.environ.get("SCADS_API_BASE")
            or _DEFAULT_BASE_URL
        )
        self.model = model or os.environ.get("SCADS_MODEL") or _DEFAULT_MODEL
        self.max_retries = max_retries

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        logger.info(
            f"ScadsAgent initialized: model={self.model}, base_url={self.base_url}"
        )

    # ------------------------------------------------------------------
    # Public interface (mirrors LLMAgent / FalconAgent)
    # ------------------------------------------------------------------

    def generate(self, prompt: str, max_new_tokens: int = 1024) -> str:
        """
        Generate a response from the ScaDS.AI LLM.

        Args:
            prompt:         User prompt string.
            max_new_tokens: Maximum tokens to generate.

        Returns:
            Generated text as a string.
        """
        messages = [{"role": "user", "content": prompt}]

        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=max_new_tokens,
                    temperature=0.0,
                )
                text = response.choices[0].message.content or ""
                if not text.strip():
                    return "I don't have enough information to provide a specific answer."
                return text

            except Exception as exc:
                if attempt == self.max_retries - 1:
                    raise RuntimeError(
                        f"ScadsAgent.generate failed after {self.max_retries} attempts: {exc}"
                    ) from exc
                wait = 2 ** attempt + 1
                logger.warning(
                    f"ScadsAgent.generate attempt {attempt + 1} failed: {exc}. "
                    f"Retrying in {wait}s…"
                )
                time.sleep(wait)

    def get_log_probs(
        self, prompt: str, target_tokens: list = None
    ) -> dict:
        """
        Approximate log-probabilities for target tokens.

        The ScaDS.AI endpoint may support the ``logprobs`` parameter for
        chat completions.  If it does, we use the actual token log-probs.
        Otherwise we fall back to a text-based heuristic: the model is asked
        to respond with one of the target tokens and we score based on whether
        the first word of the reply matches.

        Args:
            prompt:        The evaluation prompt.
            target_tokens: Tokens to score, default ["Yes", "No"].

        Returns:
            Dict mapping each token to an estimated log-probability (float).
        """
        if target_tokens is None:
            target_tokens = ["Yes", "No"]

        # Strategy 1: Try native logprobs (OpenAI-compatible endpoints often
        # expose this via logprobs=True + top_logprobs).
        try:
            scores = self._get_log_probs_native(prompt, target_tokens)
            if scores is not None:
                return scores
        except Exception as exc:
            logger.debug(f"Native logprobs not available: {exc}")

        # Strategy 2: Text-based heuristic fallback.
        return self._get_log_probs_heuristic(prompt, target_tokens)

    def batch_process(
        self, prompts: list, generate: bool = True, max_new_tokens: int = 256
    ) -> list:
        """
        Process a list of prompts sequentially.

        Args:
            prompts:        List of prompt strings.
            generate:       If True, call generate(); else call get_log_probs().
            max_new_tokens: Max tokens per generation call.

        Returns:
            List of results.
        """
        results = []
        for p in prompts:
            if generate:
                results.append(self.generate(p, max_new_tokens))
            else:
                results.append(self.get_log_probs(p))
        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_log_probs_native(self, prompt: str, target_tokens: list) -> dict | None:
        """
        Attempt to retrieve token log-probs via the API's logprobs parameter.
        Returns None if the endpoint does not support it.
        """
        messages = [{"role": "user", "content": prompt}]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=1,
            temperature=0.0,
            logprobs=True,
            top_logprobs=min(20, len(target_tokens) + 5),
        )

        choice = response.choices[0]
        # Parse top_logprobs from the first generated token
        if (
            choice.logprobs is None
            or not choice.logprobs.content
        ):
            return None

        first_token_info = choice.logprobs.content[0]
        top_lp = {
            lp.token.strip(): lp.logprob
            for lp in (first_token_info.top_logprobs or [])
        }

        scores = {}
        for tok in target_tokens:
            # Try exact match, then lowercase
            if tok in top_lp:
                scores[tok] = top_lp[tok]
            elif tok.lower() in top_lp:
                scores[tok] = top_lp[tok.lower()]
            else:
                # Token not in top-k → assign a very low probability
                scores[tok] = math.log(1e-6)

        return scores

    def _get_log_probs_heuristic(self, prompt: str, target_tokens: list) -> dict:
        """
        Text-based fallback: ask the model to answer with the target token.
        Score is log(1.0) if the reply starts with that token, log(0.1) otherwise.
        """
        # Ask model to answer with exactly one of the target tokens
        token_list = " or ".join(f'"{t}"' for t in target_tokens)
        constrained_prompt = (
            f"{prompt}\n\n"
            f"Answer with exactly one word — {token_list}. "
            f"Do not add punctuation or explanation."
        )

        try:
            reply = self.generate(constrained_prompt, max_new_tokens=5).strip()
        except Exception:
            reply = ""

        first_word = reply.split()[0].strip(".,!?") if reply else ""

        scores = {}
        for tok in target_tokens:
            if first_word.lower() == tok.lower():
                scores[tok] = 0.0          # log(1.0)
            else:
                scores[tok] = math.log(0.1)  # log(0.1) ≈ -2.3
        return scores
