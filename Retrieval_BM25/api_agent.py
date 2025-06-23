import requests
import time
import torch


class FalconAgent:
    """
    LLM agent that uses Falcon3-10B-Instruct through the AI71 API.

    This agent implements the same interface as LLMAgent but uses the
    AI71 API to access the Falcon model instead of loading it locally.
    """

    def __init__(self, api_key):
        """
        Initialize the Falcon agent with an API key.

        Args:
            api_key: AI71 API key for accessing the Falcon model
        """
        self.api_key = api_key
        self.api_url = "https://api.ai71.ai/v1/models/falcon-3-10b-instruct/completions"
        print("FalconAgent initialized (using AI71 API)")


def generate(self, prompt, max_new_tokens=256):
    """
    Generate text using the Falcon model via the AI71 API.

    Args:
        prompt: The input prompt
        max_new_tokens: Maximum number of tokens to generate

    Returns:
        Generated text as a string
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {self.api_key}",
    }

    # Format as a chat message compatible with Falcon API
    # The API may have a different format than the local model
    # so we'll need to check their documentation

    # Option 1: Try formatting as messages
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_new_tokens,
        "temperature": 0.0,  # Use greedy decoding as in MAIN-RAG
    }

    # Add retry logic for robustness
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # First try with messages format
            response = requests.post(self.api_url, headers=headers, json=payload)

            # If that fails, fall back to prompt format
            if response.status_code != 200:
                print("API doesn't support messages format, trying prompt format")
                # Format the prompt as a chat message
                payload = {
                    "prompt": f"User: {prompt}\nAssistant:",
                    "max_tokens": max_new_tokens,
                    "temperature": 0.0,
                }
                response = requests.post(self.api_url, headers=headers, json=payload)

            response.raise_for_status()  # Raise exception for HTTP errors

            text_response = response.json()["choices"][0]["text"]

            # Check if the response is empty or just contains formatting tokens
            if not text_response or text_response.strip() in ["", "<|assistant|>"]:
                return "I don't have enough information to provide a specific answer."

            return text_response
        except Exception as e:
            if attempt == max_retries - 1:
                raise Exception(
                    f"Failed to generate text after {max_retries} attempts: {e}"
                )
            wait_time = 2**attempt + 1  # Exponential backoff
            print(f"API call failed, retrying in {wait_time}s... ({str(e)})")
            time.sleep(wait_time)

    def get_log_probs(self, prompt, target_tokens=["Yes", "No"]):
        """
        Calculate log probabilities for target tokens.

        Note: The AI71 API may not provide direct access to token probabilities.
        This method approximates them with a heuristic approach.

        Args:
            prompt: The input prompt
            target_tokens: List of tokens to get probabilities for

        Returns:
            Dictionary mapping tokens to their log probabilities
        """
        # We'll use generations with biased prompts to approximate probabilities
        scores = {}

        for token in target_tokens:
            # Create a prompt that guides toward the target token
            biased_prompt = f"{prompt}\n\nBased on the above information, I should answer '{token}'."

            try:
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                }

                payload = {
                    "prompt": biased_prompt,
                    "max_tokens": 10,
                    "temperature": 0.1,  # Low temperature to maximize confidence
                }

                response = requests.post(self.api_url, headers=headers, json=payload)
                response.raise_for_status()

                generation = response.json()["choices"][0]["text"].strip()

                # Score based on whether the generation starts with the token
                if generation.startswith(token):
                    scores[token] = 0.0  # log(1.0)
                else:
                    scores[token] = -1.0  # log(0.368)
            except Exception as e:
                print(f"Error getting log probs for token '{token}': {e}")
                # Default to a low probability on failure
                scores[token] = -2.0  # log(0.135)

        return scores

    def batch_process(self, prompts, generate=True, max_new_tokens=256):
        """
        Process a batch of prompts.

        Args:
            prompts: List of prompt strings
            generate: If True, generate text; if False, return log probs for Yes/No
            max_new_tokens: Maximum new tokens for generation

        Returns:
            List of responses or log probs
        """
        # This is not an efficient implementation as we'll just call the API
        # for each prompt sequentially, but it maintains the same interface
        results = []

        for prompt in prompts:
            if generate:
                results.append(self.generate(prompt, max_new_tokens))
            else:
                results.append(self.get_log_probs(prompt, ["Yes", "No"]))

        return results
