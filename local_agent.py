import torch
import os
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM

logger = logging.getLogger(__name__)


class LLMAgent:
    """LLM agent that uses language models directly on the local system with model sharing support"""

    def __init__(
        self,
        model_name_or_instance,
        device="cuda",
        precision="bfloat16",
        tokenizer=None,
        shared_model=False,
    ):
        """
        Initialize the LLM agent with either a model name or pre-loaded model

        Args:
            model_name_or_instance: Either model identifier (string) or pre-loaded model instance
            device: Device to use (cuda or cpu)
            precision: Model precision (bfloat16, float16, or float32)
            tokenizer: Pre-loaded tokenizer (optional)
            shared_model: Whether this is using a shared model instance
        """

        # Handle pre-loaded model instance
        if hasattr(model_name_or_instance, "generate") and hasattr(
            model_name_or_instance, "config"
        ):
            self.model = model_name_or_instance
            self.model_name = getattr(
                model_name_or_instance, "name_or_path", "shared_model"
            )
            self.shared_model = True

            if tokenizer is not None:
                self.tokenizer = tokenizer
            else:
                model_name = getattr(
                    model_name_or_instance, "name_or_path", model_name_or_instance
                )
                token = os.environ.get("HUGGING_FACE_HUB_TOKEN")
                if token:
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        model_name, use_auth_token=token
                    )
                else:
                    self.tokenizer = AutoTokenizer.from_pretrained(model_name)

            self.device = next(self.model.parameters()).device
            return

        # Load new model instance
        model_name = model_name_or_instance
        self.model_name = model_name
        self.shared_model = shared_model

        token = os.environ.get("HUGGING_FACE_HUB_TOKEN")

        # Load tokenizer
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            if token:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_name, use_auth_token=token
                )
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Determine torch dtype
        if precision == "bfloat16" and torch.cuda.is_bf16_supported():
            torch_dtype = torch.bfloat16
        elif precision == "float16":
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32

        # Load model
        model_kwargs = {
            "torch_dtype": torch_dtype,
            "device_map": "auto",
            "trust_remote_code": True,
        }

        if token:
            model_kwargs["use_auth_token"] = token

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

        self.device = (
            "cuda" if torch.cuda.is_available() and device == "cuda" else "cpu"
        )

    def generate(self, prompt, max_new_tokens=1024):
        """Generate text using the local model with proper chat formatting"""
        # Format as chat messages using the tokenizer's chat template
        messages = [{"role": "user", "content": prompt}]

        formatted_prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        model_inputs = self.tokenizer([formatted_prompt], return_tensors="pt").to(
            self.device
        )

        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Extract only newly generated tokens
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[
            0
        ]

        # Fallback for empty responses
        if not response or response.strip() == "":
            response = "I don't have enough information to provide a specific answer."

        return response

    def get_log_probs(self, prompt, target_tokens=["Yes", "No"]):
        """Calculate log probabilities for specific tokens"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Get logits for the last token position
        logits = outputs.logits[0, -1, :]

        # Get token IDs for target tokens
        target_ids = []
        for token in target_tokens:
            token_ids = self.tokenizer.encode(" " + token, add_special_tokens=False)
            target_ids.append(
                token_ids[0] if token_ids else self.tokenizer.unk_token_id
            )

        # Calculate log probabilities
        log_probs = torch.log_softmax(logits, dim=0)
        target_log_probs = {
            token: log_probs[tid].item()
            for token, tid in zip(target_tokens, target_ids)
        }

        return target_log_probs

    def batch_process(self, prompts, generate=True, max_new_tokens=256):
        """Process a batch of prompts"""
        if not prompts:
            return []

        results = []
        for prompt in prompts:
            if generate:
                results.append(self.generate(prompt, max_new_tokens))
            else:
                results.append(self.get_log_probs(prompt, ["Yes", "No"]))

        return results


class SharedModelManager:
    """Manager class for sharing model instances across multiple agents"""

    def __init__(self, model_name, device="cuda", precision="bfloat16"):
        """Initialize and load the shared model and tokenizer"""
        self.model_name = model_name
        self.device = device
        self.precision = precision

        token = os.environ.get("HUGGING_FACE_HUB_TOKEN")

        # Load tokenizer
        if token:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, use_auth_token=token
            )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Determine torch dtype
        if precision == "bfloat16" and torch.cuda.is_bf16_supported():
            torch_dtype = torch.bfloat16
        elif precision == "float16":
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32

        # Load model
        model_kwargs = {
            "torch_dtype": torch_dtype,
            "device_map": "auto",
            "trust_remote_code": True,
        }

        if token:
            model_kwargs["use_auth_token"] = token

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    def create_agent(self):
        """Create a new LLMAgent using the shared model instance"""
        return LLMAgent(
            model_name_or_instance=self.model,
            tokenizer=self.tokenizer,
            shared_model=True,
        )
