#!/usr/bin/env python3
"""
ArXiv Dataset Generator using University Llama API
Modified to use your university's LLM instead of OpenAI for dataset generation
"""

import os
import json
import plyvel
import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Union
import tempfile
import re
import random
from openai import OpenAI
from pydantic import BaseModel

from deepeval.synthesizer import Synthesizer
from deepeval.synthesizer.config import (
    StylingConfig,
    FiltrationConfig,
    EvolutionConfig,
    ContextConstructionConfig,
)
from deepeval.synthesizer.types import Evolution
from deepeval.dataset import EvaluationDataset
from deepeval.models.base_model import DeepEvalBaseLLM
import logging
from deepeval.models.base_model import DeepEvalBaseEmbeddingModel
from sentence_transformers import SentenceTransformer
from typing import List

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class HFEmbeddingModel(DeepEvalBaseEmbeddingModel):
    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        self.model_name = model_name
        self.model = None

        print(f"Initializing HFEmbeddingModel with {model_name}")

        # Load model directly instead of relying on parent class
        self._load_model_directly()

        # Call parent init after our model is loaded
        super().__init__()

        # Verify model is working
        self._test_model()

    def _load_model_directly(self):
        """Load the model directly with proper error handling"""
        try:
            print(f"Loading SentenceTransformer: {self.model_name}")

            # Import here to ensure proper loading
            from sentence_transformers import SentenceTransformer

            # Load with explicit cache handling
            self.model = SentenceTransformer(
                self.model_name,
                cache_folder=None,  # Use default cache
                trust_remote_code=False,
            )

            print(f"Model loaded successfully")
            print(f"   Model type: {type(self.model)}")
            print(f"   Model device: {self.model.device}")
            print(
                f"   Model modules: {len(self.model._modules) if self.model._modules else 0}"
            )

            return self.model

        except Exception as e:
            print(f"Error loading model: {e}")
            print(f"   Model name: {self.model_name}")
            import traceback

            traceback.print_exc()
            raise

    def _test_model(self):
        """Test that the model works with a simple embedding"""
        try:
            print("Testing model with sample text...")
            test_embedding = self.model.encode(
                "This is a test sentence.", show_progress_bar=False
            )
            print(f"Model test successful. Embedding shape: {test_embedding.shape}")
        except Exception as e:
            print(f"Model test failed: {e}")
            raise RuntimeError(
                f"Model {self.model_name} failed functionality test: {e}"
            )

    def load_model(self):
        """Return the already-loaded model (required by DeepEval)"""
        if self.model is None:
            print("Model is None in load_model(), loading now...")
            self._load_model_directly()
        return self.model

    def get_model_name(self) -> str:
        return f"HF–{self.model_name}"

    def embed_text(self, text: str) -> List[float]:
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        try:
            return self.model.encode(text, show_progress_bar=False).tolist()
        except Exception as e:
            print(f"Error in embed_text: {e}")
            print(f"   Text length: {len(text)}")
            print(f"   Model state: {self.model is not None}")
            raise

    async def a_embed_text(self, text: str) -> List[float]:
        return self.embed_text(text)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        if not texts:
            return []

        try:
            print(f"Embedding {len(texts)} texts...")
            embeddings = self.model.encode(
                texts,
                batch_size=32,
                show_progress_bar=True,
                convert_to_tensor=False,  # Ensure numpy arrays
                normalize_embeddings=False,
            )
            result = embeddings.tolist()
            print(f"Successfully embedded {len(texts)} texts")
            return result

        except Exception as e:
            print(f"Error in embed_texts: {e}")
            print(f"   Number of texts: {len(texts)}")
            print(
                f"   First text preview: {texts[0][:100]}..." if texts else "No texts"
            )
            print(f"   Model state: {self.model is not None}")
            import traceback

            traceback.print_exc()
            raise

    async def a_embed_texts(self, texts: List[str]) -> List[List[float]]:
        return self.embed_texts(texts)


class UniversityLLMForGeneration(DeepEvalBaseLLM):
    """
    University LLM adapted for DeepEval dataset generation
    Based on your existing UniversityLLM but optimized for synthesis tasks
    """

    def __init__(self, api_key=None):
        """Initialize University LLM for dataset generation"""
        # Load API key (same as your existing code)
        if not api_key:
            path_to_key = os.path.join(os.path.expanduser("~"), ".scadsai-api-key")
            if os.path.exists(path_to_key):
                with open(path_to_key) as keyfile:
                    api_key = keyfile.readline().strip()

        if not api_key:
            raise ValueError("No API key provided or found in ~/.scadsai-api-key")

        # Initialize OpenAI client with university endpoint
        self.client = OpenAI(base_url="https://llm.scads.ai/v1", api_key=api_key)

        # Find available model
        self.model_name = self._find_model()
        logger.info(f"Using generation model: {self.model_name}")

    def _find_model(self):
        """Find the best available model for dataset generation"""
        try:
            models = self.client.models.list()
            # Prefer larger Llama models for generation quality
            for model in models.data:
                if "llama" in model.id.lower() and "70b" in model.id.lower():
                    return model.id

            # Fallback to any Llama model
            for model in models.data:
                if "llama" in model.id.lower():
                    return model.id

            # Ultimate fallback
            if models.data:
                return models.data[0].id

        except Exception as e:
            logger.warning(f"Could not fetch models: {e}")

        # Default fallback
        return "meta-llama/Llama-3.3-70B-Instruct"

    def load_model(self):
        """Return the client (required by DeepEval)"""
        return self.client

    def generate(self, prompt: str, schema: BaseModel = None) -> Union[str, BaseModel]:
        """
        Enhanced generate method optimized for dataset generation with error handling
        """
        client = self.load_model()

        max_retries = 3
        retry_delay = 2

        for attempt in range(max_retries):
            try:
                if schema is not None:
                    # Enhanced prompt for JSON generation
                    json_prompt = self._create_json_prompt(prompt, schema)

                    response = client.chat.completions.create(
                        model=self.model_name,
                        messages=[{"role": "user", "content": json_prompt}],
                        temperature=0.1,
                        max_tokens=3000,
                    )

                    result_text = response.choices[0].message.content.strip()
                    return self._parse_to_schema(result_text, schema)

                else:
                    # Dataset generation
                    response = client.chat.completions.create(
                        model=self.model_name,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.3,
                        max_tokens=3000,
                    )

                    return response.choices[0].message.content.strip()

            except Exception as e:
                logger.warning(f"Generation attempt {attempt + 1} failed: {e}")

                if attempt < max_retries - 1:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    logger.error(f"All generation attempts failed: {e}")

                    # Return fallback schema if expected
                    if schema is not None:
                        return self._create_fallback_schema(schema)

                    return f"Error after {max_retries} attempts: {str(e)}"

    async def a_generate(
        self, prompt: str, schema: BaseModel = None
    ) -> Union[str, BaseModel]:
        """Async version - reuses sync for simplicity"""
        return self.generate(prompt, schema)

    def _create_json_prompt(self, original_prompt: str, schema: BaseModel) -> str:
        """Create a prompt that encourages valid JSON output (same as your existing code)"""
        try:
            schema_dict = schema.model_json_schema()
            properties = schema_dict.get("properties", {})
            required = schema_dict.get("required", [])

            example_json = self._create_example_json(properties)

            json_prompt = f"""{original_prompt}

CRITICAL: Respond with ONLY valid JSON in this exact format:

{json.dumps(example_json, indent=2)}

Required fields: {', '.join(required)}

Rules:
1. Output ONLY the JSON object, no other text
2. Use double quotes for all strings
3. Include all required fields
4. Use reasonable values for the evaluation

JSON Response:"""

            return json_prompt

        except Exception as e:
            logger.warning(f"Could not create JSON prompt: {e}")
            return f"{original_prompt}\n\nRespond with valid JSON only."

    def _create_example_json(self, properties: dict) -> dict:
        """Create example JSON from schema properties (same as your existing code)"""
        example = {}

        for prop_name, prop_info in properties.items():
            prop_type = prop_info.get("type", "string")

            if prop_name == "statements":
                example[prop_name] = ["statement 1", "statement 2"]
            elif prop_name == "verdicts":
                example[prop_name] = [{"verdict": "yes"}, {"verdict": "no"}]
            elif prop_name == "reason":
                example[prop_name] = "explanation of the evaluation"
            elif prop_name == "score":
                example[prop_name] = 0.8
            elif prop_type == "string":
                example[prop_name] = "example text"
            elif prop_type == "number":
                example[prop_name] = 0.5
            elif prop_type == "integer":
                example[prop_name] = 1
            elif prop_type == "boolean":
                example[prop_name] = True
            elif prop_type == "array":
                example[prop_name] = ["item1", "item2"]
            else:
                example[prop_name] = None

        return example

    def _parse_to_schema(self, response_text: str, schema: BaseModel) -> BaseModel:
        """Parse response text to schema instance (same as your existing code)"""
        try:
            json_text = self._extract_json_from_text(response_text)
            json_data = json.loads(json_text)
            return schema(**json_data)

        except json.JSONDecodeError as e:
            logger.warning(f"JSON parsing failed: {e}")
            return self._create_fallback_schema(schema)

        except Exception as e:
            logger.warning(f"Schema creation failed: {e}")
            return self._create_fallback_schema(schema)

    def _extract_json_from_text(self, text: str) -> str:
        """Extract JSON object from response text (same as your existing code)"""
        import re

        text = re.sub(r"```json\s*", "", text)
        text = re.sub(r"```\s*$", "", text)

        json_pattern = r"\{.*\}"
        matches = re.findall(json_pattern, text, re.DOTALL)

        if matches:
            return matches[0]

        text = text.strip()
        if text.startswith("{") and text.endswith("}"):
            return text

        return text

    def _create_fallback_schema(self, schema: BaseModel) -> BaseModel:
        """Create fallback schema instance when parsing fails (same as your existing code)"""
        try:
            return schema()
        except Exception:
            try:
                schema_dict = schema.model_json_schema()
                properties = schema_dict.get("properties", {})

                defaults = {}
                for prop_name, prop_info in properties.items():
                    prop_type = prop_info.get("type", "string")

                    if prop_name == "statements":
                        defaults[prop_name] = ["Unable to evaluate properly"]
                    elif prop_name == "verdicts":
                        defaults[prop_name] = [{"verdict": "unknown"}]
                    elif prop_name == "reason":
                        defaults[prop_name] = "Schema parsing failed"
                    elif prop_name == "score":
                        defaults[prop_name] = 0.5
                    elif prop_type == "string":
                        defaults[prop_name] = "default"
                    elif prop_type == "number":
                        defaults[prop_name] = 0.5
                    elif prop_type == "integer":
                        defaults[prop_name] = 1
                    elif prop_type == "boolean":
                        defaults[prop_name] = True
                    elif prop_type == "array":
                        defaults[prop_name] = []
                    else:
                        defaults[prop_name] = None

                return schema(**defaults)

            except Exception as e:
                logger.error(f"Could not create fallback schema: {e}")

                # Final fallback: create minimal object
                class MinimalFallback:
                    def __init__(self):
                        self.statements = ["Evaluation failed"]
                        self.verdicts = [{"verdict": "unknown"}]
                        self.reason = "Schema creation failed"
                        self.score = 0.0

                return MinimalFallback()

    def get_model_name(self):
        """Return model name (required by DeepEval)"""
        return f"University LLM ({self.model_name})"


class UniversityArXivDatasetGenerator:
    """ArXiv Dataset Generator using University Llama API"""

    def __init__(self, db_path: str, output_dir: str = "datasets"):
        """
        Initialize the dataset generator with University LLM

        Args:
            db_path: Path to your LevelDB database containing arXiv papers
            output_dir: Directory to save generated datasets
        """
        self.db_path = db_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Initialize database connection
        self.db = plyvel.DB(db_path, create_if_missing=False)
        logger.info(f"Connected to database at {db_path}")

        # Initialize University LLM for generation
        self.university_llm = UniversityLLMForGeneration()
        logger.info("University LLM initialized for dataset generation")

    def extract_clean_text_from_paper(self, paper_content: str, paper_id: str) -> str:
        """
        Extract and clean text content from arXiv paper data

        Your format: "Content for math/0004036:\nThe asymptotic behavior..."
        """
        try:
            # Your exact format: remove "Content for [paper_id]:" header
            clean_text = re.sub(r"Content for [^:]*:\s*", "", paper_content)

            # Remove any opening LevelDB messages if they somehow got included
            lines = clean_text.split("\n")
            content_lines = []
            skip_prefixes = ["Opening LevelDB", "Opened LevelDB", "Time taken to open"]

            for line in lines:
                # Skip system messages
                if any(line.strip().startswith(prefix) for prefix in skip_prefixes):
                    continue
                content_lines.append(line)

            clean_text = "\n".join(content_lines).strip()

            # Preserve academic structure but clean excessive whitespace
            clean_text = re.sub(r"[ \t]+", " ", clean_text)  # Normalize spaces/tabs
            clean_text = re.sub(r"\n{3,}", "\n\n", clean_text)  # Max 2 newlines

            # Validate content length for academic papers
            if len(clean_text) < 1000:  # Academic papers should be substantial
                logger.warning(
                    f"Paper {paper_id} has short content ({len(clean_text)} chars)"
                )
                return None

            # Validate it looks like academic content
            if not any(
                keyword in clean_text.lower()
                for keyword in [
                    "abstract",
                    "introduction",
                    "method",
                    "result",
                    "conclusion",
                    "theorem",
                    "proof",
                ]
            ):
                logger.warning(f"Paper {paper_id} may not be academic content")

            return clean_text.strip()

        except Exception as e:
            logger.error(f"Error cleaning text for paper {paper_id}: {e}")
            return None

    def load_paper_ids_from_file(self, ids_file: str) -> List[str]:
        """Load paper IDs from a previously saved file"""
        try:
            with open(ids_file, "r", encoding="utf-8") as f:
                paper_ids = [line.strip() for line in f if line.strip()]

            logger.info(f"Loaded {len(paper_ids)} paper IDs from {ids_file}")
            return paper_ids

        except Exception as e:
            logger.error(f"Error loading paper IDs from file: {e}")
            return []

    def save_papers_as_temp_files(
        self,
        paper_ids: List[str] = None,
        max_papers: int = 50,
        random_selection: bool = False,
        random_seed: int = None,
    ) -> List[str]:
        """Extract papers from database and save as temporary text files"""
        temp_files = []
        papers_processed = 0

        logger.info(f"Extracting up to {max_papers} papers from database...")
        if random_selection:
            logger.info("Using random selection")
            if random_seed:
                random.seed(random_seed)
                logger.info(f"Random seed: {random_seed}")

        # Create temporary directory
        temp_dir = tempfile.mkdtemp(prefix="arxiv_papers_")
        logger.info(f"Using temporary directory: {temp_dir}")

        try:
            if paper_ids:
                # Use provided paper IDs directly (most efficient)
                logger.info(f"Using {len(paper_ids)} provided paper IDs")
                for i, paper_id in enumerate(paper_ids):
                    if papers_processed >= max_papers:
                        break

                    try:
                        paper_content = self.db.get(paper_id.encode("utf-8"))
                        if paper_content:
                            paper_content = paper_content.decode("utf-8")
                            clean_text = self.extract_clean_text_from_paper(
                                paper_content, paper_id
                            )

                            if clean_text:
                                # Save to temporary file
                                temp_file = os.path.join(
                                    temp_dir, f"{paper_id.replace('/', '_')}.txt"
                                )

                                with open(temp_file, "w", encoding="utf-8") as f:
                                    f.write(f"# arXiv Paper: {paper_id}\n\n")
                                    f.write(clean_text)

                                temp_files.append(temp_file)
                                papers_processed += 1

                                # Progress indicator
                                if papers_processed % 10 == 0 or papers_processed <= 10:
                                    logger.info(
                                        f"Extracted paper {papers_processed}/{min(max_papers, len(paper_ids))}: {paper_id}"
                                    )
                            else:
                                logger.debug(f"Skipped invalid paper: {paper_id}")

                    except Exception as e:
                        logger.error(f"Error extracting paper {paper_id}: {e}")
                        continue

            elif random_selection:
                # FAST: First pass - collect keys only (no content reading)
                logger.info("Collecting paper IDs (fast key-only scan)...")
                all_paper_ids = []
                for key, value in self.db:
                    paper_id = key.decode("utf-8")
                    all_paper_ids.append(paper_id)

                logger.info(f"Found {len(all_paper_ids)} papers in database")

                # Randomly select more than needed (account for invalid papers)
                # Select 2-3x more than needed to account for short/invalid papers
                oversample_factor = 3
                sample_size = min(max_papers * oversample_factor, len(all_paper_ids))

                selected_paper_ids = random.sample(all_paper_ids, sample_size)
                logger.info(
                    f"Pre-selected {len(selected_paper_ids)} papers for validation"
                )

                # Second pass: validate and extract until we have enough
                for paper_id in selected_paper_ids:
                    if papers_processed >= max_papers:
                        break

                    try:
                        paper_content = self.db.get(paper_id.encode("utf-8"))
                        if paper_content:
                            paper_content = paper_content.decode("utf-8")
                            clean_text = self.extract_clean_text_from_paper(
                                paper_content, paper_id
                            )

                            if clean_text:
                                # Save to temporary file
                                temp_file = os.path.join(
                                    temp_dir, f"{paper_id.replace('/', '_')}.txt"
                                )

                                with open(temp_file, "w", encoding="utf-8") as f:
                                    f.write(f"# arXiv Paper: {paper_id}\n\n")
                                    f.write(clean_text)

                                temp_files.append(temp_file)
                                papers_processed += 1

                                logger.info(
                                    f"Extracted paper {papers_processed}/{max_papers}: {paper_id}"
                                )
                            else:
                                logger.debug(f"Skipped invalid paper: {paper_id}")

                    except Exception as e:
                        logger.error(f"Error extracting paper {paper_id}: {e}")
                        continue

            else:
                # Original sequential/filtered selection
                for key, value in self.db:
                    if papers_processed >= max_papers:
                        break

                    paper_id = key.decode("utf-8")

                    # Filter by specific paper IDs if provided
                    if paper_ids and paper_id not in paper_ids:
                        continue

                    paper_content = value.decode("utf-8")

                    # Clean the text
                    clean_text = self.extract_clean_text_from_paper(
                        paper_content, paper_id
                    )
                    if not clean_text:
                        continue

                    # Save to temporary file
                    temp_file = os.path.join(
                        temp_dir, f"{paper_id.replace('/', '_')}.txt"
                    )

                    try:
                        with open(temp_file, "w", encoding="utf-8") as f:
                            f.write(f"# arXiv Paper: {paper_id}\n\n")
                            f.write(clean_text)

                        temp_files.append(temp_file)
                        papers_processed += 1

                        logger.info(
                            f"Extracted paper {papers_processed}/{max_papers}: {paper_id}"
                        )

                    except Exception as e:
                        logger.error(f"Error saving paper {paper_id}: {e}")
                        continue

        except Exception as e:
            logger.error(f"Error iterating through database: {e}")

        logger.info(f"Successfully extracted {len(temp_files)} papers")
        return temp_files

    def _get_timestamp(self) -> str:
        """Generate timestamp string for file naming"""
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def _manual_save_dataset(self, dataset: EvaluationDataset, output_file: Path):
        """Fallback method to manually save dataset as JSON"""
        try:
            import json

            # Convert goldens to serializable format
            goldens_data = []
            for golden in dataset.goldens:
                golden_dict = {
                    "input": golden.input,
                    "expected_output": golden.expected_output,
                    "context": golden.context if golden.context else [],
                    "additional_metadata": (
                        golden.additional_metadata
                        if hasattr(golden, "additional_metadata")
                        else {}
                    ),
                }
                goldens_data.append(golden_dict)

            # Save to JSON
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "goldens": goldens_data,
                        "metadata": {
                            "total_goldens": len(goldens_data),
                            "generated_with": "University LLM",
                            "timestamp": time.time(),
                        },
                    },
                    f,
                    indent=2,
                    ensure_ascii=False,
                )

            logger.info(f"Manually saved {len(goldens_data)} goldens to {output_file}")

        except Exception as e:
            logger.error(f"Failed to manually save dataset: {e}")
            raise

    def generate_research_dataset_with_university_llm(
        self,
        document_paths: List[str],
        config_name: str = "research_uni",
        batch_size: int = 10,
    ) -> EvaluationDataset:
        """
        Generate a research dataset using University Llama API with batch processing

        Args:
            document_paths: List of document file paths
            config_name: Name for this configuration
            batch_size: Process documents in batches to avoid timeouts

        Returns:
            Generated EvaluationDataset
        """
        logger.info(f"Generating research dataset with University LLM: {config_name}")
        logger.info(
            f"Using {len(document_paths)} documents in batches of {batch_size}..."
        )

        # Use optimized embedding model for faster performance
        embedder = HFEmbeddingModel("sentence-transformers/all-MiniLM-L6-v2")

        context_config = ContextConstructionConfig(
            embedder=embedder,
            critic_model=self.university_llm,
            max_contexts_per_document=1,  # One context per document = one golden per paper
            chunk_size=256,
            chunk_overlap=25,
        )

        # Research-focused styling configuration optimized for Llama
        styling_config = StylingConfig(
            input_format=(
                "Generate clear, straightforward questions about this research paper. "
                "Ask about the main findings, methods used, and key results. "
                "Make questions that a curious student would ask."
            ),
            expected_output_format=(
                "Provide clear answers based on the research paper. "
                "Use simple language and explain technical terms when needed. "
                "Give specific examples from the paper."
            ),
        )

        # Quality filtration optimized for academic content
        filtration_config = FiltrationConfig(
            critic_model=self.university_llm, synthetic_input_quality_threshold=0.78
        )

        # Evolution configuration for academic complexity
        evolution_config = EvolutionConfig(
            num_evolutions=1,
            evolutions={
                Evolution.REASONING: 1.0,
            },
        )

        logger.info("Generating goldens using University Llama model...")

        # Process in batches to avoid timeouts and memory issues
        all_goldens = []
        total_batches = (len(document_paths) + batch_size - 1) // batch_size

        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(document_paths))
            batch_documents = document_paths[start_idx:end_idx]

            logger.info(
                f"Processing batch {batch_idx + 1}/{total_batches} ({len(batch_documents)} documents)"
            )

            try:
                # Create a new synthesizer for each batch to avoid memory buildup
                batch_synthesizer = Synthesizer(
                    model=self.university_llm,
                    styling_config=styling_config,
                    filtration_config=filtration_config,
                    evolution_config=evolution_config,
                )

                # Generate goldens for this batch
                batch_synthesizer.generate_goldens_from_docs(
                    document_paths=batch_documents,
                    include_expected_output=True,
                    max_goldens_per_context=1,
                    context_construction_config=context_config,
                )

                batch_goldens = batch_synthesizer.synthetic_goldens
                all_goldens.extend(batch_goldens)

                logger.info(
                    f"Batch {batch_idx + 1} completed: {len(batch_goldens)} goldens generated"
                )
                logger.info(
                    f"Total progress: {len(all_goldens)}/{len(document_paths)} goldens"
                )

                # Small delay between batches to avoid rate limiting
                if batch_idx < total_batches - 1:
                    time.sleep(2)

            except Exception as e:
                logger.error(f"Error processing batch {batch_idx + 1}: {e}")
                logger.info("Continuing with next batch...")
                import traceback

                traceback.print_exc()
                continue

        # Create final dataset
        dataset = EvaluationDataset(goldens=all_goldens)

        logger.info(
            f"Generated {len(dataset.goldens)} total goldens using University LLM"
        )

        # Add timestamp to filename
        timestamp = self._get_timestamp()
        output_file = self.output_dir / f"arxiv_{config_name}_{timestamp}_dataset.json"

        try:
            # Try new API first
            dataset.save_as(file_path=str(output_file), file_type="json")
        except TypeError:
            try:
                # Try older API
                dataset.save_as(str(output_file), "json")
            except:
                # Fallback: manual JSON save
                self._manual_save_dataset(dataset, output_file)

        logger.info(f"Saved dataset to {output_file}")

        return dataset

    def generate_rag_evaluation_dataset_with_university_llm(
        self,
        document_paths: List[str],
        config_name: str = "rag_uni",
        batch_size: int = 10,
    ) -> EvaluationDataset:
        """
        Generate a RAG evaluation dataset using University Llama API with batch processing

        Args:
            document_paths: List of document file paths
            config_name: Name for this configuration
            batch_size: Process documents in batches to avoid timeouts

        Returns:
            Generated EvaluationDataset
        """
        logger.info(
            f"Generating RAG evaluation dataset with University LLM: {config_name}"
        )
        logger.info(
            f"Using {len(document_paths)} documents in batches of {batch_size}..."
        )

        # RAG-focused styling configuration for Llama
        styling_config = StylingConfig(
            input_format=(
                "Generate questions that test information retrieval and answer generation capabilities. "
                "Include factual questions, definition requests, methodology inquiries, and result summaries. "
                "Questions should be directly answerable from the paper content without external knowledge."
            ),
            expected_output_format=(
                "Provide clear, direct answers based on the provided research context. "
                "Focus on factual accuracy and faithfulness to the source material. "
                "Avoid adding information not present in the context. "
                "Structure answers to be easily verifiable against the source."
            ),
        )

        # Use optimized embedding model for faster performance
        embedder = HFEmbeddingModel("sentence-transformers/all-MiniLM-L6-v2")

        # One context per document = one golden per paper
        context_config = ContextConstructionConfig(
            embedder=embedder,
            critic_model=self.university_llm,
            max_contexts_per_document=1,
            chunk_size=512,
            chunk_overlap=50,
        )

        # Moderate filtration for RAG testing
        filtration_config = FiltrationConfig(
            critic_model=self.university_llm, synthetic_input_quality_threshold=0.72
        )

        # Evolution focused on RAG-relevant patterns
        evolution_config = EvolutionConfig(
            num_evolutions=1,
            evolutions={
                Evolution.MULTICONTEXT: 1.0,
                Evolution.CONCRETIZING: 1.0,
            },
        )

        logger.info("Generating RAG evaluation cases using University Llama model...")

        # Process in batches to avoid timeouts and memory issues
        all_goldens = []
        total_batches = (len(document_paths) + batch_size - 1) // batch_size

        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(document_paths))
            batch_documents = document_paths[start_idx:end_idx]

            logger.info(
                f"Processing batch {batch_idx + 1}/{total_batches} ({len(batch_documents)} documents)"
            )

            try:
                # Create a new synthesizer for each batch
                batch_synthesizer = Synthesizer(
                    model=self.university_llm,
                    styling_config=styling_config,
                    filtration_config=filtration_config,
                    evolution_config=evolution_config,
                )

                # Generate goldens for this batch
                batch_synthesizer.generate_goldens_from_docs(
                    document_paths=batch_documents,
                    include_expected_output=True,
                    max_goldens_per_context=1,
                    context_construction_config=context_config,
                )

                batch_goldens = batch_synthesizer.synthetic_goldens
                all_goldens.extend(batch_goldens)

                logger.info(
                    f"Batch {batch_idx + 1} completed: {len(batch_goldens)} goldens generated"
                )
                logger.info(
                    f"Total progress: {len(all_goldens)}/{len(document_paths)} goldens"
                )

                # Small delay between batches
                if batch_idx < total_batches - 1:
                    time.sleep(2)

            except Exception as e:
                logger.error(f"Error processing batch {batch_idx + 1}: {e}")
                logger.info("Continuing with next batch...")
                import traceback

                traceback.print_exc()
                continue

        # Create final dataset
        dataset = EvaluationDataset(goldens=all_goldens)

        logger.info(
            f"Generated {len(dataset.goldens)} total RAG evaluation cases using University LLM"
        )

        # Add timestamp to filename
        timestamp = self._get_timestamp()
        output_file = self.output_dir / f"arxiv_{config_name}_{timestamp}_dataset.json"

        try:
            # Try new API first
            dataset.save_as(file_path=str(output_file), file_type="json")
        except TypeError:
            try:
                # Try older API
                dataset.save_as(str(output_file), "json")
            except:
                # Fallback: manual JSON save
                self._manual_save_dataset(dataset, output_file)

        logger.info(f"Saved dataset to {output_file}")

        return dataset

    def show_generation_statistics(self, dataset: EvaluationDataset):
        """Show detailed statistics about the generated dataset"""

        logger.info(f"\nDATASET GENERATION STATISTICS")
        logger.info("=" * 60)

        goldens = dataset.goldens

        if not goldens:
            logger.info("No goldens generated")
            return

        # Basic statistics
        total_goldens = len(goldens)
        input_lengths = [len(g.input) for g in goldens]
        output_lengths = [len(g.expected_output) for g in goldens if g.expected_output]
        context_counts = [len(g.context) for g in goldens if g.context]

        # Evolution statistics
        all_evolutions = []
        for golden in goldens:
            if hasattr(golden, "additional_metadata") and golden.additional_metadata:
                evolutions = golden.additional_metadata.get("evolutions", [])
                all_evolutions.extend(evolutions)

        unique_evolutions = list(set(all_evolutions))

        logger.info(
            f"Generated with University LLM: {self.university_llm.get_model_name()}"
        )
        logger.info(f"Total goldens: {total_goldens}")
        logger.info(
            f"Average input length: {sum(input_lengths)/len(input_lengths):.1f} chars"
        )
        logger.info(
            f"Average output length: {sum(output_lengths)/len(output_lengths):.1f} chars"
        )
        logger.info(
            f"Average contexts per golden: {sum(context_counts)/len(context_counts):.1f}"
        )
        logger.info(f"Evolution types applied: {unique_evolutions}")
        logger.info(f"Total evolution applications: {len(all_evolutions)}")

        # Show sample questions
        logger.info(f"\nSample Generated Questions:")
        for i, golden in enumerate(goldens[:3]):
            logger.info(f"\n{i+1}. {golden.input}")
            if golden.expected_output:
                preview = (
                    golden.expected_output[:150] + "..."
                    if len(golden.expected_output) > 150
                    else golden.expected_output
                )
                logger.info(f"   Answer: {preview}")

    def cleanup_temp_files(self, temp_files: List[str]):
        """Clean up temporary files"""
        for temp_file in temp_files:
            try:
                os.remove(temp_file)
            except Exception as e:
                logger.warning(f"Could not remove temp file {temp_file}: {e}")

        # Remove temp directory if empty
        if temp_files:
            temp_dir = os.path.dirname(temp_files[0])
            try:
                os.rmdir(temp_dir)
                logger.info(f"Cleaned up temporary directory: {temp_dir}")
            except Exception:
                pass

    def close(self):
        """Close database connection"""
        if hasattr(self, "db"):
            self.db.close()
            logger.info("Database connection closed")


def main():
    """Main function demonstrating University LLM dataset generation"""
    parser = argparse.ArgumentParser(
        description="Generate datasets using University Llama API"
    )
    parser.add_argument(
        "--db_path", type=str, required=True, help="Path to LevelDB database"
    )
    parser.add_argument(
        "--max_papers",
        type=int,
        default=3,
        help="Maximum papers to use (default: 3 for testing)",
    )
    parser.add_argument(
        "--dataset_type",
        choices=["research", "rag", "both"],
        default="both",
        help="Type of dataset to generate",
    )
    parser.add_argument(
        "--output_dir", type=str, default="university_datasets", help="Output directory"
    )
    parser.add_argument("--paper_ids", nargs="+", help="Specific paper IDs to use")

    # Random selection options
    parser.add_argument(
        "--random",
        action="store_true",
        help="Randomly select papers instead of sequential",
    )
    parser.add_argument(
        "--random_seed", type=int, help="Random seed for reproducible selection"
    )

    # NEW: Paper ID extraction options
    parser.add_argument(
        "--extract_ids_only",
        action="store_true",
        help="Only extract and save all paper IDs, don't generate datasets",
    )
    parser.add_argument(
        "--load_ids_file", type=str, help="Load paper IDs from a previously saved file"
    )
    parser.add_argument(
        "--select_random_from_all",
        type=int,
        help="Randomly select N papers from all available papers (e.g., 500)",
    )

    # NEW: Batch processing options
    parser.add_argument(
        "--batch_size",
        type=int,
        default=10,
        help="Process documents in batches (default: 10)",
    )

    args = parser.parse_args()

    logger.info("University LLM Dataset Generation")
    logger.info("=" * 60)

    # Initialize generator with University LLM
    generator = UniversityArXivDatasetGenerator(args.db_path, args.output_dir)

    try:
        # NEW WORKFLOW: Extract all IDs first if requested
        if args.extract_ids_only:
            logger.info("Extracting all paper IDs from database...")
            all_ids = generator.extract_all_paper_ids(save_to_file=True)
            logger.info(f"Extraction complete! Found {len(all_ids)} papers total.")
            logger.info(
                f"Use --select_random_from_all 500 to randomly select 500 papers from these IDs"
            )
            return

        # Determine which papers to use
        selected_paper_ids = None

        if args.load_ids_file:
            # Load IDs from file
            all_available_ids = generator.load_paper_ids_from_file(args.load_ids_file)

            if args.select_random_from_all:
                # Randomly select from loaded IDs
                if args.random_seed:
                    random.seed(args.random_seed)
                    logger.info(f"Using random seed: {args.random_seed}")

                select_count = min(args.select_random_from_all, len(all_available_ids))
                selected_paper_ids = random.sample(all_available_ids, select_count)
                logger.info(
                    f"Randomly selected {len(selected_paper_ids)} papers from {len(all_available_ids)} available"
                )
            else:
                selected_paper_ids = all_available_ids[: args.max_papers]

        elif args.select_random_from_all:
            # Extract all IDs and randomly select
            logger.info("Extracting all paper IDs for random selection...")
            all_available_ids = generator.extract_all_paper_ids(save_to_file=True)

            if args.random_seed:
                random.seed(args.random_seed)
                logger.info(f"Using random seed: {args.random_seed}")

            select_count = min(args.select_random_from_all, len(all_available_ids))
            selected_paper_ids = random.sample(all_available_ids, select_count)
            logger.info(
                f"Randomly selected {len(selected_paper_ids)} papers from {len(all_available_ids)} total"
            )

        # Extract papers as temporary files
        temp_files = generator.save_papers_as_temp_files(
            paper_ids=selected_paper_ids or args.paper_ids,
            max_papers=args.max_papers,
            random_selection=args.random,
            random_seed=args.random_seed,
        )

        if not temp_files:
            logger.error("No papers extracted. Exiting.")
            return

        # Generate datasets based on type
        if args.dataset_type in ["research", "both"]:
            logger.info("\nGenerating research dataset...")
            start_time = time.time()
            research_dataset = generator.generate_research_dataset_with_university_llm(
                temp_files, "research_uni", batch_size=args.batch_size
            )
            research_time = time.time() - start_time
            logger.info(f"Research dataset generated in {research_time:.1f} seconds")
            generator.show_generation_statistics(research_dataset)

        if args.dataset_type in ["rag", "both"]:
            logger.info("\nGenerating RAG evaluation dataset...")
            start_time = time.time()
            rag_dataset = generator.generate_rag_evaluation_dataset_with_university_llm(
                temp_files, "rag_uni", batch_size=args.batch_size
            )
            rag_time = time.time() - start_time
            logger.info(f"RAG dataset generated in {rag_time:.1f} seconds")
            generator.show_generation_statistics(rag_dataset)

        logger.info(f"\nDataset generation completed using University LLM!")
        logger.info(f"Results saved to: {args.output_dir}")
        logger.info(f"Ready for integration with your existing RAG evaluation system")

    except Exception as e:
        logger.error(f"Error during dataset generation: {e}")
        import traceback

        traceback.print_exc()

    finally:
        # Cleanup
        if "temp_files" in locals():
            generator.cleanup_temp_files(temp_files)
        generator.close()


if __name__ == "__main__":
    main()
