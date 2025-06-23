#!/usr/bin/env python3
"""
Parallel RAG Evaluator - Enhanced with None fallback for failed metrics
Updated to return None for failed evaluations and ignore them in averages
Run with: python rag_evaluation_0806.py cite_RAG_0906_litsearch.jsonl --max-items 100 --workers 16 --output cite_RAG_0906_litsearch_0806eval_result_test1.jsonl
"""

import os
import json
import argparse
import sys
import time
from typing import List, Dict, Any, Union
from pathlib import Path
from openai import OpenAI
from pydantic import BaseModel
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualRelevancyMetric,
)
from deepeval.test_case import LLMTestCase

# Add parallel processing imports
from concurrent.futures import ThreadPoolExecutor, as_completed
import concurrent.futures


class UniversityLLM(DeepEvalBaseLLM):
    """Fixed University LLM with proper DeepEval schema support."""

    def __init__(self, api_key=None):
        """Initialize University LLM with DeepEval compatibility."""
        # Load API key
        if not api_key:
            path_to_key = os.path.join(os.path.expanduser("~"), ".scadsai-api-key")
            if os.path.exists(path_to_key):
                with open(path_to_key) as keyfile:
                    api_key = keyfile.readline().strip()

        if not api_key:
            raise ValueError("No API key provided or found in ~/.scadsai-api-key")

        # Initialize OpenAI client
        self.client = OpenAI(base_url="https://llm.scads.ai/v1", api_key=api_key)

        # Find available model
        self.model_name = self._find_model()
        print(f"🔧 Using evaluation model: {self.model_name}")

    def _find_model(self):
        """Find the best available model."""
        try:
            models = self.client.models.list()
            # Prefer Llama models
            for model in models.data:
                if "llama" in model.id.lower() and "70b" in model.id.lower():
                    return model.id

            # Fallback to any Llama model
            for model in models.data:
                if "llama" in model.id.lower():
                    return model.id

            # Fallback to first available
            if models.data:
                return models.data[0].id

        except Exception as e:
            print(f"Warning: Could not fetch models: {e}")

        # Ultimate fallback
        return "meta-llama/Llama-3.3-70B-Instruct"

    def load_model(self):
        """Return the client (required by DeepEval)."""
        return self.client

    def generate(self, prompt: str, schema: BaseModel = None) -> Union[str, BaseModel]:
        """Enhanced generate method with schema support."""
        client = self.load_model()

        try:
            if schema is not None:
                # Enhanced prompt for JSON generation
                json_prompt = self._create_json_prompt(prompt, schema)

                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": json_prompt}],
                    temperature=0.0,  # Very low for consistent JSON
                    max_tokens=2000,
                )

                result_text = response.choices[0].message.content.strip()

                # Parse and return schema instance
                return self._parse_to_schema(result_text, schema)

            else:
                # Normal string generation
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=2000,
                )

                return response.choices[0].message.content.strip()

        except Exception as e:
            print(f"Generation error: {e}")

            # Return fallback schema if expected
            if schema is not None:
                return self._create_fallback_schema(schema)

            return f"Error: {str(e)}"

    async def a_generate(
        self, prompt: str, schema: BaseModel = None
    ) -> Union[str, BaseModel]:
        """Async version - reuses sync for simplicity."""
        return self.generate(prompt, schema)

    def _create_json_prompt(self, original_prompt: str, schema: BaseModel) -> str:
        """Create a prompt that encourages valid JSON output."""

        try:
            # Get schema information
            schema_dict = schema.model_json_schema()
            properties = schema_dict.get("properties", {})
            required = schema_dict.get("required", [])

            # Create example JSON
            example_json = self._create_example_json(properties)

            # Enhanced prompt
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
            print(f"Warning: Could not create JSON prompt: {e}")
            return f"{original_prompt}\n\nRespond with valid JSON only."

    def _create_example_json(self, properties: dict) -> dict:
        """Create example JSON from schema properties."""

        example = {}

        for prop_name, prop_info in properties.items():
            prop_type = prop_info.get("type", "string")

            # Handle specific DeepEval schema fields
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
        """Parse response text to schema instance."""

        try:
            # Extract JSON from response
            json_text = self._extract_json_from_text(response_text)

            # Parse JSON
            json_data = json.loads(json_text)

            # Create and return schema instance
            return schema(**json_data)

        except json.JSONDecodeError as e:
            print(f"JSON parsing failed: {e}")
            print(f"Response was: {response_text[:200]}...")
            return self._create_fallback_schema(schema)

        except Exception as e:
            print(f"Schema creation failed: {e}")
            return self._create_fallback_schema(schema)

    def _extract_json_from_text(self, text: str) -> str:
        """Extract JSON object from response text."""

        import re

        # Remove code block markers
        text = re.sub(r"```json\s*", "", text)
        text = re.sub(r"```\s*$", "", text)

        # Find JSON object pattern
        json_pattern = r"\{.*\}"
        matches = re.findall(json_pattern, text, re.DOTALL)

        if matches:
            return matches[0]

        # If no JSON found, try to clean the text
        text = text.strip()
        if text.startswith("{") and text.endswith("}"):
            return text

        # Last resort: return the whole text
        return text

    def _create_fallback_schema(self, schema: BaseModel) -> BaseModel:
        """Create fallback schema instance when parsing fails."""

        try:
            # Try empty initialization
            return schema()

        except Exception:
            try:
                # Try with common defaults
                schema_dict = schema.model_json_schema()
                properties = schema_dict.get("properties", {})

                defaults = {}
                for prop_name, prop_info in properties.items():
                    prop_type = prop_info.get("type", "string")

                    # Special handling for known DeepEval schema fields
                    if prop_name == "statements":
                        defaults[prop_name] = ["Unable to evaluate properly"]
                    elif prop_name == "verdicts":
                        defaults[prop_name] = [{"verdict": "unknown"}]
                    elif prop_name == "reason":
                        defaults[prop_name] = "Schema parsing failed"
                    elif prop_name == "score":
                        # CHANGED: Return None instead of 0.5 for failed evaluations
                        defaults[prop_name] = None
                    elif prop_type == "string":
                        defaults[prop_name] = "default"
                    elif prop_type == "number":
                        defaults[prop_name] = None  # CHANGED: None instead of 0.5
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
                print(f"Could not create fallback schema: {e}")

                # Final fallback: create minimal object with common attributes
                class MinimalFallback:
                    def __init__(self):
                        self.statements = ["Evaluation failed"]
                        self.verdicts = [{"verdict": "unknown"}]
                        self.reason = "Schema creation failed"
                        self.score = None  # CHANGED: None instead of 0.0

                return MinimalFallback()

    def get_model_name(self):
        """Return model name (required by DeepEval)."""
        return f"University LLM ({self.model_name})"


class ParallelRAGEvaluator:
    """Enhanced RAG evaluator with parallel processing capabilities and None fallback."""

    def __init__(self, max_workers=8):
        """Initialize evaluator with parallel processing support."""
        self.llm = UniversityLLM()
        self.max_workers = max_workers
        self.setup_metrics()
        self.results = []
        self.errors = []

    def setup_metrics(self):
        """Setup evaluation metrics."""
        self.answer_relevancy = AnswerRelevancyMetric(model=self.llm, threshold=0.5)
        self.faithfulness = FaithfulnessMetric(model=self.llm, threshold=0.5)
        self.contextual_relevancy = ContextualRelevancyMetric(
            model=self.llm, threshold=0.5
        )
        print("✅ Evaluation metrics initialized")

    def load_parsed_data(self, input_file: str) -> List[Dict[str, Any]]:
        """Load parsed RAG data from file."""

        print(f"📖 Loading parsed data from: {input_file}")

        parsed_data = []

        try:
            with open(input_file, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        data = json.loads(line)

                        # Validate required fields
                        required_fields = [
                            "id",
                            "input",
                            "actual_output",
                            "retrieval_context",
                        ]
                        missing_fields = [
                            field for field in required_fields if field not in data
                        ]

                        if missing_fields:
                            raise ValueError(
                                f"Missing required fields: {missing_fields}"
                            )

                        parsed_data.append(data)

                    except (json.JSONDecodeError, ValueError) as e:
                        error_msg = f"Line {line_num}: {e}"
                        self.errors.append(error_msg)
                        print(f"⚠️  {error_msg}")
                        continue

        except FileNotFoundError:
            print(f"❌ Error: File {input_file} not found")
            return []
        except Exception as e:
            print(f"❌ Error reading file: {e}")
            return []

        print(f"✅ Loaded {len(parsed_data)} valid records")
        return parsed_data

    def create_test_cases(self, parsed_data: List[Dict[str, Any]]) -> List[LLMTestCase]:
        """Convert parsed data to DeepEval test cases."""

        test_cases = []

        for data in parsed_data:
            test_case = LLMTestCase(
                input=data["input"],
                actual_output=data["actual_output"],
                retrieval_context=data["retrieval_context"],
            )
            test_cases.append(test_case)

        return test_cases

    def evaluate_data_parallel(
        self, parsed_data: List[Dict[str, Any]], max_items: int = None
    ) -> List[Dict[str, Any]]:
        """NEW: Parallel evaluation method with progress tracking."""

        # Limit data if requested
        if max_items:
            parsed_data = parsed_data[:max_items]
            print(f"🔢 Limiting evaluation to first {max_items} items")

        print(
            f"\n🚀 PARALLEL Evaluating {len(parsed_data)} results with {self.max_workers} workers..."
        )
        print("=" * 80)

        test_cases = self.create_test_cases(parsed_data)
        evaluation_results = []

        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all evaluation tasks
            future_to_data = {}
            for i, (test_case, original_data) in enumerate(
                zip(test_cases, parsed_data)
            ):
                future = executor.submit(
                    self._evaluate_single_item, test_case, original_data, i
                )
                future_to_data[future] = (test_case, original_data, i)

            # Collect results as they complete
            completed = 0
            for future in as_completed(future_to_data):
                test_case, original_data, original_index = future_to_data[future]
                completed += 1

                try:
                    result = future.result(timeout=300)  # 5 minute timeout per item
                    if result:
                        evaluation_results.append(result)

                        # Print progress with scores
                        overall_score = result.get("overall_score")
                        score_str = (
                            f"{overall_score:.3f}"
                            if overall_score is not None
                            else "N/A"
                        )
                        print(
                            f"✅ {completed}/{len(test_cases)} completed: {original_data['id']} (Score: {score_str})"
                        )
                    else:
                        print(
                            f"❌ {completed}/{len(test_cases)} failed: {original_data['id']}"
                        )

                except concurrent.futures.TimeoutError:
                    error_msg = f"Timeout evaluating {original_data['id']}"
                    self.errors.append(error_msg)
                    print(
                        f"⏰ {completed}/{len(test_cases)} timeout: {original_data['id']}"
                    )

                except Exception as e:
                    error_msg = (
                        f"Error evaluating {original_data['id']}: {str(e)[:100]}"
                    )
                    self.errors.append(error_msg)
                    print(
                        f"❌ {completed}/{len(test_cases)} error: {original_data['id']} - {str(e)[:50]}"
                    )

        # Sort results back to original order
        evaluation_results.sort(key=lambda x: x.get("_original_index", 0))

        # Remove the ordering field
        for result in evaluation_results:
            result.pop("_original_index", None)

        self.results = evaluation_results
        return evaluation_results

    def _evaluate_single_item(self, test_case, original_data, original_index):
        """Evaluate a single item - designed for parallel execution."""

        try:
            # Evaluate metrics one by one with error handling
            ar_result = self._safe_evaluate_metric(
                self.answer_relevancy, test_case, "Answer Relevancy"
            )
            f_result = self._safe_evaluate_metric(
                self.faithfulness, test_case, "Faithfulness"
            )
            cr_result = self._safe_evaluate_metric(
                self.contextual_relevancy, test_case, "Contextual Relevancy"
            )

            # CHANGED: Calculate overall score ignoring None values
            valid_scores = [
                score
                for score in [
                    ar_result["score"],
                    f_result["score"],
                    cr_result["score"],
                ]
                if score is not None and isinstance(score, (int, float))
            ]

            # CHANGED: Return None if no valid scores, otherwise calculate average
            overall_score = (
                sum(valid_scores) / len(valid_scores) if valid_scores else None
            )

            result = {
                "id": original_data["id"],
                "input": test_case.input,
                "actual_output": test_case.actual_output,
                "num_contexts": len(test_case.retrieval_context),
                "evaluation": {
                    "answer_relevancy": ar_result,
                    "faithfulness": f_result,
                    "contextual_relevancy": cr_result,
                },
                "overall_score": overall_score,  # Can now be None
                "metadata": original_data.get("metadata", {}),
                "_original_index": original_index,  # For sorting later
            }

            return result

        except Exception as e:
            print(f"Single evaluation error for {original_data['id']}: {e}")
            return None

    def evaluate_data(
        self, parsed_data: List[Dict[str, Any]], max_items: int = None
    ) -> List[Dict[str, Any]]:
        """Original sequential evaluation method (kept for compatibility)."""

        # Limit data if requested
        if max_items:
            parsed_data = parsed_data[:max_items]
            print(f"🔢 Limiting evaluation to first {max_items} items")

        print(f"\n🔍 SEQUENTIAL Evaluating {len(parsed_data)} results...")
        print("=" * 80)

        test_cases = self.create_test_cases(parsed_data)
        evaluation_results = []

        for i, (test_case, original_data) in enumerate(zip(test_cases, parsed_data)):
            print(f"\n📋 Evaluating {i+1}/{len(test_cases)}: {original_data['id']}")
            print("-" * 60)
            print(f"Question: {test_case.input[:80]}...")
            print(f"Answer: {test_case.actual_output[:80]}...")
            print(f"Contexts: {len(test_case.retrieval_context)}")

            try:
                # Evaluate metrics one by one with error handling
                ar_result = self._safe_evaluate_metric(
                    self.answer_relevancy, test_case, "Answer Relevancy"
                )
                f_result = self._safe_evaluate_metric(
                    self.faithfulness, test_case, "Faithfulness"
                )
                cr_result = self._safe_evaluate_metric(
                    self.contextual_relevancy, test_case, "Contextual Relevancy"
                )

                # CHANGED: Calculate overall score ignoring None values
                valid_scores = [
                    score
                    for score in [
                        ar_result["score"],
                        f_result["score"],
                        cr_result["score"],
                    ]
                    if score is not None and isinstance(score, (int, float))
                ]

                # CHANGED: Return None if no valid scores
                overall_score = (
                    sum(valid_scores) / len(valid_scores) if valid_scores else None
                )

                result = {
                    "id": original_data["id"],
                    "input": test_case.input,
                    "actual_output": test_case.actual_output,
                    "num_contexts": len(test_case.retrieval_context),
                    "evaluation": {
                        "answer_relevancy": ar_result,
                        "faithfulness": f_result,
                        "contextual_relevancy": cr_result,
                    },
                    "overall_score": overall_score,  # Can now be None
                    "metadata": original_data.get("metadata", {}),
                }

                evaluation_results.append(result)

                # CHANGED: Print summary with None handling
                ar_score_str = (
                    f"{ar_result['score']:.3f}"
                    if ar_result["score"] is not None
                    else "FAILED"
                )
                f_score_str = (
                    f"{f_result['score']:.3f}"
                    if f_result["score"] is not None
                    else "FAILED"
                )
                cr_score_str = (
                    f"{cr_result['score']:.3f}"
                    if cr_result["score"] is not None
                    else "FAILED"
                )
                overall_score_str = (
                    f"{overall_score:.3f}" if overall_score is not None else "FAILED"
                )

                print(
                    f"  Answer Relevancy: {ar_score_str} {'✅' if ar_result['passed'] else '❌'}"
                )
                print(
                    f"  Faithfulness: {f_score_str} {'✅' if f_result['passed'] else '❌'}"
                )
                print(
                    f"  Contextual Relevancy: {cr_score_str} {'✅' if cr_result['passed'] else '❌'}"
                )
                print(f"  Overall Score: {overall_score_str}")

            except Exception as e:
                error_msg = f"Error evaluating {original_data['id']}: {e}"
                self.errors.append(error_msg)
                print(f"❌ {error_msg}")
                continue

        self.results = evaluation_results
        return evaluation_results

    def _safe_evaluate_metric(self, metric, test_case, metric_name):
        """Safely evaluate a single metric with error handling."""

        try:
            # Clear any previous state to avoid comparison errors
            if hasattr(metric, "score"):
                metric.score = None
            if hasattr(metric, "reason"):
                metric.reason = None

            metric.measure(test_case)

            # Validate the result
            score = getattr(metric, "score", None)
            reason = getattr(metric, "reason", None)

            # CHANGED: If score is None or invalid, return None
            if score is None or not isinstance(score, (int, float)):
                return {
                    "score": None,
                    "reason": f"Evaluation returned invalid score: {score}",
                    "passed": False,
                }

            # Ensure score is in valid range
            score = max(0.0, min(1.0, float(score)))

            # Safe comparison for passed calculation
            try:
                passed = score >= 0.5
            except (TypeError, ValueError):
                passed = False

            return {
                "score": score,
                "reason": (
                    reason
                    if isinstance(reason, str)
                    else f"{metric_name} evaluation completed"
                ),
                "passed": passed,
            }

        except Exception as e:
            print(f"    ⚠️  {metric_name} failed: {e}")
            # CHANGED: Return None instead of 0.5 for failed evaluations
            return {
                "score": None,
                "reason": f"Evaluation failed: {str(e)}",
                "passed": False,
            }

    def print_summary(self, evaluation_results: List[Dict[str, Any]]):
        """Print evaluation summary with None handling."""

        if not evaluation_results:
            print("\n❌ No results to summarize")
            return

        print(f"\n📊 EVALUATION SUMMARY")
        print("=" * 80)

        # Calculate statistics
        total = len(evaluation_results)

        # CHANGED: Filter out None scores for all metrics
        ar_scores = [
            r["evaluation"]["answer_relevancy"]["score"]
            for r in evaluation_results
            if r["evaluation"]["answer_relevancy"]["score"] is not None
        ]
        f_scores = [
            r["evaluation"]["faithfulness"]["score"]
            for r in evaluation_results
            if r["evaluation"]["faithfulness"]["score"] is not None
        ]
        cr_scores = [
            r["evaluation"]["contextual_relevancy"]["score"]
            for r in evaluation_results
            if r["evaluation"]["contextual_relevancy"]["score"] is not None
        ]

        # CHANGED: Filter out None overall scores
        overall_scores = [
            r["overall_score"]
            for r in evaluation_results
            if r["overall_score"] is not None
        ]

        print(f"Total Evaluated: {total}")

        # CHANGED: Show success rates and averages
        if ar_scores:
            print(
                f"Answer Relevancy: {sum(ar_scores)/len(ar_scores):.3f} avg ({len(ar_scores)}/{total} successful)"
            )
        else:
            print(f"Answer Relevancy: No successful evaluations (0/{total} successful)")

        if f_scores:
            print(
                f"Faithfulness: {sum(f_scores)/len(f_scores):.3f} avg ({len(f_scores)}/{total} successful)"
            )
        else:
            print(f"Faithfulness: No successful evaluations (0/{total} successful)")

        if cr_scores:
            print(
                f"Contextual Relevancy: {sum(cr_scores)/len(cr_scores):.3f} avg ({len(cr_scores)}/{total} successful)"
            )
        else:
            print(
                f"Contextual Relevancy: No successful evaluations (0/{total} successful)"
            )

        if overall_scores:
            print(
                f"Overall Score: {sum(overall_scores)/len(overall_scores):.3f} avg ({len(overall_scores)}/{total} successful)"
            )
        else:
            print(f"Overall Score: No successful evaluations (0/{total} successful)")

        # Pass rates (only count successful evaluations)
        ar_pass = sum(
            1
            for r in evaluation_results
            if r["evaluation"]["answer_relevancy"]["passed"]
            and r["evaluation"]["answer_relevancy"]["score"] is not None
        )
        f_pass = sum(
            1
            for r in evaluation_results
            if r["evaluation"]["faithfulness"]["passed"]
            and r["evaluation"]["faithfulness"]["score"] is not None
        )
        cr_pass = sum(
            1
            for r in evaluation_results
            if r["evaluation"]["contextual_relevancy"]["passed"]
            and r["evaluation"]["contextual_relevancy"]["score"] is not None
        )

        print(f"\nPass Rates (threshold ≥ 0.5, among successful evaluations):")
        if ar_scores:
            print(
                f"  Answer Relevancy: {ar_pass}/{len(ar_scores)} ({ar_pass/len(ar_scores):.1%})"
            )
        if f_scores:
            print(
                f"  Faithfulness: {f_pass}/{len(f_scores)} ({f_pass/len(f_scores):.1%})"
            )
        if cr_scores:
            print(
                f"  Contextual Relevancy: {cr_pass}/{len(cr_scores)} ({cr_pass/len(cr_scores):.1%})"
            )

        # CHANGED: Show failure statistics
        ar_failures = total - len(ar_scores)
        f_failures = total - len(f_scores)
        cr_failures = total - len(cr_scores)
        overall_failures = total - len(overall_scores)

        if any([ar_failures, f_failures, cr_failures]):
            print(f"\nEvaluation Failures:")
            if ar_failures > 0:
                print(
                    f"  Answer Relevancy: {ar_failures}/{total} failed ({ar_failures/total:.1%})"
                )
            if f_failures > 0:
                print(
                    f"  Faithfulness: {f_failures}/{total} failed ({f_failures/total:.1%})"
                )
            if cr_failures > 0:
                print(
                    f"  Contextual Relevancy: {cr_failures}/{total} failed ({cr_failures/total:.1%})"
                )
            if overall_failures > 0:
                print(
                    f"  Overall: {overall_failures}/{total} items had no successful metrics ({overall_failures/total:.1%})"
                )

        if self.errors:
            print(f"\n⚠️  Processing errors encountered: {len(self.errors)}")


def main():
    """Enhanced main function with parallel processing options."""

    parser = argparse.ArgumentParser(
        description="Evaluate RAG results using University LLM with None fallback for failed metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "input_file", help="Input file with parsed RAG data (JSONL format)"
    )
    parser.add_argument(
        "--max-items", type=int, help="Maximum number of items to evaluate"
    )
    parser.add_argument("--output", "-o", help="Output file for results (JSON format)")

    # NEW: Parallel processing options
    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=8,
        help="Number of parallel workers (default: 8, use 1 for sequential)",
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Force sequential processing (same as --workers 1)",
    )

    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        print(f"❌ Error: File {args.input_file} not found")
        sys.exit(1)

    # Determine processing mode
    if args.sequential:
        args.workers = 1

    # Initialize evaluator
    print(
        "🚀 Initializing Enhanced RAG Evaluator with None Fallback for Failed Metrics"
    )
    if args.workers > 1:
        print(f"🔄 PARALLEL MODE: Using {args.workers} workers")
    else:
        print("🔄 SEQUENTIAL MODE: Using 1 worker")
    print("=" * 80)

    try:
        evaluator = ParallelRAGEvaluator(max_workers=args.workers)

        # Load data
        parsed_data = evaluator.load_parsed_data(args.input_file)

        if not parsed_data:
            print("❌ No valid data to evaluate")
            sys.exit(1)

        # Run evaluation with timing
        start_time = time.time()

        if args.workers > 1:
            print("🚀 Using PARALLEL processing...")
            evaluation_results = evaluator.evaluate_data_parallel(
                parsed_data, args.max_items
            )
        else:
            print("🔍 Using SEQUENTIAL processing...")
            evaluation_results = evaluator.evaluate_data(parsed_data, args.max_items)

        end_time = time.time()
        elapsed_time = end_time - start_time

        # Print timing information
        print(
            f"\n⏱️  Total evaluation time: {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)"
        )
        if evaluation_results:
            speed = len(evaluation_results) / elapsed_time
            print(f"📈 Processing speed: {speed:.2f} items/second")

        # Print summary
        evaluator.print_summary(evaluation_results)

        # Save results if requested
        if args.output:
            print(f"\n💾 Saving results to: {args.output}")

            output_data = {
                "metadata": {
                    "total_items": len(evaluation_results),
                    "input_file": args.input_file,
                    "evaluation_model": evaluator.llm.get_model_name(),
                    "processing_mode": "parallel" if args.workers > 1 else "sequential",
                    "workers_used": args.workers,
                    "evaluation_time_seconds": elapsed_time,
                    "items_per_second": (
                        len(evaluation_results) / elapsed_time
                        if evaluation_results
                        else 0
                    ),
                    "errors": len(evaluator.errors),
                    "fallback_behavior": "Failed metrics return None and are excluded from averages",
                },
                "results": evaluation_results,
                "errors": evaluator.errors,
            }

            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)

            print(f"✅ Results saved to {args.output}")

        print(f"\n🎉 EVALUATION COMPLETED!")

    except KeyboardInterrupt:
        print(f"\n⏹️  Evaluation interrupted by user")
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
