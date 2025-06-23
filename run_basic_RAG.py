#!/usr/bin/env python3
"""
Strategy-Aware Basic RAG adapted to use your local hybrid Retriever
SUPPORTS: E5-only, BM25-only, and Hybrid strategies with configurable alpha
IMPROVED: Better output format with retrieved passages and document titles
"""

import argparse
import json
import time
import datetime
import os
import logging
import numpy as np
from tqdm import tqdm
import re
import random
import string


# Generate a unique ID for log filename
def get_unique_log_filename():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    random_str = "".join(random.choices(string.ascii_lowercase + string.digits, k=6))
    return f"logs/strategy_basic_rag_runner_{timestamp}_{random_str}.log"


# Create logs directory
os.makedirs("logs", exist_ok=True)

# Configure logging with unique filename
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(get_unique_log_filename()), logging.StreamHandler()],
)
logger = logging.getLogger("STRATEGY_BASIC_RAG_Runner")

# Import YOUR Retriever class AND configuration
from hybrid_retriever import Retriever

try:
    from config import E5_INDEX_DIR, BM25_INDEX_DIR, DB_PATH

    USING_CONFIG = True
    logger.info("Using configuration from config.py")
except ImportError:
    logger.warning("config.py not found, using command line arguments")
    E5_INDEX_DIR = None
    BM25_INDEX_DIR = None
    DB_PATH = None
    USING_CONFIG = False


class StrategyBasicRAG:
    def __init__(
        self,
        retriever,
        agent_model=None,
        top_k=10,
        falcon_api_key=None,
        strategy="hybrid",
    ):
        """
        Strategy-aware Basic RAG implementation using configurable Retriever.

        Args:
            retriever: Retriever instance with strategy support
            agent_model: Model name or pre-initialized agent
            top_k: Number of documents to retrieve and use
            falcon_api_key: API key for Falcon if using API
            strategy: Retrieval strategy ("e5", "bm25", "hybrid")
        """
        self.retriever = retriever
        self.top_k = top_k
        self.strategy = strategy

        # Initialize LLM agent
        if isinstance(agent_model, str):
            if "falcon" in agent_model.lower() and falcon_api_key:
                # Initialize Falcon agent if using Falcon API
                from api_agent import FalconAgent

                self.agent = FalconAgent(api_key=falcon_api_key)
                logger.info("Using Falcon agent with API")
            else:
                # Initialize local LLM agent
                from local_agent import LLMAgent

                self.agent = LLMAgent(agent_model)
                logger.info(f"Using local LLM agent with model {agent_model}")
        else:
            # Use pre-initialized agent
            self.agent = agent_model
            logger.info("Using pre-initialized agent")

        logger.info(
            f"Strategy-aware Basic RAG initialized with {strategy.upper()} strategy"
        )

    def close(self):
        """Clean up resources"""
        try:
            if hasattr(self.retriever, "close"):
                self.retriever.close()
                logger.info("Retriever closed successfully")
        except Exception as e:
            logger.warning(f"Error closing retriever: {e}")

        try:
            # Clear any agent resources if needed
            if hasattr(self.agent, "close"):
                self.agent.close()
        except Exception as e:
            logger.debug(f"Agent cleanup: {e}")

        logger.info(f"{self.strategy.upper()} Basic RAG system closed")

    def _extract_document_title(self, doc_text, doc_id):
        """
        Extract document title from the document text or metadata.
        """
        # Try to extract title from the beginning of the document
        lines = doc_text.split("\n")

        # Look for title patterns in first few lines
        for i, line in enumerate(lines[:5]):
            line = line.strip()
            if len(line) > 10 and len(line) < 200:
                # Check if it looks like a title (not too short, not too long)
                if not line.startswith(
                    ("Abstract", "Introduction", "Methods", "Results")
                ):
                    # Remove common prefixes and clean up
                    title = re.sub(r"^(Title|TITLE):\s*", "", line)
                    title = re.sub(r'^["\']|["\']$', "", title)  # Remove quotes
                    if len(title) > 10:
                        return title

        # Fallback: use first 100 characters as title
        first_line = doc_text.split("\n")[0].strip()
        if len(first_line) > 100:
            return first_line[:100] + "..."
        elif len(first_line) > 10:
            return first_line
        else:
            return f"Document {doc_id}"

    # When referencing information from the documents, please mention the document title when possible.

    def _create_rag_prompt(self, query, documents_with_titles):
        """
        Create strategy-aware prompt for the LLM with retrieved documents and titles.
        """
        # Clean and format documents with titles
        docs_text = ""
        for i, (doc_text, doc_id, doc_title) in enumerate(documents_with_titles):
            # Basic text cleaning
            clean_text = self._clean_document_text(doc_text)
            docs_text += f"\nDocument {i+1} - Title: {doc_title}\nID: {doc_id}\nContent: {clean_text}\n"

        return f"""You are an accurate and helpful AI assistant. Answer the question based ONLY on the information provided in the documents below. If the documents don't contain the necessary information to answer the question, simply state that you don't have enough information.

Documents:
{docs_text}

Question: {query}

Answer:"""

    def _clean_document_text(self, text: str) -> str:
        """Clean document text for better presentation"""
        # Remove common JSON artifacts and technical markup
        import re

        # Remove JSON-like structures
        text = re.sub(r"\{[^}]*\}", "", text)

        # Remove section markers
        text = re.sub(r"'section':\s*'[^']*',\s*'text':\s*'", "", text)

        # Remove LaTeX commands and math markup
        text = re.sub(r"\\[a-zA-Z]+\{[^}]*\}", "", text)
        text = re.sub(r"\$[^$]*\$", "[MATH]", text)

        # Clean up whitespace
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"\n\s*\n", "\n", text)

        # Take first 500 characters to avoid overwhelming the prompt
        if len(text) > 500:
            text = text[:500] + "..."

        return text.strip()

    def answer_query(self, query):
        """
        Process a query using strategy-aware basic RAG approach.
        IMPROVED: Now includes document titles and detailed passage information
        """
        logger.info(f"Processing query with {self.strategy.upper()} basic RAG: {query}")

        # Step 1: Use strategy-aware retrieve_abstracts method
        logger.info(
            f"Retrieving top-{self.top_k} documents using {self.strategy.upper()} strategy..."
        )

        # The retriever now handles strategy internally
        retrieved_abstracts = self.retriever.retrieve_abstracts(query, top_k=self.top_k)
        logger.info(
            f"Retrieved {len(retrieved_abstracts)} abstracts using {self.strategy.upper()}"
        )

        # Step 2: Extract titles and prepare documents
        docs_with_titles = []
        docs_for_prompt = []

        for abstract_text, doc_id in retrieved_abstracts:
            # Extract document title
            doc_title = self._extract_document_title(abstract_text, doc_id)

            docs_with_titles.append((abstract_text, doc_id, doc_title))
            docs_for_prompt.append((abstract_text, doc_id, doc_title))

        # DEBUG: Log what documents were actually retrieved with titles
        logger.info(f"DEBUG: {self.strategy.upper()} retrieved documents with titles:")
        for i, (abstract_text, doc_id, doc_title) in enumerate(docs_with_titles):
            logger.info(f"   Doc {i+1} (ID: {doc_id}): {doc_title}")
            logger.info(f"   Content: {abstract_text[:150]}...")

        # Step 3: Create strategy-aware prompt with documents and titles
        prompt = self._create_rag_prompt(query, docs_for_prompt)

        # DEBUG: Log the actual prompt sent to LLM (truncated)
        logger.info(f"DEBUG: {self.strategy.upper()} prompt sent to LLM:")
        logger.info(f"{prompt[:800]}...")

        # Step 4: Generate answer
        logger.info(
            f"Generating answer from LLM using {self.strategy.upper()} context..."
        )
        answer = self.agent.generate(prompt)
        logger.info(f"Answer: {answer}")

        # Step 5: Build comprehensive debug information with titles
        debug_info = {
            "raw_answer": answer,
            "retrieved_docs": [
                (abstract_text, doc_id) for abstract_text, doc_id, _ in docs_for_prompt
            ],
            "retrieved_docs_with_titles": docs_with_titles,
            "prompt": prompt,
            "retrieval_method": f"{self.strategy}_strategy",
            "strategy": self.strategy,
            "docs_count": len(docs_for_prompt),
            "alpha": (
                getattr(self.retriever, "alpha", None)
                if self.strategy == "hybrid"
                else None
            ),
            # IMPROVED: Add detailed passage information with titles
            "passages_detail": [
                {
                    "doc_id": doc_id,
                    "title": doc_title,
                    "text": abstract_text,
                    "rank": i + 1,
                    "length": len(abstract_text),
                    "preview": (
                        abstract_text[:200] + "..."
                        if len(abstract_text) > 200
                        else abstract_text
                    ),
                }
                for i, (abstract_text, doc_id, doc_title) in enumerate(docs_with_titles)
            ],
        }

        return answer, debug_info


def load_questions(file_path):
    """Load questions from JSON or JSONL file."""
    is_jsonl = file_path.lower().endswith(".jsonl")

    try:
        questions = []

        if is_jsonl:
            logger.info(f"Loading questions from JSONL file: {file_path}")
            with open(file_path, "r", encoding="utf-8") as f:
                line_num = 0
                for line in f:
                    line_num += 1
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        question = json.loads(line)
                        if "id" not in question:
                            question["id"] = line_num
                        questions.append(question)
                    except json.JSONDecodeError as e:
                        logger.error(f"Error parsing JSON at line {line_num}: {e}")
        else:
            logger.info(f"Loading questions from JSON file: {file_path}")
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

                if isinstance(data, list):
                    questions = data
                    for i, question in enumerate(questions):
                        if "id" not in question:
                            question["id"] = i + 1
                elif isinstance(data, dict):
                    if "questions" in data:
                        questions = data["questions"]
                    elif "question" in data:
                        questions = [data]
                    else:
                        questions = [data]

        logger.info(f"Loaded {len(questions)} questions")
        return questions

    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error loading questions: {e}")
        return []


def format_result_comprehensive(result):
    """
    IMPROVED: Comprehensive result formatting that includes retrieved passages with titles.
    This replaces the original format_result function.
    """
    # Extract passage information with titles
    retrieved_passages = []
    if "retrieved_docs_with_titles" in result:
        for i, (doc_text, doc_id, doc_title) in enumerate(
            result["retrieved_docs_with_titles"]
        ):
            passage = {
                "passage_id": doc_id,
                "passage_title": doc_title,
                "passage_text": doc_text,
                "passage_rank": i + 1,
                "passage_length": len(doc_text),
                "passage_preview": (
                    doc_text[:200] + "..." if len(doc_text) > 200 else doc_text
                ),
            }
            retrieved_passages.append(passage)
    elif "retrieved_docs" in result:
        # Fallback for old format without titles
        for i, (doc_text, doc_id) in enumerate(result["retrieved_docs"]):
            passage = {
                "passage_id": doc_id,
                "passage_title": f"Document {doc_id}",
                "passage_text": doc_text,
                "passage_rank": i + 1,
                "passage_length": len(doc_text),
                "passage_preview": (
                    doc_text[:200] + "..." if len(doc_text) > 200 else doc_text
                ),
            }
            retrieved_passages.append(passage)

    # Create comprehensive result
    formatted_result = {
        # Basic information
        "question_id": result.get("id", 0),
        "question": result.get("question", ""),
        "generated_answer": result.get("model_answer", ""),
        "reference_answer": result.get("reference_answer", ""),
        # Retrieval strategy information
        "retrieval_strategy": {
            "strategy_type": result.get("strategy", "unknown"),
            "method": result.get("retrieval_method", "strategy_basic_rag"),
            "total_documents": len(retrieved_passages),
            "alpha_weight": result.get("alpha"),  # for hybrid strategy
        },
        # IMPROVED: Retrieved passages with titles and full details
        "retrieved_passages": retrieved_passages,
        # Performance metrics
        "performance_metrics": {
            "processing_time_seconds": result.get("process_time", 0),
            "total_content_length": sum(
                len(passage["passage_text"]) for passage in retrieved_passages
            ),
            "average_passage_length": (
                sum(len(passage["passage_text"]) for passage in retrieved_passages)
                / max(len(retrieved_passages), 1)
            ),
            "retrieval_efficiency": len(retrieved_passages)
            / max(result.get("process_time", 1), 0.1),
        },
        # Metadata
        "timestamp": result.get("timestamp"),
        "model_used": result.get("model_name", "unknown"),
    }

    return formatted_result


def write_comprehensive_results_to_jsonl(results, output_file):
    """Write comprehensive results to JSONL file."""
    with open(output_file, "w", encoding="utf-8") as f:
        for result in results:
            formatted_result = format_result_comprehensive(result)
            f.write(json.dumps(formatted_result, ensure_ascii=False) + "\n")
    logger.info(f"Comprehensive results written to {output_file}")


def write_comprehensive_result_to_json(result, output_file):
    """Write a comprehensive result to JSON file."""
    formatted_result = format_result_comprehensive(result)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(formatted_result, f, indent=2, ensure_ascii=False)
    logger.info(f"Comprehensive result written to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Strategy-Aware Basic RAG with E5/BM25/Hybrid Support"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="tiiuae/Falcon3-10B-Instruct",
        help="Model for LLM agent",
    )
    parser.add_argument(
        "--falcon_api_key",
        type=str,
        default=None,
        help="API key for Falcon API (only needed if using Falcon API)",
    )
    # Strategy parameters
    parser.add_argument(
        "--retriever_type",
        choices=["e5", "bm25", "hybrid"],
        default="hybrid",
        help="Retrieval strategy: e5 (E5 only), bm25 (BM25 only), hybrid (E5+BM25)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.65,
        help="Weight for E5 in hybrid mode (0.0=BM25 only, 1.0=E5 only, default=0.65)",
    )
    # Index directories
    parser.add_argument(
        "--e5_index_dir",
        type=str,
        default="/data/horse/ws/inbe405h-unarxive/faiss_index",
        help="Directory containing FAISS E5 index",
    )
    parser.add_argument(
        "--bm25_index_dir",
        type=str,
        default="/data/horse/ws/inbe405h-unarxive/bm25_retriever",
        help="Directory containing BM25 index",
    )
    parser.add_argument(
        "--index_dir",
        type=str,
        default="/data/horse/ws/inbe405h-unarxive/test_index",
        help="Main index directory (for compatibility)",
    )
    parser.add_argument(
        "--top_k", type=int, default=10, help="Number of documents to retrieve and use"
    )
    parser.add_argument(
        "--data_file",
        type=str,
        default="quick_test_questions.jsonl",
        help="File containing questions (JSON or JSONL)",
    )
    parser.add_argument(
        "--single_question",
        type=str,
        default=None,
        help="Process a single question instead of the entire dataset",
    )
    parser.add_argument(
        "--output_format",
        choices=["json", "jsonl"],
        default="jsonl",
        help="Output format: 'json' for single file, 'jsonl' for line-delimited JSON",
    )
    parser.add_argument(
        "--output_dir", type=str, default="results", help="Directory to save results"
    )
    args = parser.parse_args()

    # Use config.py paths if available, otherwise use command line arguments
    if USING_CONFIG:
        e5_index_dir = E5_INDEX_DIR
        bm25_index_dir = BM25_INDEX_DIR
        logger.info(f"Using config.py paths:")
        logger.info(f"   E5 index: {e5_index_dir}")
        logger.info(f"   BM25 index: {bm25_index_dir}")
    else:
        e5_index_dir = args.e5_index_dir
        bm25_index_dir = args.bm25_index_dir
        logger.info(f"Using command line paths:")
        logger.info(f"   E5 index: {e5_index_dir}")
        logger.info(f"   BM25 index: {bm25_index_dir}")

    # Initialize strategy-aware Retriever
    logger.info(f"Initializing {args.retriever_type.upper()} Retriever...")
    if args.retriever_type == "hybrid":
        logger.info(
            f"Hybrid strategy with alpha={args.alpha} (E5={args.alpha:.2f}, BM25={1-args.alpha:.2f})"
        )

    retriever = Retriever(
        e5_index_directory=e5_index_dir,
        bm25_index_directory=bm25_index_dir,
        top_k=args.top_k,
        strategy=args.retriever_type,  # Strategy parameter
        alpha=args.alpha,  # Alpha parameter
    )

    # Show retriever status
    if hasattr(retriever, "get_bm25_status"):
        bm25_status = retriever.get_bm25_status()
        logger.info(f"BM25 Status: {bm25_status['method']} ({bm25_status['status']})")

    # Initialize Strategy-aware BasicRAG
    logger.info(
        f"Initializing strategy-aware basic RAG with {args.retriever_type.upper()} and top_k={args.top_k}..."
    )
    rag = StrategyBasicRAG(
        retriever,
        agent_model=args.model,
        top_k=args.top_k,
        falcon_api_key=args.falcon_api_key,
        strategy=args.retriever_type,
    )

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Process a single question if specified
    if args.single_question:
        logger.info(
            f"Processing single question with {args.retriever_type.upper()} strategy: {args.single_question}"
        )
        start_time = time.time()

        try:
            # Process the query
            answer, debug_info = rag.answer_query(args.single_question)

            # Calculate processing time
            process_time = time.time() - start_time

            # Create comprehensive result object with titles
            result = {
                "id": f"single_question_{args.retriever_type}",
                "question": args.single_question,
                "model_answer": answer,
                "process_time": process_time,
                "retrieved_docs": debug_info["retrieved_docs"],
                "retrieved_docs_with_titles": debug_info["retrieved_docs_with_titles"],
                "retrieval_method": debug_info["retrieval_method"],
                "strategy": debug_info["strategy"],
                "alpha": debug_info["alpha"],
                "passages_detail": debug_info["passages_detail"],
                "timestamp": datetime.datetime.now().isoformat(),
                "model_name": args.model,
            }

            logger.info(f"Answer: {answer}")
            logger.info(f"Processing time: {process_time:.2f} seconds")
            logger.info(
                f"Documents used: {len(debug_info['retrieved_docs_with_titles'])}"
            )
            if debug_info["alpha"] is not None:
                logger.info(f"Alpha used: {debug_info['alpha']}")

            # Show document titles
            logger.info("Document titles used:")
            for i, (_, doc_id, doc_title) in enumerate(
                debug_info["retrieved_docs_with_titles"]
            ):
                logger.info(f"   {i+1}. {doc_title} (ID: {doc_id})")

            # Save result
            output_file = os.path.join(
                args.output_dir,
                f"comprehensive_rag_{args.retriever_type}_single_{timestamp}.json",
            )
            write_comprehensive_result_to_json(result, output_file)

        except Exception as e:
            logger.error(f"Error processing question: {e}", exc_info=True)
        finally:
            rag.close()

        return

    # Load questions
    questions = load_questions(args.data_file)
    if not questions:
        logger.error("No questions found. Exiting.")
        return

    # Process each question
    results = []

    for i, item in enumerate(questions):
        question_id = item.get("id", i + 1)
        question_text = item.get("question") or item.get("text")
        logger.info(
            f"Processing question {i+1}/{len(questions)} with {args.retriever_type.upper()}: {question_text}"
        )
        start_time = time.time()

        try:
            # Process the query
            answer, debug_info = rag.answer_query(question_text)

            # Calculate processing time
            process_time = time.time() - start_time

            # IMPROVED: Save comprehensive result with titles and passages
            result = {
                "id": question_id,
                "question": question_text,
                "reference_answer": item.get("answer", ""),
                "model_answer": answer,
                "process_time": process_time,
                "retrieved_docs": debug_info["retrieved_docs"],
                "retrieved_docs_with_titles": debug_info["retrieved_docs_with_titles"],
                "retrieval_method": debug_info["retrieval_method"],
                "strategy": debug_info["strategy"],
                "alpha": debug_info["alpha"],
                "passages_detail": debug_info["passages_detail"],
                "timestamp": datetime.datetime.now().isoformat(),
                "model_name": args.model,
            }
            results.append(result)

            logger.info(f"Answer: {answer}")
            logger.info(f"Processing time: {process_time:.2f} seconds")
            logger.info(
                f"Documents used: {len(debug_info['retrieved_docs_with_titles'])}"
            )

            # Show document titles
            for j, (_, doc_id, doc_title) in enumerate(
                debug_info["retrieved_docs_with_titles"][:3]
            ):
                logger.info(f"   Doc {j+1}: {doc_title}")

        except Exception as e:
            logger.error(f"Error processing question {question_id}: {e}", exc_info=True)

    # Clean up
    rag.close()

    # Save all results with comprehensive format
    if results:
        random_num = "".join(
            random.choices(string.ascii_lowercase + string.digits, k=6)
        )
        if args.output_format == "jsonl":
            output_file = os.path.join(
                args.output_dir,
                f"comprehensive_rag_{args.retriever_type}_answers_{timestamp}_{random_num}.jsonl",
            )
            write_comprehensive_results_to_jsonl(results, output_file)
        else:  # json
            for result in results:
                question_id = result["id"]
                output_file = os.path.join(
                    args.output_dir,
                    f"comprehensive_rag_{args.retriever_type}_answer_{question_id}_{timestamp}.json",
                )
                write_comprehensive_result_to_json(result, output_file)

    logger.info(
        f"Processed {len(results)} questions with {args.retriever_type.upper()} strategy."
    )

    # Print summary statistics with titles info
    if results:
        avg_time = sum(r["process_time"] for r in results) / len(results)
        avg_docs = sum(len(r["retrieved_docs_with_titles"]) for r in results) / len(
            results
        )
        strategy_info = (
            f" (alpha={args.alpha})" if args.retriever_type == "hybrid" else ""
        )

        logger.info(f"Summary for {args.retriever_type.upper()}{strategy_info}:")
        logger.info(f"   Average processing time: {avg_time:.2f} seconds")
        logger.info(f"   Average documents used: {avg_docs:.1f}")
        logger.info(f"   Total questions processed: {len(results)}")
        logger.info(f"   Document titles extracted: Yes")


if __name__ == "__main__":
    main()
