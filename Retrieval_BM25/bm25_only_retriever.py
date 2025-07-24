#!/usr/bin/env python3
"""
BM25-Only Retriever - No Haystack Dependencies
"""

import json
import logging
import time
import os
from typing import List, Tuple, Dict, Any
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np

MAIN_DATA_DIR = open(f"{os.getenv('HOME')}/data_dir").read().strip() if (lambda f: f and f.strip())(open(f"{os.getenv('HOME')}/data_dir").read()) else "/data/horse/ws/inbe405h-unarxive"

try:
    from fast_llamaindex_retriever import FastLlamaIndexBM25Retriever

    FAST_BM25_AVAILABLE = True
except ImportError:
    FAST_BM25_AVAILABLE = False

logger = logging.getLogger(__name__)


class BM25OnlyRetriever:
    """BM25-Only Retriever"""

    def __init__(self, bm25_index_directory: str, top_k: int = 5, alpha: float = 0.65):
        """
        Initialize BM25-Only retriever

        Args:
            bm25_index_directory: Path to your LlamaIndex BM25 index
            top_k: Number of top-documents to retrieve
            alpha: Ignored (kept for compatibility)
        """
        self.strategy = "bm25"
        self.alpha = alpha
        self.top_k = top_k
        self.bm25_index_directory = bm25_index_directory
        self.e5 = None

        # Initialize BM25
        if FAST_BM25_AVAILABLE:
            try:
                self._fast_bm25 = FastLlamaIndexBM25Retriever(
                    bm25_index_directory, top_k, preload=True
                )
                self._use_fast_bm25 = True
            except Exception as e:
                logger.warning(f"Fast BM25 failed: {e}, falling back to subprocess")
                self._init_subprocess_fallback()
        else:
            self._init_subprocess_fallback()

        # Caching and threading
        self._doc_cache = {}
        self._abstract_cache = {}
        self._cache_size = 100
        self._executor = ThreadPoolExecutor(max_workers=4)
        self._retrieval_times = []

    def _init_subprocess_fallback(self):
        """Initialize subprocess fallback"""
        self._fast_bm25 = None
        self._use_fast_bm25 = False
        self.bm25_python = "bm25_env/bin/python"
        self.bm25_script = "bm25_worker.py"

    def retrieve_abstracts(
        self, query: str, top_k: int = None
    ) -> List[Tuple[str, str]]:
        """BM25-only abstract retrieval"""
        start_time = time.time()

        if top_k is None:
            top_k = self.top_k

        # Check cache
        cache_key = f"bm25_{query.lower().strip()}_{top_k}"
        if cache_key in self._abstract_cache:
            return self._abstract_cache[cache_key]

        result = self._retrieve_bm25_only(query, top_k)

        # Cache result
        if len(self._abstract_cache) >= self._cache_size:
            oldest_key = next(iter(self._abstract_cache))
            del self._abstract_cache[oldest_key]
        self._abstract_cache[cache_key] = result

        elapsed = time.time() - start_time
        self._retrieval_times.append(elapsed)
        return result

    def _retrieve_bm25_only(self, query: str, top_k: int) -> List[Tuple[str, str]]:
        """Pure BM25-only retrieval"""
        if self._use_fast_bm25 and self._fast_bm25:
            try:
                abstracts = self._fast_bm25.retrieve_abstracts(query, top_k)
                result = []
                for abstract_text, doc_id in abstracts:
                    self._doc_cache[doc_id] = {
                        "abstract": abstract_text,
                        "e5_doc": None,
                        "bm25_node": {"paper_id": doc_id, "text": abstract_text},
                        "score": 1.0,
                    }
                    result.append((abstract_text, doc_id))
                return result
            except Exception as e:
                logger.warning(f"Fast BM25 failed: {e}, falling back to subprocess")
                self._use_fast_bm25 = False

        return self._retrieve_bm25_subprocess(query, top_k)

    def _retrieve_bm25_subprocess(
        self, query: str, top_k: int
    ) -> List[Tuple[str, str]]:
        """Subprocess BM25 fallback"""
        if not hasattr(self, "bm25_script"):
            logger.error("BM25 subprocess not initialized")
            return []

        bm25_items = self._get_bm25_results_subprocess(query, top_k)
        result = []

        for item in bm25_items:
            doc_id = item["paper_id"]
            abstract_text = item["text"]

            self._doc_cache[doc_id] = {
                "abstract": abstract_text,
                "e5_doc": None,
                "bm25_node": item,
                "score": item["score"],
            }
            result.append((abstract_text, doc_id))

        return result

    def _get_bm25_results_subprocess(self, query: str, top_k: int) -> List[Dict]:
        """Subprocess BM25 method"""
        try:
            import subprocess

            cmd = [
                self.bm25_python,
                self.bm25_script,
                query,
                self.bm25_index_directory,
                str(top_k),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

            if result.returncode == 0:
                return json.loads(result.stdout)
            else:
                logger.warning(f"BM25 subprocess failed: {result.stderr}")
                return []
        except Exception as e:
            logger.warning(f"BM25 subprocess error: {e}")
            return []

    def get_full_texts(self, doc_ids: List[str], db=None) -> List[Tuple[str, str]]:
        """Get full texts for documents"""
        if not doc_ids:
            return []

        # Try fast BM25 first
        if self._use_fast_bm25 and self._fast_bm25:
            try:
                result = self._fast_bm25.get_full_texts(doc_ids, db)
                if result:
                    return result
            except Exception as e:
                logger.warning(f"Fast BM25 full text retrieval failed: {e}")

        return self._get_full_texts_from_db(doc_ids, db)

    def _get_full_texts_from_db(self, doc_ids: List[str], db) -> List[Tuple[str, str]]:
        """Get full texts from LevelDB"""
        if db is None:
            logger.error("No LevelDB provided for full text retrieval")
            return []

        results = []
        for doc_id in doc_ids:
            try:
                content = db.get(doc_id.encode("utf-8"))
                if content:
                    full_text = content.decode("utf-8")
                    results.append((full_text, doc_id))
                else:
                    logger.warning(f"Full text not found for {doc_id}")
            except Exception as e:
                logger.error(f"Error retrieving full text for {doc_id}: {e}")

        return results

    def retrieve(self, query: str, top_k: int = None) -> List[Dict]:
        """Legacy method for backward compatibility"""
        abstracts = self.retrieve_abstracts(query, top_k)
        results = []
        for abstract_text, doc_id in abstracts:
            result = {"id": doc_id, "abstract": abstract_text, "semantic_score": 1.0}
            results.append(result)
        return results

    def get_bm25_status(self):
        """Diagnostic method to check BM25 status"""
        if self._use_fast_bm25:
            return {
                "method": "fast_inmemory",
                "available": True,
                "status": "FAST_BM25_ACTIVE",
                "conflicts": "none",
            }
        else:
            return {
                "method": "subprocess",
                "available": True,
                "status": "SUBPROCESS_FALLBACK",
                "conflicts": "none",
            }

    def get_performance_stats(self):
        """Get performance statistics"""
        base_stats = {
            "retriever_type": "BM25_ONLY",
            "strategy": "bm25",
            "fast_bm25_active": self._use_fast_bm25,
            "cache_sizes": {
                "abstract_cache": len(self._abstract_cache),
                "doc_cache": len(self._doc_cache),
            },
        }

        if self._retrieval_times:
            avg_time = sum(self._retrieval_times) / len(self._retrieval_times)
            base_stats.update(
                {
                    "avg_retrieval_time": avg_time,
                    "total_retrievals": len(self._retrieval_times),
                }
            )

        base_stats["bm25_status"] = self.get_bm25_status()
        return base_stats

    def close(self):
        """Clean up resources"""
        if hasattr(self, "_fast_bm25") and self._fast_bm25:
            try:
                self._fast_bm25.close()
            except Exception as e:
                logger.error(f"Error closing Fast BM25: {e}")

        if hasattr(self, "_executor"):
            self._executor.shutdown(wait=True)

        self._doc_cache.clear()
        self._abstract_cache.clear()


def create_bm25_only_retriever(
    bm25_index_directory: str, top_k: int = 5
) -> BM25OnlyRetriever:
    """
    Create a BM25-only retriever without Haystack dependencies

    Args:
        bm25_index_directory: Path to your LlamaIndex BM25 index
        top_k: Number of documents to retrieve

    Returns:
        BM25OnlyRetriever instance
    """
    return BM25OnlyRetriever(bm25_index_directory, top_k)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python bm25_only_retriever.py <bm25_index_dir> [query]")
        print(
            f"Example: python bm25_only_retriever.py {MAIN_DATA_DIR}/bm25_retriever"
        )
        sys.exit(1)

    index_dir = sys.argv[1]
    test_query = sys.argv[2] if len(sys.argv) > 2 else "discrete quantum walks control"

    logging.basicConfig(level=logging.INFO)

    print(f"Testing BM25-ONLY retriever")
    print(f"Index: {index_dir}")

    retriever = create_bm25_only_retriever(index_dir, top_k=5)

    print(f"\nTesting query: '{test_query}'")

    # Test retrieval speed
    start_time = time.time()
    results = retriever.retrieve_abstracts(test_query, top_k=5)
    first_time = time.time() - start_time

    start_time = time.time()
    results = retriever.retrieve_abstracts(test_query, top_k=5)
    cached_time = time.time() - start_time

    print(f"First query: {first_time:.3f} seconds")
    print(f"Cached query: {cached_time:.3f} seconds")

    if first_time > 0 and cached_time > 0:
        print(f"Speedup: {first_time/cached_time:.1f}x")

    print(f"\nRetrieved {len(results)} documents:")
    for i, (text, doc_id) in enumerate(results, 1):
        print(f"[{i}] {doc_id}")
        print(f"    {text[:150]}...")

    stats = retriever.get_performance_stats()
    print(f"\nPerformance stats: {stats}")

    retriever.close()
    print("\nTest completed successfully!")
