#!/usr/bin/env python3
"""
Fast LlamaIndex BM25 Retriever - Keeps index loaded in memory
"""

MAIN_DATA_DIR = "/data/horse/ws/inbe405h-unarxive"

import json
import logging
import time
import os
from typing import List, Tuple, Dict, Any
from pathlib import Path

try:
    from llama_index.retrievers.bm25 import BM25Retriever
    from llama_index.core import Document

    LLAMAINDEX_AVAILABLE = True
except ImportError:
    print(
        "LlamaIndex BM25 not available. Install with: pip install llama-index-retrievers-bm25"
    )
    LLAMAINDEX_AVAILABLE = False

logger = logging.getLogger(__name__)


class FastLlamaIndexBM25Retriever:
    """Fast LlamaIndex BM25 retriever that keeps index loaded in memory"""

    def __init__(self, persist_dir: str, top_k: int = 5, preload: bool = True):
        """
        Initialize with existing LlamaIndex BM25 index

        Args:
            persist_dir: BM25 index directory
            top_k: Number of documents to retrieve
            preload: Whether to load the index immediately
        """
        if not LLAMAINDEX_AVAILABLE:
            raise ImportError("LlamaIndex BM25 not available")

        self.persist_dir = Path(persist_dir)
        self.top_k = top_k
        self.retriever = None
        self._cache = {}
        self._cache_size = 100

        if preload:
            self._load_retriever()

    def _load_retriever(self):
        """Load the LlamaIndex BM25 retriever once and keep in memory"""
        if not self.persist_dir.exists():
            raise FileNotFoundError(f"BM25 index not found at {self.persist_dir}")

        self.retriever = BM25Retriever.from_persist_dir(str(self.persist_dir))

    def retrieve_abstracts(
        self, query: str, top_k: int = None
    ) -> List[Tuple[str, str]]:
        """
        Retrieve abstracts using LlamaIndex BM25

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of (abstract_text, doc_id) tuples
        """
        if top_k is None:
            top_k = self.top_k

        # Check cache
        cache_key = f"{query.lower().strip()}_{top_k}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        if not self.retriever:
            self._load_retriever()

        results = self.retriever.retrieve(query)

        # Convert to expected format
        formatted_results = []
        for result in results[:top_k]:
            paper_id = result.node.metadata.get("paper_id", "unknown")
            text = result.node.get_text()
            formatted_results.append((text, paper_id))

        # Cache result
        if len(self._cache) >= self._cache_size:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        self._cache[cache_key] = formatted_results

        return formatted_results

    def get_bm25_results(self, query: str, top_k: int) -> List[Dict]:
        """Get BM25 results in bm25_worker.py format"""
        if not self.retriever:
            self._load_retriever()

        results = self.retriever.retrieve(query)

        out = [
            {
                "paper_id": r.node.metadata.get("paper_id"),
                "text": r.node.get_text(),
                "score": r.score,
            }
            for r in results[:top_k]
        ]

        return out

    def get_full_texts(self, doc_ids: List[str], db=None) -> List[Tuple[str, str]]:
        """Get full texts for documents"""
        results = []

        if db is not None:
            # Use LevelDB if available
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
        else:
            # Fallback: try to get full_text from metadata
            if not self.retriever:
                self._load_retriever()

            for doc_id in doc_ids:
                try:
                    temp_results = self.retriever.retrieve(f"paper_id:{doc_id}")
                    if temp_results:
                        for result in temp_results:
                            if result.node.metadata.get("paper_id") == doc_id:
                                full_text = result.node.metadata.get(
                                    "full_text", result.node.get_text()
                                )
                                results.append((full_text, doc_id))
                                break
                except Exception as e:
                    logger.debug(f"Could not retrieve full text for {doc_id}: {e}")

        return results

    def retrieve(self, query: str, top_k: int = None) -> List[Dict]:
        """Legacy method for backward compatibility"""
        abstracts = self.retrieve_abstracts(query, top_k)

        results = []
        for abstract_text, doc_id in abstracts:
            result = {"id": doc_id, "abstract": abstract_text, "semantic_score": 1.0}
            results.append(result)

        return results

    def get_performance_stats(self):
        """Get performance statistics"""
        return {
            "retriever_type": "FAST_LLAMAINDEX_BM25",
            "index_loaded": self.retriever is not None,
            "cache_size": len(self._cache),
            "persist_dir": str(self.persist_dir),
        }

    def close(self):
        """Clean up resources"""
        self._cache.clear()


class FastLlamaIndexRetriever:
    """Drop-in replacement for subprocess-based retriever"""

    def __init__(
        self,
        e5_index_directory: str,
        bm25_index_directory: str,
        top_k: int = 5,
        strategy: str = "bm25",
        alpha: float = 0.65,
    ):
        """Compatible with existing Retriever class interface"""
        self.strategy = strategy
        self.alpha = alpha
        self.top_k = top_k

        # Initialize BM25
        if strategy in ["hybrid", "bm25"]:
            self.bm25 = FastLlamaIndexBM25Retriever(bm25_index_directory, top_k)
        else:
            self.bm25 = None

        # E5 initialization (if needed later)
        if strategy in ["hybrid", "e5"]:
            logger.warning("E5 not implemented in fast version yet")
            self.e5 = None
        else:
            self.e5 = None

        # Caching
        self._doc_cache = {}
        self._abstract_cache = {}
        self._cache_size = 100

    def retrieve_abstracts(
        self, query: str, top_k: int = None
    ) -> List[Tuple[str, str]]:
        """Main method to replace subprocess-based BM25 calls"""
        if top_k is None:
            top_k = self.top_k

        if self.strategy == "bm25" and self.bm25:
            return self.bm25.retrieve_abstracts(query, top_k)
        else:
            logger.error(f"Strategy {self.strategy} not supported in fast version yet")
            return []

    def get_full_texts(self, doc_ids: List[str], db=None) -> List[Tuple[str, str]]:
        """Get full texts for documents"""
        if self.bm25:
            return self.bm25.get_full_texts(doc_ids, db)
        else:
            return []

    def retrieve(self, query: str, top_k: int = None):
        """Legacy method for backward compatibility"""
        if self.bm25:
            return self.bm25.retrieve(query, top_k)
        else:
            return []

    def get_performance_stats(self):
        """Get performance statistics"""
        if self.bm25:
            return self.bm25.get_performance_stats()
        else:
            return {"error": "No retriever initialized"}

    def close(self):
        """Clean up resources"""
        if self.bm25:
            self.bm25.close()


def create_fast_llamaindex_retriever(
    bm25_index_directory: str, top_k: int = 5
) -> FastLlamaIndexRetriever:
    """
    Create a fast LlamaIndex BM25 retriever to replace subprocess approach

    Args:
        bm25_index_directory: BM25 index directory path
        top_k: Number of documents to retrieve

    Returns:
        FastLlamaIndexRetriever instance
    """
    return FastLlamaIndexRetriever("", bm25_index_directory, top_k, strategy="bm25")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python fast_llamaindex_retriever.py <bm25_index_dir> [query]")
        print(
            f"Example: python fast_llamaindex_retriever.py {MAIN_DATA_DIR}/bm25_retriever"
        )
        sys.exit(1)

    index_dir = sys.argv[1]
    test_query = sys.argv[2] if len(sys.argv) > 2 else "discrete quantum walks control"

    logging.basicConfig(level=logging.INFO)

    print(f"Testing Fast LlamaIndex BM25 with index: {index_dir}")

    retriever = create_fast_llamaindex_retriever(index_dir, top_k=5)

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

    # Test BM25 format compatibility
    print(f"\nTesting BM25 worker format compatibility:")
    bm25_results = retriever.bm25.get_bm25_results(test_query, 3)
    print(json.dumps(bm25_results[:1], indent=2))

    stats = retriever.get_performance_stats()
    print(f"\nPerformance stats: {stats}")

    retriever.close()
