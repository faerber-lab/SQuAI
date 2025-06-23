from haystack import Document
from haystack_retriever import HaystackRetriever
import subprocess
import json
import shlex
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import os

logger = logging.getLogger(__name__)


def normalize(scores):
    """Normalize scores to 0-1 range"""
    if not scores:
        return []
    min_score = min(scores)
    max_score = max(scores)
    if max_score - min_score == 0:
        return [1.0 for _ in scores]
    return [(s - min_score) / (max_score - min_score) for s in scores]


class Retriever:
    """Hybrid retriever with strategy support: E5, BM25, or Hybrid"""

    def __init__(
        self,
        e5_index_directory: str,
        bm25_index_directory: str,
        top_k: int = 5,
        strategy: str = "hybrid",
        alpha: float = 0.65,
    ):
        """
        Initialize retriever with strategy support

        Args:
            strategy: "hybrid", "e5", or "bm25"
            alpha: Weight for E5 in hybrid mode (default 0.65)
        """
        self.strategy = strategy
        self.alpha = alpha
        self.top_k = top_k

        # Initialize E5 if needed
        if strategy in ["hybrid", "e5"]:
            self.e5 = HaystackRetriever(e5_index_directory)
        else:
            self.e5 = None

        # Initialize BM25 if needed
        if strategy in ["hybrid", "bm25"]:
            self.bm25_python = "bm25_env/bin/python"
            self.bm25_script = "bm25_worker.py"
            self.bm25_index_directory = bm25_index_directory
            self._bm25_retriever = None
        else:
            self._bm25_retriever = None

        # Caching and threading
        self._doc_cache = {}
        self._abstract_cache = {}
        self._cache_size = 100
        self._executor = ThreadPoolExecutor(max_workers=4)
        self._retrieval_times = []

    def _search_bm25_subprocess_optimized(self, query: str, top_k: int = 10):
        """BM25 subprocess method"""
        try:
            cmd = [
                self.bm25_python,
                self.bm25_script,
                query,
                self.bm25_index_directory,
                str(top_k),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                return json.loads(result.stdout)
            else:
                logger.warning(f"BM25 subprocess failed: {result.stderr}")
                return []
        except Exception as e:
            logger.warning(f"BM25 subprocess error: {e}")
            return []

    def _get_bm25_results(self, query, top_k):
        """Get BM25 results using subprocess method only"""
        return self._search_bm25_subprocess_optimized(query, top_k)

    def _get_e5_results(self, query, top_k):
        """Get E5 results"""
        try:
            return self.e5.retrieve(query, top_k=top_k)
        except Exception as e:
            logger.warning(f"E5 retrieval failed: {e}")
            return []

    def _fast_normalize(self, scores):
        """Fast numpy-based normalization"""
        if len(scores) == 0:
            return scores
        min_score = np.min(scores)
        max_score = np.max(scores)
        if max_score - min_score == 0:
            return np.ones_like(scores)
        return (scores - min_score) / (max_score - min_score)

    def retrieve_abstracts(self, query: str, top_k: int = None) -> list:
        """Retrieve abstracts based on configured strategy"""
        start_time = time.time()

        if top_k is None:
            top_k = self.top_k

        # Check cache
        cache_key = f"{self.strategy}_{query.lower().strip()}_{top_k}"
        if cache_key in self._abstract_cache:
            return self._abstract_cache[cache_key]

        # Route to strategy
        if self.strategy == "e5":
            result = self._retrieve_e5_only(query, top_k)
        elif self.strategy == "bm25":
            result = self._retrieve_bm25_only(query, top_k)
        else:  # hybrid
            result = self._retrieve_hybrid(query, top_k)

        # Cache result
        if len(self._abstract_cache) >= self._cache_size:
            oldest_key = next(iter(self._abstract_cache))
            del self._abstract_cache[oldest_key]
        self._abstract_cache[cache_key] = result

        elapsed = time.time() - start_time
        self._retrieval_times.append(elapsed)
        return result

    def _retrieve_e5_only(self, query: str, top_k: int) -> list:
        """E5-only retrieval"""
        if not self.e5:
            logger.error("E5 not initialized")
            return []

        e5_docs = self.e5.retrieve(query, top_k=top_k)
        result = []

        for doc in e5_docs:
            doc_id = doc.meta["paper_id"]
            abstract_text = doc.content

            self._doc_cache[doc_id] = {
                "abstract": abstract_text,
                "e5_doc": doc,
                "bm25_node": None,
                "score": doc.score,
            }

            result.append((abstract_text, doc_id))

        return result

    def _retrieve_bm25_only(self, query: str, top_k: int) -> list:
        """BM25-only retrieval"""
        if not hasattr(self, "bm25_script"):
            logger.error("BM25 not initialized")
            return []

        bm25_items = self._get_bm25_results(query, top_k)
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

    def _retrieve_hybrid(self, query: str, top_k: int) -> list:
        """Hybrid retrieval with configurable alpha"""
        if not self.e5 or not hasattr(self, "bm25_script"):
            logger.error("Both E5 and BM25 required for hybrid")
            return []

        # Parallel retrieval
        e5_future = self._executor.submit(self._get_e5_results, query, top_k * 2)
        bm25_future = self._executor.submit(self._get_bm25_results, query, top_k * 2)

        e5_docs = e5_future.result()
        bm25_items = bm25_future.result()

        # Build mappings
        e5_map = {doc.meta["paper_id"]: doc for doc in e5_docs}
        e5_scores = {doc.meta["paper_id"]: doc.score for doc in e5_docs}
        bm25_map = {it["paper_id"]: it for it in bm25_items}
        bm25_scores = {it["paper_id"]: it["score"] for it in bm25_items}

        all_ids = list(set(e5_scores.keys()).union(bm25_scores.keys()))

        if not all_ids:
            logger.warning("No documents retrieved")
            return []

        # Normalize and combine scores using alpha
        e5_score_array = np.array([e5_scores.get(pid, 0.0) for pid in all_ids])
        bm25_score_array = np.array([bm25_scores.get(pid, 0.0) for pid in all_ids])

        e5_norm = self._fast_normalize(e5_score_array)
        bm25_norm = self._fast_normalize(bm25_score_array)

        if bm25_items:
            combined_scores = self.alpha * e5_norm + (1 - self.alpha) * bm25_norm
        else:
            combined_scores = e5_norm

        # Build results
        combined = []
        for i, pid in enumerate(all_ids):
            final_score = combined_scores[i]

            if pid in e5_map:
                doc = e5_map[pid]
                abstract_text = doc.content
            elif pid in bm25_map:
                node = bm25_map[pid]
                abstract_text = node.get("text", "")
            else:
                continue

            self._doc_cache[pid] = {
                "abstract": abstract_text,
                "e5_doc": e5_map.get(pid),
                "bm25_node": bm25_map.get(pid),
                "score": final_score,
            }

            combined.append((final_score, abstract_text, pid))

        combined.sort(key=lambda x: x[0], reverse=True)
        result = [(text, doc_id) for _, text, doc_id in combined[:top_k]]

        return result

    def get_full_texts(self, doc_ids: list, db=None) -> list:
        """Get full texts based on configured strategy"""
        if not doc_ids:
            return []

        # Route to strategy
        if self.strategy == "e5":
            result = self._get_full_texts_e5_only(doc_ids, db)
        elif self.strategy == "bm25":
            result = self._get_full_texts_bm25_only(doc_ids, db)
        else:  # hybrid
            result = self._get_full_texts_hybrid(doc_ids, db)

        return result

    def _get_full_texts_e5_only(self, doc_ids: list, db=None) -> list:
        """E5-only: Use cached abstracts as full texts"""
        result = []
        for doc_id in doc_ids:
            if doc_id in self._doc_cache:
                abstract_text = self._doc_cache[doc_id]["abstract"]
                result.append((abstract_text, doc_id))

        return result

    def _get_full_texts_bm25_only(self, doc_ids: list, db=None) -> list:
        """BM25 strategy also uses LevelDB for full texts"""
        return self._get_full_texts_hybrid(doc_ids, db)

    def _get_full_texts_hybrid(self, doc_ids: list, db=None) -> list:
        """Use LevelDB for full text retrieval"""
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

    def retrieve(self, query: str, top_k: int = None):
        """Legacy method for backward compatibility"""
        if top_k is None:
            top_k = self.top_k

        cache_key = f"legacy_{query.lower().strip()}_{top_k}"
        if hasattr(self, "_legacy_cache") and cache_key in self._legacy_cache:
            return self._legacy_cache[cache_key]

        e5_future = self._executor.submit(self._get_e5_results, query, top_k * 2)
        bm25_future = self._executor.submit(self._get_bm25_results, query, top_k)

        e5_docs = e5_future.result()
        bm25_items = bm25_future.result()

        e5_map = {doc.meta["paper_id"]: doc for doc in e5_docs}
        e5_scores = {doc.meta["paper_id"]: doc.score for doc in e5_docs}
        bm25_map = {it["paper_id"]: it for it in bm25_items}
        bm25_scores = {it["paper_id"]: it["score"] for it in bm25_items}

        all_ids = set(e5_scores.keys()).union(bm25_scores.keys())

        all_ids_list = list(all_ids)
        e5_score_array = np.array([e5_scores.get(pid, 0.0) for pid in all_ids_list])
        bm25_score_array = np.array([bm25_scores.get(pid, 0.0) for pid in all_ids_list])

        e5_norm = self._fast_normalize(e5_score_array)
        bm25_norm = self._fast_normalize(bm25_score_array)

        combined = []
        for i, pid in enumerate(all_ids_list):
            final_score = 0.65 * e5_norm[i] + 0.35 * bm25_norm[i]
            if pid in e5_map:
                doc = e5_map[pid]
            else:
                node = bm25_map[pid]
                doc = Document(
                    id=node.get("paper_id", ""), content=node.get("text", "")
                )
            combined.append((final_score, doc))

        combined.sort(key=lambda x: x[0], reverse=True)

        result = [
            {
                "id": d.meta.get("paper_id") if hasattr(d, "meta") else d.id,
                "abstract": d.content,
                "semantic_score": getattr(d, "score", final_score),
            }
            for final_score, d in combined[:top_k]
        ]

        if not hasattr(self, "_legacy_cache"):
            self._legacy_cache = {}
        if len(self._legacy_cache) >= 50:
            oldest_key = next(iter(self._legacy_cache))
            del self._legacy_cache[oldest_key]
        self._legacy_cache[cache_key] = result

        return result

    def get_bm25_status(self):
        """Diagnostic method to check BM25 status"""
        return {"method": "subprocess", "available": True, "status": "SUBPROCESS_ONLY"}

    def get_performance_stats(self):
        """Get performance statistics"""
        if self._retrieval_times:
            avg_time = sum(self._retrieval_times) / len(self._retrieval_times)
            return {
                "avg_retrieval_time": avg_time,
                "total_retrievals": len(self._retrieval_times),
                "cache_sizes": {
                    "abstract_cache": len(self._abstract_cache),
                    "doc_cache": len(self._doc_cache),
                },
                "bm25_status": self.get_bm25_status(),
            }
        return {"no_data": True}

    def close(self):
        """Clean up resources"""
        try:
            if hasattr(self.e5, "close"):
                self.e5.close()
        except Exception as e:
            logger.error(f"Error closing E5 retriever: {e}")

        if hasattr(self, "_executor"):
            self._executor.shutdown(wait=True)

        self._doc_cache.clear()
        self._abstract_cache.clear()
