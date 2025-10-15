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
import get_paths

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
    """
    SIMPLE Hybrid retriever with strategy support: E5, BM25, or Hybrid
    """

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

        logger.info(f"Initializing {strategy.upper()} retriever...")
        if strategy == "hybrid":
            logger.info(f"Hybrid weights: E5={alpha:.2f}, BM25={1-alpha:.2f}")

        # Initialize E5 if needed
        if strategy in ["hybrid", "e5"]:
            self.e5 = HaystackRetriever(e5_index_directory)
            logger.info("E5 initialized")
        else:
            self.e5 = None

        # Initialize BM25 if needed
        if strategy in ["hybrid", "bm25"]:
            self.bm25_python = get_paths.get_bm25_python_path()
            self.bm25_script = "bm25_worker.py"
            self.bm25_fulltext_script = "bm25_fulltext_worker.py"
            self.bm25_index_directory = bm25_index_directory
            self._bm25_retriever = self._load_bm25_into_memory()
            logger.info("BM25 initialized")
        else:
            self._bm25_retriever = None

        # Caching and threading (always needed)
        self._doc_cache = {}
        self._abstract_cache = {}
        self._fulltext_cache = {}
        self._cache_size = 100
        self._executor = ThreadPoolExecutor(max_workers=4)
        self._retrieval_times = []

    def _load_bm25_into_memory(self):
        """Load BM25 index into memory if possible"""
        logger.info("Attempting to load BM25 index into memory...")

        try:
            start_time = time.time()
            from llama_index.indices.bm25_retriever import BM25Retriever

            retriever = BM25Retriever.from_persist_dir(self.bm25_index_directory)
            elapsed = time.time() - start_time
            logger.info(f"BM25 index loaded into memory in {elapsed:.2f}s!")
            return retriever
        except Exception as e:
            logger.warning(f"Could not load BM25 into memory: {e}")
            logger.info("Will use subprocess method instead")
            return None

    def _search_bm25_memory(self, query: str, top_k: int = 10):
        """Fast BM25 search using in-memory index"""
        if not self._bm25_retriever:
            return []

        try:
            start_time = time.time()
            results = self._bm25_retriever.retrieve(query)

            formatted_results = []
            for result in results[:top_k]:
                formatted_results.append(
                    {
                        "paper_id": result.node.metadata.get(
                            "paper_id", result.node.id_
                        ),
                        "title": result.node.metadata.get("title", ""),
                        "text": result.node.get_content(),
                        "score": result.score,
                    }
                )

            elapsed = time.time() - start_time
            logger.info(
                f"⚡ BM25 in-memory: {len(formatted_results)} results in {elapsed:.3f}s"
            )
            return formatted_results
        except Exception as e:
            logger.warning(f"BM25 memory search failed: {e}")
            return []

    def _search_bm25_subprocess_optimized(self, query: str, top_k: int = 10):
        """BM25 subprocess method without timeout"""
        try:
            start_time = time.time()
            cmd = [
                self.bm25_python,
                self.bm25_script,
                query,
                self.bm25_index_directory,
                str(top_k),
            ]

            # No timeout - let BM25 run as long as needed
            result = subprocess.run(cmd, capture_output=True, text=True)

            elapsed = time.time() - start_time

            if result.returncode == 0:
                bm25_results = json.loads(result.stdout)
                logger.info(
                    f"BM25 subprocess: {len(bm25_results)} results in {elapsed:.2f}s"
                )
                return bm25_results
            else:
                logger.warning(f"BM25 subprocess failed: {result.stderr}")
                return []
        except Exception as e:
            logger.warning(f"BM25 subprocess error: {e}")
            return []

    def _get_bm25_results(self, query, top_k):
        """Get BM25 results using best available method"""
        if self._bm25_retriever:
            results = self._search_bm25_memory(query, top_k)
            if results:
                return results
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
        """
        STRATEGY-AWARE: Retrieve abstracts based on configured strategy
        """
        start_time = time.time()

        if top_k is None:
            top_k = self.top_k

        # Check cache
        cache_key = f"{self.strategy}_{query.lower().strip()}_{top_k}"
        if cache_key in self._abstract_cache:
            logger.info(f"⚡ {self.strategy.upper()} cache hit!")
            return self._abstract_cache[cache_key]

        logger.info(f"🔍 {self.strategy.upper()} retrieval for query: {query}")

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
        logger.info(
            f"{self.strategy.upper()}: Retrieved {len(result)} abstracts in {elapsed:.2f}s"
        )
        return result

    def _retrieve_e5_only(self, query: str, top_k: int) -> list:
        """E5-only retrieval"""
        if not self.e5:
            logger.error("E5 not initialized!")
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

        logger.info(f"E5-only: {len(result)} documents")
        return result

    def _retrieve_bm25_only(self, query: str, top_k: int) -> list:
        """BM25-only retrieval"""
        if not self._bm25_retriever and not hasattr(self, "bm25_script"):
            logger.error("BM25 not initialized!")
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

        logger.info(f"BM25-only: {len(result)} documents")
        return result

    def _retrieve_hybrid(self, query: str, top_k: int) -> list:
        """Hybrid retrieval with configurable alpha"""
        if not self.e5 or not hasattr(self, "bm25_script"):
            logger.error("Both E5 and BM25 required for hybrid!")
            return []

        bm25_method = "in-memory" if self._bm25_retriever else "subprocess"
        logger.info(f"BM25 method: {bm25_method}")

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
            logger.info(
                f"True hybrid: E5({len(e5_docs)}) + BM25({len(bm25_items)}) with α={self.alpha}"
            )
        else:
            combined_scores = e5_norm
            logger.info(f"E5-only fallback: BM25 unavailable")

        # Build results
        combined = []
        for i, pid in enumerate(all_ids):
            final_score = combined_scores[i]

            if pid in e5_map:
                doc = e5_map[pid]
                abstract_text = doc.content
            elif pid in bm25_map:
                node = bm25_map[pid]
                print("node bm25: ", node)
                title = node.get("title", "")
                abstract_text = node.get("text", "")
            else:
                continue

            self._doc_cache[pid] = {
                "abstract": abstract_text,
                "title": title,
                "e5_doc": e5_map.get(pid),
                "bm25_node": bm25_map.get(pid),
                "score": final_score,
            }

            combined.append((final_score, abstract_text, title, pid))

        combined.sort(key=lambda x: x[0], reverse=True)
        result = [(text, doc_id) for _, text, doc_id in combined[:top_k]]

        logger.info(f"Hybrid: {len(result)} documents")
        return result

    def get_full_texts(self, doc_ids: list, db=None) -> list:
        """
        STRATEGY-AWARE: Get full texts based on configured strategy
        """
        if not doc_ids:
            return []

        start_time = time.time()
        logger.info(
            f"{self.strategy.upper()}: Retrieving full texts for {len(doc_ids)} documents"
        )

        # Route to strategy
        if self.strategy == "e5":
            result = self._get_full_texts_e5_only(doc_ids, db)
        elif self.strategy == "bm25":
            result = self._get_full_texts_bm25_only(doc_ids, db)
        else:  # hybrid
            result = self._get_full_texts_hybrid(doc_ids, db)

        elapsed = time.time() - start_time
        total_chars = sum(len(text) for text, _ in result)
        avg_length = total_chars // max(len(result), 1)
        logger.info(
            f"{self.strategy.upper()}: Retrieved {len(result)} full texts in {elapsed:.2f}s (avg {avg_length} chars/doc)"
        )

        return result

    def _get_full_texts_e5_only(self, doc_ids: list, db=None) -> list:
        """E5-only: Use cached abstracts as full texts"""
        logger.info("E5-only: Using abstracts as full texts")

        result = []
        for doc_id in doc_ids:
            if doc_id in self._doc_cache:
                abstract_text = self._doc_cache[doc_id]["abstract"]
                result.append((abstract_text, doc_id))
            else:
                logger.debug(f"No cached content for {doc_id}, skipping")

        return result

    def _get_full_texts_bm25_only(self, doc_ids: list, db=None) -> list:
        """BM25-only: Use BM25 system for full text retrieval"""
        logger.info("BM25-only: Using BM25 for full text retrieval")

        # Use the existing BM25 full text retrieval logic
        return self._get_full_texts_hybrid(doc_ids, db)  # BM25 logic is the same

    def _get_full_texts_hybrid(self, doc_ids: list, db=None) -> list:
        """Hybrid: Use existing optimized batch retrieval"""
        logger.info(f"Hybrid: Using optimized batch retrieval (α={self.alpha})")

        # Step 1: Check cache first
        cached_results = {}
        missing_ids = []

        for doc_id in doc_ids:
            if doc_id in self._fulltext_cache:
                cached_results[doc_id] = self._fulltext_cache[doc_id]
            else:
                missing_ids.append(doc_id)

        logger.info(
            f"Cache hits: {len(cached_results)}, Need to fetch: {len(missing_ids)}"
        )

        # Step 2: Batch retrieve missing documents
        if missing_ids:
            batch_results = self._optimized_batch_retrieve(missing_ids)
            cached_results.update(batch_results)

            # Update cache with size management
            for doc_id, text in batch_results.items():
                if len(self._fulltext_cache) >= self._cache_size:
                    oldest_key = next(iter(self._fulltext_cache))
                    del self._fulltext_cache[oldest_key]
                self._fulltext_cache[doc_id] = text

        # Step 3: Prepare final results
        final_texts = []

        for doc_id in doc_ids:
            if doc_id not in cached_results:
                # Fallback to abstract if full text not available
                if doc_id in self._doc_cache:
                    text = self._doc_cache[doc_id]["abstract"]
                    logger.debug(f"Using abstract as fallback for {doc_id}")
                else:
                    continue
            else:
                text = cached_results[doc_id]

            final_texts.append((text, doc_id))

        return final_texts

    def _optimized_batch_retrieve(self, doc_ids: list) -> dict:
        """Multi-strategy batch retrieval"""
        results = {}

        # Try batch worker without timeout
        batch_results = self._try_batch_worker_with_timeout(doc_ids, timeout=None)
        if batch_results:
            results.update(batch_results)
            successfully_retrieved = set(batch_results.keys())
            remaining_ids = [
                doc_id for doc_id in doc_ids if doc_id not in successfully_retrieved
            ]
        else:
            remaining_ids = doc_ids

        # Parallel individual retrieval for remaining docs
        if remaining_ids:
            logger.info(
                f"Using parallel individual retrieval for {len(remaining_ids)} remaining docs"
            )
            parallel_results = self._parallel_individual_retrieve(remaining_ids)
            results.update(parallel_results)

        return results

    def _try_batch_worker_with_timeout(self, doc_ids: list, timeout=None) -> dict:
        """Try batch worker"""
        worker_scripts = ["bm25_fulltext_worker_batch.py"]

        if os.path.exists("bm25_fulltext_worker_optimized.py"):
            worker_scripts.insert(0, "bm25_fulltext_worker_optimized.py")

        for worker_script in worker_scripts:
            try:
                cmd = [
                    self.bm25_python,
                    worker_script,
                    json.dumps(doc_ids),
                    self.bm25_index_directory,
                ]

                if timeout is not None:
                    result = subprocess.run(
                        cmd, capture_output=True, text=True, timeout=timeout
                    )
                else:
                    result = subprocess.run(cmd, capture_output=True, text=True)

                if result.returncode == 0:
                    batch_results = json.loads(result.stdout.strip())
                    if "error" not in batch_results and batch_results:
                        logger.info(
                            f"Batch worker {worker_script} retrieved {len(batch_results)} texts"
                        )
                        return batch_results

                logger.debug(f"Batch worker {worker_script} failed: {result.stderr}")
            except Exception as e:
                logger.debug(f"Batch worker {worker_script} error: {e}")

        return {}

    def _parallel_individual_retrieve(self, doc_ids: list) -> dict:
        """Parallel individual retrieval using thread pool"""
        results = {}

        def get_single_doc(doc_id):
            try:
                text = self._get_full_text_for_doc(doc_id)
                return doc_id, text if text and text != "NOT_FOUND" else None
            except Exception as e:
                logger.debug(f"Individual retrieval failed for {doc_id}: {e}")
                return doc_id, None

        with ThreadPoolExecutor(max_workers=6) as executor:
            future_to_id = {
                executor.submit(get_single_doc, doc_id): doc_id for doc_id in doc_ids
            }

            for future in as_completed(future_to_id):  # No timeout
                try:
                    doc_id, text = future.result()
                    if text:
                        results[doc_id] = text
                except Exception as e:
                    logger.debug(f"Future failed: {e}")

        logger.info(
            f"Parallel individual retrieval got {len(results)}/{len(doc_ids)} texts"
        )
        return results

    def _get_full_text_for_doc(self, doc_id: str) -> str:
        """Get full text for a document"""
        try:
            cmd = [
                self.bm25_python,
                self.bm25_fulltext_script,
                doc_id,
                self.bm25_index_directory,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)  # No timeout

            if result.returncode == 0:
                full_text = result.stdout.strip()
                if (
                    full_text
                    and full_text != "NOT_FOUND"
                    and not full_text.startswith("ERROR:")
                ):
                    return full_text
        except Exception as e:
            logger.debug(f"BM25 individual lookup failed for {doc_id}: {e}")

        return self._try_fallback_sources(doc_id)

    def _try_fallback_sources(self, doc_id: str) -> str:
        """Try multiple fallback sources for full text"""
        if doc_id in self._doc_cache and "bm25_node" in self._doc_cache[doc_id]:
            bm25_node = self._doc_cache[doc_id]["bm25_node"]
            if bm25_node and "full_text" in bm25_node:
                return bm25_node["full_text"]

        if doc_id in self._doc_cache and "e5_doc" in self._doc_cache[doc_id]:
            e5_doc = self._doc_cache[doc_id]["e5_doc"]
            if e5_doc and "full_text" in e5_doc.meta:
                return e5_doc.meta["full_text"]
            elif e5_doc and len(e5_doc.content) > 1000:
                return e5_doc.content

        if doc_id in self._doc_cache:
            return self._doc_cache[doc_id]["abstract"]

        return ""

    def retrieve(self, query: str, top_k: int = None):
        """Legacy method for backward compatibility"""
        if top_k is None:
            top_k = self.top_k

        cache_key = f"legacy_{query.lower().strip()}_{top_k}"
        if hasattr(self, "_legacy_cache") and cache_key in self._legacy_cache:
            logger.info("⚡ Legacy cache hit!")
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
                    id=node.get("paper_id", ""),
                    title=node.get("title", ""),
                    content=node.get("text", ""),
                )
            combined.append((final_score, doc))

        combined.sort(key=lambda x: x[0], reverse=True)

        result = [
            {
                "id": d.meta.get("paper_id") if hasattr(d, "meta") else d.id,
                "title": d.meta.get("title") if hasattr(d, "meta") else None,
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
        if self._bm25_retriever:
            return {"method": "in-memory", "available": True, "status": "OPTIMIZED"}
        else:
            return {"method": "subprocess", "available": True, "status": "SLOW"}

    def get_performance_stats(self):
        """Get performance statistics"""
        if self._retrieval_times:
            avg_time = sum(self._retrieval_times) / len(self._retrieval_times)
            return {
                "avg_retrieval_time": avg_time,
                "total_retrievals": len(self._retrieval_times),
                "cache_sizes": {
                    "abstract_cache": len(self._abstract_cache),
                    "fulltext_cache": len(self._fulltext_cache),
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
        self._fulltext_cache.clear()

        stats = self.get_performance_stats()
        if "avg_retrieval_time" in stats:
            logger.info(
                f"Final stats: {stats['total_retrievals']} retrievals, avg {stats['avg_retrieval_time']:.2f}s"
            )

        bm25_status = self.get_bm25_status()
        logger.info(
            f"BM25 final status: {bm25_status['method']} ({bm25_status['status']})"
        )

        logger.info("Retriever closed")
