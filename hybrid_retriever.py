# hybrid_retriever.py

from pathlib import Path
import logging
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    Wraps UnifiedArxivRetriever to be fully compatible with the old interface.
    """
    
    def __init__(self, e5_index_directory, bm25_index_directory, 
                 top_k=5, strategy="hybrid", alpha=0.65):
        
        logger.info(f"Initializing {strategy.upper()} retriever...")
        
        # Using the optimized retriever
        try:
            from unified_arxiv_retriever import UnifiedArxivRetriever
            self._inner = UnifiedArxivRetriever(
                e5_index_directory=e5_index_directory,
                bm25_index_directory=bm25_index_directory,
                leveldb_path=None,  
                strategy=strategy,
                alpha=alpha,
                top_k=top_k
            )
            self._using_new = True
            logger.info("Using optimized UnifiedArxivRetriever")
        except Exception as e:
            logger.warning(f"Failed to load UnifiedArxivRetriever: {e}")
            self._using_new = False
            raise
        
        self.strategy = strategy
        self.alpha = alpha
        self.top_k = top_k
        
        self._doc_cache = {}
        self._abstract_cache = {}
        self._fulltext_cache = {}
        self._cache_size = 100
        self._executor = ThreadPoolExecutor(max_workers=4)
        self._retrieval_times = []
        
        self.e5 = self._inner.e5 if hasattr(self._inner, 'e5') else None
        self.bm25 = self._inner.bm25 if hasattr(self._inner, 'bm25') else None
        self._bm25_retriever = self.bm25
        
    def retrieve_abstracts(self, query: str, top_k: int = None) -> list:
        """Retrieve abstracts with caching"""
        if top_k is None:
            top_k = self.top_k
            
        start_time = time.time()

        # Check cache
        cache_key = f"{self.strategy}_{query.lower().strip()}_{top_k}"
        if cache_key in self._abstract_cache:
            logger.info(f"⚡ {self.strategy.upper()} cache hit!")
            return self._abstract_cache[cache_key]
        
        logger.info(f"🔍 {self.strategy.upper()} retrieval for query: {query}")

        # Use new retriever
        result = self._inner.retrieve_abstracts(query, top_k)
        
        # Cache result
        if len(self._abstract_cache) >= self._cache_size:
            oldest_key = next(iter(self._abstract_cache))
            del self._abstract_cache[oldest_key]
        self._abstract_cache[cache_key] = result
        
        # Save retrieval time
        elapsed = time.time() - start_time
        self._retrieval_times.append(elapsed)
        if len(self._retrieval_times) > 100:
            self._retrieval_times = self._retrieval_times[-100:]
        
        logger.info(f"{self.strategy.upper()}: Retrieved {len(result)} abstracts in {elapsed:.2f}s")
        
        return result
    
    def get_full_texts(self, doc_ids: list, db=None) -> list:
        """获取全文 - 保持接口不变"""
        if not doc_ids:
            return []
        
        start_time = time.time()
        logger.info(f"{self.strategy.upper()}: Retrieving full texts for {len(doc_ids)} documents")
        
        # Use new retriever
        result = self._inner.get_full_texts(doc_ids, db)
        
        elapsed = time.time() - start_time
        if result:
            total_chars = sum(len(text) for text, _ in result)
            avg_length = total_chars // len(result)
            logger.info(f"{self.strategy.upper()}: Retrieved {len(result)} full texts in {elapsed:.2f}s (avg {avg_length} chars/doc)")
        
        return result
    
    def retrieve(self, query: str, top_k: int = None):
        """Old retrieve method - for compatibility"""
        if top_k is None:
            top_k = self.top_k
            
        # Obtain abstracts
        abstracts = self.retrieve_abstracts(query, top_k)
        
        # Transform to old format
        results = []
        for i, (text, doc_id) in enumerate(abstracts):
            results.append({
                "id": doc_id,
                "paper_id": doc_id,
                "title": "Unknown",  
                "abstract": text,
                "semantic_score": 1.0 / (i + 1)  # Simple ranking score
            })
        
        return results
    
    def close(self):
        """Clean up resources"""
        try:
            if hasattr(self._inner, 'close'):
                self._inner.close()
        except:
            pass
        
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=True)
        
        self._doc_cache.clear()
        self._abstract_cache.clear()
        self._fulltext_cache.clear()
        
        stats = self.get_performance_stats()
        if "avg_retrieval_time" in stats:
            logger.info(f"Final stats: {stats['total_retrievals']} retrievals, avg {stats['avg_retrieval_time']:.2f}s")
        
        logger.info("Retriever closed")
    
    def get_bm25_status(self):
        """Diagnostic method"""
        if self._using_new:
            return {"method": "optimized", "available": True, "status": "FAST"}
        else:
            return {"method": "unknown", "available": False, "status": "UNKNOWN"}
    
    def get_performance_stats(self):
        """Performance statistics"""
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
                "bm25_status": self.get_bm25_status()
            }
        return {"no_data": True}
    
    def _fast_normalize(self, scores):
        """Rapid normalization using numpy"""
        if len(scores) == 0:
            return scores
        min_score = np.min(scores)
        max_score = np.max(scores)
        if max_score - min_score == 0:
            return np.ones_like(scores)
        return (scores - min_score) / (max_score - min_score)
    
    def _get_bm25_results(self, query, top_k):
        """Compatible method - if there is code, call this directly"""
        if self.bm25:
            # Call new retriever's BM25
            return self._inner._retrieve_bm25(query, top_k)
        return []
    
    def _get_e5_results(self, query, top_k):
        """Compatible method - if there is code, call this directly"""
        if self.e5:
            # Call new retriever's E5
            docs = self._inner.e5.retrieve(query, top_k)
            # Transform format
            return docs
        return []
    
    def _load_bm25_into_memory(self):
        """Compatible method - return None because new system handles it automatically"""
        return None