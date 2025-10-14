#!/usr/bin/env python3
"""
Fast LlamaIndex BM25 Retriever - Keeps index loaded in memory
Drop-in replacement for your subprocess approach
"""
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
    print("LlamaIndex BM25 not available. Install with: pip install llama-index-retrievers-bm25")
    LLAMAINDEX_AVAILABLE = False

logger = logging.getLogger(__name__)

class FastLlamaIndexBM25Retriever:
    """
    Fast LlamaIndex BM25 retriever that keeps index loaded in memory
    Compatible with your existing persist format
    """
    
    def __init__(self, persist_dir: str, top_k: int = 5, preload: bool = True):
        """
        Initialize with your existing LlamaIndex BM25 index
        
        Args:
            persist_dir: Your BM25_INDEX_DIR (/data/horse/ws/inbe405h-unarxive/bm25_retriever)
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
        
        logger.info(f"Initializing Fast LlamaIndex BM25 from {persist_dir}")
        
        if preload:
            self._load_retriever()
    
    def _load_retriever(self):
        """Load the LlamaIndex BM25 retriever once and keep in memory"""
        start_time = time.time()
        
        if not self.persist_dir.exists():
            raise FileNotFoundError(f"BM25 index not found at {self.persist_dir}")
        
        logger.info("Loading LlamaIndex BM25 retriever...")
        self.retriever = BM25Retriever.from_persist_dir(str(self.persist_dir))
        
        elapsed = time.time() - start_time
        logger.info(f"LlamaIndex BM25 loaded in {elapsed:.2f}s")
    
    def retrieve_abstracts(self, query: str, top_k: int = None) -> List[Tuple[str, str]]:
        """
        Retrieve abstracts using LlamaIndex BM25 (compatible with your interface)
        
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
            logger.debug("LlamaIndex BM25 cache hit")
            return self._cache[cache_key]
        
        start_time = time.time()
        
        if not self.retriever:
            self._load_retriever()
        
        logger.info(f"LlamaIndex BM25 retrieving for query: {query}")
        
        # Use LlamaIndex retriever
        results = self.retriever.retrieve(query)
        
        # Convert to your expected format
        formatted_results = []
        for result in results[:top_k]:
            # Extract paper_id and text
            paper_id = result.node.metadata.get("paper_id", "unknown")
            text = result.node.get_text()
            
            formatted_results.append((text, paper_id))
        
        # Cache result
        if len(self._cache) >= self._cache_size:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        self._cache[cache_key] = formatted_results
        
        elapsed = time.time() - start_time
        logger.info(f"Fast LlamaIndex BM25: Retrieved {len(formatted_results)} documents in {elapsed:.3f}s")
        
        return formatted_results
    
    def get_bm25_results(self, query: str, top_k: int) -> List[Dict]:
        """
        Get BM25 results in the same format as your bm25_worker.py
        
        Returns:
            List of dicts with paper_id, text, score
        """
        if not self.retriever:
            self._load_retriever()
        
        results = self.retriever.retrieve(query)
        
        # Format exactly like your bm25_worker.py output
        out = [
            {
                "paper_id": r.node.metadata.get("paper_id"),
                "text": r.node.get_text(),
                "score": r.score
            } for r in results[:top_k]
        ]
        
        return out
    
    def get_full_texts(self, doc_ids: List[str], db=None) -> List[Tuple[str, str]]:
        """
        Get full texts for documents
        """
        results = []
        
        if db is not None:
            # Use LevelDB if available
            logger.info(f"Retrieving full texts from LevelDB for {len(doc_ids)} documents")
            
            for doc_id in doc_ids:
                try:
                    content = db.get(doc_id.encode('utf-8'))
                    if content:
                        full_text = content.decode('utf-8')
                        results.append((full_text, doc_id))
                    else:
                        logger.warning(f"Full text not found in LevelDB for {doc_id}")
                except Exception as e:
                    logger.error(f"Error retrieving full text for {doc_id}: {e}")
        else:
            # Fallback: try to get full_text from metadata
            logger.info("No LevelDB available, trying to get full_text from BM25 metadata")
            
            # Get all documents from retriever and find matches
            if not self.retriever:
                self._load_retriever()
            
            # This is a workaround - LlamaIndex BM25Retriever doesn't have direct doc access
            # We'll need to query for each doc_id
            for doc_id in doc_ids:
                try:
                    # Try querying by paper_id (this is a hack but might work)
                    temp_results = self.retriever.retrieve(f"paper_id:{doc_id}")
                    if temp_results:
                        for result in temp_results:
                            if result.node.metadata.get("paper_id") == doc_id:
                                full_text = result.node.metadata.get("full_text", result.node.get_text())
                                results.append((full_text, doc_id))
                                break
                except Exception as e:
                    logger.debug(f"Could not retrieve full text for {doc_id}: {e}")
        
        return results
    
    def retrieve(self, query: str, top_k: int = None) -> List[Dict]:
        """
        Legacy method for backward compatibility with your existing code
        """
        abstracts = self.retrieve_abstracts(query, top_k)
        
        results = []
        for abstract_text, doc_id in abstracts:
            result = {
                "id": doc_id,
                "abstract": abstract_text,
                "semantic_score": 1.0  # BM25 score could be added
            }
            results.append(result)
        
        return results
    
    def get_performance_stats(self):
        """Get performance statistics"""
        return {
            "retriever_type": "FAST_LLAMAINDEX_BM25",
            "index_loaded": self.retriever is not None,
            "cache_size": len(self._cache),
            "persist_dir": str(self.persist_dir)
        }
    
    def close(self):
        """Clean up resources"""
        self._cache.clear()
        # LlamaIndex retriever doesn't need explicit cleanup
        logger.info("Fast LlamaIndex BM25 retriever closed")


# Drop-in replacement for your subprocess approach
class FastLlamaIndexRetriever:
    """
    Drop-in replacement for your hybrid_retriever.py that uses subprocess
    """
    
    def __init__(self, e5_index_directory: str, bm25_index_directory: str, top_k: int = 5, strategy: str = "bm25", alpha: float = 0.65):
        """
        Compatible with your existing Retriever class interface
        """
        self.strategy = strategy
        self.alpha = alpha
        self.top_k = top_k
        
        logger.info(f"Initializing Fast LlamaIndex {strategy.upper()} retriever...")
        
        # Only initialize BM25 for now (your focus)
        if strategy in ["hybrid", "bm25"]:
            self.bm25 = FastLlamaIndexBM25Retriever(bm25_index_directory, top_k)
            logger.info("Fast LlamaIndex BM25 initialized successfully")
        else:
            self.bm25 = None
        
        # E5 initialization (if needed later)
        if strategy in ["hybrid", "e5"]:
            # You can add E5 initialization here if needed
            logger.warning("E5 not implemented in fast version yet")
            self.e5 = None
        else:
            self.e5 = None
        
        # Caching
        self._doc_cache = {}
        self._abstract_cache = {}
        self._cache_size = 100
    
    def retrieve_abstracts(self, query: str, top_k: int = None) -> List[Tuple[str, str]]:
        """
        Main method to replace your subprocess-based BM25 calls
        """
        if top_k is None:
            top_k = self.top_k
        
        if self.strategy == "bm25" and self.bm25:
            return self.bm25.retrieve_abstracts(query, top_k)
        else:
            logger.error(f"Strategy {self.strategy} not supported in fast version yet")
            return []
    
    def get_full_texts(self, doc_ids: List[str], db=None) -> List[Tuple[str, str]]:
        """
        Get full texts for documents
        """
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
        logger.info("Fast LlamaIndex retriever closed")


# Factory function to create fast retriever
def create_fast_llamaindex_retriever(bm25_index_directory: str, top_k: int = 5) -> FastLlamaIndexRetriever:
    """
    Create a fast LlamaIndex BM25 retriever to replace subprocess approach
    
    Args:
        bm25_index_directory: Your BM25_INDEX_DIR path
        top_k: Number of documents to retrieve
    
    Returns:
        FastLlamaIndexRetriever instance
    """
    return FastLlamaIndexRetriever("", bm25_index_directory, top_k, strategy="bm25")


if __name__ == "__main__":
    # Test with your actual index
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python fast_llamaindex_retriever.py <bm25_index_dir> [query]")
        print("Example: python fast_llamaindex_retriever.py /data/horse/ws/inbe405h-unarxive/bm25_retriever")
        sys.exit(1)
    
    index_dir = sys.argv[1]
    test_query = sys.argv[2] if len(sys.argv) > 2 else "discrete quantum walks control"
    
    logging.basicConfig(level=logging.INFO)
    
    print(f"Testing Fast LlamaIndex BM25 with your index: {index_dir}")
    
    # Test the fast retriever
    retriever = create_fast_llamaindex_retriever(index_dir, top_k=5)
    
    # Test retrieval speed
    print(f"\nTesting query: '{test_query}'")
    
    # First query (loading index)
    start_time = time.time()
    results = retriever.retrieve_abstracts(test_query, top_k=5)
    first_time = time.time() - start_time
    
    # Second query (using cached index)
    start_time = time.time()
    results = retriever.retrieve_abstracts(test_query, top_k=5)
    cached_time = time.time() - start_time
    
    print(f"First query (loading): {first_time:.3f} seconds")
    print(f"Second query (cached): {cached_time:.3f} seconds")
    
    if first_time > 0 and cached_time > 0:
        print(f"Speedup after loading: {first_time/cached_time:.1f}x")
    
    print(f"\nRetrieved {len(results)} documents:")
    for i, (text, doc_id) in enumerate(results, 1):
        print(f"[{i}] {doc_id}")
        print(f"    {text[:150]}...")
    
    # Test the BM25 format compatibility
    print(f"\nTesting BM25 worker format compatibility:")
    bm25_results = retriever.bm25.get_bm25_results(test_query, 3)
    print(json.dumps(bm25_results[:1], indent=2))  # Show first result
    
    # Performance stats
    stats = retriever.get_performance_stats()
    print(f"\nPerformance stats: {stats}")
    
    retriever.close()