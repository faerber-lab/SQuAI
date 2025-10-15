#!/usr/bin/env python3
"""
Unified Retriever - E5 + BM25 Hybrid Retrieval
Optimized based on test results, adapted for your arXiv index structure
"""

import json
import sqlite3
import logging
import time
import pickle
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

import faiss
from sentence_transformers import SentenceTransformer
import torch  # Add this import

logger = logging.getLogger(__name__)


class E5DirectRetriever:
    """
    Optimized version: Uses cached mapping to avoid rebuilding each time
    First run takes 5 minutes to build mapping, subsequent runs only take seconds to load
    """
    
    def __init__(self, index_directory: str, model_name: str = "intfloat/e5-large-v2"):
        self.index_dir = Path(index_directory)
        self.model_name = model_name
        
        # Mapping cache file path
        self.mapping_cache_file = self.index_dir / "faiss_document_mapping.pkl"
        
        logger.info(f"Initializing E5 retriever: {index_directory}")
        
        # Load FAISS index
        self._load_index()
        
        # Load or build mapping
        if self.mapping_cache_file.exists():
            self._load_cached_mapping()
        else:
            logger.info("First run, need to build mapping (about 5 minutes)...")
            self._connect_db()
            self._build_and_save_mapping()
        
        # Load and optimize E5 model
        self._load_model()
        
        # Query cache
        self._cache = {}
        self._cache_size = 100
        
        logger.info(f"E5 retriever initialization complete")
    
    def _load_index(self):
        """Load FAISS index"""
        index_path = self.index_dir / "faiss_index"
        self.index = faiss.read_index(str(index_path))
        logger.info(f"FAISS index loaded: {self.index.ntotal:,} vectors, dimension {self.index.d}")
    
    def _load_cached_mapping(self):
        """Load mapping from cache file (fast)"""
        logger.info(f"Loading mapping from cache: {self.mapping_cache_file}")
        start_time = time.time()
        
        try:
            with open(self.mapping_cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            
            # Validate the loaded data
            if 'faiss_to_doc' not in cache_data or 'doc_metadata' not in cache_data:
                raise ValueError("Cache file is missing required data")
            
            self.faiss_to_doc = cache_data['faiss_to_doc']
            self.doc_metadata = cache_data['doc_metadata']
            
            elapsed = time.time() - start_time
            logger.info(f"Mapping loaded: {len(self.faiss_to_doc):,} entries ({elapsed:.1f}s)")
        except Exception as e:
            logger.error(f"Failed to load cache: {e}")
            logger.info("Rebuilding mapping...")
            self._connect_db()
            self._build_and_save_mapping()
    
    def _connect_db(self):
        """Connect to SQLite database"""
        db_path = self.index_dir / "document_metadata.db"
        self.conn = sqlite3.connect(str(db_path))
        self.cursor = self.conn.cursor()
    
    def _build_and_save_mapping(self):
        """Build and save FAISS index to document ID mapping (for first run)"""
        logger.info("Building FAISS to document mapping...")
        start_time = time.time()
        
        # Get all document mappings
        self.cursor.execute("SELECT faiss_index, doc_id, title, abstract FROM document_embeddings")
        
        self.faiss_to_doc = {}
        self.doc_metadata = {}
        
        for faiss_idx, doc_id, title, abstract in self.cursor:
            self.faiss_to_doc[faiss_idx] = doc_id
            self.doc_metadata[doc_id] = {
                'title': title,
                'abstract': abstract
            }
        
        # Save to cache
        cache_data = {
            'faiss_to_doc': self.faiss_to_doc,
            'doc_metadata': self.doc_metadata
        }
        
        with open(self.mapping_cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
        
        elapsed = time.time() - start_time
        logger.info(f"Mapping built and saved: {len(self.faiss_to_doc):,} entries ({elapsed:.1f}s)")
        
        # Close database connection (no longer needed after caching)
        self.conn.close()
    
    def _load_model(self):
        """Load E5 model"""
        logger.info(f"Loading E5 model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        
        # Enable GPU if available
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            logger.info("E5 model moved to GPU")
        else:
            logger.info("Using CPU (GPU not available)")
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[str, str]]:
        """
        Retrieve documents
        Returns: List of (abstract, doc_id) tuples
        """
        # Check cache
        cache_key = f"{query}_{top_k}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Add E5 prefix
        prefixed_query = f"query: {query}"
        
        # Generate query embedding
        query_embedding = self.model.encode(
            prefixed_query,
            convert_to_tensor=True,
            normalize_embeddings=True
        )
        
        # Convert to numpy
        if torch.cuda.is_available():
            query_embedding = query_embedding.cpu().numpy()
        else:
            query_embedding = query_embedding.numpy()
        
        # Search
        distances, indices = self.index.search(
            query_embedding.reshape(1, -1),
            top_k
        )
        
        # Build results
        results = []
        for idx in indices[0]:
            if idx in self.faiss_to_doc:
                doc_id = self.faiss_to_doc[idx]
                if doc_id in self.doc_metadata:
                    metadata = self.doc_metadata[doc_id]
                    results.append((metadata['abstract'], doc_id))
        
        # Cache results
        if len(self._cache) < self._cache_size:
            self._cache[cache_key] = results
        
        return results
    
    def close(self):
        """Clean up resources"""
        if hasattr(self, 'conn'):
            self.conn.close()
        self._cache.clear()
        logger.info("E5 retriever closed")

class UnifiedArxivRetriever:
    """
    Unified arXiv retriever - supports E5, BM25, and hybrid modes
    """
    
    def __init__(self, 
                 e5_index_directory: str,
                 bm25_index_directory: str, 
                 leveldb_path: str = None,
                 strategy: str = "hybrid",
                 alpha: float = 0.65,
                 top_k: int = 5):
        """
        Initialize unified retriever
        
        Args:
            e5_index_directory: E5 FAISS index directory
            bm25_index_directory: BM25 LlamaIndex index directory
            leveldb_path: LevelDB full-text storage path (optional)
            strategy: "e5", "bm25", or "hybrid"
            alpha: E5 weight (hybrid mode)
            top_k: Default number of documents to return
        """
        self.strategy = strategy
        self.alpha = alpha
        self.top_k = top_k
        
        logger.info(f"Initializing unified retriever - strategy: {strategy}")
        
        # Initialize E5 (if needed)
        self.e5 = None
        if strategy in ["e5", "hybrid"]:
            self.e5 = E5DirectRetriever(e5_index_directory)
            logger.info("E5 retriever loaded")
        
        # Initialize BM25 (if needed)
        self.bm25 = None
        if strategy in ["bm25", "hybrid"]:
            try:
                # Try using your fast implementation
                from fast_llamaindex_retriever import FastLlamaIndexBM25Retriever
                self.bm25 = FastLlamaIndexBM25Retriever(bm25_index_directory, top_k)
                logger.info("BM25 retriever loaded (fast version)")
            except ImportError:
                # Or use your BM25OnlyRetriever
                try:
                    from bm25_only_retriever import BM25OnlyRetriever
                    self.bm25 = BM25OnlyRetriever(bm25_index_directory, top_k)
                    logger.info("BM25 retriever loaded (pure BM25 version)")
                except ImportError:
                    logger.warning("BM25 retriever not available")
        
        # LevelDB (for full text)
        self.leveldb = None
        if leveldb_path:
            try:
                import plyvel
                self.leveldb = plyvel.DB(leveldb_path, create_if_missing=False)
                logger.info(f"LevelDB connected: {leveldb_path}")
            except Exception as e:
                logger.warning(f"LevelDB connection failed: {e}")
        
        # Parallel executor
        self._executor = ThreadPoolExecutor(max_workers=2)
        
        # Cache
        self._cache = {}
        self._cache_size = 100
    
    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """Normalize scores to [0, 1]"""
        if not scores:
            return []
        min_score = min(scores)
        max_score = max(scores)
        if max_score - min_score == 0:
            return [1.0] * len(scores)
        return [(s - min_score) / (max_score - min_score) for s in scores]
    
    def _retrieve_e5(self, query: str, top_k: int) -> List[Tuple[str, str, float]]:
        """Retrieve using E5"""
        if not self.e5:
            return []
        
        results = self.e5.retrieve(query, top_k)
        # Add normalized scores
        scores = [1.0 / (i + 1) for i in range(len(results))]  # Simple ranking score
        scores_norm = self._normalize_scores(scores)
        
        return [(abstract, doc_id, score) 
                for (abstract, doc_id), score in zip(results, scores_norm)]
    
    def _retrieve_bm25(self, query: str, top_k: int) -> List[Tuple[str, str, float]]:
        """Retrieve using BM25"""
        if not self.bm25:
            return []
        
        # Call BM25's retrieve_abstracts method
        if hasattr(self.bm25, 'retrieve_abstracts'):
            results = self.bm25.retrieve_abstracts(query, top_k)
        else:
            # Fallback
            return []
        
        # Convert format and add scores
        formatted_results = []
        for i, (abstract, doc_id) in enumerate(results):
            score = 1.0 / (i + 1)  # Simple ranking score
            formatted_results.append((abstract, doc_id, score))
        
        # Normalize scores
        if formatted_results:
            scores = [r[2] for r in formatted_results]
            scores_norm = self._normalize_scores(scores)
            formatted_results = [(r[0], r[1], s) 
                                for r, s in zip(formatted_results, scores_norm)]
        
        return formatted_results
    
    def retrieve_abstracts(self, query: str, top_k: int = None) -> List[Tuple[str, str]]:
        """
        Main retrieval method - returns (abstract, doc_id) pairs
        """
        if top_k is None:
            top_k = self.top_k
        
        # Check cache
        cache_key = f"{self.strategy}_{query}_{top_k}"
        if cache_key in self._cache:
            logger.info(f"Cache hit for: {query[:50]}...")
            return self._cache[cache_key]
        
        logger.info(f"Retrieving with {self.strategy}: {query[:50]}...")
        start_time = time.time()
        
        if self.strategy == "e5":
            results = self._retrieve_e5(query, top_k)
            final_results = [(r[0], r[1]) for r in results]
            
        elif self.strategy == "bm25":
            results = self._retrieve_bm25(query, top_k)
            final_results = [(r[0], r[1]) for r in results]
            
        elif self.strategy == "hybrid":
            # Parallel retrieval
            with self._executor as executor:
                e5_future = executor.submit(self._retrieve_e5, query, top_k * 2)
                bm25_future = executor.submit(self._retrieve_bm25, query, top_k * 2)
                
                e5_results = e5_future.result()
                bm25_results = bm25_future.result()
            
            # Merge results
            final_results = self._merge_results(e5_results, bm25_results, top_k)
        
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        
        elapsed = time.time() - start_time
        logger.info(f"Retrieved {len(final_results)} documents ({elapsed:.2f}s)")
        
        # Cache results
        if len(self._cache) < self._cache_size:
            self._cache[cache_key] = final_results
        
        return final_results
    
    def _merge_results(self, e5_results: List, bm25_results: List, top_k: int) -> List[Tuple[str, str]]:
        """Merge E5 and BM25 results"""
        doc_scores = {}
        doc_abstracts = {}
        
        # Process E5 results
        for abstract, doc_id, score in e5_results:
            doc_scores[doc_id] = self.alpha * score
            doc_abstracts[doc_id] = abstract
        
        # Process BM25 results
        for abstract, doc_id, score in bm25_results:
            if doc_id in doc_scores:
                doc_scores[doc_id] += (1 - self.alpha) * score
            else:
                doc_scores[doc_id] = (1 - self.alpha) * score
                doc_abstracts[doc_id] = abstract
        
        # Sort and return top-k
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        
        results = []
        for doc_id, _ in sorted_docs[:top_k]:
            results.append((doc_abstracts[doc_id], doc_id))
        
        return results
    
    def retrieve_fulltexts(self, query: str, top_k: int = None) -> List[Tuple[str, str]]:
        """Retrieve full texts"""
        # First get abstracts
        abstracts = self.retrieve_abstracts(query, top_k)
        
        if not self.leveldb:
            logger.warning("LevelDB not available, returning abstracts")
            return abstracts
        
        # Get full texts
        results = []
        for _, doc_id in abstracts:
            try:
                full_text = self.leveldb.get(doc_id.encode())
                if full_text:
                    results.append((full_text.decode('utf-8'), doc_id))
                else:
                    results.append((f"[Full text not found: {doc_id}]", doc_id))
            except Exception as e:
                logger.error(f"Failed to retrieve full text for {doc_id}: {e}")
                results.append((f"[Error retrieving: {doc_id}]", doc_id))
        
        return results
    
    def get_full_texts(self, doc_ids: List[str]) -> List[Tuple[str, str]]:
        """Get full texts for specific document IDs"""
        if not self.leveldb:
            return [(f"[LevelDB not available]", doc_id) for doc_id in doc_ids]
        
        results = []
        for doc_id in doc_ids:
            try:
                full_text = self.leveldb.get(doc_id.encode())
                if full_text:
                    results.append((full_text.decode('utf-8'), doc_id))
                else:
                    results.append((f"[Not found: {doc_id}]", doc_id))
            except Exception as e:
                logger.error(f"Failed to get {doc_id}: {e}")
                results.append((f"[Error: {doc_id}]", doc_id))
        
        return results
    
    def close(self):
        """Clean up resources"""
        if self.e5:
            self.e5.close()
        if self.bm25 and hasattr(self.bm25, 'close'):
            self.bm25.close()
        if self.leveldb:
            self.leveldb.close()
        self._executor.shutdown(wait=True)
        logger.info("Unified retriever closed")


# Usage example
if __name__ == "__main__":
    import sys
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Configure paths (adjust to your actual paths)
    E5_INDEX = "/data/horse/ws/s3811141-faiss/inbe405h-unarxive/faiss_index"
    BM25_INDEX = "/data/horse/ws/s3811141-faiss/inbe405h-unarxive/bm25_retriever"
    LEVELDB = "/data/horse/ws/inbe405h-unarxive/full_text_db"  # If available
    
    # Test different strategies
    strategies = ["e5", "bm25", "hybrid"] if len(sys.argv) <= 1 else [sys.argv[1]]
    
    for strategy in strategies:
        print(f"\n{'='*60}")
        print(f"Testing {strategy.upper()} strategy")
        print('='*60)
        
        retriever = UnifiedArxivRetriever(
            e5_index_directory=E5_INDEX,
            bm25_index_directory=BM25_INDEX,
            leveldb_path=LEVELDB if Path(LEVELDB).exists() else None,
            strategy=strategy,
            alpha=0.65
        )
        
        # Test queries
        queries = [
            "quantum computing algorithms",
            "deep learning transformers",
            "protein folding prediction"
        ]
        
        for query in queries:
            print(f"\nQuery: '{query}'")
            start = time.time()
            results = retriever.retrieve_abstracts(query, top_k=3)
            elapsed = time.time() - start
            
            print(f"Found {len(results)} results (time: {elapsed:.3f}s)")
            for i, (abstract, paper_id) in enumerate(results, 1):
                preview = abstract[:100] + "..." if len(abstract) > 100 else abstract
                print(f"  [{i}] {paper_id}: {preview}")
        
        # Test full text retrieval (if LevelDB is configured)
        if retriever.leveldb and results:
            print("\nTesting full text retrieval...")
            doc_ids = [paper_id for _, paper_id in results[:2]]
            full_texts = retriever.get_full_texts(doc_ids)
            for full_text, doc_id in full_texts:
                print(f"  {doc_id}: {len(full_text)} characters")
        
        retriever.close()
    
    print("\nAll tests complete!")