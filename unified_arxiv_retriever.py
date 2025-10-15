#!/usr/bin/env python3
"""
E5 + BM25 hybrid retriever with optimized caching and mapping
Maintains compatibility with existing interfaces
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
    E5 direct retriever with FAISS and SQLite mapping
    """
    
    def __init__(self, index_directory: str, model_name: str = "intfloat/e5-large-v2"):
        self.index_dir = Path(index_directory)
        self.model_name = model_name
        
        # mapping cache file
        self.mapping_cache_file = self.index_dir / "faiss_document_mapping.pkl"
        
        logger.info(f"Initialize E5: {index_directory}")
        
        # load FAISS index
        self._load_index()
        
        # load or build mapping
        if self.mapping_cache_file.exists():
            self._load_cached_mapping()
        else:
            logger.info("Building mapping...")
            self._connect_db()
            self._build_and_save_mapping()
        
        # load and optimize E5 model
        self._load_model()
        
        # query cache
        self._cache = {}
        self._cache_size = 100
        
        logger.info(f"E5 retriever ready")
    
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
            logger.info(f"✓ Mapping loaded: {len(self.faiss_to_doc):,} documents (耗时 {elapsed:.1f}秒)")

        except (EOFError, pickle.UnpicklingError, ValueError) as e:
            logger.warning(f"Cache file is not complete: {e}")
            logger.info("Deleting corrupted cache and rebuilding mapping...")
            
            # Delete corrupted cache
            if self.mapping_cache_file.exists():
                self.mapping_cache_file.unlink()
            
            # Rebuild
            self._connect_db()
            self._build_and_save_mapping()
    
    def _connect_db(self):
        """Connect to SQLite database (only when mapping needs to be built)"""
        db_path = self.index_dir / "index_store.db"
        self.db_conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.db_cursor = self.db_conn.cursor()
        
        # Validate connection
        self.db_cursor.execute("SELECT COUNT(*) FROM document")
        doc_count = self.db_cursor.fetchone()[0]
        logger.info(f"SQLite database connection: {doc_count:,} documents")

    def _build_and_save_mapping(self):
        """Build mapping and save to cache file (only needs to be run once)"""
        logger.info("Building document mapping (this will take about 5 minutes, but only needs to be done once)...")
        start_time = time.time()

        # Creating mapping: FAISS index -> Document information
        self.faiss_to_doc = {}
        
        logger.info("  1/3 Getting vector_id mapping...")
        self.db_cursor.execute("""
            SELECT id, vector_id, content 
            FROM document 
            WHERE vector_id IS NOT NULL
        """)
        
        all_docs = self.db_cursor.fetchall()
        total_docs = len(all_docs)
        
        logger.info(f"  2/3 process {total_docs:,} documents...")
        for i, (doc_id, vector_id_str, content) in enumerate(all_docs):
            faiss_idx = int(vector_id_str)
            self.faiss_to_doc[faiss_idx] = {
                'id': doc_id,
                'content': content
            }
            
            if (i + 1) % 100000 == 0:
                logger.info(f"    processed {i+1:,}/{total_docs:,} documents...")

        # Loading metadata
        logger.info("  3/3 Loading document metadata...")
        self.doc_metadata = {}
        
        self.db_cursor.execute("""
            SELECT document_id, name, value 
            FROM meta_document
            WHERE name = 'paper_id'
        """)
        
        for doc_id, name, value in self.db_cursor.fetchall():
            if doc_id not in self.doc_metadata:
                self.doc_metadata[doc_id] = {}
            
            if value:
                value = value.strip('"')
            self.doc_metadata[doc_id]['paper_id'] = value

        # Save to cache file
        logger.info(f"Saving mapping to cache file: {self.mapping_cache_file}")
        cache_data = {
            'faiss_to_doc': self.faiss_to_doc,
            'doc_metadata': self.doc_metadata,
            'created_at': time.time(),
            'doc_count': len(self.faiss_to_doc)
        }
        
        with open(self.mapping_cache_file, 'wb') as f:
            pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Close DB connection
        self.db_conn.close()
        
        elapsed = time.time() - start_time
        logger.info(f"✓ Mapping built and saved successfully (elapsed time {elapsed:.1f} seconds)")
        logger.info(f"  Next run will load cache directly, taking only a few seconds!")
    
    def _load_model(self):
        """load and optimize E5 model"""
        logger.info(f"Loading E5 model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            logger.info("✓ E5 model running on GPU")
            self.device = 'cuda'
        else:
            self.model = self.model.cpu()
            logger.info("⚠ E5 model running on CPU (slower)")
            self.device = 'cpu'
        
        self.model.eval()
        
        with torch.no_grad():
            _ = self.model.encode("warmup", convert_to_numpy=True, show_progress_bar=False)
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        using E5 to retrieve documents
        """
        # Check cache
        cache_key = f"{query}_{top_k}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        start_time = time.time()
        
        query_text = f"query: {query}"
        
        with torch.no_grad():
            query_embedding = self.model.encode(
                query_text, 
                convert_to_numpy=True, 
                show_progress_bar=False,
                normalize_embeddings=True
            )
        
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        
        # FAISS search
        distances, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for dist, faiss_idx in zip(distances[0], indices[0]):
            if faiss_idx == -1:
                continue
            
            if faiss_idx in self.faiss_to_doc:
                doc_info = self.faiss_to_doc[faiss_idx]
                doc_id = doc_info['id']
                content = doc_info['content']
                
                metadata = self.doc_metadata.get(doc_id, {})
                paper_id = metadata.get('paper_id', doc_id)
                
                results.append({
                    "id": doc_id,
                    "content": content,
                    "score": float(dist),
                    "paper_id": paper_id,
                    "metadata": metadata
                })
            else:
                logger.debug(f"FAISS index {faiss_idx} not found in mapping")
        
        if len(self._cache) >= self._cache_size:
            self._cache.pop(next(iter(self._cache)))
        self._cache[cache_key] = results
        
        elapsed = time.time() - start_time
        
        if elapsed > 5:
            logger.warning(f"E5 retrieval slow: {elapsed:.3f} seconds")
        else:
            logger.info(f"E5 retrieval: {len(results)} results, elapsed time {elapsed:.3f} seconds")

        return results
    
    def retrieve_abstracts(self, query: str, top_k: int = 5) -> List[Tuple[str, str]]:
        """
        Retrieve abstracts (compatible interface)
        
        Returns:
            List of (abstract_text, paper_id) tuples
        """
        docs = self.retrieve(query, top_k)
        return [(doc["content"], doc["paper_id"]) for doc in docs]
    
    def rebuild_mapping_cache(self):
        """rebuild mapping cache from scratch"""
        logger.info("Rebuilding mapping cache...")
        
        # Remove old cache
        if self.mapping_cache_file.exists():
            self.mapping_cache_file.unlink()
            logger.info("Old cache file deleted")
        
        # Rebuild
        self._connect_db()
        self._build_and_save_mapping()
    
    def close(self):
        """Clean up resources"""
        if hasattr(self, 'db_conn'):
            self.db_conn.close()
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
            alpha: E5 weight (hybrid)
            top_k: return top-k results
        """
        self.strategy = strategy
        self.alpha = alpha
        self.top_k = top_k

        logger.info(f"initialize retriever - 策略: {strategy}")

        # Initialize E5 (if needed)
        self.e5 = None
        if strategy in ["e5", "hybrid"]:
            self.e5 = E5DirectRetriever(e5_index_directory)
            logger.info("E5 retriever loaded")
        
        self.bm25 = None
        if strategy in ["bm25", "hybrid"]:
            try:
                from fast_llamaindex_retriever import FastLlamaIndexBM25Retriever
                self.bm25 = FastLlamaIndexBM25Retriever(bm25_index_directory, top_k)
                logger.info("BM25 retriever loaded (fast version)")
            except ImportError:
                try:
                    from bm25_only_retriever import BM25OnlyRetriever
                    self.bm25 = BM25OnlyRetriever(bm25_index_directory, top_k)
                    logger.info("BM25 retriever loaded (pure BM25 version)")
                except ImportError:
                    logger.warning("BM25 retriever not available")
        
        # LevelDB
        self.leveldb = None
        if leveldb_path:
            try:
                import plyvel
                self.leveldb = plyvel.DB(leveldb_path, create_if_missing=False)
                logger.info(f"LevelDB connect: {leveldb_path}")
            except Exception as e:
                logger.warning(f"LevelDB connect failed: {e}")

        self._executor = ThreadPoolExecutor(max_workers=2)
        
        # caching
        self._cache = {}
        self._cache_size = 50
    
    def retrieve_abstracts(self, query: str, top_k: int = None) -> List[Tuple[str, str]]:
        """
        retrieve abstracts based on strategy
        
        Returns:
            List of (abstract_text, paper_id) tuples
        """
        if top_k is None:
            top_k = self.top_k
        
        cache_key = f"{self.strategy}_{query}_{top_k}"
        if cache_key in self._cache:
            logger.info("caching hit")
            return self._cache[cache_key]
        
        logger.info(f"{self.strategy.upper()} retrieve: '{query}'")
        
        if self.strategy == "e5":
            result = self._retrieve_e5(query, top_k)
        elif self.strategy == "bm25":
            result = self._retrieve_bm25(query, top_k)
        else:  # hybrid
            result = self._retrieve_hybrid(query, top_k)
        
        if len(self._cache) >= self._cache_size:
            self._cache.pop(next(iter(self._cache)))
        self._cache[cache_key] = result
        
        return result
    
    def _retrieve_e5(self, query: str, top_k: int) -> List[Tuple[str, str]]:
        """E5 retrieve"""
        if not self.e5:
            logger.error("E5 not initialized")
            return []
        return self.e5.retrieve_abstracts(query, top_k)
    
    def _retrieve_bm25(self, query: str, top_k: int) -> List[Tuple[str, str]]:
        """BM25 retrieve"""
        if not self.bm25:
            logger.error("BM25 not initialized")
            return []
        return self.bm25.retrieve_abstracts(query, top_k)
    
    def _retrieve_hybrid(self, query: str, top_k: int) -> List[Tuple[str, str]]:
        """Hybrid retrieve"""
        if not self.e5 or not self.bm25:
            logger.error("Hybrid mode requires E5 and BM25")
            return self._retrieve_e5(query, top_k) if self.e5 else []
        
        # Parallel retrieval
        e5_future = self._executor.submit(self.e5.retrieve, query, top_k * 2)
        bm25_future = self._executor.submit(self.bm25.retrieve_abstracts, query, top_k * 2)
        
        e5_results = e5_future.result()
        bm25_results = bm25_future.result()
        
        doc_scores = {}
        doc_contents = {}
        
        # E5 results
        for doc in e5_results:
            paper_id = doc["paper_id"]
            doc_scores[paper_id] = doc_scores.get(paper_id, 0) + self.alpha * doc["score"]
            doc_contents[paper_id] = doc["content"]

        # BM25 results
        for i, (content, paper_id) in enumerate(bm25_results):
            # BM25 score: use reciprocal rank as score
            bm25_score = 1.0 / (i + 1)
            doc_scores[paper_id] = doc_scores.get(paper_id, 0) + (1 - self.alpha) * bm25_score
            if paper_id not in doc_contents:
                doc_contents[paper_id] = content

        # Sort and select top_k
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        
        results = []
        for paper_id, _ in sorted_docs[:top_k]:
            if paper_id in doc_contents:
                results.append((doc_contents[paper_id], paper_id))

        logger.info(f"Hybrid retrieve: E5({len(e5_results)}) + BM25({len(bm25_results)}) = {len(results)} results")

        return results
    
    def get_full_texts(self, doc_ids: List[str], db=None) -> List[Tuple[str, str]]:
        """
        acquire full texts - add db parameter for flexibility
        
        Args:
            doc_ids: document IDs
            db: LevelDB instance (optional)
        
        Returns:
            List of (full_text, doc_id) tuples
        """
        if not self.leveldb and db:
            results = []
            for doc_id in doc_ids:
                try:
                    content = db.get(doc_id.encode('utf-8'))
                    if content:
                        results.append((content.decode('utf-8'), doc_id))
                except Exception as e:
                    logger.error(f"acquire {doc_id} failed: {e}")
            return results
        elif self.leveldb:
            results = []
            for doc_id in doc_ids:
                try:
                    content = self.leveldb.get(doc_id.encode('utf-8'))
                    if content:
                        results.append((content.decode('utf-8'), doc_id))
                except Exception as e:
                    logger.error(f"acquire {doc_id} failed: {e}")
            return results
        else:
            logger.warning("no LevelDB available for full text retrieval")
            return []
    
    def close(self):
        """clean up resources"""
        if self.e5:
            self.e5.close()
        if self.bm25 and hasattr(self.bm25, 'close'):
            self.bm25.close()
        if self.leveldb:
            self.leveldb.close()
        self._executor.shutdown(wait=True)
        logger.info("retriever closed")


if __name__ == "__main__":
    import sys
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    E5_INDEX = "/data/horse/ws/s3811141-faiss/inbe405h-unarxive/faiss_index"
    BM25_INDEX = "/data/horse/ws/s3811141-faiss/inbe405h-unarxive/bm25_retriever"
    LEVELDB = "/data/horse/ws/inbe405h-unarxive/full_text_db"
    
    strategies = ["e5", "bm25", "hybrid"] if len(sys.argv) <= 1 else [sys.argv[1]]
    
    for strategy in strategies:
        print(f"\n{'='*60}")
        print(f"test {strategy.upper()} strategy")
        print('='*60)
        
        retriever = UnifiedArxivRetriever(
            e5_index_directory=E5_INDEX,
            bm25_index_directory=BM25_INDEX,
            leveldb_path=LEVELDB if Path(LEVELDB).exists() else None,
            strategy=strategy,
            alpha=0.65
        )
        
        queries = [
            "quantum computing algorithms",
            "deep learning transformers",
            "protein folding prediction"
        ]
        
        for query in queries:
            print(f"\nquery: '{query}'")
            start = time.time()
            results = retriever.retrieve_abstracts(query, top_k=3)
            elapsed = time.time() - start

            print(f"found {len(results)} results (time: {elapsed:.3f}s)")
            for i, (abstract, paper_id) in enumerate(results, 1):
                preview = abstract[:100] + "..." if len(abstract) > 100 else abstract
                print(f"  [{i}] {paper_id}: {preview}")
        
        if retriever.leveldb and results:
            print("\ntest full text retrieval...")
            doc_ids = [paper_id for _, paper_id in results[:2]]
            full_texts = retriever.get_full_texts(doc_ids)
            for full_text, doc_id in full_texts:
                print(f"  {doc_id}: {len(full_text)} characters")
        
        retriever.close()

    print("\nAll tests completed!")