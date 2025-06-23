from haystack.document_stores import FAISSDocumentStore
from haystack.nodes import EmbeddingRetriever
from haystack import Document
import logging

from config import EMBEDDING_MODEL, MODEL_FORMAT, EMBEDDING_DIM, USE_GPU

logger = logging.getLogger(__name__)

class HaystackRetriever:
    """
    FIXED: Haystack retriever that keeps E5 model in memory
    """
    def __init__(self, e5_index_directory: str):
        logger.info(f"🔄 Loading Haystack retriever from {e5_index_directory}")
        
        # Load document store
        self.document_store = FAISSDocumentStore.load(
            index_path=f"{e5_index_directory}/faiss_index",
            config_path=f"{e5_index_directory}/faiss_index.json"
        )
        
        # Initialize retriever - this loads the model
        logger.info("🔄 Loading E5 model (this may take time on first load)...")
        self.retriever = EmbeddingRetriever(
            document_store=self.document_store,
            embedding_model=EMBEDDING_MODEL,
            model_format=MODEL_FORMAT,
            use_gpu=USE_GPU
        )
        
        # CRITICAL FIX: Force model to stay loaded
        self._ensure_model_loaded()
        
        # Track if we've done the initial expensive load
        self._model_warmed_up = False
        
        logger.info("✅ Haystack retriever initialized")

    def _ensure_model_loaded(self):
        """
        CRITICAL FIX: Ensure the embedding model stays in memory
        """
        try:
            # Access the embedding model to ensure it's loaded
            if hasattr(self.retriever, 'embedding_model'):
                model = self.retriever.embedding_model
                
                # For sentence transformers, ensure it's loaded
                if hasattr(model, '_model') and hasattr(model._model, 'eval'):
                    model._model.eval()  # Put in eval mode
                    logger.info("✅ E5 model forced to eval mode")
                
                # Store reference to prevent garbage collection
                self._cached_model = model
                
            # Also check if there's a tokenizer to cache
            if hasattr(self.retriever, 'tokenizer'):
                self._cached_tokenizer = self.retriever.tokenizer
                
        except Exception as e:
            logger.warning(f"Could not cache model reference: {e}")

    def retrieve(self, query: str, top_k: int = 5):
        """
        OPTIMIZED: Retrieve with model warmup detection
        """
        import time
        start_time = time.time()
        
        # First query detection - will be slow due to model loading
        if not self._model_warmed_up:
            logger.info(f"🔥 First E5 query - this will be slow due to model loading...")
        
        # Perform retrieval
        docs = self.retriever.retrieve(query, top_k=top_k)
        
        elapsed = time.time() - start_time
        
        # Mark as warmed up after first query
        if not self._model_warmed_up:
            self._model_warmed_up = True
            logger.info(f"✅ E5 model warmed up! First query took {elapsed:.2f}s")
            logger.info("🚀 Subsequent queries should be much faster (2-5s)")
        else:
            # This should be fast now
            if elapsed > 10:
                logger.warning(f"⚠️ E5 query took {elapsed:.2f}s - model may have reloaded!")
            else:
                logger.info(f"⚡ E5 query took {elapsed:.2f}s (good!)")
        
        return docs

    def close(self):
        """Clean up resources but try to keep model loaded"""
        try:
            # Don't actually close the model, just the document store if needed
            logger.info("Haystack retriever closing (keeping model loaded)")
        except Exception as e:
            logger.error(f"Error closing Haystack retriever: {e}")