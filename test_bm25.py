# test_bm25_import.py
import sys
import logging
logging.basicConfig(level=logging.DEBUG)

print("Testing BM25 imports...")

# Test 1: Check llama_index imports
print("\n1. Testing llama_index imports:")
try:
    from llama_index.retrievers.bm25 import BM25Retriever
    print("✓ llama_index.retrievers.bm25 imported successfully")
except ImportError as e:
    print(f"✗ llama_index.retrievers.bm25 import failed: {e}")

try:
    from llama_index.core import Document
    print("✓ llama_index.core imported successfully")
except ImportError as e:
    print(f"✗ llama_index.core import failed: {e}")

# Test 2: Try importing fast_llamaindex_retriever
print("\n2. Testing fast_llamaindex_retriever:")
try:
    from SQuAI.fast_llamaindex_retriever import FastLlamaIndexBM25Retriever
    print("✓ FastLlamaIndexBM25Retriever imported successfully")
    
    # Try creating instance
    bm25_dir = "/data/horse/ws/s3811141-faiss/inbe405h-unarxive/bm25_retriever"
    retriever = FastLlamaIndexBM25Retriever(bm25_dir, top_k=5)
    print("✓ FastLlamaIndexBM25Retriever instance created successfully")
    
except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Try importing bm25_only_retriever
print("\n3. Testing bm25_only_retriever:")
try:
    from bm25_only_retriever import BM25OnlyRetriever
    print("✓ BM25OnlyRetriever imported successfully")
except ImportError as e:
    print(f"✗ bm25_only_retriever import failed: {e}")