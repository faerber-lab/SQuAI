import os
import torch
from get_paths import get_main_data_dir

MAIN_DATA_DIR = get_main_data_dir()

EMBEDDING_MODEL = "intfloat/e5-large-v2"
MODEL_FORMAT = "sentence_transformers"
EMBEDDING_DIM = 1024
# Default: CPU for demo deployment. Set USE_GPU=1 to enable GPU when available.
USE_GPU = os.environ.get("USE_GPU", "0") == "1" and torch.cuda.is_available()

# Configuration paths
DATA_DIR = f"{MAIN_DATA_DIR}_extended_data"
E5_INDEX_DIR = f"{MAIN_DATA_DIR}/faiss_index"
# E5_INDEX_DIR = f"/data/horse/ws/jihe529c-main-rag/BM25/faiss_index"
BM25_INDEX_DIR = f"{MAIN_DATA_DIR}/bm25_retriever"
DB_PATH = f"{MAIN_DATA_DIR}/full_text_db"
# DB_PATH = f"/data/horse/ws/jihe529c-main-rag/BM25/db2"
