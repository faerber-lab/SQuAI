import torch

EMBEDDING_MODEL = "intfloat/e5-large-v2"
MODEL_FORMAT = "sentence_transformers"
EMBEDDING_DIM = 1024
USE_GPU = torch.cuda.is_available()

# Configuration paths
DATA_DIR = "/data/horse/ws/inbe405h-unarxive/processed_unarxive_extended_data"
DOC_INDEX_DIR = "/data/horse/ws/inbe405h-unarxive/faiss_index"
E5_INDEX_DIR = "/data/horse/ws/inbe405h-unarxive/faiss_index"
BM25_INDEX_DIR = "/data/horse/ws/inbe405h-unarxive/bm25_retriever"
DB_PATH = "/data/horse/ws/inbe405h-unarxive/full_text_db"
