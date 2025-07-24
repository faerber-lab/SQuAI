import torch

MAIN_DATA_DIR = open("/etc/data_dir").read().strip() if (lambda f: f and f.strip())(open("/etc/data_dir").read()) else "/data/horse/ws/inbe405h-unarxive"

EMBEDDING_MODEL = "intfloat/e5-large-v2"
MODEL_FORMAT = "sentence_transformers"
EMBEDDING_DIM = 1024
USE_GPU = torch.cuda.is_available()

# Configuration paths
DATA_DIR = f"{MAIN_DATA_DIR}_extended_data"
E5_INDEX_DIR = f"{MAIN_DATA_DIR}/faiss_index"
BM25_INDEX_DIR = f"{MAIN_DATA_DIR}/bm25_retriever"
DB_PATH = f"{MAIN_DATA_DIR}/full_text_db"
