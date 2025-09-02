import os
import torch
import sys

def get_main_data_dir():
    fallback_paths = [
        f"{os.getenv('HOME')}/data_dir",
        "/data/horse/ws/inbe405h-unarxive",
        "/data/horse/ws/s3811141-faiss/inbe405h-unarxive"
    ]

    resolved_path = None

    for path in fallback_paths:
        # Erst prüfen, ob es eine "data_dir"-Datei ist
        if os.path.isfile(path):
            try:
                with open(path, "r") as f:
                    content = f.read().strip()
                if content and os.path.isdir(content):
                    resolved_path = content
                    break
            except (OSError, IOError) as e:
                sys.stderr.write(f"Fehler beim Lesen von {path}: {e}\n")
                continue

        # Wenn es direkt ein Verzeichnis ist
        if os.path.isdir(path):
            resolved_path = path
            break

    if resolved_path is None:
        sys.stderr.write(
            "Kein gültiges Datenverzeichnis gefunden. "
            f"Versuchte Pfade: {', '.join(fallback_paths)}\n"
        )
        sys.exit(1)

    return resolved_path


MAIN_DATA_DIR = get_main_data_dir()

EMBEDDING_MODEL = "intfloat/e5-large-v2"
MODEL_FORMAT = "sentence_transformers"
EMBEDDING_DIM = 1024
USE_GPU = torch.cuda.is_available()

# Configuration paths
DATA_DIR = f"{MAIN_DATA_DIR}_extended_data"
E5_INDEX_DIR = f"{MAIN_DATA_DIR}/faiss_index"
BM25_INDEX_DIR = f"{MAIN_DATA_DIR}/bm25_retriever"
DB_PATH = f"{MAIN_DATA_DIR}/full_text_db"
