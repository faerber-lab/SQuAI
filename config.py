import os
import torch
from get_paths import get_main_data_dir

# MAIN_DATA_DIR holds faiss_index/, bm25_retriever/ and full_text_db/.
# Env override keeps the repo portable across deployments; falls back to get_paths.
MAIN_DATA_DIR = os.environ.get("MAIN_DATA_DIR") or get_main_data_dir()

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

# --------------------------------------------------------------------------- #
# Passage-level context selection (replaces the TOP/BOTTOM char slice in Agent 4)
# --------------------------------------------------------------------------- #
# Turn the passage ranker on/off. When off, Agent 4 uses the legacy char-slice path.
USE_PASSAGES = os.environ.get("USE_PASSAGES", "1") == "1"
# Backend: "bm25" (CPU, zero-dep, default) | "scads_rerank" | "scads_embed".
# The scads_* backends offload semantic ranking to ScaDS GPUs (no local model load)
# and automatically fall back to BM25 order if the API is unavailable.
PASSAGE_BACKEND = os.environ.get("PASSAGE_BACKEND", "bm25")
PASSAGE_TOP_K = int(os.environ.get("PASSAGE_TOP_K", "12"))      # passages sent to Agent 4
PASSAGE_CANDIDATES = int(os.environ.get("PASSAGE_CANDIDATES", "24"))  # BM25 -> rerank shortlist
# ScaDS API models (see https://llm.scads.ai/docs/models/).
SCADS_API_BASE = os.environ.get("SCADS_API_BASE", "https://llm.scads.ai/v1")
RERANK_MODEL = os.environ.get("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "Qwen/Qwen3-Embedding-4B")
# Phase-1 latency: run the per-abstract Agent2->Agent3 chains concurrently instead of
# in two sequential loops. AGENT_CONCURRENCY bounds simultaneous remote LLM calls (a
# separate pool from the sub-question executor). AGENT2_MAX_TOKENS caps Agent 2's
# answer length (it only feeds Agent 3's judgement, so it needs far fewer than 1024).
AGENT_CONCURRENCY = int(os.environ.get("AGENT_CONCURRENCY", "8"))
AGENT2_MAX_TOKENS = int(os.environ.get("AGENT2_MAX_TOKENS", "256"))
# Sentence-level attribution: map each answer sentence to its source passage span.
# "lexical" = fast word-overlap (no API). "scads_rerank" = use the bge reranker for a
# stronger entailment-style grounding (one rerank call per sentence; falls back to lexical).
ATTRIBUTION_VERIFY = os.environ.get("ATTRIBUTION_VERIFY", "lexical")
