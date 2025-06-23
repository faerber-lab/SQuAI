#!/usr/bin/env python3
"""
bm25_llamaindex.py

1) Parses all JSONL under DATA_DIR into LlamaIndex Documents (in parallel)
2) Builds a BM25Retriever and persists it to ./bm25_index/
3) Demonstrates re-loading and querying the saved index
"""

import os
import glob
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

# 1) Install dependencies
#    pip install llama-index llama-index-retrievers-bm25
from llama_index.core import Document
from llama_index.retrievers.bm25 import BM25Retriever
import Stemmer
from llama_index.core.storage.docstore import SimpleDocumentStore
# ─── USER CONFIG ───────────────────────────────────────────────────────────────
DATA_DIR = "/data/horse/ws/inbe405h-unarxive/test"   # JSONL root directory
PERSIST_DIR = "/data/horse/ws/inbe405h-unarxive/bm25_retriever" # where the BM25 index will be saved
DOCSTORE_PATH = os.path.join(PERSIST_DIR, "docstore.json")
TOP_K = 5                              # how many docs to return per query
# ────────────────────────────────────────────────────────────────────────────────

def _parse_jsonl_file(file_path):
    """
    Load a JSONL file and return a list of llama_index.Document objects.
    Executed in parallel.
    """
    docs = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                paper = json.loads(line.strip())
                meta = paper.get("metadata", {}) or {}
                title = meta.get("title", "")
                abstract = paper.get("abstract", "")
                sections = paper.get("sections", {}) or {}
                sections_text = " ".join(
                    f"{sec}: {obj.get('text','')}" for sec, obj in sections.items() if isinstance(obj, dict)
                )
                content = f"{title}. {abstract}"
                full_text = f"{title}. {abstract}. {sections_text}" 
                docs.append(Document(
                    content=content,
                    meta={
                        "paper_id": paper.get("paper_id", ""),
                        "title": title,
                        "authors": meta.get("authors", []),
                        "full_text": full_text
                    }
                ))
            except json.JSONDecodeError:
                continue
    return docs

def load_documents(data_dir):
    """Walk data_dir for .jsonl, parse each in parallel, return flat list of Documents."""
    pattern = os.path.join(data_dir, "**", "*.jsonl")
    files = glob.glob(pattern, recursive=True)
    print(f"Found {len(files)} JSONL files under {data_dir}")
    docs = []
    workers = multiprocessing.cpu_count()
    print(f"Parsing with {workers} workers…")
    with ProcessPoolExecutor(max_workers=workers) as exe:
        for batch in exe.map(_parse_jsonl_file, files):
            docs.extend(batch)
    print(f"Loaded a total of {len(docs)} documents")
    return docs

def build_and_persist_bm25(docs, persist_dir):
    """Builds a BM25Retriever over `docs` and persists it to disk."""
    # 1) Create the retriever
    print("[INFO] Saving docstore...")
    docstore = SimpleDocumentStore()
    docstore.add_documents(docs)
    retriever = BM25Retriever.from_defaults(
        docstore=docstore,
        similarity_top_k=TOP_K,
        stemmer=Stemmer.Stemmer("english"),
        language="english"
    )
    # 2) Persist the index
    retriever.persist(persist_dir)
    print(f"BM25 index built and saved to {persist_dir}")
    return retriever

def load_bm25(persist_dir):
    """Reloads a persisted BM25Retriever from disk."""
    retriever = BM25Retriever.from_persist_dir(persist_dir)
    print(f"BM25 index loaded from {persist_dir}")
    return retriever

def demo_query(retriever, query):
    """Run a sample query and print results."""
    print(f"\n Query: “{query}”\n")
    results = retriever.retrieve(query)
    for i, res in enumerate(results, 1):
        text_snip = res.node.get_text()[:].replace("\n"," ")
        print(f"{i}. [score={res.score:.4f}] {text_snip}…")
    print()

if __name__ == "__main__":
    # Step 1: Parse and load
    

    # Step 2: Build & persist
    if not os.path.exists(PERSIST_DIR):
        os.makedirs(PERSIST_DIR, exist_ok=True)
        documents = load_documents(DATA_DIR)
        bm25 = build_and_persist_bm25(documents, PERSIST_DIR)
    else:
        print(f"{PERSIST_DIR} already exists; skipping build.")

    # Step 3: Reload and demo
    retr = load_bm25(PERSIST_DIR)
    demo_query(retr, "explain to me How we can Controll discrete quantum walks")
