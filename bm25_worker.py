#!/usr/bin/env python
import sys, json
from llama_index.retrievers.bm25 import BM25Retriever
import inspect
import time
query     = sys.argv[1]
index_dir = sys.argv[2]
top_k     = int(sys.argv[3])

retr = BM25Retriever.from_persist_dir(index_dir)

start_time= time.time()
results = retr.retrieve(query)
top_results = results[:top_k] 
out = [
    {
        "paper_id": r.node.metadata.get("paper_id"),
        "title": r.node.metadata.get("title"),
        "text":     r.node.get_text(),
        "score":    r.score
    } for r in top_results
]

print(json.dumps(out))

