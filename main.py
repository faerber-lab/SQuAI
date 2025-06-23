from fastapi import FastAPI
from pydantic import BaseModel
import plyvel
from run_SQuAI import Enhanced4AgentRAG, initialize_retriever
from config import DB_PATH, BM25_INDEX_DIR, E5_INDEX_DIR
from typing import Optional, List

app = FastAPI()

# Default config values
DEFAULT_MODEL = "tiiuae/Falcon3-10B-Instruct"
DEFAULT_RETRIEVER = "bm25"
DEFAULT_N_VALUE = 0.5
DEFAULT_TOP_K = 5
DEFAULT_ALPHA = 0.65

# Global objects
db = None
ragent = None

@app.on_event("startup")
def startup_event():
    global db, ragent
    db = plyvel.DB(DB_PATH, create_if_missing=False)
    
    retriever = initialize_retriever(
        retriever_type=DEFAULT_RETRIEVER,
        e5_index_dir=E5_INDEX_DIR,
        bm25_index_dir=BM25_INDEX_DIR,
        db_path=DB_PATH,
        top_k=DEFAULT_TOP_K,
        alpha=DEFAULT_ALPHA
    )

    ragent = Enhanced4AgentRAG(
        retriever=retriever,
        agent_model=DEFAULT_MODEL,
        n=DEFAULT_N_VALUE,
        index_dir=BM25_INDEX_DIR,  # Change if needed
        max_workers=6
    )

@app.on_event("shutdown")
def shutdown_event():
    if db is not None:
        db.close()

# This model is used for dynamic POST requests
class QueryRequest(BaseModel):
    question: str
    should_split: Optional[bool] = None
    sub_questions: Optional[List[str]] = None
    model: Optional[str] = DEFAULT_MODEL
    retrieval_method: Optional[str] = DEFAULT_RETRIEVER
    n_value: Optional[float] = DEFAULT_N_VALUE
    top_k: Optional[int] = DEFAULT_TOP_K
    alpha: Optional[float] = DEFAULT_ALPHA    

@app.post("/split")
def split_question(req: QueryRequest):
    should_split, sub_questions = ragent.question_splitter.analyze_and_split(req.question)
    return {
        "should_split": should_split,
        "sub_questions": sub_questions if should_split else [],
        "original_question": req.question
    }

@app.post("/ask")
def ask_question(req: QueryRequest):
    result, references, debug_info = ragent.answer_query(req.question, db, should_split=req.should_split, sub_questions=req.sub_questions)
    return {
        "answer": result,
        "references": references,
        "debug_info": debug_info
    }

