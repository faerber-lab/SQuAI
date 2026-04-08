import sys
import os
import socket
from fastapi import FastAPI
from pydantic import BaseModel
import plyvel
from run_SQuAI import Enhanced4AgentRAG, initialize_retriever
from config import DB_PATH, BM25_INDEX_DIR, E5_INDEX_DIR
from typing import Optional, List

# Import language detection library
from langdetect import detect, LangDetectException

# SCADS AI agent (CPU deployment – no local GPU required)
from scads_agent import ScadsAgent


app = FastAPI()

# Default config values
# Model is served via SCADS AI API; set SCADS_MODEL env var to override.
DEFAULT_MODEL = os.environ.get("SCADS_MODEL", "meta-llama/Llama-4-Scout-17B-16E-Instruct")
DEFAULT_RETRIEVER = "hybrid"
DEFAULT_N_VALUE = 0.5
DEFAULT_TOP_K = 5
DEFAULT_ALPHA = 0.65

# Non-English query rejection message
NON_ENGLISH_MESSAGE = (
    "It looks like your question isn't in English. "
    "If you could translate it to English, I'll do my best "
    "to give you a complete answer."
)


# Language detection function
def detect_language(text: str) -> str:
    """
    Detect query language
    Returns: 'en', 'de', 'zh', 'ja', 'es', 'fr', etc.
    """
    try:
        return detect(text)
    except LangDetectException:
        return "en"  # Default to English on error


# Global objects
db = None
ragent = None
_scads_agents = {}  # cache: model_name -> ScadsAgent


def _get_scads_agent(model: str) -> "ScadsAgent":
    """Return a cached ScadsAgent for the given model, creating one if needed."""
    if model not in _scads_agents:
        _scads_agents[model] = ScadsAgent(model=model)
    return _scads_agents[model]


@app.on_event("startup")
def startup_event():
    global db, ragent
    write_host_and_port_file()
    try:
        db = plyvel.DB(DB_PATH, create_if_missing=False)
        retriever = initialize_retriever(
            retriever_type=DEFAULT_RETRIEVER,
            e5_index_dir=E5_INDEX_DIR,
            bm25_index_dir=BM25_INDEX_DIR,
            db_path=DB_PATH,
            top_k=DEFAULT_TOP_K,
            alpha=DEFAULT_ALPHA,
        )
        # Instantiate SCADS AI agent (reads SCADS_API_KEY from environment)
        scads_agent = _get_scads_agent(DEFAULT_MODEL)
        ragent = Enhanced4AgentRAG(
            retriever=retriever,
            agent_model=scads_agent,  # pass pre-built agent object
            n=DEFAULT_N_VALUE,
            index_dir=BM25_INDEX_DIR,
            max_workers=6,
        )
    except plyvel._plyvel.IOError as e:
        print(f"Error: {e}. Cannot continue.")
        sys.exit(1)


def write_host_and_port_file():
    port = os.getenv("uvicorn_port", "8000")
    host = socket.gethostname()
    filepath = os.path.expanduser("~/hpc_server_host_and_file")
    try:
        with open(filepath, "w") as f:
            f.write(f"{host}:{port}\n")
    except OSError as e:
        print(f"Failed to write {filepath}: {e}")


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


def _swap_agents_if_needed(model: str):
    """Switch ragent's underlying LLM agents to the requested model."""
    agent = _get_scads_agent(model)
    ragent.agent1 = agent
    ragent.agent2 = agent
    ragent.agent3 = agent
    ragent.agent4 = agent
    ragent.question_splitter.agent = agent


@app.post("/split")
def split_question(req: QueryRequest):
    # Detect query language
    detected_lang = detect_language(req.question)

    if detected_lang != "en":
        # Non-English query, return marker for frontend
        return {
            "should_split": False,
            "sub_questions": [],
            "original_question": req.question,
            "detected_language": detected_lang,
            "is_non_english": True
        }

    # Switch to the requested model if different from current
    _swap_agents_if_needed(req.model)

    # English query, proceed with normal split logic
    should_split, sub_questions = ragent.question_splitter.analyze_and_split(
        req.question
    )
    return {
        "should_split": should_split,
        "sub_questions": sub_questions if should_split else [],
        "original_question": req.question,
        "detected_language": "en"
    }


@app.post("/ask")
def ask_question(req: QueryRequest):
    # Detect query language (critical step)
    detected_lang = detect_language(req.question)
    
    if detected_lang != "en":
        # Non-English query detected, return rejection message immediately
        # Format matches normal response so frontend displays it correctly
        return {
            "answer": NON_ENGLISH_MESSAGE,
            "references": [],  # Empty reference list
            "debug_info": {
                "original_query": req.question,
                "detected_language": detected_lang,
                "status": "rejected_non_english",
                "was_split": False,
                "sub_questions": [],
                "questions_processed": 0,
                "full_texts_retrieved": 0,
                "total_filtered_docs": 0,
                "total_citations": 0
            }
        }

    # Switch to the requested model if different from current
    _swap_agents_if_needed(req.model)

    # English query, proceed with normal RAG pipeline
    result, references, debug_info = ragent.answer_query(
        req.question, db, should_split=req.should_split, sub_questions=req.sub_questions
    )
    return {"answer": result, "references": references, "debug_info": debug_info}