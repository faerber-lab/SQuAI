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
from langdetect import detect, detect_langs, DetectorFactory, LangDetectException

# Make langdetect deterministic so the same query yields the same verdict
# across runs (default behavior is non-deterministic).
DetectorFactory.seed = 0

# SCADS AI agent (CPU deployment – no local GPU required)
from scads_agent import ScadsAgent


app = FastAPI()

# Default config values
# Model is served via SCADS AI API; set SCADS_MODEL env var to override.
DEFAULT_MODEL = os.environ.get("SCADS_MODEL", "google/gemma-4-31B-it")
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
# Tuned to reduce false positives on short scientific queries like
# "dense retrieval" or "BM25" that langdetect frequently misclassifies.
MIN_DETECT_CHARS = 20      # below this, default to English
MIN_CONFIDENCE = 0.85      # below this on ASCII input, default to English

def detect_language(text: str) -> str:
    """
    Detect query language.
    Returns: 'en', 'de', 'zh', 'ja', 'es', 'fr', etc.

    Heuristics on top of langdetect:
    1. Very short input -> 'en' (langdetect is unreliable on < ~20 chars).
    2. Pure ASCII letters/punct -> require high confidence to leave 'en',
       otherwise default to 'en' (scientific English terms often trip
       langdetect into Catalan/Indonesian/etc.).
    3. Any non-ASCII (CJK, Cyrillic, Arabic, ...) -> trust full detection.
    """
    text = (text or "").strip()
    if len(text) < MIN_DETECT_CHARS:
        return "en"

    if text.isascii():
        try:
            candidates = detect_langs(text)
            if not candidates:
                return "en"
            top = candidates[0]
            if top.lang == "en" or top.prob < MIN_CONFIDENCE:
                return "en"
            return top.lang
        except LangDetectException:
            return "en"

    try:
        return detect(text)
    except LangDetectException:
        return "en"


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
        # NOTE: index_dir is intentionally NOT set to BM25_INDEX_DIR.
        # BM25_INDEX_DIR contains a 7.5 GB corpus.jsonl whose schema does not
        # match what EnhancedCitationHandler._load_arxiv_papers expects, so
        # parsing it produced no usable metadata while costing ~82s per query.
        # Letting index_dir default to a non-existent path makes the arxiv
        # metadata loader a no-op; BM25 retrieval itself is unaffected because
        # it goes through the `retriever` object (not self.index_dir).
        ragent = Enhanced4AgentRAG(
            retriever=retriever,
            agent_model=scads_agent,  # pass pre-built agent object
            n=DEFAULT_N_VALUE,
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
    # Multi-turn chat: prior turns [{"question": str, "answer": str}, ...]. When present,
    # the new question is rewritten into a standalone question before retrieval. When
    # absent/empty, behaviour is identical to single-query mode (fully backward compatible).
    chat_history: Optional[List[dict]] = None


def contextualize_question(history, question, agent) -> str:
    """Rewrite a follow-up into a self-contained, standalone question using prior turns.

    The retriever needs a standalone question (a bare "what about its limitations?" has no
    subject to retrieve on). Returns the original question unchanged when there is no
    history, or on any failure.
    """
    if not history:
        return question
    turns = []
    for h in history[-3:]:                      # last 3 turns keep the prompt small
        q = (h.get("question") or "").strip()
        a = (h.get("answer") or "").strip()
        if q:
            turns.append(f"User: {q}")
        if a:
            turns.append(f"Assistant: {a[:400]}")
    if not turns:
        return question
    prompt = (
        "You rewrite a user's follow-up question into a fully self-contained, standalone "
        "question for a scientific search system. Resolve every pronoun and reference "
        "(it, they, that, this method, the model, etc.) using the conversation. If the "
        "follow-up is already standalone, return it unchanged. Output ONLY the rewritten "
        "question, with no preamble or quotes.\n\n"
        f"Conversation:\n{chr(10).join(turns)}\n\n"
        f"Follow-up question: {question}\n\n"
        "Standalone question:"
    )
    try:
        rewritten = (agent.generate(prompt, max_new_tokens=128) or "").strip().strip('"').strip()
        if rewritten and len(rewritten) <= 500:
            return rewritten
    except Exception:
        pass
    return question


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

    # Multi-turn: rewrite a follow-up into a standalone question using chat history.
    # No history -> standalone == req.question, so single-query behaviour is unchanged.
    standalone = contextualize_question(req.chat_history, req.question, ragent.agent1)
    rewritten = standalone != req.question
    # A rewritten question must be (re)analysed for splitting; ignore stale client split.
    should_split = None if rewritten else req.should_split
    sub_questions = None if rewritten else req.sub_questions

    result, references, debug_info = ragent.answer_query(
        standalone, db, should_split=should_split, sub_questions=sub_questions
    )
    if isinstance(debug_info, dict):
        debug_info["asked_question"] = req.question
        debug_info["standalone_question"] = standalone
    return {"answer": result, "references": references, "debug_info": debug_info}
