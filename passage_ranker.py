#!/usr/bin/env python3
"""
passage_ranker.py — query-relevant passage extraction + grounded attribution for SQuAI.

Replaces the blind TOP/BOTTOM char-slice in run_SQuAI._prepare_documents_for_agent4
with section-aware passages that are ranked against the query, so Agent 4 sees only
the most relevant ~12 passages (instead of ~48k tokens of mostly-irrelevant text) and
every passage carries an exact source span for sentence-level attribution.

Design (CPU-friendly demo):
  chunk filtered papers  (string ops, free)
    -> BM25 first-pass over an EXPANDED query        (CPU, ~20 ms / few hundred passages)
    -> optional ScaDS rerank or embed                (GPU-side via API, ~1 round-trip)
    -> top-k passages -> numbered [E#] evidence + passage_table for attribution

Backends (selectable):
  "bm25"         : hand-rolled BM25, zero dependency, always available (default / fallback)
  "scads_rerank" : BM25 -> top-N candidates -> BAAI/bge-reranker-v2-m3 via ScaDS rerank route
  "scads_embed"  : BM25 -> top-N candidates -> Qwen/Qwen3-Embedding-4B cosine via ScaDS /embeddings

The data format (verified over 2000 papers in the LevelDB full_text_db):
  line 1            = title                       (100%)
  "abstract: ..."   = abstract header             (100%)
  "<Section>: body" = section headers             (median 7/paper; 0% zero-header after masking)
  inline math       = "{{formula:<uuid>}}"        (~366/paper — MUST be normalised)
  also {{figure:}}, {{table:}}, {{cite:}}
Section bodies are often long (50% > 2000 chars) so long sections are windowed.
"""

from __future__ import annotations

import os
import re
import json
import math
import logging
import urllib.request
import urllib.error
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Dict, Optional, Iterable

import numpy as np

logger = logging.getLogger(__name__)


def _read_key_file(path: str) -> Optional[str]:
    """Read an API key from a file, returning None if it is missing/unreadable."""
    try:
        if os.path.exists(path):
            return open(path).read().strip() or None
    except Exception:
        pass
    return None

# --------------------------------------------------------------------------- #
# Placeholder handling
# --------------------------------------------------------------------------- #
# unarXive embeds non-text objects as "{{type:<uuid>}}". They are extremely dense
# (~366 formulas/paper) and contain an internal ':' that breaks naive header
# parsing, so we MASK them (offset-preserving) before structural parsing and
# render them as readable tokens for the text we actually embed/show.
_PLACEHOLDER_RE = re.compile(r"\{\{([a-zA-Z]+):[^{}]*\}\}")
_DISPLAY_TOKEN = {
    "formula": "[FORMULA]",
    "figure": "[FIGURE]",
    "table": "[TABLE]",
    "cite": "[REF]",
}


def _mask_placeholders(text: str) -> str:
    """Replace every {{type:uuid}} with an EQUAL-LENGTH run of spaces.

    Equal length keeps every character offset identical to the raw blob, so the
    section/window spans we compute index correctly back into the original text
    for display. The spaces also remove the internal ':' that would otherwise
    fool the section-header regex.
    """
    return _PLACEHOLDER_RE.sub(lambda m: " " * len(m.group(0)), text)


# A window edge can slice a {{formula:uuid}}, leaving a half-open fragment such as
# '{formula:uuid}}' or a bare uuid tail '6df2d-59b9-...269}}'. These regexes mop up
# whatever survives the full-placeholder substitution at the two window boundaries.
_UUID_FRAG = re.compile(r"\{*\b[0-9a-fA-F]{4,}(?:-[0-9a-fA-F]{2,}){2,}\b\}*")
_ORPHAN_KW = re.compile(r"\{*\b(?:formula|figure|table|cite)\s*:?\s*(?=\[|$|\s)", re.I)


def normalize_for_model(text: str) -> str:
    """Readable text for embedding / reranking / display: placeholders -> short tokens."""
    text = _PLACEHOLDER_RE.sub(lambda m: f" {_DISPLAY_TOKEN.get(m.group(1).lower(), '[OBJ]')} ", text)
    text = _UUID_FRAG.sub(" [FORMULA] ", text)          # edge-cut uuid tails / heads
    text = text.replace("{{", " ").replace("}}", " ")
    text = _ORPHAN_KW.sub(" ", text)                    # leftover 'formula:' / 'cite:' words
    text = re.sub(r"(\[FORMULA\]\s*){2,}", "[FORMULA] ", text)  # collapse runs
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# --------------------------------------------------------------------------- #
# Section structure
# --------------------------------------------------------------------------- #
# A header is a line-leading label of up to 120 non-colon chars ending in ": ".
# Anchored on the masked text it yields 0% zero-header over the corpus.
_HEADER_RE = re.compile(r"(?m)^([^\n:]{1,120}):[ \t]")

# Section-type buckets (used for the query-intent prior). Keys are matched as
# substrings of the lower-cased, de-numbered header name.
_SECTION_TYPE = {
    "abstract": ["abstract", "summary"],
    "intro": ["introduction", "background", "motivation", "overview", "preliminaries"],
    "method": ["method", "model", "approach", "architecture", "algorithm", "framework",
               "formulation", "design", "implementation", "the case", "theory", "proof"],
    "results": ["result", "experiment", "evaluation", "numerical", "observation",
                "data", "analysis", "performance", "ablation", "benchmark"],
    "discussion": ["discussion", "conclusion", "concluding", "future", "limitation", "remarks"],
}
# Sections that almost never help answer a question — down-weighted, never boosted.
_LOW_VALUE = ("acknowledg", "reference", "bibliograph", "funding", "conflict",
              "author contribution", "appendix", "supplementary", "declaration")


def classify_section(name: str) -> str:
    n = re.sub(r"^\d+(\.\d+)*\.?\s*", "", (name or "").strip().lower())
    for stype, keys in _SECTION_TYPE.items():
        if any(k in n for k in keys):
            return stype
    return "other"


def _is_low_value(name: str) -> bool:
    return any(k in (name or "").lower() for k in _LOW_VALUE)


# Map a coarse query intent -> section types to boost. Intent is optional; if not
# supplied we still apply the abstract/intro mild boost and low-value penalty.
_INTENT_BOOST = {
    "definition": {"abstract", "intro"},
    "method": {"method", "intro"},
    "results": {"results", "discussion"},
    "comparison": {"results", "discussion"},
    "limitations": {"discussion"},
}


# --------------------------------------------------------------------------- #
# Passage
# --------------------------------------------------------------------------- #
@dataclass
class Passage:
    paper_id: str
    section: str
    section_type: str
    char_start: int          # offset into the RAW blob (for exact display)
    char_end: int
    raw_text: str            # verbatim slice of the blob
    text: str = ""           # normalized text (placeholders -> tokens) for model + display
    score: float = 0.0
    eid: str = ""            # "E1", "E2", ... assigned at evidence-build time

    def __post_init__(self):
        if not self.text:
            self.text = normalize_for_model(self.raw_text)

    def display(self, max_chars: int = 600) -> str:
        t = self.text
        return t if len(t) <= max_chars else t[:max_chars].rsplit(" ", 1)[0] + "…"


# --------------------------------------------------------------------------- #
# Chunker
# --------------------------------------------------------------------------- #
class PaperChunker:
    """Split a LevelDB full-text blob into section-aware, offset-tracked passages."""

    def __init__(self, window_chars: int = 900, overlap_chars: int = 150,
                 min_chars: int = 200, max_windows_per_section: int = 4,
                 body_cap: int = 20, max_per_paper: int = 48):
        self.window = window_chars
        self.overlap = overlap_chars
        self.min_chars = min_chars
        self.max_windows_per_section = max_windows_per_section
        self.body_cap = body_cap
        self.max_per_paper = max_per_paper

    def _mk(self, blob: str, pid: str, name: str, stype: str, ws: int, we: int) -> Optional[Passage]:
        raw = blob[ws:we]
        if len(raw.strip()) < self.min_chars and stype != "abstract":
            return None
        norm = normalize_for_model(raw)
        if not norm:
            return None
        label = (name or "body").strip()
        text = norm if label.lower() in ("abstract", "title", "body") else f"{label}: {norm}"
        return Passage(pid, label, stype, ws, we, raw, text)

    def chunk(self, blob: str, paper_id: str) -> List[Passage]:
        if not blob:
            return []
        masked = _mask_placeholders(blob)          # offset-preserving
        title = blob.split("\n", 1)[0].strip()
        hits = list(_HEADER_RE.finditer(masked))
        passages: List[Passage] = []

        if len(hits) <= 1:
            # Run-on paper (~19%): at most an 'abstract:' header, no internal sections.
            # Emit a real abstract span, then window the remaining body as 'body'
            # (NOT 'abstract', so the section prior + attribution stay honest).
            astart = hits[0].end() if hits else len(title)
            abs_end = min(astart + 1800, len(blob))
            dot = masked.rfind(". ", astart + 400, abs_end)
            if dot != -1:
                abs_end = dot + 1
            p = self._mk(blob, paper_id, "abstract", "abstract", astart, abs_end)
            if p:
                passages.append(p)
            for ws, we in self._windows(masked[abs_end:len(blob)], abs_end)[: self.body_cap]:
                p = self._mk(blob, paper_id, "body", "other", ws, we)
                if p:
                    passages.append(p)
        else:
            for i, m in enumerate(hits):
                name = m.group(1).strip()
                bstart, bend = m.end(), (hits[i + 1].start() if i + 1 < len(hits) else len(blob))
                body_masked = masked[bstart:bend]
                if len(body_masked.strip()) < 40:
                    continue
                stype = classify_section(name)
                for ws, we in self._windows(body_masked, bstart)[: self.max_windows_per_section]:
                    p = self._mk(blob, paper_id, name, stype, ws, we)
                    if p:
                        passages.append(p)

        if not passages:   # degenerate paper: fall back to title only
            passages.append(Passage(paper_id, "title", "abstract", 0, len(title), title))
        return passages[: self.max_per_paper]

    def _windows(self, body_masked: str, base: int) -> List[Tuple[int, int]]:
        """Yield (abs_start, abs_end) windows over a section body, snapping ends to
        sentence boundaries found in the masked text (so we never cut a placeholder)."""
        n = len(body_masked)
        # strip leading whitespace offset
        lead = len(body_masked) - len(body_masked.lstrip())
        i = lead
        out: List[Tuple[int, int]] = []
        if n - lead <= self.window:
            return [(base + lead, base + n)]
        while i < n:
            end = min(i + self.window, n)
            if end < n:
                dot = body_masked.rfind(". ", i + int(self.window * 0.6), end)
                if dot != -1:
                    end = dot + 1
            out.append((base + i, base + end))
            if end >= n:
                break
            i = max(end - self.overlap, i + 1)
        return out


# --------------------------------------------------------------------------- #
# BM25 (zero-dependency first-pass)
# --------------------------------------------------------------------------- #
_STOP = set("a an the of to in on for and or is are be as by with that this we our it its "
            "from at which using use used can may show shows results result method methods "
            "paper approach based given these those such also into than then they their".split())
_WORD_RE = re.compile(r"[a-z0-9]+")


def _tok(s: str) -> List[str]:
    return [w for w in _WORD_RE.findall(s.lower()) if len(w) > 1 and w not in _STOP]


class BM25:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b

    def fit(self, docs_tokens: List[List[str]]):
        self.docs = docs_tokens
        self.N = max(1, len(docs_tokens))
        self.dl = np.array([len(d) for d in docs_tokens], dtype=float)
        self.avgdl = float(self.dl.mean()) if self.N else 1.0
        df: Dict[str, int] = {}
        self.tf: List[Dict[str, int]] = []
        for d in docs_tokens:
            c: Dict[str, int] = {}
            for w in d:
                c[w] = c.get(w, 0) + 1
            self.tf.append(c)
            for w in c:
                df[w] = df.get(w, 0) + 1
        self.idf = {w: math.log(1 + (self.N - f + 0.5) / (f + 0.5)) for w, f in df.items()}
        return self

    def scores(self, query_tokens: List[str]) -> np.ndarray:
        out = np.zeros(self.N, dtype=float)
        q = [w for w in query_tokens if w in self.idf]
        if not q:
            return out
        for i, tf in enumerate(self.tf):
            dl = self.dl[i]
            s = 0.0
            for w in q:
                f = tf.get(w, 0)
                if f:
                    s += self.idf[w] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            out[i] = s
        return out


# --------------------------------------------------------------------------- #
# ScaDS API backends (GPU-side; no local model load)
# --------------------------------------------------------------------------- #
class ScadsBackends:
    """Thin clients for the ScaDS embedding + rerank endpoints (OpenAI-compatible).

    Rerank model:    BAAI/bge-reranker-v2-m3   (cross-encoder, ~0.15 s)
    Embedding model: Qwen/Qwen3-Embedding-4B   (~2560-dim)
    """

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None,
                 rerank_model: str = "BAAI/bge-reranker-v2-m3",
                 embed_model: str = "Qwen/Qwen3-Embedding-4B", timeout: int = 30):
        # Resolve the key from the SAME sources as ScadsAgent (env, then key files), so
        # the reranker/embedder works wherever the LLM already has a key — no new config.
        self.api_key = (
            api_key
            or os.environ.get("SCADS_API_KEY")
            or os.environ.get("SCADSAI_API_KEY")
            or _read_key_file("/etc/scads_api_key")
            or _read_key_file(os.path.expanduser("~/.scads_api_key"))
        )
        self.base_url = (base_url or os.environ.get("SCADS_API_BASE") or "https://llm.scads.ai/v1").rstrip("/")
        self.rerank_model = rerank_model
        self.embed_model = embed_model
        self.timeout = timeout

    # -- rerank ------------------------------------------------------------- #
    def rerank(self, query: str, documents: List[str], top_n: int) -> Optional[List[Tuple[int, float]]]:
        """Return [(orig_index, score), ...] best-first, or None on failure."""
        payload = {"model": self.rerank_model, "query": query,
                   "documents": documents, "top_n": min(top_n, len(documents))}
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        for path in ("/rerank", "/v2/rerank"):
            url = self.base_url + path
            try:
                req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                             headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    data = json.loads(r.read().decode())
                results = data.get("results") or data.get("data") or []
                out = []
                for item in results:
                    idx = item.get("index", item.get("document", {}).get("index"))
                    score = item.get("relevance_score", item.get("score", 0.0))
                    if idx is not None:
                        out.append((int(idx), float(score)))
                if out:
                    out.sort(key=lambda x: x[1], reverse=True)
                    return out
            except urllib.error.HTTPError as e:
                logger.debug(f"rerank {url} HTTP {e.code}")
            except Exception as e:
                logger.debug(f"rerank {url} failed: {e}")
        logger.warning("ScaDS rerank unavailable; caller should fall back to BM25 order")
        return None

    # -- embeddings --------------------------------------------------------- #
    def embed(self, texts: List[str]) -> Optional[np.ndarray]:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            resp = client.embeddings.create(model=self.embed_model, input=texts)
            vecs = np.array([d.embedding for d in resp.data], dtype="float32")
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            return vecs / np.clip(norms, 1e-9, None)
        except Exception as e:
            logger.warning(f"ScaDS embed unavailable ({e}); caller should fall back to BM25")
            return None


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
class PassageRanker:
    def __init__(self, backend: str = "bm25", top_k: int = 12, candidates: int = 24,
                 chunker: Optional[PaperChunker] = None, scads: Optional[ScadsBackends] = None):
        assert backend in ("bm25", "scads_rerank", "scads_embed")
        self.backend = backend
        self.top_k = top_k
        self.candidates = max(candidates, top_k)
        self.chunker = chunker or PaperChunker()
        self.scads = scads or ScadsBackends()

    def rank(self, query: str, full_texts: List[Tuple[str, str]],
             expansion_text: str = "", intent: Optional[str] = None) -> List[Passage]:
        """full_texts: [(blob, paper_id), ...] (the filtered papers). Returns top_k Passages."""
        passages: List[Passage] = []
        for blob, pid in full_texts:
            passages.extend(self.chunker.chunk(blob, pid))
        if not passages:
            return []

        # ---- Stage 1: BM25 first-pass over an expanded query (free, CPU) ----
        bm = BM25().fit([_tok(p.text) for p in passages])
        q_expanded = " ".join(filter(None, [query, expansion_text]))
        base = bm.scores(_tok(q_expanded))
        prior = np.array([self._section_prior(p, intent) for p in passages])
        first = base * prior
        order = np.argsort(first)[::-1]
        cand_idx = list(order[: self.candidates])
        for j in cand_idx:
            passages[j].score = float(first[j])
        candidates = [passages[j] for j in cand_idx]

        # ---- Stage 2: optional semantic re-ranking on the candidates (GPU API) ----
        if self.backend == "scads_rerank":
            ranked = self.scads.rerank(query, [c.text for c in candidates], self.top_k)
            if ranked is not None:
                picked = []
                for idx, score in ranked[: self.top_k]:
                    candidates[idx].score = score
                    picked.append(candidates[idx])
                return picked
            # else fall through to BM25 order

        elif self.backend == "scads_embed":
            qv = self.scads.embed([f"query: {query}"])
            pv = self.scads.embed([f"passage: {c.text}" for c in candidates])
            if qv is not None and pv is not None:
                sims = (pv @ qv[0])
                o = np.argsort(sims)[::-1]
                picked = []
                for k in o[: self.top_k]:
                    candidates[k].score = float(sims[k])
                    picked.append(candidates[k])
                return picked
            # else fall through to BM25 order

        return candidates[: self.top_k]

    @staticmethod
    def _section_prior(p: Passage, intent: Optional[str]) -> float:
        if _is_low_value(p.section):
            return 0.35
        w = 1.0
        if p.section_type in ("abstract", "intro"):
            w *= 1.15
        if intent and p.section_type in _INTENT_BOOST.get(intent, set()):
            w *= 1.25
        return w

    # ---- evidence formatting + attribution table ----
    @staticmethod
    def build_evidence(passages: List[Passage]) -> Tuple[str, Dict[str, dict]]:
        """Assign E# ids, render the numbered evidence block, and return a passage_table
        keyed by E# with exact source spans for attribution."""
        lines, table = [], {}
        for i, p in enumerate(passages, 1):
            p.eid = f"E{i}"
            lines.append(f"[{p.eid}] (paper {p.paper_id} — {p.section}): {p.display()}")
            table[p.eid] = {
                "paper_id": p.paper_id, "section": p.section, "section_type": p.section_type,
                "char_start": p.char_start, "char_end": p.char_end,
                "text": p.text, "score": round(p.score, 4),
            }
        return "\n\n".join(lines), table

    @staticmethod
    def verify_sentence(sentence: str, passage_text: str) -> bool:
        """Cheap lexical grounding check (word overlap), mirroring the existing
        EnhancedCitationHandler heuristic. Replace with the ScaDS reranker/NLI for a
        stronger gate if desired."""
        a = set(_tok(sentence))
        b = set(_tok(passage_text))
        if not a:
            return False
        inter = len(a & b)
        return inter > 4 or inter / max(len(a), 1) > 0.25


# --------------------------------------------------------------------------- #
# Sentence-level attribution (post-hoc; does NOT alter the answer text)
# --------------------------------------------------------------------------- #
# Split on sentence enders followed by whitespace + a capital / opening bracket /
# citation marker. Avoid splitting after a lone capital-letter initial ("J. Smith").
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[])")


def split_sentences(text: str) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    out = []
    for part in _SENT_SPLIT.split(text):
        part = part.strip()
        if part:
            out.append(part)
    return out


def attribute_sentences(answer_text: str, passage_table: Dict[str, dict],
                        verify: str = "lexical", reranker: "Optional[ScadsBackends]" = None,
                        min_ratio: float = 0.18, min_words: int = 4) -> List[dict]:
    """Map each answer sentence to the source passage span that best supports it.

    Purely additive: reads the [E#] passage_table built during context preparation and
    the [n] document citations already in the answer; it does NOT modify the answer or
    the references. Returns one record per sentence with the exact (paper_id, section,
    char_start, char_end) of its supporting passage and a `verified` flag.

    Candidate passages for a sentence are restricted to the documents it cites with
    [n]; if it cites none, all fed passages are considered.
    """
    if not passage_table:
        return []
    by_cite: Dict[int, list] = {}
    for eid, row in passage_table.items():
        by_cite.setdefault(row.get("citation_num"), []).append((eid, row))
    all_rows = list(passage_table.items())

    results = []
    for si, sent in enumerate(split_sentences(answer_text)):
        # Handles single [n] and multi-citations like [1, 2, 3].
        cites = sorted({int(n) for grp in re.findall(r"\[([\d,\s]+)\]", sent)
                        for n in re.findall(r"\d+", grp)})
        clean = re.sub(r"\[\d+\]", "", sent).strip()
        if len(clean) < 8:
            continue
        candidates = []
        for c in cites:
            candidates.extend(by_cite.get(c, []))
        if not candidates:
            candidates = all_rows
        if not candidates:
            continue

        best_eid = best_row = None
        best_score = -1.0
        verified = False

        if verify == "scads_rerank" and reranker is not None:
            ranked = reranker.rerank(clean, [r["text"] for _, r in candidates], top_n=1)
            if ranked:
                idx, score = ranked[0]
                best_eid, best_row = candidates[idx]
                best_score = round(float(score), 4)
                verified = score >= 0.05

        if best_row is None:                      # lexical (default or rerank fallback)
            sw = set(_tok(clean))
            for eid, row in candidates:
                inter = len(sw & set(_tok(row["text"])))
                ratio = inter / max(len(sw), 1)
                score = ratio + 0.01 * inter
                if score > best_score:
                    best_score, best_eid, best_row = score, eid, row
            if best_row is not None:
                inter = len(sw & set(_tok(best_row["text"])))
                ratio = inter / max(len(sw), 1)
                best_score = round(ratio, 3)
                verified = ratio >= min_ratio or inter >= min_words

        rec = {"sentence_index": si, "sentence": sent, "cited_docs": cites,
               "verified": bool(verified)}
        if best_row is not None:
            rec.update({
                "eid": best_eid, "citation_num": best_row.get("citation_num"),
                "paper_id": best_row.get("paper_id"), "section": best_row.get("section"),
                "char_start": best_row.get("char_start"), "char_end": best_row.get("char_end"),
                "score": best_score,
                "evidence_preview": best_row["text"][:240] + ("…" if len(best_row["text"]) > 240 else ""),
            })
        results.append(rec)
    return results


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import time, sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    import plyvel
    DB = sys.argv[1] if len(sys.argv) > 1 else \
        "/data/horse/ws/jihe529c-test-QA/jihe529c-hora-1778972422/full_text_db"

    db = plyvel.DB(DB, create_if_missing=False)
    sample = []
    for i, (k, v) in enumerate(db.iterator()):
        sample.append((v.decode("utf-8", "replace"), k.decode()))
        if i >= 4:
            break
    db.close()

    query = "How does the method handle rare or out-of-vocabulary tokens?"
    ck = PaperChunker()
    allp = []
    for blob, pid in sample:
        ps = ck.chunk(blob, pid)
        allp.extend(ps)
        secs = [p.section for p in ps][:6]
        print(f"\n{pid}: {len(blob):>7,} chars -> {len(ps):>3} passages | sections: {secs}")

    print(f"\nTOTAL passages from {len(sample)} papers: {len(allp)}")

    pr = PassageRanker(backend="bm25", top_k=8, candidates=16)
    t0 = time.time()
    top = pr.rank(query, sample, expansion_text="rare words subword tokenization vocabulary")
    dt = (time.time() - t0) * 1000
    print(f"\nBM25 rank ({len(allp)} passages) -> top {len(top)} in {dt:.1f} ms")
    ev, table = pr.build_evidence(top)
    for p in top:
        print(f"  {p.eid} [{p.score:6.2f}] {p.paper_id} §{p.section[:30]:30s} | {p.text[:90]!r}")
    print("\n--- sample evidence block (first 600 chars) ---")
    print(ev[:600])

    # live API smoke test only if a key is present
    if os.environ.get("SCADS_API_KEY") or os.environ.get("SCADSAI_API_KEY"):
        for be in ("scads_rerank", "scads_embed"):
            t0 = time.time()
            r = PassageRanker(backend=be, top_k=8, candidates=16).rank(query, sample)
            print(f"\n[{be}] top {len(r)} in {(time.time()-t0)*1000:.0f} ms; "
                  f"top1={r[0].paper_id}/{r[0].section[:25]!r} score={r[0].score:.3f}" if r else f"[{be}] no result")
    else:
        print("\n(no SCADS_API_KEY set — skipped live rerank/embed smoke test)")
