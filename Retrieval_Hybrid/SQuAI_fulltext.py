#!/usr/bin/env python3
"""4-Agent RAG System with Question Splitting and Parallel Processing"""
import plyvel
import argparse
import json
import time
import datetime
import os
from tqdm import tqdm
import logging
import numpy as np
import random
import string
import re
from typing import List, Tuple, Dict, Any, Optional
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import E5_INDEX_DIR, BM25_INDEX_DIR, DB_PATH


def get_unique_log_filename():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    random_str = "".join(random.choices(string.ascii_lowercase + string.digits, k=6))
    return f"logs/4agent_rag_{timestamp}_{random_str}.log"


os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(get_unique_log_filename()), logging.StreamHandler()],
)
logger = logging.getLogger("4Agent_RAG")


class QuestionSplitter:
    """Agent 1: Intelligent Question Splitting Agent"""

    def __init__(self, agent_model):
        self.agent = agent_model

    def _create_splitting_prompt(self, query: str) -> str:
        """Create prompt for question splitting analysis"""
        return f"""You are an intelligent question analyzer. Your task is to determine if a query contains multiple distinct sub-questions that would benefit from separate retrieval and research.

SPLITTING CRITERIA:
- Split if query contains multiple distinct topics connected by "and", "also", "what about"
- Split if query asks for comparisons between different concepts
- Split if query has multiple question words (what, how, why, when, where)
- DO NOT split simple clarifications or related aspects of the same topic

Examples:
Query: "What is quantum computing and how is it used in cryptography?"
Split: YES
Sub-questions: ["What is quantum computing?", "How is quantum computing used in cryptography?"]

Query: "What are the applications and limitations of machine learning?"
Split: NO (same topic, different aspects)
Sub-questions: []

Now analyze this query:
Query: "{query}"

Respond with ONLY this format:
Split: YES/NO
Sub-questions: [list of questions] (empty list if Split: NO)"""

    def analyze_and_split(self, query: str) -> Tuple[bool, List[str]]:
        """Analyze query and split into sub-questions if beneficial"""
        if not self._quick_split_check(query):
            return False, []

        prompt = self._create_splitting_prompt(query)
        response = self.agent.generate(prompt)

        return self._parse_splitting_response(response, query)

    def _quick_split_check(self, query: str) -> bool:
        """Fast heuristic check to avoid LLM calls for obvious cases"""
        query_lower = query.lower()

        if len(query.split()) < 6:
            return False

        split_indicators = [
            " and what ",
            " and how ",
            " and why ",
            " and when ",
            " and where ",
            "what about",
            "also what",
            "also how",
            "also why",
            "? and ",
            "? what",
            "? how",
            "? why",
            "? when",
            "? where",
        ]

        for indicator in split_indicators:
            if indicator in query_lower:
                return True

        question_words = ["what", "how", "why", "when", "where", "which", "who"]
        question_count = sum(1 for word in question_words if word in query_lower)

        return question_count >= 2

    def _parse_splitting_response(
        self, response: str, original_query: str
    ) -> Tuple[bool, List[str]]:
        """Parse the LLM response for splitting decision"""
        try:
            lines = response.strip().split("\n")
            should_split = False
            sub_questions = []

            for line in lines:
                line = line.strip()
                if line.startswith("Split:"):
                    should_split = "YES" in line.upper()
                elif line.startswith("Sub-questions:"):
                    list_part = line.split(":", 1)[1].strip()
                    if list_part and list_part != "[]":
                        try:
                            if list_part.startswith("[") and list_part.endswith("]"):
                                sub_questions = json.loads(list_part)
                            else:
                                sub_questions = [
                                    q.strip().strip('"').strip("'")
                                    for q in list_part.split(",")
                                ]
                        except:
                            sub_questions = []

            if should_split and sub_questions:
                valid_questions = []
                for q in sub_questions:
                    q = q.strip()
                    if len(q) > 10 and q.endswith("?"):
                        valid_questions.append(q)

                if len(valid_questions) < 2:
                    return False, []

                return True, valid_questions

            return False, []

        except Exception as e:
            logger.warning(f"Error parsing splitting response: {e}")
            return False, []


class PaperTitleExtractor:
    """Utility class for extracting paper titles from document text"""

    @staticmethod
    def extract_title_from_text(doc_text: str, doc_id: str) -> str:
        """Extract paper title from document text using multiple patterns"""
        try:
            leveldb_pattern = r"Content for [^:]*:\s*\n([^\n]+)"
            match = re.search(leveldb_pattern, doc_text)
            if match:
                title_candidate = match.group(1).strip()
                if (
                    len(title_candidate) > 10
                    and len(title_candidate) < 300
                    and not title_candidate.lower().startswith(
                        ("abstract:", "introduction:", "the abstract", "in this", "we ")
                    )
                ):
                    return title_candidate

            lines = doc_text.split("\n")
            for i, line in enumerate(lines[:5]):
                line = line.strip()

                if not line or line.lower().startswith(
                    ("content for", "time taken", "opening")
                ):
                    continue

                if (
                    len(line) > 10
                    and len(line) < 300
                    and not line.lower().startswith(
                        (
                            "abstract:",
                            "introduction:",
                            "the abstract",
                            "in this",
                            "we ",
                            "this paper",
                            "{",
                        )
                    )
                    and not re.match(r"^\d+", line)
                    and not line.endswith(":")
                    and line.count(" ") >= 2
                ):
                    return line

            for line in lines[:5]:
                line = line.strip()
                if len(line) > 15 and len(line) < 200:
                    return line[:150] + "..." if len(line) > 150 else line

            return f"Document {doc_id}"

        except Exception as e:
            return f"Document {doc_id}"

    @staticmethod
    def format_title_for_log(title: str, max_length: int = 80) -> str:
        """Format title for logging with length limit"""
        if len(title) <= max_length:
            return title
        return title[: max_length - 3] + "..."


class MultiPaperCitationHandler:
    """Citation handler with multi-paper extraction"""

    def __init__(self, index_dir: str = "test_index", agent_model=None):
        self.doc_to_citation = {}
        self.citation_to_doc = {}
        self.next_citation_num = 1
        self.index_dir = Path(index_dir)
        self.agent_model = agent_model

        self.arxiv_papers = self._load_arxiv_papers()

    def _load_arxiv_papers(self):
        """Load arXiv papers for metadata extraction"""
        papers = {}
        try:
            jsonl_files = list(self.index_dir.glob("*.jsonl"))

            for jsonl_file in jsonl_files:
                with open(jsonl_file, "r") as f:
                    for line in f:
                        try:
                            paper = json.loads(line.strip())
                            paper_id = paper.get("paper_id", "")

                            metadata = paper.get("metadata", {})
                            title = metadata.get("title", "Unknown Title")
                            authors = metadata.get("authors", "Unknown")

                            year = "Unknown"
                            versions = paper.get("versions", [])
                            if versions:
                                created = versions[0].get("created", "")
                                year_match = re.search(r"(\d{4})", created)
                                if year_match:
                                    year = year_match.group(1)

                            if "authors_parsed" in paper:
                                authors_list = paper["authors_parsed"]
                                if authors_list and len(authors_list) > 0:
                                    first_author = authors_list[0]
                                    if len(first_author) >= 2:
                                        formatted_author = (
                                            f"{first_author[0]}, {first_author[1][0]}."
                                            if first_author[1]
                                            else first_author[0]
                                        )
                                        if len(authors_list) > 1:
                                            authors = f"{formatted_author} et al."
                                        else:
                                            authors = formatted_author

                            papers[paper_id] = {
                                "title": title,
                                "authors": authors,
                                "year": year,
                                "paper_id": paper_id,
                                "abstract": paper.get("abstract", {}).get("text", ""),
                            }
                        except:
                            continue

            return papers
        except Exception as e:
            return {}

    def _extract_paper_info(
        self, doc_text: str, doc_id: str, metadata: Dict = None
    ) -> Dict:
        """Extract paper metadata"""
        paper_info = {
            "title": "Unknown Title",
            "authors": "Unknown",
            "venue": "arXiv",
            "year": "Unknown",
            "paper_id": doc_id,
        }

        try:
            paper_info["title"] = PaperTitleExtractor.extract_title_from_text(
                doc_text, doc_id
            )

            if "{" in doc_text and '"metadata"' in doc_text:
                try:
                    json_match = re.search(r'\{.*?"metadata".*?\}', doc_text, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(0)
                        paper_data = json.loads(json_str)

                        if "metadata" in paper_data:
                            meta = paper_data["metadata"]
                            if "authors" in meta:
                                paper_info["authors"] = meta["authors"]

                        if "paper_id" in paper_data:
                            paper_info["paper_id"] = paper_data["paper_id"]

                        if "versions" in paper_data and paper_data["versions"]:
                            created = paper_data["versions"][0].get("created", "")
                            year_match = re.search(r"(\d{4})", created)
                            if year_match:
                                paper_info["year"] = year_match.group(1)

                except Exception as e:
                    pass

            if doc_id in self.arxiv_papers:
                arxiv_data = self.arxiv_papers[doc_id]
                if (
                    paper_info["title"] == "Unknown Title"
                    or paper_info["title"] == f"Document {doc_id}"
                ):
                    paper_info["title"] = arxiv_data["title"]
                if paper_info["authors"] == "Unknown":
                    paper_info["authors"] = arxiv_data["authors"]
                if paper_info["year"] == "Unknown":
                    paper_info["year"] = arxiv_data["year"]

            if len(paper_info["title"]) > 150:
                paper_info["title"] = paper_info["title"][:150] + "..."

            if not paper_info["paper_id"]:
                paper_info["paper_id"] = doc_id

        except Exception as e:
            pass

        return paper_info

    def _basic_text_cleaning(self, text: str) -> str:
        """Basic text cleaning for better model processing"""
        text = re.sub(r"'section':\s*'[^']*',\s*'text':\s*'", "", text)
        text = re.sub(r"^\s*\{.*?'text':\s*'", "", text)
        text = re.sub(r"\{[^}]*\}", "", text)

        text = re.sub(r"\{\{[^}]+\}\}", "[REF]", text)
        text = re.sub(r"\$[^$]+\$", "[MATH]", text)
        text = re.sub(r"\\[a-zA-Z]+\{[^}]*\}", "[LATEX]", text)

        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"\n\s*\n", "\n\n", text)

        lines = text.split("\n")
        cleaned_lines = []
        for line in lines:
            if len(line) > 1000 and line.count(" ") < 10:
                continue
            cleaned_lines.append(line)

        result = "\n".join(cleaned_lines).strip()
        return result

    def add_document(self, doc_text: str, doc_id: str, metadata: Dict = None) -> int:
        """Add a document and return its citation number"""
        if doc_id not in self.doc_to_citation:
            citation_num = self.next_citation_num
            self.doc_to_citation[doc_id] = citation_num

            paper_info = self._extract_paper_info(doc_text, doc_id, metadata)

            self.citation_to_doc[citation_num] = {
                "doc_id": doc_id,
                "paper_info": paper_info,
                "text": doc_text,
            }

            self.next_citation_num += 1
            return citation_num
        else:
            return self.doc_to_citation[doc_id]

    def get_citation_map(self) -> Dict[str, int]:
        """Get mapping from doc_id to citation number"""
        return self.doc_to_citation.copy()

    def _find_all_used_citations(self, answer_text: str) -> List[int]:
        """Find all citation numbers used in the answer"""
        citation_matches = re.findall(r"\[(\d+)\]", answer_text)
        used_citations = sorted(set(int(num) for num in citation_matches))
        return used_citations

    def _extract_claims_for_citation(
        self, answer_text: str, citation_num: int
    ) -> List[str]:
        """Extract specific claims attributed to a particular citation"""
        sentences = re.split(r"[.!?]+", answer_text)

        claims = []
        citation_pattern = f"\\[{citation_num}\\]"

        for sentence in sentences:
            if re.search(citation_pattern, sentence):
                clean_sentence = re.sub(r"\[\d+\]", "", sentence).strip()
                if len(clean_sentence) > 10:
                    claims.append(clean_sentence)

        return claims

    def _create_comprehensive_extraction_prompt(
        self, answer_text: str, used_citations: List[int]
    ) -> str:
        """Create a comprehensive prompt that analyzes ALL citations at once"""
        documents_section = "RESEARCH PAPERS:\n"
        documents_section += "=" * 80 + "\n\n"

        for citation_num in used_citations:
            if citation_num in self.citation_to_doc:
                doc_info = self.citation_to_doc[citation_num]
                paper_info = doc_info["paper_info"]

                clean_content = self._basic_text_cleaning(doc_info["text"])
                if len(clean_content) > 8000:
                    clean_content = clean_content[:8000] + "... [content truncated]"

                documents_section += (
                    f"PAPER [{citation_num}]: \"{paper_info['title']}\"\n"
                )
                documents_section += f"Authors: {paper_info['authors']}\n"
                documents_section += f"Year: {paper_info['year']}\n"
                documents_section += f"Content:\n{clean_content}\n\n"
                documents_section += "-" * 50 + "\n\n"

        claims_section = "CLAIMS TO ANALYZE:\n"
        claims_section += "=" * 40 + "\n"

        for citation_num in used_citations:
            claims = self._extract_claims_for_citation(answer_text, citation_num)
            if claims:
                claims_section += (
                    f"\nCitation [{citation_num}] supports these claims:\n"
                )
                for i, claim in enumerate(claims, 1):
                    claims_section += f'  {i}. "{claim}"\n'

        prompt = f"""You are an expert citation analyzer. Your task is to identify the EXACT passages from research papers that support specific claims in an academic answer.

{documents_section}

{claims_section}

COMPLETE ANSWER WITH CITATIONS:
{answer_text}

TASK: For each citation [1], [2], [3], etc., identify the specific passages from the corresponding paper that directly support the claims attributed to that citation.

RESPONSE FORMAT (follow this structure exactly):

Citation [1]:
Passage: "[exact text from Paper [1] that supports the claims for citation 1]"

Citation [2]: 
Passage: "[exact text from Paper [2] that supports the claims for citation 2]"

Citation [3]:
Passage: "[exact text from Paper [3] that supports the claims for citation 3]"

[Continue for all citations found...]

Begin analysis now:"""

        return prompt

    def _parse_comprehensive_response(
        self, model_response: str, used_citations: List[int]
    ) -> Dict[int, str]:
        """Parse the model's comprehensive response into individual citation passages"""
        citation_passages = {}

        try:
            for citation_num in used_citations:
                primary_pattern = (
                    rf'Citation \[{citation_num}\]:\s*\n?\s*Passage:\s*"([^"]+)"'
                )
                match = re.search(
                    primary_pattern, model_response, re.DOTALL | re.IGNORECASE
                )

                if match:
                    passage = match.group(1).strip()
                    passage = re.sub(r"\s+", " ", passage)
                    passage = re.sub(r"\n+", " ", passage)

                    if passage and not passage.endswith((".", "!", "?")):
                        passage += "."

                    citation_passages[citation_num] = passage

                else:
                    alt_patterns = [
                        rf'\[{citation_num}\][:\s]*Passage[:\s]*"([^"]+)"',
                        rf'Citation {citation_num}[:\s]*"([^"]+)"',
                        rf'Paper \[{citation_num}\][:\s]*"([^"]+)"',
                        rf'\[{citation_num}\][:\s]*"([^"]+)"',
                    ]

                    found = False
                    for alt_pattern in alt_patterns:
                        alt_match = re.search(
                            alt_pattern, model_response, re.DOTALL | re.IGNORECASE
                        )
                        if alt_match:
                            passage = alt_match.group(1).strip()
                            passage = re.sub(r"\s+", " ", passage)

                            if passage and not passage.endswith((".", "!", "?")):
                                passage += "."

                            citation_passages[citation_num] = passage
                            found = True
                            break

                    if not found:
                        citation_passages[citation_num] = self._get_fallback_passage(
                            citation_num
                        )

            return citation_passages

        except Exception as e:
            logger.error(f"Error parsing comprehensive response: {e}")
            return self._fallback_extraction_all(used_citations)

    def _get_fallback_passage(self, citation_num: int) -> str:
        """Fallback extraction for a specific citation when model extraction fails"""
        if citation_num in self.citation_to_doc:
            doc_text = self.citation_to_doc[citation_num]["text"]
            clean_text = self._basic_text_cleaning(doc_text)
            sentences = re.split(r"[.!?]+", clean_text)
            sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

            fallback_sentences = sentences[:3] if len(sentences) >= 3 else sentences
            fallback = ". ".join(fallback_sentences)
            if fallback and not fallback.endswith((".", "!", "?")):
                fallback += "."

            return fallback

        return f"Content from citation [{citation_num}] (fallback extraction)."

    def _fallback_extraction_all(self, used_citations: List[int]) -> Dict[int, str]:
        """Fallback when comprehensive extraction completely fails"""
        fallback_results = {}
        for citation_num in used_citations:
            fallback_results[citation_num] = self._get_fallback_passage(citation_num)
        return fallback_results

    def extract_all_citations_comprehensive(self, answer_text: str) -> Dict[int, str]:
        """Analyze ALL citations in the answer simultaneously in a single model call"""
        try:
            if not self.agent_model or not self.citation_to_doc:
                return {}

            used_citations = self._find_all_used_citations(answer_text)
            if not used_citations:
                return {}

            extraction_prompt = self._create_comprehensive_extraction_prompt(
                answer_text, used_citations
            )
            model_response = self.agent_model.generate(extraction_prompt)
            citation_passages = self._parse_comprehensive_response(
                model_response, used_citations
            )

            return citation_passages

        except Exception as e:
            logger.error(f"Error in comprehensive citation extraction: {e}")
            return self._fallback_extraction_all(used_citations)

    def format_references_comprehensive(self, answer_text: str = None) -> str:
        """Format references using comprehensive multi-paper extraction"""
        if not self.citation_to_doc:
            return ""

        if not answer_text:
            return self._format_references_simple()

        all_passages = self.extract_all_citations_comprehensive(answer_text)

        if not all_passages:
            return self._format_references_simple()

        references = "\n\n## References\n\n"

        for citation_num in sorted(all_passages.keys()):
            if citation_num not in self.citation_to_doc:
                continue

            doc_info = self.citation_to_doc[citation_num]
            paper_info = doc_info["paper_info"]

            ref_line = f"[{citation_num}] "

            if paper_info["authors"] != "Unknown":
                ref_line += f"{paper_info['authors']}. "

            title = paper_info["title"].replace('"', "'")
            ref_line += f'"{title}." '

            if paper_info.get("paper_id") and paper_info["paper_id"] != "Unknown":
                if str(paper_info["paper_id"]).startswith("arXiv:"):
                    ref_line += f"{paper_info['paper_id']}"
                else:
                    ref_line += f"arXiv:{paper_info['paper_id']}"
            else:
                ref_line += f"{paper_info['venue']}"

            if paper_info["year"] != "Unknown":
                ref_line += f" ({paper_info['year']})"

            passage = all_passages[citation_num]
            ref_line += f'\n    Passage: "{passage}"'

            references += ref_line + "\n\n"

        return references

    def _format_references_simple(self) -> str:
        """Simple fallback reference formatting"""
        references = "\n\n## References\n\n"

        for citation_num, doc_info in self.citation_to_doc.items():
            paper_info = doc_info["paper_info"]

            ref_line = f"[{citation_num}] "

            if paper_info["authors"] != "Unknown":
                ref_line += f"{paper_info['authors']}. "

            title = paper_info["title"].replace('"', "'")
            ref_line += f'"{title}." '

            if paper_info.get("paper_id") and paper_info["paper_id"] != "Unknown":
                if str(paper_info["paper_id"]).startswith("arXiv:"):
                    ref_line += f"{paper_info['paper_id']}"
                else:
                    ref_line += f"arXiv:{paper_info['paper_id']}"

            if paper_info["year"] != "Unknown":
                ref_line += f" ({paper_info['year']})"

            fallback_passage = self._get_fallback_passage(citation_num)
            ref_line += f'\n    Passage: "{fallback_passage}"'

            references += ref_line + "\n\n"

        return references


class FourAgentRAG:
    """4-Agent RAG System with Full-Text Context Strategy"""

    def __init__(
        self,
        retriever,
        agent_model=None,
        n=0.0,
        falcon_api_key=None,
        index_dir="test_index",
        max_workers=4,
        max_context_chars=100000,
    ):
        """Initialize with 4-agent architecture and shared model loading"""
        self.retriever = retriever
        self.n = n
        self.index_dir = index_dir
        self.max_workers = max_workers
        self.max_context_chars = max_context_chars

        # Initialize agents with shared model loading
        if isinstance(agent_model, str):
            if "falcon" in agent_model.lower() and falcon_api_key:
                from api_agent import FalconAgent

                self.agent1 = FalconAgent(falcon_api_key)
                self.agent2 = FalconAgent(falcon_api_key)
                self.agent3 = FalconAgent(falcon_api_key)
                self.agent4 = FalconAgent(falcon_api_key)
            else:
                from local_agent import LLMAgent

                shared_agent = LLMAgent(agent_model)
                self.agent1 = shared_agent
                self.agent2 = shared_agent
                self.agent3 = shared_agent
                self.agent4 = shared_agent
        else:
            self.agent1 = agent_model
            self.agent2 = agent_model
            self.agent3 = agent_model
            self.agent4 = agent_model

        self.question_splitter = QuestionSplitter(self.agent1)
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

        # Pre-warming
        try:
            dummy_abstracts = self.retriever.retrieve_abstracts("test", top_k=1)
            if hasattr(self.agent1, "generate"):
                self.agent1.generate("test")
        except Exception as e:
            pass

    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation: ~4 chars per token"""
        return len(text) // 4

    def _create_agent2_prompt(self, query, document):
        """Agent-2 prompt: Answer generation from abstracts"""
        return f"""You are an accurate and reliable AI assistant that can answer questions with the help of external documents. You should only provide the correct answer without repeating the question and instruction.

Document: {document}

Question: {query}

Answer:"""

    def _create_agent3_prompt(self, query, document, answer):
        """Agent-3 prompt: Document evaluation"""
        return f"""You are a noisy document evaluator that can judge if the external document is noisy for the query with unrelated or misleading information. Given a retrieved Document, a Question, and an Answer generated by an LLM (LLM Answer), you should judge whether both the following two conditions are reached: (1) the Document provides specific information for answering the Question; (2) the LLM Answer directly answers the question based on the retrieved Document. Please note that external documents may contain noisy or factually incorrect information. If the information in the document does not contain the answer, you should point it out with evidence. You should answer with "Yes" or "No" with evidence of your judgment, where "No" means one of the conditions (1) and (2) are unreached and indicates it is a noisy document.

Document: {document}

Question: {query}

LLM Answer: {answer}

Is this document relevant and supportive for answering the question?"""

    def _clean_full_text(self, doc_text: str, doc_id: str) -> str:
        """Clean full text for better context utilization"""
        try:
            clean_text = re.sub(r"Content for [^:]*:\s*\n", "", doc_text)

            if clean_text.startswith("{") and '"paper_id"' in clean_text[:500]:
                json_end = clean_text.find("\n\n")
                if json_end > 0:
                    clean_text = clean_text[json_end:].strip()

            clean_text = clean_text.strip()
            return clean_text

        except Exception as e:
            return doc_text

    def _smart_truncate_document(self, text: str, max_chars: int, doc_id: str) -> str:
        """Smart document truncation that tries to preserve structure"""
        if len(text) <= max_chars:
            return text

        truncation_notice = f"\n\n... [Document {doc_id} truncated to fit group limit]"
        available_chars = max_chars - len(truncation_notice)

        if available_chars <= 0:
            return truncation_notice

        paragraphs = text.split("\n\n")
        truncated = ""

        for paragraph in paragraphs:
            if len(truncated) + len(paragraph) + 2 <= available_chars:
                if truncated:
                    truncated += "\n\n"
                truncated += paragraph
            else:
                remaining_space = available_chars - len(truncated)
                if remaining_space > 100:
                    if truncated:
                        truncated += "\n\n"
                        remaining_space -= 2

                    sentences = paragraph.split(". ")
                    for sentence in sentences:
                        if len(truncated) + len(sentence) + 2 <= available_chars:
                            truncated += sentence + ". "
                        else:
                            break
                break

        if not truncated.strip():
            truncated = text[:available_chars]

        return truncated + truncation_notice

    def _prepare_documents_for_agent4_fulltext(
        self,
        full_texts: List[Tuple[str, str]],
        citation_handler,
        was_split: bool = False,
        sub_question_groups: List[List[str]] = None,
    ) -> List[str]:
        """Prepare documents for Agent 4 - handles any number of sub-questions"""
        docs_with_citations = []

        if was_split and sub_question_groups and len(sub_question_groups) >= 2:
            num_groups = len(sub_question_groups)
            context_limit_per_group = self.max_context_chars // num_groups

            total_context_used = 0

            for group_idx, doc_ids_group in enumerate(sub_question_groups):
                group_docs = [
                    (text, doc_id)
                    for text, doc_id in full_texts
                    if doc_id in doc_ids_group
                ]

                group_context_used = 0

                for doc_idx, (doc_text, doc_id) in enumerate(group_docs):
                    clean_text = self._clean_full_text(doc_text, doc_id)
                    estimated_doc_size = len(clean_text) + 300
                    remaining_group_space = context_limit_per_group - group_context_used

                    if estimated_doc_size > remaining_group_space:
                        if remaining_group_space > 500:
                            max_chars_for_content = remaining_group_space - 300
                            truncated_text = self._smart_truncate_document(
                                clean_text, max_chars_for_content, doc_id
                            )
                            estimated_doc_size = len(truncated_text) + 300
                            clean_text = truncated_text
                        else:
                            break

                    citation_num = citation_handler.add_document(clean_text, doc_id)
                    paper_info = citation_handler.citation_to_doc[citation_num][
                        "paper_info"
                    ]
                    doc_title = (
                        paper_info["title"][:80] + "..."
                        if len(paper_info["title"]) > 80
                        else paper_info["title"]
                    )

                    formatted_doc = f'Document [{citation_num}] (Group {group_idx + 1}/{num_groups}) - "{doc_title}":\n{clean_text}'
                    docs_with_citations.append(formatted_doc)

                    group_context_used += estimated_doc_size
                    total_context_used += estimated_doc_size

                    if group_context_used >= context_limit_per_group:
                        break

            total_context = total_context_used
            total_docs = len(docs_with_citations)

        else:
            total_context_used = 0
            documents_used = 0

            for doc_idx, (doc_text, doc_id) in enumerate(full_texts):
                clean_text = self._clean_full_text(doc_text, doc_id)
                estimated_doc_size = len(clean_text) + 300
                remaining_space = self.max_context_chars - total_context_used

                if estimated_doc_size > remaining_space:
                    if remaining_space > 500:
                        max_chars_for_content = remaining_space - 300
                        truncated_text = self._smart_truncate_document(
                            clean_text, max_chars_for_content, doc_id
                        )
                        estimated_doc_size = len(truncated_text) + 300
                        clean_text = truncated_text
                    else:
                        if documents_used == 0:
                            pass
                        else:
                            break

                citation_num = citation_handler.add_document(clean_text, doc_id)
                paper_info = citation_handler.citation_to_doc[citation_num][
                    "paper_info"
                ]
                doc_title = (
                    paper_info["title"][:80] + "..."
                    if len(paper_info["title"]) > 80
                    else paper_info["title"]
                )

                formatted_doc = (
                    f'Document [{citation_num}] - "{doc_title}":\n{clean_text}'
                )
                docs_with_citations.append(formatted_doc)

                total_context_used += estimated_doc_size
                documents_used += 1

                if total_context_used >= self.max_context_chars:
                    break

            total_context = total_context_used
            total_docs = documents_used

        return docs_with_citations

    def _create_agent4_prompt_with_citations_fulltext(
        self,
        original_query,
        full_texts,
        citation_handler,
        was_split: bool = False,
        sub_question_groups: List[List[str]] = None,
    ):
        """Agent-4 prompt with stronger citation enforcement"""
        docs_with_citations = self._prepare_documents_for_agent4_fulltext(
            full_texts, citation_handler, was_split, sub_question_groups
        )

        docs_text = "\n\n" + "=" * 50 + "\n\n".join(docs_with_citations)
        available_citations = [str(i) for i in range(1, len(docs_with_citations) + 1)]
        citation_examples = ", ".join(available_citations)

        return f"""You are a precise academic AI assistant. Answer questions using ONLY the provided documents with MANDATORY academic citations.

DOCUMENTS:
{docs_text}

MANDATORY CITATION RULES - FOLLOW EXACTLY:

CORRECT EXAMPLES:
"Pebble games characterize sparse graphs [1]. These algorithms find rigid structures efficiently [2]. The method extends to hypergraphs [3]."

"Machine learning uses pattern recognition [1], neural networks [2], and requires large datasets [3]."

WRONG EXAMPLES (DO NOT DO THIS):
"Machine learning is powerful. It uses algorithms and requires data. [1, 2, 3]" ← WRONG!
"The research shows important findings [1, 2, 3]." ← WRONG!
"Algorithms are useful." ← WRONG! (no citation)

REQUIREMENTS:
1. Every sentence with facts MUST end with [1], [2], [3], etc.
2. Use different numbers - don't repeat [1] for everything
3. NO GROUP CITATIONS like [1, 2, 3] at the end
4. Available citations: {citation_examples}
5. Write 4-6 paragraphs with citations throughout

QUESTION: {original_query}

Write your answer now with proper individual citations after each claim:"""

    def _reorder_full_texts_by_agent3_scores(
        self, full_texts: List[Tuple[str, str]], agent3_scores: Dict[str, float]
    ) -> List[Tuple[str, str]]:
        """Reorder full texts by Agent 3 scores (highest scores first)"""

        def get_score(item):
            doc_text, doc_id = item
            return agent3_scores.get(doc_id, 0.0)

        reordered = sorted(full_texts, key=get_score, reverse=True)
        return reordered

    def _group_documents_by_subquestions(
        self,
        filtered_doc_ids_per_question: List[List[str]],
        all_filtered_doc_ids: List[str],
    ) -> List[List[str]]:
        """Group document IDs by sub-questions for context allocation"""
        if not filtered_doc_ids_per_question or len(filtered_doc_ids_per_question) < 2:
            return [all_filtered_doc_ids]

        groups = []
        for sub_question_docs in filtered_doc_ids_per_question:
            group = [
                doc_id for doc_id in sub_question_docs if doc_id in all_filtered_doc_ids
            ]
            groups.append(group)

        return groups

    def _process_single_question_with_scores(
        self, query: str, db=None
    ) -> Tuple[List[Tuple], List, Dict[str, float]]:
        """Process a single question and return (abstracts, filtered_documents, agent3_scores)"""
        retrieved_abstracts = self.retriever.retrieve_abstracts(query, top_k=5)

        # Agent-2 generates answers from ABSTRACTS
        doc_answers = []
        for abstract_text, doc_id in tqdm(retrieved_abstracts):
            prompt = self._create_agent2_prompt(query, abstract_text)
            answer = self.agent2.generate(prompt)
            doc_answers.append((abstract_text, doc_id, answer))

        # Agent-3 evaluates documents using ABSTRACTS
        scores = []
        agent3_scores = {}

        for abstract_text, doc_id, answer in tqdm(doc_answers):
            prompt = self._create_agent3_prompt(query, abstract_text, answer)
            log_probs = self.agent3.get_log_probs(prompt, ["Yes", "No"])
            score = log_probs["Yes"] - log_probs["No"]
            scores.append(score)
            agent3_scores[doc_id] = score

        # Calculate adaptive judge bar
        tau_q = np.mean(scores)
        sigma = np.std(scores)
        adjusted_tau_q = tau_q - self.n * sigma

        # Filter documents based on abstract evaluation
        filtered_doc_ids = []
        filtered_abstracts = []
        for i, (abstract_text, doc_id, _) in enumerate(doc_answers):
            if scores[i] >= adjusted_tau_q:
                filtered_doc_ids.append(doc_id)
                filtered_abstracts.append((abstract_text, doc_id, scores[i]))

        filtered_abstracts.sort(key=lambda x: x[2], reverse=True)

        return retrieved_abstracts, filtered_doc_ids, agent3_scores

    def answer_query(self, query, db=None, choices=None):
        """Process query with full-text context strategy"""
        citation_handler = MultiPaperCitationHandler(self.index_dir, self.agent4)

        # Agent-1 Question Splitting
        should_split, sub_questions = self.question_splitter.analyze_and_split(query)

        if should_split and sub_questions:
            questions_to_process = sub_questions
        else:
            questions_to_process = [query]

        # Parallel Processing of Questions with Score Tracking
        all_filtered_doc_ids = []
        all_agent3_scores = {}
        filtered_doc_ids_per_question = []

        if len(questions_to_process) > 1:
            future_to_question = {}
            for sub_query in questions_to_process:
                future = self.executor.submit(
                    self._process_single_question_with_scores, sub_query, db
                )
                future_to_question[future] = sub_query

            for future in as_completed(future_to_question):
                sub_query = future_to_question[future]
                try:
                    retrieved_abstracts, filtered_doc_ids, agent3_scores = (
                        future.result()
                    )
                    all_filtered_doc_ids.extend(filtered_doc_ids)
                    all_agent3_scores.update(agent3_scores)
                    filtered_doc_ids_per_question.append(filtered_doc_ids)
                except Exception as e:
                    logger.error(f"Error processing sub-question '{sub_query}': {e}")
        else:
            retrieved_abstracts, filtered_doc_ids, agent3_scores = (
                self._process_single_question_with_scores(questions_to_process[0], db)
            )
            all_filtered_doc_ids = filtered_doc_ids
            all_agent3_scores = agent3_scores
            filtered_doc_ids_per_question = [filtered_doc_ids]

        # Remove duplicates while preserving order
        seen = set()
        unique_filtered_doc_ids = []
        for doc_id in all_filtered_doc_ids:
            if doc_id not in seen:
                seen.add(doc_id)
                unique_filtered_doc_ids.append(doc_id)

        # Get FULL TEXTS and Reorder by Agent 3 Scores
        if unique_filtered_doc_ids:
            full_texts = self.retriever.get_full_texts(unique_filtered_doc_ids, db=db)
            full_texts = self._reorder_full_texts_by_agent3_scores(
                full_texts, all_agent3_scores
            )
        else:
            fallback_abstracts, fallback_ids, fallback_scores = (
                self._process_single_question_with_scores(query, db)
            )
            full_texts = self.retriever.get_full_texts(fallback_ids[:3], db=db)
            all_agent3_scores.update(fallback_scores)
            if not full_texts:
                full_texts = [
                    (abstract_text, doc_id)
                    for abstract_text, doc_id in fallback_abstracts[:3]
                ]

        # Agent-4 generates final answer with full-text context
        sub_question_groups = None
        if should_split and len(filtered_doc_ids_per_question) >= 2:
            sub_question_groups = self._group_documents_by_subquestions(
                filtered_doc_ids_per_question, unique_filtered_doc_ids
            )

        prompt = self._create_agent4_prompt_with_citations_fulltext(
            query, full_texts, citation_handler, should_split, sub_question_groups
        )
        raw_answer = self.agent4.generate(prompt)

        # Generate references with comprehensive context passages
        references = citation_handler.format_references_comprehensive(raw_answer)

        if "## References" in raw_answer:
            raw_answer = re.split(r"\n\s*## References", raw_answer)[0]

        cited_answer = raw_answer.strip() + references
        citation_map = citation_handler.get_citation_map()

        debug_info = {
            "original_query": query,
            "was_split": should_split,
            "sub_questions": sub_questions if should_split else [],
            "questions_processed": len(questions_to_process),
            "total_filtered_docs": len(unique_filtered_doc_ids),
            "full_texts_retrieved": len(full_texts),
            "total_citations": len(citation_map),
            "citation_map": citation_map,
            "passages_used": self._extract_passages_used(raw_answer, citation_handler),
            "document_metadata": self._extract_document_metadata(citation_handler),
            "agent3_scores": all_agent3_scores,
            "context_stats": {
                "max_context_chars": self.max_context_chars,
                "total_chars_available": sum(len(text) for text, _ in full_texts),
                "docs_available": len(full_texts),
                "strategy": "FULL-TEXT",
                "context_allocation": (
                    "32K single" if not should_split else "16K per sub-question"
                ),
            },
        }

        return cited_answer, debug_info

    def _extract_passages_used(self, answer_text: str, citation_handler) -> List[Dict]:
        """Extract the specific passages used in the answer"""
        citation_matches = re.findall(r"\[(\d+)\]", answer_text)
        used_citations = set(int(num) for num in citation_matches)

        passages_used = []

        if hasattr(citation_handler, "extract_all_citations_comprehensive"):
            try:
                comprehensive_passages = (
                    citation_handler.extract_all_citations_comprehensive(answer_text)
                )

                for citation_num in used_citations:
                    if citation_num in citation_handler.citation_to_doc:
                        doc_info = citation_handler.citation_to_doc[citation_num]
                        context_passage = comprehensive_passages.get(
                            citation_num, "No passage extracted"
                        )

                        passages_used.append(
                            {
                                "citation_num": citation_num,
                                "doc_id": doc_info["doc_id"],
                                "paper_title": doc_info["paper_info"]["title"],
                                "paper_id": doc_info["paper_info"]["paper_id"],
                                "authors": doc_info["paper_info"]["authors"],
                                "year": doc_info["paper_info"]["year"],
                                "context_passage": context_passage,
                                "passage_preview": (
                                    context_passage[:200] + "..."
                                    if len(context_passage) > 200
                                    else context_passage
                                ),
                                "extraction_method": "comprehensive",
                            }
                        )

                return passages_used

            except Exception as e:
                pass

        # Fallback for when comprehensive extraction fails
        for citation_num in used_citations:
            if citation_num in citation_handler.citation_to_doc:
                doc_info = citation_handler.citation_to_doc[citation_num]

                try:
                    if hasattr(citation_handler, "_get_fallback_passage"):
                        context_passage = citation_handler._get_fallback_passage(
                            citation_num
                        )
                    else:
                        clean_text = citation_handler._basic_text_cleaning(
                            doc_info["text"]
                        )
                        sentences = re.split(r"[.!?]+", clean_text)
                        sentences = [
                            s.strip() for s in sentences if len(s.strip()) > 20
                        ]
                        context_passage = ". ".join(sentences[:2]) + "."
                except Exception as e:
                    context_passage = f"Error extracting passage: {str(e)}"

                passages_used.append(
                    {
                        "citation_num": citation_num,
                        "doc_id": doc_info["doc_id"],
                        "paper_title": doc_info["paper_info"]["title"],
                        "paper_id": doc_info["paper_info"]["paper_id"],
                        "authors": doc_info["paper_info"]["authors"],
                        "year": doc_info["paper_info"]["year"],
                        "context_passage": context_passage,
                        "passage_preview": (
                            context_passage[:200] + "..."
                            if len(context_passage) > 200
                            else context_passage
                        ),
                        "extraction_method": "fallback",
                    }
                )

        return passages_used

    def _extract_document_metadata(self, citation_handler):
        """Extract document metadata for all citations"""
        metadata = {}
        for citation_num, doc_info in citation_handler.citation_to_doc.items():
            metadata[citation_num] = {
                "doc_id": doc_info["doc_id"],
                "paper_info": doc_info["paper_info"],
            }
        return metadata

    def close(self):
        """Clean up resources"""
        if hasattr(self, "executor"):
            self.executor.shutdown(wait=True)


def load_datamorgana_questions(file_path):
    """Load questions from file"""
    is_jsonl = file_path.lower().endswith(".jsonl")

    try:
        questions = []

        if is_jsonl:
            with open(file_path, "r", encoding="utf-8") as f:
                line_num = 0
                for line in f:
                    line_num += 1
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        question = json.loads(line)
                        if "id" not in question:
                            question["id"] = line_num
                        questions.append(question)
                    except json.JSONDecodeError as e:
                        logger.error(f"Error parsing JSON at line {line_num}: {e}")
        else:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

                if isinstance(data, list):
                    questions = data
                    for i, question in enumerate(questions):
                        if "id" not in question:
                            question["id"] = i + 1
                elif isinstance(data, dict):
                    if "questions" in data:
                        questions = data["questions"]
                    elif "question" in data:
                        questions = [data]
                    else:
                        questions = [data]

        return questions

    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error loading questions: {e}")
        return []


def format_result_to_schema(result):
    """Format result with 4-agent information"""
    formatted_result = {
        "id": result.get("id", 0),
        "question": result.get("question", ""),
        "answer": result.get("model_answer", ""),
        "was_split": result.get("was_split", False),
        "sub_questions": result.get("sub_questions", []),
        "questions_processed": result.get("questions_processed", 1),
        "citation_count": result.get("total_citations", 0),
        "total_filtered_docs": result.get("total_filtered_docs", 0),
        "full_texts_used": result.get("full_texts_retrieved", 0),
        "processing_time": result.get("process_time", 0),
        "retriever_type": result.get("retriever_type", "hybrid"),
        "passages_used": result.get("passages_used", []),
        "document_metadata": result.get("document_metadata", {}),
        "agent3_scores": result.get("agent3_scores", {}),
        "context_strategy": result.get("context_stats", {}).get(
            "strategy", "FULL-TEXT"
        ),
        "context_allocation": result.get("context_stats", {}).get(
            "context_allocation", "32K"
        ),
        "total_chars_available": result.get("context_stats", {}).get(
            "total_chars_available", 0
        ),
    }

    return formatted_result


def write_results_to_jsonl(results, output_file):
    """Write results to JSONL file"""
    with open(output_file, "w", encoding="utf-8") as f:
        for result in results:
            formatted_result = format_result_to_schema(result)
            f.write(json.dumps(formatted_result, ensure_ascii=False) + "\n")


def write_result_to_json(result, output_file):
    """Write single result to JSON file"""
    formatted_result = format_result_to_schema(result)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(formatted_result, f, indent=2, ensure_ascii=False)


def main():
    """Main function with 4-agent support"""
    parser = argparse.ArgumentParser(
        description="4-Agent RAG with Question Splitting and Parallel Processing"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="tiiuae/Falcon3-10B-Instruct",
        help="Model for LLM agents",
    )
    parser.add_argument(
        "--n", type=float, default=0.5, help="Adjustment factor for adaptive judge bar"
    )
    parser.add_argument(
        "--retriever_type",
        choices=["e5", "bm25", "hybrid"],
        default="hybrid",
        help="Type of retriever",
    )
    parser.add_argument(
        "--index_dir",
        type=str,
        default="test_index",
        help="Directory containing metadata",
    )
    parser.add_argument(
        "--top_k", type=int, default=5, help="Number of documents to retrieve"
    )
    parser.add_argument(
        "--data_file",
        type=str,
        default="quick_test_questions.jsonl",
        help="File containing questions",
    )
    parser.add_argument(
        "--single_question", type=str, default=None, help="Process a single question"
    )
    parser.add_argument(
        "--output_format",
        choices=["json", "jsonl", "debug"],
        default="jsonl",
        help="Output format",
    )
    parser.add_argument(
        "--output_dir", type=str, default="results", help="Directory to save results"
    )
    parser.add_argument(
        "--max_workers", type=int, default=4, help="Maximum number of parallel workers"
    )
    parser.add_argument(
        "--db_path",
        type=str,
        default=None,
        help="Path to LevelDB database (overrides config)",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.65, help="Weight for E5 in hybrid mode"
    )
    parser.add_argument(
        "--max_context_chars",
        type=int,
        default=100000,
        help="Maximum context characters",
    )
    parser.add_argument(
        "--retriever_strategy",
        choices=["e5", "bm25", "hybrid"],
        default="hybrid",
        help="Retriever strategy",
    )
    args = parser.parse_args()

    db_path_to_use = args.db_path if args.db_path else DB_PATH

    try:
        db = plyvel.DB(db_path_to_use, create_if_missing=False)
    except Exception as e:
        alt_db_path = os.path.join(os.path.dirname(__file__), "local_db")
        db = plyvel.DB(alt_db_path, create_if_missing=True)

    from hybrid_retriever import Retriever

    retriever = Retriever(
        e5_index_directory=E5_INDEX_DIR,
        bm25_index_directory=BM25_INDEX_DIR,
        top_k=args.top_k,
        strategy="hybrid",
        alpha=args.alpha,
    )

    ragent = FourAgentRAG(
        retriever,
        agent_model=args.model,
        n=args.n,
        index_dir=args.index_dir,
        max_workers=args.max_workers,
        max_context_chars=args.max_context_chars,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "debug"), exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Process single question
    if args.single_question:
        start_time = time.time()

        try:
            cited_answer, debug_info = ragent.answer_query(args.single_question, db)
            process_time = time.time() - start_time

            result = {
                "id": f"single_question_4agent_{args.retriever_type}",
                "question": args.single_question,
                "model_answer": cited_answer,
                "was_split": debug_info["was_split"],
                "sub_questions": debug_info["sub_questions"],
                "questions_processed": debug_info["questions_processed"],
                "total_citations": debug_info["total_citations"],
                "total_filtered_docs": debug_info["total_filtered_docs"],
                "full_texts_retrieved": debug_info["full_texts_retrieved"],
                "passages_used": debug_info["passages_used"],
                "document_metadata": debug_info["document_metadata"],
                "process_time": process_time,
                "retriever_type": args.retriever_type,
            }

            if args.output_format == "debug":
                debug_output_file = os.path.join(
                    args.output_dir,
                    "debug",
                    f"4agent_single_{args.retriever_type}_debug_{timestamp}.json",
                )
                with open(debug_output_file, "w", encoding="utf-8") as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
            else:
                output_file = os.path.join(
                    args.output_dir,
                    f"4agent_single_{args.retriever_type}_{timestamp}.json",
                )
                write_result_to_json(result, output_file)

        except Exception as e:
            logger.error(f"Error processing question: {e}", exc_info=True)
        finally:
            ragent.close()
            retriever.close()

        return

    # Process question file
    questions = load_datamorgana_questions(args.data_file)
    if not questions:
        logger.error("No questions found. Exiting.")
        return

    results = []

    for i, item in enumerate(questions):
        question_id = item.get("id", i + 1)
        start_time = time.time()

        try:
            cited_answer, debug_info = ragent.answer_query(item["question"], db)
            process_time = time.time() - start_time

            result = {
                "id": question_id,
                "question": item["question"],
                "model_answer": cited_answer,
                "was_split": debug_info["was_split"],
                "sub_questions": debug_info["sub_questions"],
                "questions_processed": debug_info["questions_processed"],
                "total_citations": debug_info["total_citations"],
                "total_filtered_docs": debug_info["total_filtered_docs"],
                "full_texts_retrieved": debug_info["full_texts_retrieved"],
                "passages_used": debug_info["passages_used"],
                "document_metadata": debug_info["document_metadata"],
                "process_time": process_time,
                "retriever_type": args.retriever_type,
            }
            results.append(result)

            debug_output_file = os.path.join(
                args.output_dir,
                "debug",
                f"4agent_question_{question_id}_{args.retriever_type}_debug_{timestamp}.json",
            )
            with open(debug_output_file, "w", encoding="utf-8") as f:
                json.dump(debug_info, f, indent=2, ensure_ascii=False)

        except Exception as e:
            logger.error(f"Error processing question {question_id}: {e}", exc_info=True)

    ragent.close()
    retriever.close()

    random_num = "".join(random.choices(string.ascii_lowercase + string.digits, k=6))

    if results:
        if args.output_format == "jsonl":
            output_file = os.path.join(
                args.output_dir,
                f"4agent_answers_{args.retriever_type}_{timestamp}_{random_num}.jsonl",
            )
            write_results_to_jsonl(results, output_file)
        elif args.output_format == "json":
            for result in results:
                question_id = result["id"]
                output_file = os.path.join(
                    args.output_dir,
                    f"4agent_answer_{question_id}_{args.retriever_type}_{random_num}.json",
                )
                write_result_to_json(result, output_file)
        else:
            output_file = os.path.join(
                args.output_dir,
                "debug",
                f"4agent_all_results_{args.retriever_type}_debug_{timestamp}.json",
            )
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

    db.close()


if __name__ == "__main__":
    main()
