#!/usr/bin/env python3
"""
Clean 4-Agent RAG System with Hybrid Retriever and Abstract-Only Processing
- Agent 1: Question Splitter
- Agent 2: Answer Generator from abstracts
- Agent 3: Document Evaluator
- Agent 4: Final Answer Generator with citations from abstracts only
"""
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

# Import configuration
from config import E5_INDEX_DIR, BM25_INDEX_DIR
from hybrid_retriever import Retriever


# Logging setup
def get_unique_log_filename():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    random_str = "".join(random.choices(string.ascii_lowercase + string.digits, k=6))
    return f"logs/simple_4agent_rag_{timestamp}_{random_str}.log"


os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(get_unique_log_filename()), logging.StreamHandler()],
)
logger = logging.getLogger("Simple_4Agent_RAG")


class QuestionSplitter:
    """Agent 1: Intelligent Question Splitting Agent"""

    def __init__(self, agent_model):
        self.agent = agent_model

    def _create_splitting_prompt(self, query: str) -> str:
        return f"""You are an intelligent question analyzer. Determine if a query contains multiple distinct sub-questions that would benefit from separate retrieval.

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

Query: "{query}"

Respond with ONLY this format:
Split: YES/NO
Sub-questions: [list of questions] (empty list if Split: NO)"""

    def analyze_and_split(self, query: str) -> Tuple[bool, List[str]]:
        if not self._quick_split_check(query):
            return False, []

        prompt = self._create_splitting_prompt(query)
        response = self.agent.generate(prompt)
        should_split, sub_questions = self._parse_splitting_response(response, query)

        if should_split:
            logger.info(f"Split into {len(sub_questions)} sub-questions")
        return should_split, sub_questions

    def _quick_split_check(self, query: str) -> bool:
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
                valid_questions = [
                    q.strip()
                    for q in sub_questions
                    if len(q.strip()) > 10 and q.strip().endswith("?")
                ]
                if len(valid_questions) < 2:
                    return False, []
                return True, valid_questions

            return False, []
        except Exception as e:
            logger.warning(f"Error parsing splitting response: {e}")
            return False, []


class AbstractTitleExtractor:
    """Utility class for extracting paper titles from abstract text"""

    @staticmethod
    def extract_title_from_abstract(abstract_text: str, doc_id: str) -> str:
        try:
            # Extract from JSON format first
            json_pattern = r"'text':\s*'([^']*(?:''[^']*)*)'"
            match = re.search(json_pattern, abstract_text)

            if match:
                title_match = re.match(r"^([^{]+)", abstract_text)
                if title_match:
                    title_candidate = title_match.group(1).strip()
                    title_candidate = re.sub(r'^["\']|["\']$', "", title_candidate)
                    title_candidate = re.sub(r"\.$", "", title_candidate)
                    if 10 < len(title_candidate) < 300:
                        return title_candidate

            # Fallback logic
            lines = abstract_text.split("\n")
            for line in lines[:5]:
                line = line.strip()
                if 10 < len(line) < 300 and not line.lower().startswith(
                    ("abstract:", "introduction:", "methods:", "results:")
                ):
                    clean_title = re.sub(r'^["\']|["\']$', "", line)
                    if len(clean_title) > 10:
                        return clean_title

            first_line = abstract_text.split("\n")[0].strip()
            if len(first_line) > 20:
                return (
                    (first_line[:150] + "...") if len(first_line) > 150 else first_line
                )

            return f"Document {doc_id}"
        except Exception:
            return f"Document {doc_id}"

    @staticmethod
    def format_title_for_log(title: str, max_length: int = 80) -> str:
        return title if len(title) <= max_length else title[: max_length - 3] + "..."


class SimpleAbstractCitationHandler:
    """Citation handler for abstract-only processing with comprehensive multi-paper extraction"""

    def __init__(self, index_dir: str = "test_index", agent_model=None):
        self.doc_to_citation = {}
        self.citation_to_doc = {}
        self.next_citation_num = 1
        self.index_dir = Path(index_dir)
        self.agent_model = agent_model
        self.arxiv_papers = self._load_arxiv_papers()

    def _load_arxiv_papers(self):
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

                            # Extract year from versions
                            year = "Unknown"
                            versions = paper.get("versions", [])
                            if versions:
                                created = versions[0].get("created", "")
                                year_match = re.search(r"(\d{4})", created)
                                if year_match:
                                    year = year_match.group(1)

                            # Format authors
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
                                        authors = (
                                            f"{formatted_author} et al."
                                            if len(authors_list) > 1
                                            else formatted_author
                                        )

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
        except Exception:
            return {}

    def _extract_paper_info(
        self, abstract_text: str, doc_id: str, metadata: Dict = None
    ) -> Dict:
        paper_info = {
            "title": "Unknown Title",
            "authors": "Unknown",
            "venue": "arXiv",
            "year": "Unknown",
            "paper_id": doc_id,
        }

        try:
            paper_info["title"] = AbstractTitleExtractor.extract_title_from_abstract(
                abstract_text, doc_id
            )

            # Extract from JSON in abstract text
            if "{" in abstract_text and '"metadata"' in abstract_text:
                try:
                    json_match = re.search(
                        r'\{.*?"metadata".*?\}', abstract_text, re.DOTALL
                    )
                    if json_match:
                        json_str = json_match.group(0)
                        paper_data = json.loads(json_str)

                        if "metadata" in paper_data:
                            meta = paper_data["metadata"]
                            if "authors" in meta:
                                paper_info["authors"] = meta["authors"]

                        if "paper_id" in paper_data:
                            paper_info["paper_id"] = paper_data["paper_id"]

                        # Extract year from versions
                        if "versions" in paper_data and paper_data["versions"]:
                            created = paper_data["versions"][0].get("created", "")
                            year_match = re.search(r"(\d{4})", created)
                            if year_match:
                                paper_info["year"] = year_match.group(1)
                except Exception:
                    pass

            # Match with loaded arXiv papers
            if doc_id in self.arxiv_papers:
                arxiv_data = self.arxiv_papers[doc_id]
                if paper_info["title"] in ["Unknown Title", f"Document {doc_id}"]:
                    paper_info["title"] = arxiv_data["title"]
                if paper_info["authors"] == "Unknown":
                    paper_info["authors"] = arxiv_data["authors"]
                if paper_info["year"] == "Unknown":
                    paper_info["year"] = arxiv_data["year"]

            # Final cleanup
            if len(paper_info["title"]) > 150:
                paper_info["title"] = paper_info["title"][:150] + "..."

            if not paper_info["paper_id"]:
                paper_info["paper_id"] = doc_id

        except Exception:
            pass

        return paper_info

    def _basic_text_cleaning(self, text: str) -> str:
        # Extract abstract text from JSON-like structure
        json_pattern = r"'text':\s*'([^']*(?:''[^']*)*)'"
        match = re.search(json_pattern, text)

        if match:
            abstract_content = match.group(1).replace("''", "'")
            title_match = re.match(r"^([^{]+)", text)
            title = title_match.group(1).strip() if title_match else ""
            text = f"{title}\n\n{abstract_content}"

        # Clean up formatting
        text = re.sub(r"\{\{[^}]+\}\}", "[REF]", text)
        text = re.sub(r"\$[^$]+\$", "[MATH]", text)
        text = re.sub(r"\\[a-zA-Z]+\{[^}]*\}", "[LATEX]", text)
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"\n\s*\n", "\n\n", text)

        return text.strip()

    def add_document(
        self, abstract_text: str, doc_id: str, metadata: Dict = None
    ) -> int:
        if doc_id not in self.doc_to_citation:
            citation_num = self.next_citation_num
            self.doc_to_citation[doc_id] = citation_num

            paper_info = self._extract_paper_info(abstract_text, doc_id, metadata)
            self.citation_to_doc[citation_num] = {
                "doc_id": doc_id,
                "paper_info": paper_info,
                "abstract_text": abstract_text,
            }
            self.next_citation_num += 1
            return citation_num
        else:
            return self.doc_to_citation[doc_id]

    def get_citation_map(self) -> Dict[str, int]:
        return self.doc_to_citation.copy()

    def _find_all_used_citations(self, answer_text: str) -> List[int]:
        citation_matches = re.findall(r"\[(\d+)\]", answer_text)
        return sorted(set(int(num) for num in citation_matches))

    def _extract_claims_for_citation(
        self, answer_text: str, citation_num: int
    ) -> List[str]:
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
        # Build abstracts section
        abstracts_section = "RESEARCH PAPER ABSTRACTS:\n" + "=" * 80 + "\n\n"

        for citation_num in used_citations:
            if citation_num in self.citation_to_doc:
                doc_info = self.citation_to_doc[citation_num]
                paper_info = doc_info["paper_info"]
                clean_abstract = self._basic_text_cleaning(doc_info["abstract_text"])

                abstracts_section += (
                    f"ABSTRACT [{citation_num}]: \"{paper_info['title']}\"\n"
                )
                abstracts_section += f"Authors: {paper_info['authors']}\n"
                abstracts_section += f"Year: {paper_info['year']}\n"
                abstracts_section += (
                    f"Abstract:\n{clean_abstract}\n\n" + "-" * 50 + "\n\n"
                )

        # Analyze claims by citation
        claims_section = "CLAIMS TO ANALYZE:\n" + "=" * 40 + "\n"
        for citation_num in used_citations:
            claims = self._extract_claims_for_citation(answer_text, citation_num)
            if claims:
                claims_section += (
                    f"\nCitation [{citation_num}] supports these claims:\n"
                )
                for i, claim in enumerate(claims, 1):
                    claims_section += f'  {i}. "{claim}"\n'

        return f"""You are an expert citation analyzer. Identify EXACT passages from research paper abstracts that support specific claims.

{abstracts_section}

{claims_section}

COMPLETE ANSWER WITH CITATIONS:
{answer_text}

TASK: For each citation, identify specific passages from the corresponding abstract that directly support the claims.

RESPONSE FORMAT:
Citation [1]:
Passage: "[exact text from Abstract [1] that supports the claims]"

Citation [2]: 
Passage: "[exact text from Abstract [2] that supports the claims]"

[Continue for all citations...]

Begin analysis:"""

    def _parse_comprehensive_response(
        self, model_response: str, used_citations: List[int]
    ) -> Dict[int, str]:
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
                    if passage and not passage.endswith((".", "!", "?")):
                        passage += "."
                    citation_passages[citation_num] = passage
                else:
                    citation_passages[citation_num] = self._get_fallback_passage(
                        citation_num
                    )

            return citation_passages
        except Exception:
            return self._fallback_extraction_all(used_citations)

    def _get_fallback_passage(self, citation_num: int) -> str:
        if citation_num in self.citation_to_doc:
            abstract_text = self.citation_to_doc[citation_num]["abstract_text"]
            clean_text = self._basic_text_cleaning(abstract_text)
            sentences = re.split(r"[.!?]+", clean_text)
            sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

            fallback_sentences = sentences[:2] if len(sentences) >= 2 else sentences
            fallback = ". ".join(fallback_sentences)
            if fallback and not fallback.endswith((".", "!", "?")):
                fallback += "."
            return fallback
        return f"Content from citation [{citation_num}] (fallback extraction)."

    def _fallback_extraction_all(self, used_citations: List[int]) -> Dict[int, str]:
        return {
            citation_num: self._get_fallback_passage(citation_num)
            for citation_num in used_citations
        }

    def extract_all_citations_comprehensive(self, answer_text: str) -> Dict[int, str]:
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

            successful_extractions = sum(
                1
                for passage in citation_passages.values()
                if not passage.startswith(("Content from citation", "fallback"))
            )

            logger.info(
                f"Citation extraction: {successful_extractions}/{len(used_citations)} successful"
            )
            return citation_passages

        except Exception as e:
            logger.error(f"Error in comprehensive citation extraction: {e}")
            return self._fallback_extraction_all(used_citations)

    def format_references_comprehensive(self, answer_text: str = None) -> str:
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


class SimpleFourAgentRAG:
    """Simple 4-Agent RAG System with Hybrid Retriever and Abstract-Only Processing"""

    def __init__(
        self,
        retriever,
        agent_model=None,
        n=0.0,
        falcon_api_key=None,
        index_dir="test_index",
        max_workers=4,
    ):
        self.retriever = retriever
        self.n = n
        self.index_dir = index_dir
        self.max_workers = max_workers

        # Initialize agents
        if isinstance(agent_model, str):
            if "falcon" in agent_model.lower() and falcon_api_key:
                from api_agent import FalconAgent

                self.agent1 = FalconAgent(falcon_api_key)
                self.agent2 = FalconAgent(falcon_api_key)
                self.agent3 = FalconAgent(falcon_api_key)
                self.agent4 = FalconAgent(falcon_api_key)
            else:
                from local_agent import LLMAgent

                self.agent1 = LLMAgent(agent_model)
                self.agent2 = LLMAgent(agent_model)
                self.agent3 = LLMAgent(agent_model)
                self.agent4 = LLMAgent(agent_model)
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
        except Exception:
            pass

    def _create_agent2_prompt(self, query, document):
        return f"""You are an accurate AI assistant that answers questions using external documents. Provide only the correct answer without repeating the question.

Document: {document}

Question: {query}

Answer:"""

    def _create_agent3_prompt(self, query, document, answer):
        return f"""You are a document evaluator. Judge if the document provides specific information for answering the question and if the LLM answer directly answers based on the document.

Document: {document}

Question: {query}

LLM Answer: {answer}

Is this document relevant and supportive for answering the question?"""

    def _clean_abstract_text(self, abstract_text: str, doc_id: str) -> str:
        try:
            clean_text = re.sub(r"Content for [^:]*:\s*\n", "", abstract_text)
            if clean_text.startswith("{") and '"paper_id"' in clean_text[:500]:
                json_end = clean_text.find("\n\n")
                if json_end > 0:
                    clean_text = clean_text[json_end:].strip()
            return clean_text.strip()
        except Exception:
            return abstract_text

    def _prepare_abstracts_for_agent4(
        self, filtered_abstracts: List[Tuple[str, str, float]], citation_handler
    ) -> List[str]:
        abstracts_with_citations = []

        for abstract_text, doc_id, score in filtered_abstracts:
            clean_abstract = self._clean_abstract_text(abstract_text, doc_id)
            citation_num = citation_handler.add_document(clean_abstract, doc_id)
            paper_info = citation_handler.citation_to_doc[citation_num]["paper_info"]

            doc_title = (
                paper_info["title"][:80] + "..."
                if len(paper_info["title"]) > 80
                else paper_info["title"]
            )

            formatted_abstract = (
                f'Abstract [{citation_num}] - "{doc_title}":\n{clean_abstract}'
            )
            abstracts_with_citations.append(formatted_abstract)

        return abstracts_with_citations

    def _create_agent4_prompt_with_citations_abstracts(
        self, original_query, filtered_abstracts, citation_handler
    ):
        abstracts_with_citations = self._prepare_abstracts_for_agent4(
            filtered_abstracts, citation_handler
        )
        abstracts_text = "\n\n" + "=" * 50 + "\n\n".join(abstracts_with_citations)

        available_citations = [
            str(i) for i in range(1, len(abstracts_with_citations) + 1)
        ]
        citation_examples = ", ".join(available_citations)

        return f"""You are an accurate AI assistant. Answer questions based ONLY on the provided abstracts with proper academic citations.

STRICT CITATION REQUIREMENTS:
1. Add [{citation_examples}] after EVERY factual claim
2. Every sentence with factual information MUST end with a citation
3. Use MULTIPLE different abstracts - don't just cite [1] repeatedly
4. Do NOT add a references section - it will be added automatically

EXAMPLE: "Machine learning involves pattern recognition [1]. Neural networks are popular [2]. Deep learning shows success [3]."

Abstracts:
{abstracts_text}

Question: {original_query}

Answer:"""

    def _process_single_question_with_scores(
        self, query: str
    ) -> Tuple[List[Tuple], List[Tuple[str, str, float]], Dict[str, float]]:
        # Retrieve abstracts
        retrieved_abstracts = self.retriever.retrieve_abstracts(query, top_k=5)

        # Agent-2 generates answers from abstracts
        doc_answers = []
        for abstract_text, doc_id in tqdm(
            retrieved_abstracts, desc="Generating answers"
        ):
            prompt = self._create_agent2_prompt(query, abstract_text)
            answer = self.agent2.generate(prompt)
            doc_answers.append((abstract_text, doc_id, answer))

        # Agent-3 evaluates documents
        scores = []
        agent3_scores = {}

        for abstract_text, doc_id, answer in tqdm(
            doc_answers, desc="Evaluating documents"
        ):
            prompt = self._create_agent3_prompt(query, abstract_text, answer)
            log_probs = self.agent3.get_log_probs(prompt, ["Yes", "No"])
            score = log_probs["Yes"] - log_probs["No"]
            scores.append(score)
            agent3_scores[doc_id] = score

        # Calculate adaptive judge bar
        tau_q = np.mean(scores)
        sigma = np.std(scores)
        adjusted_tau_q = tau_q - self.n * sigma

        # Filter documents
        filtered_abstracts = []
        for i, (abstract_text, doc_id, _) in enumerate(doc_answers):
            if scores[i] >= adjusted_tau_q:
                filtered_abstracts.append((abstract_text, doc_id, scores[i]))

        filtered_abstracts.sort(key=lambda x: x[2], reverse=True)
        return retrieved_abstracts, filtered_abstracts, agent3_scores

    def answer_query(self, query, choices=None):
        logger.info(f"Processing query: {query}")

        citation_handler = SimpleAbstractCitationHandler(self.index_dir, self.agent4)

        # Agent-1 Question Splitting
        should_split, sub_questions = self.question_splitter.analyze_and_split(query)
        questions_to_process = (
            sub_questions if should_split and sub_questions else [query]
        )

        # Process questions (parallel if multiple)
        all_filtered_abstracts = []
        all_agent3_scores = {}

        if len(questions_to_process) > 1:
            future_to_question = {}
            for sub_query in questions_to_process:
                future = self.executor.submit(
                    self._process_single_question_with_scores, sub_query
                )
                future_to_question[future] = sub_query

            for future in as_completed(future_to_question):
                try:
                    retrieved_abstracts, filtered_abstracts, agent3_scores = (
                        future.result()
                    )
                    all_filtered_abstracts.extend(filtered_abstracts)
                    all_agent3_scores.update(agent3_scores)
                except Exception as e:
                    logger.error(f"Error processing sub-question: {e}")
        else:
            retrieved_abstracts, filtered_abstracts, agent3_scores = (
                self._process_single_question_with_scores(questions_to_process[0])
            )
            all_filtered_abstracts = filtered_abstracts
            all_agent3_scores = agent3_scores

        # Remove duplicates
        seen = set()
        unique_filtered_abstracts = []
        for abstract_text, doc_id, score in all_filtered_abstracts:
            if doc_id not in seen:
                seen.add(doc_id)
                unique_filtered_abstracts.append((abstract_text, doc_id, score))

        # Fallback if no abstracts passed filter
        if not unique_filtered_abstracts:
            fallback_abstracts, fallback_filtered, fallback_scores = (
                self._process_single_question_with_scores(query)
            )
            unique_filtered_abstracts = (
                fallback_filtered[:3]
                if fallback_filtered
                else [
                    (abstract_text, doc_id, 0.0)
                    for abstract_text, doc_id in fallback_abstracts[:3]
                ]
            )
            all_agent3_scores.update(fallback_scores)

        # Sort by scores
        unique_filtered_abstracts = sorted(
            unique_filtered_abstracts, key=lambda x: x[2], reverse=True
        )

        # Agent-4 generates final answer
        prompt = self._create_agent4_prompt_with_citations_abstracts(
            query, unique_filtered_abstracts, citation_handler
        )
        raw_answer = self.agent4.generate(prompt)

        # Generate references
        references = citation_handler.format_references_comprehensive(raw_answer)

        # Clean up answer
        if "## References" in raw_answer:
            raw_answer = re.split(r"\n\s*## References", raw_answer)[0]

        cited_answer = raw_answer.strip() + references
        citation_map = citation_handler.get_citation_map()

        debug_info = {
            "original_query": query,
            "was_split": should_split,
            "sub_questions": sub_questions if should_split else [],
            "questions_processed": len(questions_to_process),
            "total_filtered_abstracts": len(unique_filtered_abstracts),
            "abstracts_used": len(unique_filtered_abstracts),
            "total_citations": len(citation_map),
            "citation_map": citation_map,
            "passages_used": self._extract_passages_used(raw_answer, citation_handler),
            "document_metadata": self._extract_document_metadata(citation_handler),
            "agent3_scores": all_agent3_scores,
            "context_stats": {
                "total_chars_available": sum(
                    len(text) for text, _, _ in unique_filtered_abstracts
                ),
                "abstracts_available": len(unique_filtered_abstracts),
                "strategy": "ABSTRACT-ONLY",
                "context_allocation": "All filtered abstracts",
            },
        }

        return cited_answer, debug_info

    def _extract_passages_used(self, answer_text: str, citation_handler) -> List[Dict]:
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
                                "extraction_method": "comprehensive_abstract",
                            }
                        )
                return passages_used
            except Exception:
                pass

        # Fallback extraction
        for citation_num in used_citations:
            if citation_num in citation_handler.citation_to_doc:
                doc_info = citation_handler.citation_to_doc[citation_num]
                try:
                    context_passage = citation_handler._get_fallback_passage(
                        citation_num
                    )
                except Exception:
                    context_passage = "Error extracting passage"

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
                        "extraction_method": "fallback_abstract",
                    }
                )

        return passages_used

    def _extract_document_metadata(self, citation_handler):
        metadata = {}
        for citation_num, doc_info in citation_handler.citation_to_doc.items():
            metadata[citation_num] = {
                "doc_id": doc_info["doc_id"],
                "paper_info": doc_info["paper_info"],
            }
        return metadata

    def close(self):
        if hasattr(self, "executor"):
            self.executor.shutdown(wait=True)


# Utility Functions
def initialize_simple_retriever(
    retriever_type: str,
    e5_index_dir: str,
    bm25_index_dir: str,
    top_k: int,
    alpha: float = 0.65,
):
    logger.info(f"Initializing {retriever_type} retriever with alpha={alpha}")
    return Retriever(
        e5_index_dir, bm25_index_dir, top_k=top_k, strategy=retriever_type, alpha=alpha
    )


def load_questions(file_path):
    is_jsonl = file_path.lower().endswith(".jsonl")
    try:
        questions = []
        if is_jsonl:
            with open(file_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
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

        logger.info(f"Loaded {len(questions)} questions")
        return questions
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        return []
    except Exception as e:
        logger.error(f"Error loading questions: {e}")
        return []


def format_simple_result_to_schema(result):
    return {
        "id": result.get("id", 0),
        "question": result.get("question", ""),
        "answer": result.get("model_answer", ""),
        "was_split": result.get("was_split", False),
        "sub_questions": result.get("sub_questions", []),
        "questions_processed": result.get("questions_processed", 1),
        "citation_count": result.get("total_citations", 0),
        "total_filtered_abstracts": result.get("total_filtered_abstracts", 0),
        "abstracts_used": result.get("abstracts_used", 0),
        "processing_time": result.get("process_time", 0),
        "retriever_type": result.get("retriever_type", "hybrid"),
        "passages_used": result.get("passages_used", []),
        "document_metadata": result.get("document_metadata", {}),
        "agent3_scores": result.get("agent3_scores", {}),
        "context_strategy": result.get("context_stats", {}).get(
            "strategy", "ABSTRACT-ONLY"
        ),
        "context_allocation": result.get("context_stats", {}).get(
            "context_allocation", "All filtered abstracts"
        ),
        "total_chars_available": result.get("context_stats", {}).get(
            "total_chars_available", 0
        ),
    }


def write_simple_results_to_jsonl(results, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        for result in results:
            formatted_result = format_simple_result_to_schema(result)
            f.write(json.dumps(formatted_result, ensure_ascii=False) + "\n")
    logger.info(f"Results written to {output_file}")


def write_simple_result_to_json(result, output_file):
    formatted_result = format_simple_result_to_schema(result)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(formatted_result, f, indent=2, ensure_ascii=False)
    logger.info(f"Result written to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Simple 4-Agent RAG with Hybrid Retriever and Abstract-Only Processing"
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
        "--alpha", type=float, default=0.65, help="Weight for E5 in hybrid mode"
    )

    args = parser.parse_args()

    retriever = initialize_simple_retriever(
        args.retriever_type, E5_INDEX_DIR, BM25_INDEX_DIR, args.top_k, args.alpha
    )

    ragent = SimpleFourAgentRAG(
        retriever,
        agent_model=args.model,
        n=args.n,
        index_dir=args.index_dir,
        max_workers=args.max_workers,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "debug"), exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Process single question
    if args.single_question:
        logger.info(f"Processing single question: {args.single_question}")
        start_time = time.time()

        try:
            cited_answer, debug_info = ragent.answer_query(args.single_question)
            process_time = time.time() - start_time

            result = {
                "id": f"single_question_simple_4agent_{args.retriever_type}",
                "question": args.single_question,
                "model_answer": cited_answer,
                "was_split": debug_info["was_split"],
                "sub_questions": debug_info["sub_questions"],
                "questions_processed": debug_info["questions_processed"],
                "total_citations": debug_info["total_citations"],
                "total_filtered_abstracts": debug_info["total_filtered_abstracts"],
                "abstracts_used": debug_info["abstracts_used"],
                "passages_used": debug_info["passages_used"],
                "document_metadata": debug_info["document_metadata"],
                "process_time": process_time,
                "retriever_type": args.retriever_type,
            }

            logger.info(f"Processing time: {process_time:.2f} seconds")
            logger.info(f"Citations used: {debug_info['total_citations']}")

            # Save result
            if args.output_format == "debug":
                debug_output_file = os.path.join(
                    args.output_dir,
                    "debug",
                    f"simple_4agent_single_{args.retriever_type}_debug_{timestamp}.json",
                )
                with open(debug_output_file, "w", encoding="utf-8") as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                logger.info(f"Debug result saved to {debug_output_file}")
            else:
                output_file = os.path.join(
                    args.output_dir,
                    f"simple_4agent_single_{args.retriever_type}_{timestamp}.json",
                )
                write_simple_result_to_json(result, output_file)

        except Exception as e:
            logger.error(f"Error processing question: {e}")
        finally:
            ragent.close()
            retriever.close()
        return

    # Process question file
    questions = load_questions(args.data_file)
    if not questions:
        logger.error("No questions found. Exiting.")
        return

    results = []
    for i, item in enumerate(questions):
        question_id = item.get("id", i + 1)
        logger.info(f"Processing question {i+1}/{len(questions)}: {item['question']}")
        start_time = time.time()

        try:
            cited_answer, debug_info = ragent.answer_query(item["question"])
            process_time = time.time() - start_time

            result = {
                "id": question_id,
                "question": item["question"],
                "model_answer": cited_answer,
                "was_split": debug_info["was_split"],
                "sub_questions": debug_info["sub_questions"],
                "questions_processed": debug_info["questions_processed"],
                "total_citations": debug_info["total_citations"],
                "total_filtered_abstracts": debug_info["total_filtered_abstracts"],
                "abstracts_used": debug_info["abstracts_used"],
                "passages_used": debug_info["passages_used"],
                "document_metadata": debug_info["document_metadata"],
                "process_time": process_time,
                "retriever_type": args.retriever_type,
            }
            results.append(result)

            logger.info(f"Processing time: {process_time:.2f} seconds")

            # Save debug info
            debug_output_file = os.path.join(
                args.output_dir,
                "debug",
                f"simple_4agent_question_{question_id}_{args.retriever_type}_debug_{timestamp}.json",
            )
            with open(debug_output_file, "w", encoding="utf-8") as f:
                json.dump(debug_info, f, indent=2, ensure_ascii=False)

        except Exception as e:
            logger.error(f"Error processing question {question_id}: {e}")

    # Clean up
    ragent.close()
    retriever.close()

    # Save all results
    random_num = "".join(random.choices(string.ascii_lowercase + string.digits, k=6))

    if results:
        if args.output_format == "jsonl":
            output_file = os.path.join(
                args.output_dir,
                f"simple_4agent_answers_{args.retriever_type}_{timestamp}_{random_num}.jsonl",
            )
            write_simple_results_to_jsonl(results, output_file)
        elif args.output_format == "json":
            for result in results:
                question_id = result["id"]
                output_file = os.path.join(
                    args.output_dir,
                    f"simple_4agent_answer_{question_id}_{args.retriever_type}_{random_num}.json",
                )
                write_simple_result_to_json(result, output_file)
        else:  # debug
            output_file = os.path.join(
                args.output_dir,
                "debug",
                f"simple_4agent_all_results_{args.retriever_type}_debug_{timestamp}.json",
            )
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info(f"Debug results saved to {output_file}")

    logger.info(f"Processed {len(results)} questions with simple 4-agent system.")

    if results:
        avg_time = sum(r["process_time"] for r in results) / len(results)
        avg_citations = sum(r["total_citations"] for r in results) / len(results)
        split_count = sum(1 for r in results if r["was_split"])

        logger.info(f"Average processing time: {avg_time:.2f} seconds")
        logger.info(
            f"Questions split: {split_count}/{len(results)} ({split_count/len(results)*100:.1f}%)"
        )
        logger.info(f"Average citations: {avg_citations:.1f}")


if __name__ == "__main__":
    main()
