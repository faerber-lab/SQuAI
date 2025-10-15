#!/usr/bin/env python3
"""
Enhanced 4-Agent RAG System with Question Splitting and Parallel Processing
- Agent 1: Question Splitter
- Agent 2: Answer Generator from abstracts
- Agent 3: Document Evaluator
- Agent 4: Final Answer Generator with citations
"""
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
import sqlite3
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
import multiprocessing as mp
from performance_monitor import monitor, time_block

# Import configuration
from config import E5_INDEX_DIR, BM25_INDEX_DIR, DB_PATH


# Your existing logging setup (unchanged)
def get_unique_log_filename():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    random_str = "".join(random.choices(string.ascii_lowercase + string.digits, k=6))
    return f"logs/enhanced_4agent_rag_{timestamp}_{random_str}.log"


os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(get_unique_log_filename()), logging.StreamHandler()],
)
logger = logging.getLogger("Enhanced_4Agent_RAG")

# Import the hybrid retriever components
from hybrid_retriever import Retriever


class QuestionSplitter:
    """
    Agent 1: Intelligent Question Splitting Agent
    Detects complex queries with multiple sub-questions and splits them appropriately
    """

    def __init__(self, agent_model):
        self.agent = agent_model
        logger.info("Agent 1 (Question Splitter) initialized")

    def _create_splitting_prompt(self, query: str) -> str:
        """Create prompt for question splitting analysis"""
        return f"""You are an intelligent question analyzer. Your task is to determine if a query contains multiple distinct sub-questions that would benefit from separate retrieval and research.

SPLITTING CRITERIA:
- Split if query contains multiple distinct topics connected by "and", "also", "what about"
- Split if query asks for comparisons between different concepts
- Split if query has multiple question words (what, how, why, when, where)
- Split if query asks about a concept AND its implications/effects/applications
- DO NOT split simple clarifications or related aspects of the same topic

Examples:
Query: "What is quantum computing and how is it used in cryptography?"
Split: YES
Sub-questions: ["What is quantum computing?", "How is quantum computing used in cryptography?"]

Query: "what is page rank algorithm and who invented it?"
Split: YES
Sub-questions: ["What is page rank algorithm?", "Who invented page rank algorithm?"]

Query: "What is reinforcement learning?"
Split: NO
Sub-questions: []

Query: "How does BERT work and what is GPT-3?"
Split: YES  
Sub-questions: ["How does BERT work?", "What is GPT-3?"]

Query: "What are neural networks and how do they learn and what are CNNs?"
Split: YES
Sub-questions: ["What are neural networks?", "How do neural networks learn?", "What are CNNs?"]

Query: "What is federated learning and its privacy implications?"
Split: YES
Sub-questions: ["What is federated learning?", "What are the privacy implications of federated learning?"]

Now analyze this query:
Query: "{query}"

Respond with ONLY this format:
Split: YES/NO
Sub-questions: [list of questions] (empty list if Split: NO)"""

    def analyze_and_split(self, query: str) -> Tuple[bool, List[str]]:
        """
        Analyze query and split into sub-questions if beneficial
        Always uses LLM for accurate analysis

        Returns:
            Tuple of (should_split: bool, sub_questions: List[str])
        """
        # Handle time_block if it exists (for compatibility)
        try:
            # Check if time_block is available
            if 'time_block' in globals():
                with time_block("agent1_question_splitting"):
                    return self._perform_split_analysis(query)
            else:
                return self._perform_split_analysis(query)
        except:
            # Fallback if time_block causes any issues
            return self._perform_split_analysis(query)

    def _perform_split_analysis(self, query: str) -> Tuple[bool, List[str]]:
        """
        Internal method to perform the actual split analysis using LLM
        """
        logger.info(f"Agent 1: Analyzing query for splitting: {query}")
        
        # Basic validation - only skip extremely short queries
        if len(query.strip()) < 10:  # Less than 10 characters is too short
            logger.info("Query too short for meaningful splitting")
            return False, []
        
        try:
            # Always use LLM for analysis (no heuristics)
            logger.info("Using LLM to analyze if query should be split...")
            
            prompt = self._create_splitting_prompt(query)
            response = self.agent.generate(prompt)
            
            # Parse response
            should_split, sub_questions = self._parse_splitting_response(response, query)
            
            if should_split:
                logger.info(f"Agent 1: LLM decided to split into {len(sub_questions)} sub-questions: {sub_questions}")
            else:
                logger.info("Agent 1: LLM decided no splitting needed")
            
            return should_split, sub_questions
            
        except Exception as e:
            logger.error(f"Error in LLM splitting analysis: {e}")
            # On error, don't split
            return False, []

    def _parse_splitting_response(self, response: str, original_query: str) -> Tuple[bool, List[str]]:
        """Parse the LLM response for splitting decision"""
        try:
            lines = response.strip().split('\n')
            should_split = False
            sub_questions = []

            for line in lines:
                line = line.strip()
                if line.startswith("Split:"):
                    should_split = "YES" in line.upper()
                elif line.startswith("Sub-questions:"):
                    # Extract list from the line
                    list_part = line.split(":", 1)[1].strip()
                    if list_part and list_part != "[]":
                        # Parse the list - handle both ["q1", "q2"] and simple comma-separated
                        try:
                            if list_part.startswith("[") and list_part.endswith("]"):
                                # JSON-like format
                                sub_questions = json.loads(list_part)
                            else:
                                # Comma-separated format
                                sub_questions = [
                                    q.strip().strip('"').strip("'")
                                    for q in list_part.split(",")
                                ]
                        except:
                            logger.warning(f"Failed to parse sub-questions: {list_part}")
                            sub_questions = []

            # Validation: ensure sub-questions are meaningful
            if should_split and sub_questions:
                valid_questions = []
                for q in sub_questions:
                    q = q.strip()
                    # Clean up and validate
                    if len(q) > 10:
                        # Add question mark if missing
                        if not q.endswith("?"):
                            q = q + "?"
                        valid_questions.append(q)

                if len(valid_questions) < 2:
                    logger.info("Not enough valid sub-questions, keeping original")
                    return False, []

                return True, valid_questions

            return False, []

        except Exception as e:
            logger.warning(f"Error parsing splitting response: {e}")
            return False, []

    def _quick_split_check(self, query: str) -> bool:
        """
        DEPRECATED: Keep for backward compatibility but not used
        This method is kept in case other parts of the code reference it
        """
        # Always return False since we're using LLM directly now
        return False

class PaperTitleExtractor:
    """
    Utility class for extracting paper titles from document text
    IMPROVED: Handles LevelDB storage format where title is on second line
    """

    @staticmethod
    def extract_title_from_text(doc_text: str, doc_id: str) -> str:
        """
        Extract paper title from document text using multiple patterns
        IMPROVED: Handles "Content for [paper_id]:\n[Title]" format from LevelDB
        """
        try:
            # Method 1: NEW - Handle LevelDB format: "Content for [paper_id]:\n[Title]"
            leveldb_pattern = r"Content for [^:]*:\s*\n([^\n]+)"
            match = re.search(leveldb_pattern, doc_text)
            if match:
                title_candidate = match.group(1).strip()
                # Validate it looks like a title (not abstract or other content)
                if (
                    len(title_candidate) > 10
                    and len(title_candidate) < 300
                    and not title_candidate.lower().startswith(
                        ("abstract:", "introduction:", "the abstract", "in this", "we ")
                    )
                ):

                    logger.debug(
                        f"Extracted title from LevelDB format: {title_candidate[:50]}..."
                    )
                    return title_candidate

            # Method 2: Look for title in first few lines (for direct title format)
            lines = doc_text.split("\n")
            for i, line in enumerate(lines[:5]):
                line = line.strip()

                # Skip empty lines and common headers
                if not line or line.lower().startswith(
                    ("content for", "time taken", "opening")
                ):
                    continue

                # Check if this line looks like a title
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
                    and not re.match(r"^\d+", line)  # Not starting with numbers
                    and not line.endswith(":")  # Not a section header
                    and line.count(" ") >= 2
                ):  # At least 3 words

                    logger.debug(f"Extracted title from line {i+1}: {line[:50]}...")
                    return line

            # Method 3: Look for "Content for [paper_id]:" pattern (legacy)
            content_pattern = r"Content for [^:]*:\s*\n([^\n]+)"
            match = re.search(content_pattern, doc_text)
            if match:
                title_candidate = match.group(1).strip()
                if len(title_candidate) > 10 and len(title_candidate) < 300:
                    title_candidate = re.sub(r'^["\']|["\']$', "", title_candidate)
                    title_candidate = re.sub(r"^\W+|\W+$", "", title_candidate)
                    if len(title_candidate) > 10:
                        return title_candidate

            # Method 4: Look for "Title. {" pattern
            title_brace_pattern = r"^([^.]+)\.\s*\{"
            match = re.search(title_brace_pattern, doc_text.strip(), re.MULTILINE)
            if match:
                title_candidate = match.group(1).strip()
                if (
                    len(title_candidate) > 10
                    and len(title_candidate) < 300
                    and not title_candidate.lower().startswith(
                        ("the ", "this ", "in ", "we ", "abstract", "introduction")
                    )
                ):
                    title_candidate = re.sub(r'^["\']|["\']$', "", title_candidate)
                    if len(title_candidate) > 10:
                        return title_candidate

            # Method 5: Extract from cleaned first sentence
            clean_text = re.sub(r"\{[^}]*\}", "", doc_text)
            clean_text = re.sub(r"Content for [^:]+:\s*", "", clean_text)
            clean_text = clean_text.strip()

            first_sentence = clean_text.split("\n")[0].strip()
            if ". {" in first_sentence:
                first_sentence = first_sentence.split(". {")[0].strip()
            elif ". " in first_sentence and len(first_sentence.split(". ")[0]) < 200:
                first_sentence = first_sentence.split(". ")[0].strip()

            if (
                len(first_sentence) > 15
                and len(first_sentence) < 300
                and not first_sentence.lower().startswith(
                    (
                        "content for",
                        "time taken",
                        "opening",
                        "the ",
                        "this ",
                        "in ",
                        "we ",
                        "abstract",
                        "introduction",
                    )
                )
                and not re.match(r"^\d+", first_sentence)
            ):
                return first_sentence

            # Method 6: Try JSON metadata
            if "{" in doc_text and '"title"' in doc_text:
                try:
                    json_match = re.search(r'\{.*?"title".*?\}', doc_text, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(0)
                        metadata = json.loads(json_str)
                        if "title" in metadata and len(metadata["title"]) > 10:
                            return metadata["title"]
                except:
                    pass

            # Fallback: use first substantial line
            for line in lines[:5]:
                line = line.strip()
                if len(line) > 15 and len(line) < 200:
                    return line[:150] + "..." if len(line) > 150 else line

            return f"Document {doc_id}"

        except Exception as e:
            logger.debug(f"Error extracting title for {doc_id}: {e}")
            return f"Document {doc_id}"

    @staticmethod
    def format_title_for_log(title: str, max_length: int = 80) -> str:
        """Format title for logging with length limit"""
        if len(title) <= max_length:
            return title
        return title[: max_length - 3] + "..."

    @staticmethod
    def extract_paper_sections(
        full_text: str, max_chars_per_section: int = 10000
    ) -> Dict[str, str]:
        """
        Extract key sections from full paper text for better context utilization

        Args:
            full_text: The full paper text
            max_chars_per_section: Limit for introduction and conclusion extraction (abstract is kept full)

        Returns:
            Dict with 'title', 'abstract', 'introduction', 'conclusion' keys
            Note: Abstract is returned in full (no artificial limits)
        """
        sections = {}

        # Extract title (first line after "Content for")
        title_match = re.search(r"Content for [^:]*:\s*\n([^\n]+)", full_text)
        if title_match:
            sections["title"] = title_match.group(1).strip()

        # Extract abstract (keep full abstract - they're naturally short and important)
        abstract_match = re.search(
            r"abstract:\s*(.+?)(?:\n\n|\nintroduction|\nrelated work|\nmethodology)",
            full_text,
            re.IGNORECASE | re.DOTALL,
        )
        if abstract_match:
            abstract_text = abstract_match.group(1).strip()
            # Keep full abstract - no artificial limits since they're naturally concise
            sections["abstract"] = abstract_text

        # Extract introduction (can be long and informative)
        intro_match = re.search(
            r"(?:^|\n)introduction[:\n]\s*(.+?)(?:\n\n[A-Z]|\nrelated work|\nmethodology|\nconclusion)",
            full_text,
            re.IGNORECASE | re.DOTALL,
        )
        if intro_match:
            intro_text = intro_match.group(1).strip()
            sections["introduction"] = intro_text[:max_chars_per_section]

        # Extract conclusion (moderate length, important summary)
        conclusion_match = re.search(
            r"(?:^|\n)conclusion[s]?[:\n]\s*(.+?)(?:\n\n[A-Z]|\nreferences|\nacknowledgments|$)",
            full_text,
            re.IGNORECASE | re.DOTALL,
        )
        if conclusion_match:
            conclusion_text = conclusion_match.group(1).strip()
            sections["conclusion"] = conclusion_text[:max_chars_per_section]

        return sections


class EnhancedCitationHandler:
    """Enhanced citation handler with proper metadata extraction and context passages"""

    def __init__(self, index_dir: str = "test_index"):
        self.doc_to_citation = {}
        self.citation_to_doc = {}
        self.next_citation_num = 1
        self.index_dir = Path(index_dir)

        # Load arXiv papers for better metadata
        self.arxiv_papers = self._load_arxiv_papers()

        # Connect to metadata database
        self.metadata_db = self._connect_metadata_db()

    def _connect_metadata_db(self):
        """Connect to metadata database"""
        try:
            import sqlite3

            db_path = self.index_dir / "index_store.db"
            conn = sqlite3.connect(str(db_path))
            conn.row_factory = sqlite3.Row
            return conn
        except:
            return None

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

                            # Extract year from versions
                            year = "Unknown"
                            versions = paper.get("versions", [])
                            if versions:
                                created = versions[0].get("created", "")
                                year_match = re.search(r"(\d{4})", created)
                                if year_match:
                                    year = year_match.group(1)

                            # Format authors properly
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
        except:
            return {}

    def _extract_document_title_improved(self, doc_text: str, doc_id: str) -> str:
        """Use the PaperTitleExtractor for consistency"""
        return PaperTitleExtractor.extract_title_from_text(doc_text, doc_id)

    def _extract_paper_info(
        self, doc_text: str, doc_id: str, metadata: Dict = None
    ) -> Dict:
        """Enhanced paper metadata extraction with improved title extraction"""
        paper_info = {
            "title": "Unknown Title",
            "authors": "Unknown",
            "venue": "arXiv",
            "year": "Unknown",
            "paper_id": doc_id,
        }

        try:
            # Use improved title extraction
            paper_info["title"] = self._extract_document_title_improved(
                doc_text, doc_id
            )

            # Extract from JSON in document text
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

                        # Extract year from versions
                        if "versions" in paper_data and paper_data["versions"]:
                            created = paper_data["versions"][0].get("created", "")
                            year_match = re.search(r"(\d{4})", created)
                            if year_match:
                                paper_info["year"] = year_match.group(1)

                        logger.debug(
                            f"Extracted metadata from JSON in text for {doc_id}"
                        )

                except Exception as e:
                    logger.debug(f"JSON parsing failed for {doc_id}: {e}")

            # Match with loaded arXiv papers by paper_id
            if doc_id in self.arxiv_papers:
                arxiv_data = self.arxiv_papers[doc_id]
                # Update info but keep improved title if it's better
                if (
                    paper_info["title"] == "Unknown Title"
                    or paper_info["title"] == f"Document {doc_id}"
                ):
                    paper_info["title"] = arxiv_data["title"]
                if paper_info["authors"] == "Unknown":
                    paper_info["authors"] = arxiv_data["authors"]
                if paper_info["year"] == "Unknown":
                    paper_info["year"] = arxiv_data["year"]
                logger.debug(
                    f"Enhanced metadata for {doc_id} from arXiv papers database"
                )

            # Final cleanup
            if len(paper_info["title"]) > 150:
                paper_info["title"] = paper_info["title"][:150] + "..."

            # Ensure we have a paper_id
            if not paper_info["paper_id"]:
                paper_info["paper_id"] = doc_id

        except Exception as e:
            logger.debug(f"Error extracting metadata for {doc_id}: {e}")

        return paper_info

    def _basic_text_cleaning(self, text: str) -> str:
        """Basic text cleaning for citation context"""
        # Remove JSON-like section markers
        text = re.sub(r"'section':\s*'[^']*',\s*'text':\s*'", "", text)
        text = re.sub(r"^\s*\{.*?'text':\s*'", "", text)
        text = re.sub(r"\{[^}]*\}", "", text)

        # Remove technical markup
        text = re.sub(r"\{\{[^}]+\}\}", "[REF]", text)
        text = re.sub(r"\$[^$]+\$", "[MATH]", text)
        text = re.sub(r"\\[a-zA-Z]+\{[^}]*\}", "[LATEX]", text)

        # Clean whitespace
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"\n\s*\n", "\n\n", text)

        return text.strip()

    def _extract_context_passage(
        self, answer_text: str, document_text: str, citation_num: int
    ) -> str:
        """Extract specific sentence(s) used in the answer plus context"""
        try:
            # Clean the document text first
            try:
                from text_cleaner import DocumentTextCleaner

                cleaner = DocumentTextCleaner()
                clean_doc_text = cleaner.clean_for_citation_matching(document_text)
            except ImportError:
                clean_doc_text = self._basic_text_cleaning(document_text)

            # Find all sentences in the clean document
            sentences = re.split(r"[.!?]+", clean_doc_text)
            sentences = [s.strip() for s in sentences if len(s.strip()) > 15]

            # Look for content that appears in the answer near this citation
            citation_pattern = f"\\[{citation_num}\\]"
            citation_matches = list(re.finditer(citation_pattern, answer_text))

            if not citation_matches:
                # Fallback: return first few clean sentences
                return (
                    ". ".join(sentences[:2]) + "."
                    if sentences
                    else clean_doc_text[:200] + "..."
                )

            # For each citation, find the preceding text that likely came from this document
            relevant_sentences = set()

            for match in citation_matches:
                # Get text before this citation (up to 150 chars back)
                start_pos = max(0, match.start() - 150)
                context_text = answer_text[start_pos : match.start()].strip()

                # Find the sentence in context_text that likely came from the document
                context_sentences = re.split(r"[.!?]+", context_text)

                for context_sent in context_sentences[
                    -2:
                ]:  # Last 1-2 sentences before citation
                    if len(context_sent.strip()) < 15:
                        continue

                    # Find similar sentences in the document
                    context_words = set(context_sent.lower().split())

                    for i, doc_sent in enumerate(sentences):
                        doc_words = set(doc_sent.lower().split())

                        # Check word overlap
                        overlap = len(context_words.intersection(doc_words))
                        overlap_ratio = overlap / max(len(context_words), 1)

                        if overlap_ratio > 0.25 or overlap > 4:  # Good match
                            # Add this sentence plus context (±1 sentence)
                            start_idx = max(0, i - 1)
                            end_idx = min(len(sentences), i + 2)

                            for j in range(start_idx, end_idx):
                                relevant_sentences.add(j)

            if relevant_sentences:
                # Sort and build context passage
                sorted_indices = sorted(relevant_sentences)
                context_parts = [sentences[i] for i in sorted_indices]
                result = ". ".join(context_parts) + "."

                # Limit length
                if len(result) > 500:
                    result = result[:500] + "..."

                return result

            # Fallback: return beginning of clean document
            fallback = ". ".join(sentences[:2]) + "."
            return fallback if len(fallback) < 300 else fallback[:300] + "..."

        except Exception as e:
            logger.debug(f"Error extracting context passage: {e}")
            # Simple fallback with basic cleaning
            clean_text = self._basic_text_cleaning(document_text)
            return clean_text[:200] + "..." if len(clean_text) > 200 else clean_text

    def format_references(self, answer_text: str = None) -> str:
        """Format references with proper metadata and context passages"""
        if not self.citation_to_doc:
            return ""

        # Get all available citations
        citations_to_show = set(self.citation_to_doc.keys())

        # If answer text provided, filter to only used citations
        if answer_text:
            citation_matches = re.findall(r"\[(\d+)\]", answer_text)
            used_citations = set(int(num) for num in citation_matches)

            if used_citations:
                citations_to_show = used_citations.intersection(
                    set(self.citation_to_doc.keys())
                )

        if not citations_to_show:
            return ""

        references = []

        for citation_num in sorted(citations_to_show):
            reference=[]
            doc_info = self.citation_to_doc[citation_num]
            paper_info = doc_info["paper_info"]

            # Format academic reference
            reference.append(f"[{citation_num}]")

            # Add title in quotes
            title = paper_info["title"].replace('Title:', "").replace('"', "").replace("'", "")
            reference.append(title)

            # Add venue and year with paper ID
            if paper_info.get("paper_id") and paper_info["paper_id"] != "Unknown":
                if str(paper_info["paper_id"]).startswith("arXiv:"):
                    reference.append(paper_info['paper_id'])
                else:
                    reference.append(f"arXiv:{paper_info['paper_id']}")
            else:
                reference.append(f"{paper_info['venue']}")

            # Add context passage with actual sentences used
            if answer_text:
                context_passage = self._extract_context_passage(
                    answer_text, doc_info["text"], citation_num
                )
            else:
                context_passage = (
                    doc_info["text"][:300] + "..."
                    if len(doc_info["text"]) > 300
                    else doc_info["text"]
                )

            reference.append(context_passage)

            references.append(reference)

        return references

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
            logger.debug(
                f"Added document {doc_id} as citation [{citation_num}]: {paper_info['title'][:50]}..."
            )
            return citation_num
        else:
            return self.doc_to_citation[doc_id]

    def get_citation_map(self) -> Dict[str, int]:
        """Get mapping from doc_id to citation number"""
        return self.doc_to_citation.copy()


class Enhanced4AgentRAG:
    """
    Enhanced 4-Agent RAG System with Question Splitting, Parallel Processing, and Context Management
    """

    def __init__(
        self,
        retriever,
        agent_model=None,
        n=0.0,
        falcon_api_key=None,
        index_dir="test_index",
        max_workers=4,
        max_context_chars=35000,
    ):
        """Initialize with enhanced 4-agent architecture and context management"""

        self.retriever = retriever
        self.n = n
        self.index_dir = index_dir
        self.max_workers = max_workers
        self.max_context_chars = max_context_chars  # Conservative limit for Falcon-10B

        logger.info(f"Context limit set to {max_context_chars} characters")

        # Initialize agents
        if isinstance(agent_model, str):
            if "falcon" in agent_model.lower() and falcon_api_key:
                from api_agent import FalconAgent

                self.agent1 = FalconAgent(falcon_api_key)  # Question Splitter
                self.agent2 = FalconAgent(falcon_api_key)  # Answer Generator
                self.agent3 = FalconAgent(falcon_api_key)  # Document Evaluator
                self.agent4 = FalconAgent(falcon_api_key)  # Final Answer Generator
                logger.info("Using Falcon agents with API for all four agent roles")
            else:
                from local_agent import LLMAgent

                self.agent1 = LLMAgent(agent_model)  # Question Splitter
                self.agent2 = LLMAgent(agent_model)  # Answer Generator
                self.agent3 = LLMAgent(agent_model)  # Document Evaluator
                self.agent4 = LLMAgent(agent_model)  # Final Answer Generator
                logger.info(f"Using local LLM agents with model {agent_model}")
        else:
            self.agent1 = agent_model  # Question Splitter
            self.agent2 = agent_model  # Answer Generator
            self.agent3 = agent_model  # Document Evaluator
            self.agent4 = agent_model  # Final Answer Generator
            logger.info("Using pre-initialized agent for all four agent roles")

        # Initialize question splitter
        self.question_splitter = QuestionSplitter(self.agent1)

        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

        # Enhanced pre-warming
        logger.info("Enhanced 4-agent pre-warming...")
        try:
            # Warm up retriever
            dummy_abstracts = self.retriever.retrieve_abstracts("test", top_k=1)
            logger.info("Retriever pre-warmed")

            # Warm up agents
            if hasattr(self.agent1, "generate"):
                self.agent1.generate("test")
                logger.info("All agents pre-warmed")

        except Exception as e:
            logger.warning(f"Pre-warming had issues: {e}")

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

    def _prepare_documents_for_agent4(
        self,
        full_texts: List[Tuple[str, str]],
        citation_handler,
        was_split: bool = False,
    ) -> List[str]:
        """
        Prepare documents for Agent 4 with dynamic context length management

        Args:
            full_texts: List of (document_text, doc_id) tuples
            citation_handler: Citation handler instance
            was_split: Whether the original question was split into sub-questions

        Returns:
            List of formatted document strings ready for the prompt
        """
        docs_with_citations = []
        total_chars = 0
        documents_used = 0

        # Dynamic context allocation - top + bottom extraction approach
        if was_split:
            # Conservative: Target ~4K total per paper
            top_chars = 2500  # Top of paper (title, abstract, intro start)
            bottom_chars = 1500  # Bottom of paper (conclusion, results)
            strategy = "CONSERVATIVE (split questions)"
            target_per_paper = "~4K"
        else:
            # Generous: Target ~8K total per paper
            top_chars = 5000  # More from top (title, abstract, intro)
            bottom_chars = 3000  # More from bottom (conclusion, results)
            strategy = "GENEROUS (single question)"
            target_per_paper = "~8K"

        logger.info(
            f"Preparing documents for Agent 4 (context limit: {self.max_context_chars} chars)"
        )
        logger.info(
            f"   Context strategy: {strategy} - targeting {target_per_paper} chars per paper"
        )
        logger.info(
            f"   Extraction: TOP({top_chars} chars) + BOTTOM({bottom_chars} chars) + Title"
        )

        for i, (doc_text, doc_id) in enumerate(full_texts):
            # New approach: Extract from top and bottom of paper
            condensed_content = []

            # Extract title first (if available)
            title = PaperTitleExtractor.extract_title_from_text(doc_text, doc_id)
            if title and not title.startswith("Document "):
                condensed_content.append(f"Title: {title}")

            # Remove "Content for [paper_id]:" line and other metadata for cleaner extraction
            clean_text = doc_text
            # Remove the "Content for" line
            clean_text = re.sub(r"Content for [^:]*:\s*\n", "", clean_text)
            # Remove any leading whitespace/newlines
            clean_text = clean_text.strip()

            # TOP EXTRACTION: Get beginning of paper (naturally includes abstract, intro start)
            top_text = clean_text[:top_chars]
            if len(clean_text) > top_chars:
                # Find a good breaking point (end of sentence)
                break_point = top_text.rfind(". ")
                if (
                    break_point > top_chars * 0.8
                ):  # If we find a sentence end in the last 20%
                    top_text = top_text[: break_point + 1]
                else:
                    top_text += "..."

            condensed_content.append(f"[TOP {len(top_text)} chars]: {top_text}")

            # BOTTOM EXTRACTION: Get end of paper (naturally includes conclusion, results)
            if (
                len(clean_text) > top_chars + 100
            ):  # Only add bottom if there's enough remaining content
                bottom_text = clean_text[-bottom_chars:]
                if len(clean_text) > bottom_chars:
                    # Find a good starting point (beginning of sentence)
                    start_point = bottom_text.find(". ")
                    if (
                        start_point > 0 and start_point < bottom_chars * 0.2
                    ):  # If we find sentence start in first 20%
                        bottom_text = bottom_text[start_point + 2 :]  # +2 to skip ". "
                    else:
                        bottom_text = "..." + bottom_text

                condensed_content.append(
                    f"[BOTTOM {len(bottom_text)} chars]: {bottom_text}"
                )

            condensed_text = "\n\n".join(condensed_content)

            # Check if adding this document would exceed context limit
            estimated_doc_size = len(condensed_text) + 200  # +200 for formatting

            if (
                total_chars + estimated_doc_size > self.max_context_chars
                and documents_used > 0
            ):
                logger.info(
                    f"Context limit reached. Using {documents_used} out of {len(full_texts)} documents"
                )
                break

            # Add document with citation
            citation_num = citation_handler.add_document(condensed_text, doc_id)

            # Get paper info for better document labeling
            paper_info = citation_handler.citation_to_doc[citation_num]["paper_info"]
            doc_title = (
                paper_info["title"][:80] + "..."
                if len(paper_info["title"]) > 80
                else paper_info["title"]
            )

            formatted_doc = (
                f'Document [{citation_num}] - "{doc_title}":\n{condensed_text}'
            )
            docs_with_citations.append(formatted_doc)

            total_chars += estimated_doc_size
            documents_used += 1

            logger.info(
                f"  Added doc [{citation_num}]: {doc_title[:60]}... ({len(condensed_text)} chars)"
            )

        logger.info(
            f"Total context size: {total_chars} chars (~{self._estimate_tokens(str(total_chars))} tokens)"
        )
        logger.info(f"Using {documents_used}/{len(full_texts)} documents for Agent 4")

        return docs_with_citations

    def _create_agent4_prompt_with_citations(
        self, original_query, full_texts, citation_handler, was_split: bool = False
    ):
        """Agent-4 prompt with context-aware document preparation"""

        # Prepare documents with dynamic context management based on question splitting
        docs_with_citations = self._prepare_documents_for_agent4(
            full_texts, citation_handler, was_split
        )

        docs_text = "\n\n" + "=" * 50 + "\n\n".join(docs_with_citations)

        # Count available citation numbers
        available_citations = [str(i) for i in range(1, len(docs_with_citations) + 1)]
        citation_examples = ", ".join(available_citations)

        return f"""You are an accurate and reliable AI assistant. Answer questions based ONLY on the provided documents with proper academic citations.

STRICT CITATION REQUIREMENTS - YOU MUST FOLLOW THESE:
1. You MUST add [{citation_examples}] after EVERY claim you make
2. Every sentence that contains factual information MUST end with a citation
3. If you mention ANY concept, method, or fact, cite the document immediately
4. Use ONLY the document numbers shown: [{citation_examples}]
5. Do NOT write ANY sentence without a citation number
6. Use MULTIPLE different documents - don't just cite [1] repeatedly
7. Do NOT add a references section - it will be added automatically
8. EXAMPLE: "Machine learning involves pattern recognition [1]. Neural networks are a popular approach [2]. Deep learning has shown success in many domains [3]."

WRONG (no citations): "Machine learning is a powerful technique."
CORRECT (with citations): "Machine learning is a powerful technique for pattern recognition [1]."

WRONG (only one citation): "ML works by finding patterns [1]. It uses algorithms [1]. It requires data [1]."
CORRECT (multiple citations): "ML works by finding patterns [1]. It uses algorithms [2]. It requires data [3]."

Documents:
{docs_text}

Question: {original_query}

Remember: Use information from MULTIPLE documents and cite each one appropriately with [{citation_examples}]. Answer:"""

    def _log_retrieved_papers(
        self, query: str, retrieved_abstracts: List[Tuple], phase: str = "RETRIEVAL"
    ):
        """Log the titles of retrieved papers with improved title extraction"""
        if not retrieved_abstracts:
            logger.info(f"{phase}: No papers retrieved for query: {query[:50]}...")
            return

        logger.info(
            f"{phase}: Retrieved {len(retrieved_abstracts)} papers for query: {query[:50]}..."
        )
        logger.info("=" * 80)

        for i, (abstract_text, doc_id) in enumerate(retrieved_abstracts, 1):
            # Extract title using improved utility
            title = PaperTitleExtractor.extract_title_from_text(abstract_text, doc_id)
            formatted_title = PaperTitleExtractor.format_title_for_log(
                title, max_length=70
            )

            logger.info(f"  [{i:2d}] {formatted_title}")
            logger.info(f"       Doc ID: {doc_id}")

        logger.info("=" * 80)

    def _log_filtered_papers(
        self, query: str, filtered_abstracts: List[Tuple], scores: List[float]
    ):
        """Log the titles of papers that passed Agent 3 filtering"""
        if not filtered_abstracts:
            logger.info(
                f"FILTERING: No papers passed Agent 3 filter for query: {query[:50]}..."
            )
            return

        logger.info(
            f"FILTERING: {len(filtered_abstracts)} papers passed Agent 3 filter for query: {query[:50]}..."
        )
        logger.info("=" * 80)

        # Sort by score for display
        combined = list(zip(filtered_abstracts, scores))
        combined.sort(key=lambda x: x[1], reverse=True)

        for i, ((abstract_text, doc_id, _), score) in enumerate(combined, 1):
            # Extract title using improved utility
            title = PaperTitleExtractor.extract_title_from_text(abstract_text, doc_id)
            formatted_title = PaperTitleExtractor.format_title_for_log(
                title, max_length=65
            )

            logger.info(f"  ✓ [{i:2d}] {formatted_title} (score: {score:.3f})")
            logger.info(f"        Doc ID: {doc_id}")

        logger.info("=" * 80)

    def _log_context_usage(
        self, full_texts: List[Tuple], docs_used: int, was_split: bool = False
    ):
        """Log context usage statistics with dynamic strategy info"""
        total_chars = sum(len(text) for text, _ in full_texts)
        avg_chars = total_chars // len(full_texts) if full_texts else 0

        strategy = (
            "CONSERVATIVE (split questions)"
            if was_split
            else "GENEROUS (single question)"
        )
        chars_per_paper = (
            "TOP(2.5K)+BOTTOM(1.5K)" if was_split else "TOP(5K)+BOTTOM(3K)"
        )

        logger.info(f"   CONTEXT USAGE [{strategy}]:")
        logger.info(f"   Available papers: {len(full_texts)}")
        logger.info(f"   Papers sent to Agent 4: {docs_used}")
        logger.info(f"   Total characters available: {total_chars:,}")
        logger.info(f"   Average per paper: {avg_chars:,} chars")
        logger.info(f"   Context limit: {self.max_context_chars:,} chars")
        logger.info(f"   Strategy: {chars_per_paper}")

        if total_chars > self.max_context_chars:
            logger.info(
                f"Full papers exceed context limit by {total_chars - self.max_context_chars:,} chars"
            )
            logger.info(f"Using condensed sections to fit within limits")

    def _process_single_question(self, query: str, db=None) -> Tuple[List[Tuple], List]:
        """Process a single question and return (abstracts, filtered_documents)"""

        # PHASE 1: Retrieve ABSTRACTS for Agent2 & Agent3 filtering
        with time_block(f"retrieve_abstracts_{query[:20]}"):
            logger.info(f"Retrieving abstracts for: {query[:50]}...")
            retrieved_abstracts = self.retriever.retrieve_abstracts(query, top_k=5)

            # ✨ NEW: Log retrieved papers with titles
            self._log_retrieved_papers(query, retrieved_abstracts, "RETRIEVAL")

        # Step 2: Agent-2 generates answers from ABSTRACTS
        with time_block(f"agent2_generation_{query[:20]}"):
            logger.info(
                f"Agent-2 generating answers from abstracts for: {query[:50]}..."
            )
            doc_answers = []
            for abstract_text, doc_id in tqdm(retrieved_abstracts):
                prompt = self._create_agent2_prompt(query, abstract_text)
                answer = self.agent2.generate(prompt)
                doc_answers.append((abstract_text, doc_id, answer))

        # Step 3: Agent-3 evaluates documents using ABSTRACTS
        with time_block(f"agent3_evaluation_{query[:20]}"):
            logger.info(f"Agent-3 evaluating abstracts for: {query[:50]}...")
            scores = []
            for abstract_text, doc_id, answer in tqdm(doc_answers):
                prompt = self._create_agent3_prompt(query, abstract_text, answer)
                log_probs = self.agent3.get_log_probs(prompt, ["Yes", "No"])
                score = log_probs["Yes"] - log_probs["No"]
                scores.append(score)

        # Step 4: Calculate adaptive judge bar
        tau_q = np.mean(scores)
        sigma = np.std(scores)
        adjusted_tau_q = tau_q - self.n * sigma
        logger.info(
            f"Adaptive judge bar for '{query[:30]}...': tau_q={tau_q:.4f}, adjusted: {adjusted_tau_q:.4f}"
        )

        # Step 5: Filter documents based on abstract evaluation
        filtered_doc_ids = []
        filtered_abstracts = []
        for i, (abstract_text, doc_id, _) in enumerate(doc_answers):
            if scores[i] >= adjusted_tau_q:
                filtered_doc_ids.append(doc_id)
                filtered_abstracts.append((abstract_text, doc_id, scores[i]))

        filtered_abstracts.sort(key=lambda x: x[2], reverse=True)

        # ✨ NEW: Log filtered papers with titles and scores
        self._log_filtered_papers(
            query, filtered_abstracts, [x[2] for x in filtered_abstracts]
        )

        return retrieved_abstracts, filtered_doc_ids

    def answer_query(self, query, db=None, choices=None, should_split=None, sub_questions=None):
        """
        ENHANCED: Process query with 4-agent approach, question splitting, and parallel processing
        """
        with time_block("total_4agent_processing"):
            logger.info(f"Processing query with enhanced 4-agent approach: {query}")

            # Initialize enhanced citation handler
            citation_handler = EnhancedCitationHandler(self.index_dir)
            
            # If should_split and sub_questions are not provided, analyze the query
            if should_split is None or sub_questions is None:
                logger.info("Analyzing query for splitting...")
                should_split, sub_questions = self.question_splitter.analyze_and_split(query)
            
            if should_split and sub_questions:
                logger.info(f"Processing {len(sub_questions)} sub-questions in parallel")
                questions_to_process = sub_questions
            else:
                logger.info("Processing single question")
                questions_to_process = [query]

            # PHASE 2: Parallel Processing of Questions
            all_filtered_doc_ids = []

            if len(questions_to_process) > 1:
                # Parallel processing using thread pool
                with time_block("parallel_question_processing"):
                    logger.info(
                        f"Processing {len(questions_to_process)} questions in parallel"
                    )

                    # Submit all questions for parallel processing
                    future_to_question = {}
                    for sub_query in questions_to_process:
                        future = self.executor.submit(
                            self._process_single_question, sub_query, db
                        )
                        future_to_question[future] = sub_query

                    # Collect results
                    for future in as_completed(future_to_question):
                        sub_query = future_to_question[future]
                        try:
                            retrieved_abstracts, filtered_doc_ids = future.result()
                            all_filtered_doc_ids.extend(filtered_doc_ids)
                            logger.info(
                                f"Completed processing: {sub_query[:50]}... -> {len(filtered_doc_ids)} docs"
                            )
                        except Exception as e:
                            logger.error(
                                f"Error processing sub-question '{sub_query}': {e}"
                            )
            else:
                # Single question processing
                retrieved_abstracts, filtered_doc_ids = self._process_single_question(
                    questions_to_process[0], db
                )
                all_filtered_doc_ids = filtered_doc_ids

            # Remove duplicates while preserving order
            seen = set()
            unique_filtered_doc_ids = []
            for doc_id in all_filtered_doc_ids:
                if doc_id not in seen:
                    seen.add(doc_id)
                    unique_filtered_doc_ids.append(doc_id)

            logger.info(
                f"Total unique filtered documents: {len(unique_filtered_doc_ids)}"
            )

            # PHASE 3: Get FULL TEXTS for Agent4
            with time_block("get_full_texts"):
                logger.info("Retrieving FULL texts for final answer generation...")
                if unique_filtered_doc_ids:
                    full_texts = self.retriever.get_full_texts(
                        unique_filtered_doc_ids, db=db
                    )

                    # Enhanced logging with context awareness
                    logger.info(
                        f"FINAL ANSWER GENERATION: Retrieved {len(full_texts)} papers:"
                    )
                    logger.info("=" * 80)
                    for i, (doc_text, doc_id) in enumerate(full_texts, 1):
                        title = PaperTitleExtractor.extract_title_from_text(
                            doc_text, doc_id
                        )
                        formatted_title = PaperTitleExtractor.format_title_for_log(
                            title, max_length=70
                        )
                        char_count = len(doc_text)
                        logger.info(
                            f"[{i:2d}] {formatted_title} ({char_count:,} chars)"
                        )
                        logger.info(f"Doc ID: {doc_id}")
                    logger.info("=" * 80)

                    # Log context usage
                    estimated_docs_used = min(
                        len(full_texts),
                        self.max_context_chars // (4000 if should_split else 8000),
                    )
                    self._log_context_usage(
                        full_texts, estimated_docs_used, should_split
                    )

                else:
                    logger.warning("No documents passed the filter, using fallback")
                    # Fallback to some documents from the original query
                    fallback_abstracts, fallback_ids = self._process_single_question(
                        query, db
                    )
                    full_texts = self.retriever.get_full_texts(fallback_ids[:3], db=db)
                    if not full_texts:
                        # Last resort: use abstracts
                        full_texts = [
                            (abstract_text, doc_id)
                            for abstract_text, doc_id in fallback_abstracts[:3]
                        ]

                    # Log fallback papers
                    logger.info(f"FALLBACK: Using {len(full_texts)} papers:")
                    for i, (doc_text, doc_id) in enumerate(full_texts, 1):
                        title = PaperTitleExtractor.extract_title_from_text(
                            doc_text, doc_id
                        )
                        formatted_title = PaperTitleExtractor.format_title_for_log(
                            title, max_length=70
                        )
                        logger.info(f"[{i:2d}] {formatted_title}")

            # PHASE 4: Agent-4 generates final answer with context management
            with time_block("agent4_generation"):
                strategy_info = (
                    "CONSERVATIVE (split questions)"
                    if should_split
                    else "GENEROUS (single question)"
                )
                logger.info(
                    f"Agent-4 generating final answer with context-aware citations... [{strategy_info}]"
                )
                prompt = self._create_agent4_prompt_with_citations(
                    query, full_texts, citation_handler, should_split
                )
                raw_answer = self.agent4.generate(prompt)

            # Generate references with enhanced context passages
            references = citation_handler.format_references(raw_answer)

            # Remove any references Agent-4 might have added
            if "Reference" in raw_answer:
                raw_answer = re.split(r"References", raw_answer)[0]

            citation_map = citation_handler.get_citation_map()

            # Enhanced debug info
            debug_info = {
                "original_query": query,
                "was_split": should_split,
                "sub_questions": sub_questions if should_split else [],
                "questions_processed": len(questions_to_process),
                "total_filtered_docs": len(unique_filtered_doc_ids),
                "full_texts_retrieved": len(full_texts),
                "total_citations": len(citation_map),
                "citation_map": citation_map,
                "passages_used": self._extract_passages_used(
                    raw_answer, citation_handler
                ),
                "document_metadata": self._extract_document_metadata(citation_handler),
                "context_stats": {
                    "max_context_chars": self.max_context_chars,
                    "total_chars_available": sum(len(text) for text, _ in full_texts),
                    "docs_available": len(full_texts),
                    "estimated_docs_used": min(
                        len(full_texts),
                        self.max_context_chars // (4000 if should_split else 8000),
                    ),
                    "strategy": (
                        "CONSERVATIVE (split questions)"
                        if should_split
                        else "GENEROUS (single question)"
                    ),
                    "chars_per_paper_limit": (
                        "TOP(2.5K)+BOTTOM(1.5K)"
                        if should_split
                        else "TOP(5K)+BOTTOM(3K)"
                    ),
                },
                "performance_stats": (
                    monitor.get_stats() if hasattr(monitor, "get_stats") else {}
                ),
            }

        return raw_answer, references, debug_info

    def _extract_passages_used(
        self, answer_text: str, citation_handler: EnhancedCitationHandler
    ):
        """Extract the specific passages used in the answer"""
        # Find all citations in the answer
        citation_matches = re.findall(r"\[(\d+)\]", answer_text)
        used_citations = set(int(num) for num in citation_matches)

        passages_used = []
        for citation_num in used_citations:
            if citation_num in citation_handler.citation_to_doc:
                doc_info = citation_handler.citation_to_doc[citation_num]

                # Extract context passage for this citation
                context_passage = citation_handler._extract_context_passage(
                    answer_text, doc_info["text"], citation_num
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
                    }
                )

        return passages_used

    def _extract_document_metadata(self, citation_handler: EnhancedCitationHandler):
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
        logger.info("Enhanced 4-Agent RAG system closed")


def initialize_retriever(
    retriever_type: str,
    e5_index_dir: str,
    bm25_index_dir: str,
    db_path: str,
    top_k: int,
    alpha: float = 0.65,
    db=None,
):
    """Initialize the retriever with strategy and alpha support"""
    logger.info(f"Initializing {retriever_type} retriever with alpha={alpha}...")
    return Retriever(
        e5_index_dir, bm25_index_dir, top_k=top_k, strategy=retriever_type, alpha=alpha
    )


# Your existing utility functions (unchanged)
def load_datamorgana_questions(file_path):
    """Load questions from file"""
    is_jsonl = file_path.lower().endswith(".jsonl")

    try:
        questions = []

        if is_jsonl:
            logger.info(f"Loading questions from JSONL file: {file_path}")
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
            logger.info(f"Loading questions from JSON file: {file_path}")
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
        logger.error(f"Unexpected error loading questions: {e}")
        return []


def format_enhanced_result_to_schema(result):
    """Format result with enhanced 4-agent information"""
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
    }

    return formatted_result


def write_enhanced_results_to_jsonl(results, output_file):
    """Write enhanced results to JSONL file"""
    with open(output_file, "w", encoding="utf-8") as f:
        for result in results:
            formatted_result = format_enhanced_result_to_schema(result)
            f.write(json.dumps(formatted_result, ensure_ascii=False) + "\n")
    logger.info(f"Enhanced results written to {output_file}")


def write_enhanced_result_to_json(result, output_file):
    """Write single enhanced result to JSON file"""
    formatted_result = format_enhanced_result_to_schema(result)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(formatted_result, f, indent=2, ensure_ascii=False)
    logger.info(f"Enhanced result written to {output_file}")


def main():
    """Main function with enhanced 4-agent support"""

    parser = argparse.ArgumentParser(
        description="Enhanced 4-Agent RAG with Question Splitting and Parallel Processing"
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
        "--alpha",
        type=float,
        default=0.65,
        help="Weight for E5 in hybrid mode (0.0=BM25 only, 1.0=E5 only)",
    )

    args = parser.parse_args()

    # Use custom DB path if provided, otherwise use config default
    db_path_to_use = args.db_path if args.db_path else DB_PATH

    logger.info(f"Opening database at {db_path_to_use}...")
    try:
        db = plyvel.DB(db_path_to_use, create_if_missing=False)
        logger.info("Database opened successfully")
    except Exception as e:
        logger.error(f"Failed to open database: {e}")
        # Try alternative path if permission denied
        alt_db_path = os.path.join(os.path.dirname(__file__), "local_db")
        logger.info(f"Trying alternative database path: {alt_db_path}")
        db = plyvel.DB(alt_db_path, create_if_missing=True)

    retriever = initialize_retriever(
        args.retriever_type,
        E5_INDEX_DIR,
        BM25_INDEX_DIR,
        DB_PATH,
        args.top_k,
        args.alpha,
    )

    logger.info(
        f"Initializing enhanced 4-agent RAG with n={args.n}, max_workers={args.max_workers}..."
    )
    ragent = Enhanced4AgentRAG(
        retriever,
        agent_model=args.model,
        n=args.n,
        index_dir=args.index_dir,
        max_workers=args.max_workers,
    )

    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "debug"), exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Process single question
    if args.single_question:
        logger.info(
            f"\nProcessing single question with enhanced 4-agent system: {args.single_question}"
        )
        start_time = time.time()

        try:
            should_split, sub_questions = ragent.question_splitter.analyze_and_split(
                query
            )
            cited_answer, references, debug_info = ragent.answer_query(args.single_question, db, should_split, sub_questions)
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

            logger.info(f"Cited Answer: {cited_answer}")
            logger.info(f"References: {references}")
            logger.info(f"Was Split: {debug_info['was_split']}")
            if debug_info["was_split"]:
                logger.info(f"Sub-questions: {debug_info['sub_questions']}")
            logger.info(f"Processing time: {process_time:.2f} seconds")
            logger.info(f"Citations used: {debug_info['total_citations']}")

            # Save result
            if args.output_format == "debug":
                debug_output_file = os.path.join(
                    args.output_dir,
                    "debug",
                    f"enhanced_4agent_single_{args.retriever_type}_debug_{timestamp}.json",
                )
                with open(debug_output_file, "w", encoding="utf-8") as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                logger.info(f"Debug result saved to {debug_output_file}")
            else:
                output_file = os.path.join(
                    args.output_dir,
                    f"enhanced_4agent_single_{args.retriever_type}_{timestamp}.json",
                )
                write_enhanced_result_to_json(result, output_file)

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
        logger.info(
            f"\nProcessing question {i+1}/{len(questions)} with enhanced 4-agent system: {item['question']}"
        )
        start_time = time.time()

        try:
            should_split, sub_questions = ragent.question_splitter.analyze_and_split(
                query
            )
            cited_answer, references, debug_info = ragent.answer_query(item["question"], db, should_split, sub_questions)
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

            logger.info(f"Cited Answer: {cited_answer[:200]}...")
            logger.info(f"References: {references}")
            logger.info(f"Was Split: {debug_info['was_split']}")
            if debug_info["was_split"]:
                logger.info(f"Sub-questions: {debug_info['sub_questions']}")
            logger.info(f"Processing time: {process_time:.2f} seconds")
            logger.info(f"Citations used: {debug_info['total_citations']}")

            # Save debug info
            debug_output_file = os.path.join(
                args.output_dir,
                "debug",
                f"enhanced_4agent_question_{question_id}_{args.retriever_type}_debug_{timestamp}.json",
            )
            with open(debug_output_file, "w", encoding="utf-8") as f:
                json.dump(debug_info, f, indent=2, ensure_ascii=False)

        except Exception as e:
            logger.error(f"Error processing question {question_id}: {e}", exc_info=True)

    # Clean up
    ragent.close()
    retriever.close()

    # Save all results
    random_num = "".join(random.choices(string.ascii_lowercase + string.digits, k=6))

    if results:
        if args.output_format == "jsonl":
            output_file = os.path.join(
                args.output_dir,
                f"enhanced_4agent_answers_{args.retriever_type}_{timestamp}_{random_num}.jsonl",
            )
            write_enhanced_results_to_jsonl(results, output_file)
        elif args.output_format == "json":
            for result in results:
                question_id = result["id"]
                output_file = os.path.join(
                    args.output_dir,
                    f"enhanced_4agent_answer_{question_id}_{args.retriever_type}_{random_num}.json",
                )
                write_enhanced_result_to_json(result, output_file)
        else:  # debug
            output_file = os.path.join(
                args.output_dir,
                "debug",
                f"enhanced_4agent_all_results_{args.retriever_type}_debug_{timestamp}.json",
            )
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info(f"Debug results saved to {output_file}")

    logger.info(f"\nProcessed {len(results)} questions with enhanced 4-agent system.")

    if results:
        avg_time = sum(r["process_time"] for r in results) / len(results)
        avg_filtered = sum(r["total_filtered_docs"] for r in results) / len(results)
        avg_citations = sum(r["total_citations"] for r in results) / len(results)
        avg_full_texts = sum(r["full_texts_retrieved"] for r in results) / len(results)
        split_count = sum(1 for r in results if r["was_split"])

        logger.info(f"Average processing time: {avg_time:.2f} seconds")
        logger.info(
            f"Questions split: {split_count}/{len(results)} ({split_count/len(results)*100:.1f}%)"
        )
        logger.info(f"Average filtered documents: {avg_filtered:.1f}")
        logger.info(f"Average citations: {avg_citations:.1f}")
        logger.info(f"Average full texts used: {avg_full_texts:.1f}")

    db.close()
    logger.info("Database closed")


if __name__ == "__main__":
    main()
