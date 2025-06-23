#!/usr/bin/env python3
"""
text_cleaner.py
Clean and normalize document text for better LLM processing
"""

import re
import logging

logger = logging.getLogger(__name__)

class DocumentTextCleaner:
    """Clean and normalize document text for better LLM processing"""
    
    def __init__(self):
        self.formula_pattern = re.compile(r'\{\{formula:[^}]+\}\}')
        self.cite_pattern = re.compile(r'\{\{cite:[^}]+\}\}')
        self.ref_pattern = re.compile(r'Refs?\.\s*\{\{cite:[^}]+\}\}(?:,\s*\{\{cite:[^}]+\}\})*')
        self.figure_pattern = re.compile(r'\{\{figure:[^}]+\}\}')
        self.section_pattern = re.compile(r"'section':\s*'[^']*',\s*'text':\s*'")
        
    def clean_document_text(self, text: str) -> str:
        """Clean document text of LaTeX markup and artifacts"""
        
        # Remove JSON-like section markers
        text = re.sub(r"'section':\s*'[^']*',\s*'text':\s*'", "", text)
        text = re.sub(r"^\s*\{.*?'text':\s*'", "", text)
        
        # Remove formula and citation markup
        text = self.formula_pattern.sub('[FORMULA]', text)
        text = self.cite_pattern.sub('[REF]', text)
        text = self.ref_pattern.sub('[REFERENCES]', text)
        text = self.figure_pattern.sub('[FIGURE]', text)
        
        # Remove excessive reference lists
        text = re.sub(r'\[REF\](?:,?\s*\[REF\]){3,}', '[MULTIPLE_REFS]', text)
        
        # Clean up extra whitespace and formatting
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        # Remove very technical notation patterns
        text = re.sub(r'\$[^$]+\$', '[MATH]', text)  # LaTeX math
        text = re.sub(r'\\[a-zA-Z]+\{[^}]*\}', '[LATEX]', text)  # LaTeX commands
        
        # Clean up sentence boundaries
        text = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', text)
        
        return text.strip()
    
    def extract_clean_sentences(self, text: str, max_sentences: int = 15) -> str:
        """Extract clean, readable sentences from document text"""
        
        cleaned_text = self.clean_document_text(text)
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', cleaned_text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        # Filter out sentences with too much technical markup
        clean_sentences = []
        for sentence in sentences:
            # Skip sentences that are mostly markup
            markup_ratio = (sentence.count('[') + sentence.count('{')) / max(len(sentence), 1)
            if markup_ratio < 0.1:  # Less than 10% markup
                clean_sentences.append(sentence)
        
        # Return first N clean sentences
        result_sentences = clean_sentences[:max_sentences]
        return '. '.join(result_sentences) + '.' if result_sentences else cleaned_text[:500]
    
    def clean_for_citation_matching(self, text: str) -> str:
        """Clean text specifically for citation-answer matching"""
        
        # Remove all markup first
        cleaned = self.clean_document_text(text)
        
        # Extract only meaningful content sentences
        sentences = re.split(r'[.!?]+', cleaned)
        
        meaningful_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            
            # Skip very short or very long sentences
            if len(sentence) < 30 or len(sentence) > 300:
                continue
                
            # Skip sentences that are mostly numbers or symbols
            alpha_ratio = sum(c.isalpha() or c.isspace() for c in sentence) / max(len(sentence), 1)
            if alpha_ratio < 0.7:  # Less than 70% alphabetic
                continue
                
            # Skip sentences with too many technical terms
            if sentence.count('[') > 3 or sentence.count('{') > 2:
                continue
                
            meaningful_sentences.append(sentence)
        
        return '. '.join(meaningful_sentences[:10]) + '.' if meaningful_sentences else cleaned[:200]