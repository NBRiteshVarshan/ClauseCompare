import re
import fitz  # PyMuPDF
import docx2txt
from typing import List, Dict, Any

class ClauseExtractor:
    """Extract clauses from legal documents using text-block strategy"""

    def __init__(self, min_clause_length: int = 60, merge_threshold: int = 30):
        """
        Initialize the extractor.

        Args:
            min_clause_length: Minimum characters for a segment to be considered a clause
            merge_threshold: If a segment is shorter than this, merge with next
        """
        self.min_clause_length = min_clause_length
        self.merge_threshold = merge_threshold

    def extract_from_pdf(self, file_content: bytes) -> List[Dict[str, Any]]:
        """Extract text from PDF file using PyMuPDF"""
        try:
            doc = fitz.open(stream=file_content, filetype="pdf")
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return self.extract_clauses(text)
        except Exception as e:
            raise Exception(f"Failed to extract from PDF: {str(e)}")

    def extract_from_docx(self, file_content: bytes) -> List[Dict[str, Any]]:
        """Extract text from DOCX file using docx2txt"""
        try:
            import tempfile
            import os

            with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as tmp_file:
                tmp_file.write(file_content)
                tmp_path = tmp_file.name

            text = docx2txt.process(tmp_path)
            os.unlink(tmp_path)  # Clean up temp file

            return self.extract_clauses(text)
        except Exception as e:
            raise Exception(f"Failed to extract from DOCX: {str(e)}")

    def extract_clauses(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract clauses from raw text using text-block strategy.
        Splits on double newlines and filters/merges segments.
        """
        # Normalize line endings
        text = text.replace('\r\n', '\n')
        text = text.replace('\r', '\n')

        # Remove common noise patterns (page numbers, headers, footers)
        text = self._clean_noise(text)

        # Split on double newlines or more
        segments = re.split(r'\n\s*\n', text)

        # Clean each segment
        cleaned_segments = []
        for seg in segments:
            seg = seg.strip()
            seg = re.sub(r'\s+', ' ', seg)
            if seg:
                cleaned_segments.append(seg)

        # Merge short segments with the next one
        merged_segments = self._merge_short_segments(cleaned_segments)

        # Filter out noise segments
        clauses = []
        for idx, segment in enumerate(merged_segments):
            if self._is_valid_clause(segment):
                clauses.append({
                    'number': str(idx + 1),
                    'text': segment,
                    'metadata': {
                        'word_count': len(segment.split()),
                        'char_count': len(segment),
                        'has_conditions': self._check_conditions(segment),
                        'has_exceptions': self._check_exceptions(segment),
                        'is_title': self._is_likely_title(segment)
                    }
                })

        # Re-number clauses sequentially
        for idx, clause in enumerate(clauses):
            clause['number'] = str(idx + 1)

        return clauses

    def _clean_noise(self, text: str) -> str:
        """Remove common document noise like page numbers and headers"""
        text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
        text = re.sub(r'Page \d+ of \d+', '', text)
        text = re.sub(r'-\s*\d+\s*-', '', text)
        text = re.sub(r'Confidential\s*[-–]\s*Draft', '', text, flags=re.IGNORECASE)
        return text

    def _merge_short_segments(self, segments: List[str]) -> List[str]:
        """Merge short segments with the next segment"""
        if not segments:
            return []

        merged = []
        i = 0
        while i < len(segments):
            current = segments[i]
            if len(current) < self.merge_threshold and i + 1 < len(segments):
                merged_segment = current + " " + segments[i + 1]
                merged.append(merged_segment)
                i += 2
            else:
                merged.append(current)
                i += 1
        return merged

    def _is_valid_clause(self, text: str) -> bool:
        """Check if a segment is a valid clause"""
        if len(text) < self.min_clause_length:
            return False
        if re.match(r'^[\d\s\.\,\;\:\-]+$', text):
            return False
        return True

    def _is_likely_title(self, text: str) -> bool:
        """Check if the segment is likely just a title/header"""
        if len(text) < 50:
            if text.isupper() or text.endswith(':') or text.endswith('.'):
                return True
        return False

    def _check_conditions(self, text: str) -> bool:
        condition_words = ['if', 'provided', 'unless', 'subject to', 'when', 'where', 'in the event', 'upon']
        text_lower = text.lower()
        return any(word in text_lower for word in condition_words)

    def _check_exceptions(self, text: str) -> bool:
        exception_words = ['except', 'excluding', 'other than', 'notwithstanding', 'unless', 'without prejudice']
        text_lower = text.lower()
        return any(word in text_lower for word in exception_words)


def get_document_summary(clauses: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Get summary statistics for extracted clauses"""
    if not clauses:
        return {'total': 0, 'avg_length': 0, 'has_conditions': 0, 'has_exceptions': 0}

    total = len(clauses)
    avg_length = sum(c['metadata']['word_count'] for c in clauses) / total if total > 0 else 0
    has_conditions = sum(1 for c in clauses if c['metadata'].get('has_conditions', False))
    has_exceptions = sum(1 for c in clauses if c['metadata'].get('has_exceptions', False))

    return {
        'total': total,
        'avg_length': round(avg_length, 2),
        'has_conditions': has_conditions,
        'has_exceptions': has_exceptions
    }