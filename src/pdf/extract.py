# =============================
# FILE: src/pdf/extract.py
# =============================
from pathlib import Path
import fitz
import re
from typing import List, Dict

class PDFExtractor:
    def __init__(self, chunk_words: int = 900, overlap_words: int = 120):
        self.chunk_words = chunk_words
        self.overlap_words = overlap_words

    def extract_pages(self, pdf_path: Path) -> list[dict]:
        pages = []
        with fitz.open(pdf_path) as doc:
            for i, page in enumerate(doc):
                text = page.get_text("text")
                text = re.sub(r"\s+", " ", text).strip()
                pages.append({"page": i + 1, "text": text})
        return pages

    def chunkify(self, pdf_id: str, pages: list[dict]) -> List[Dict]:
        # simple concatenation across pages; track page ranges
        words, idx_to_page = [], []
        for p in pages:
            toks = p["text"].split()
            words.extend(toks)
            idx_to_page.extend([p["page"]] * len(toks))
        if not words:
            return []
        res = []
        step = max(1, self.chunk_words - self.overlap_words)
        for i in range(0, len(words), step):
            span = words[i : i + self.chunk_words]
            if not span:
                break
            start_page = idx_to_page[i]
            end_page = idx_to_page[min(i + self.chunk_words - 1, len(idx_to_page) - 1)]
            chunk_id = f"{pdf_id}:pp{start_page}-{end_page}"
            res.append({"chunk_id": chunk_id, "pages": [start_page, end_page], "text": " ".join(span)})
            if i + self.chunk_words >= len(words):
                break
        return res
