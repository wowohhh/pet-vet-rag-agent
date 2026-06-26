"""PDF parsing and text extraction for CNKI papers."""

import re
from pathlib import Path
import fitz  # PyMuPDF


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract full text from a PDF file.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Extracted text with basic cleaning applied.
    """
    doc = fitz.open(str(pdf_path))
    all_text: list[str] = []

    for page in doc:
        text = page.get_text("text")
        all_text.append(text)

    doc.close()
    raw = "\n".join(all_text)
    return _clean_text(raw)


def _clean_text(text: str) -> str:
    """Basic text cleaning for CNKI papers.

    - Remove excessive whitespace
    - Remove header/footer artifacts (page numbers, journal names)
    - Normalize newlines
    """
    # Remove page numbers and common CNKI headers
    text = re.sub(r"\n\s*\d+\s*\n", "\n", text)
    text = re.sub(r"版权所有\S*", "", text)
    text = re.sub(r"中国知网\S*", "", text)

    # Normalize whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove empty lines
    text = re.sub(r"\n\s*\n\s*\n", "\n\n", text)

    return text.strip()


def extract_metadata(pdf_path: Path) -> dict[str, str]:
    """Extract metadata from a CNKI paper PDF.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Dict with title, journal, year, doi, authors (if available).
    """
    doc = fitz.open(str(pdf_path))
    meta = doc.metadata or {}
    doc.close()

    # CNKI papers often embed title in metadata
    return {
        "title": meta.get("title", pdf_path.stem),
        "source": pdf_path.name,
        "journal": meta.get("subject", ""),
        "year": _extract_year(meta),
        "doi": _extract_doi(meta, doc),
    }


def _extract_year(meta: dict) -> str:
    for key in ("creationDate", "modDate"):
        val = meta.get(key, "")
        match = re.search(r"20\d{2}", val)
        if match:
            return match.group(0)
    return ""


def _extract_doi(meta: dict, doc) -> str:
    # Try metadata first, then scan first page
    for key in ("doi", "DOI"):
        if key in meta:
            return meta[key]
    return ""
