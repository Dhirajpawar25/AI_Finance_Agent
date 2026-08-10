"""Financial document processing — PDF extraction, summarization, Q&A."""
from __future__ import annotations

import io
import logging
import re

from pypdf import PdfReader

from app.models import Document, User
from app.services.ai import get_reply

logger = logging.getLogger(__name__)


def extract_pdf_text(content: bytes, filename: str = "document.pdf") -> str:
    """Extract text from a PDF byte stream."""
    try:
        reader = PdfReader(io.BytesIO(content))
        parts = []
        for page in reader.pages:
            try:
                text = page.extract_text() or ""
            except Exception:  # noqa: BLE001
                text = ""
            if text.strip():
                parts.append(text)
        full_text = "\n".join(parts)
        if not full_text.strip():
            return ""
        # Basic cleanup
        full_text = re.sub(r"[ \t]+", " ", full_text)
        full_text = re.sub(r"\n{3,}", "\n\n", full_text)
        return full_text
    except Exception as exc:  # noqa: BLE001
        logger.exception("PDF extraction failed for %s", filename)
        raise ValueError(f"Could not read the PDF: {exc}") from exc


def chunk_text(text: str, chunk_chars: int = 6000) -> list[str]:
    """Split text into overlapping chunks for summarization."""
    if len(text) <= chunk_chars:
        return [text]
    chunks = []
    for i in range(0, len(text), chunk_chars):
        chunk = text[i : i + chunk_chars]
        # try to break at sentence boundary
        if i + chunk_chars < len(text):
            last_period = max(chunk.rfind(". "), chunk.rfind("\n\n"))
            if last_period > chunk_chars // 2:
                chunk = chunk[: last_period + 1]
        chunks.append(chunk.strip())
    return chunks


def _trim_for_ai(text: str, max_chars: int = 8000) -> str:
    return text[:max_chars]


def summarize_document_text(text: str, doc_name: str = "document") -> str:
    """Generate an executive summary of a financial document."""
    if not text.strip():
        return "The document appears to have no extractable text (possibly a scanned PDF)."

    excerpt = _trim_for_ai(text)
    prompt = f"""
You are a senior financial analyst. I'm giving you the text of a financial document called "{doc_name}".

Produce a concise executive summary with these sections:
1. **Overview** – What is this document and who is it about? (2-3 sentences)
2. **Key Financials** – The most important numbers/trends (revenue, margins, cash, debt, growth).
3. **What Matters** – The 3-5 most important takeaways a finance professional should know.
4. **Risks / Red Flags** – Anything concerning (accounting issues, declining trends, high debt, guidance cuts).
5. **Key Metrics Comparison** – If figures for multiple periods are present, show a small YoY/QoQ comparison.

Keep it tight and skimmable. Use bullets and bold headings. Do not invent numbers that are not in the text. If data is missing, say so.

DOCUMENT TEXT:
{excerpt}
"""
    try:
        return get_reply(prompt, system="You are a precise financial document analyst.", max_tokens=800)
    except Exception as exc:  # noqa: BLE001
        logger.exception("summarization failed")
        return f"Could not summarize the document automatically.\nReason: {exc}"


def answer_question_about_document(text: str, question: str, doc_name: str = "document") -> str:
    """Answer a user's question grounded in the document text."""
    excerpt = _trim_for_ai(text, max_chars=12000)
    prompt = f"""
I asked a question about a financial document named "{doc_name}".

QUESTION:
{question}

Answer concisely but thoroughly, citing the specific numbers/passages from the document you are relying on. If the document does not contain the answer, say so clearly — do not invent information.

DOCUMENT TEXT:
{excerpt}
"""
    try:
        return get_reply(prompt, system="You are a precise financial document analyst grounded only in provided text.", max_tokens=600)
    except Exception as exc:  # noqa: BLE001
        logger.exception("document Q&A failed")
        return f"I couldn't analyze the document right now.\nReason: {exc}"


def compare_documents(docs: list[tuple[str, str]]) -> str:
    """Compare multiple documents (name, text) and highlight differences."""
    if not docs:
        return "No documents to compare."
    blocks = []
    for name, text in docs:
        blocks.append(f"=== {name} ===\n{_trim_for_ai(text, max_chars=6000)}")
    combined = "\n\n".join(blocks)

    prompt = f"""
You are a senior financial analyst. Compare the following {len(docs)} financial documents.

For each document briefly summarize its key financials, then provide a **Comparison** section highlighting:
- Differences in revenue, margins, cash, debt, growth trajectory
- Diverging strategic narratives or risk profiles
- Unusual trends or anomalies
- Which document reflects the strongest / weakest financial position and why

Be concise, use bullets, and anchor every claim to the actual numbers present.

DOCUMENTS:
{combined}
"""
    try:
        return get_reply(prompt, system="You are a rigorous financial analyst comparing documents.", max_tokens=900)
    except Exception as exc:  # noqa: BLE001
        logger.exception("document comparison failed")
        return f"Could not compare the documents.\nReason: {exc}"


def save_uploaded_document(user: User, storage_path: str, filename: str, text_content: str) -> Document:
    """Persist a processed document on the user's record."""
    from app.database import SessionLocal

    db = SessionLocal()
    try:
        doc = Document(
            user_id=user.id,
            filename=filename,
            file_type="pdf" if filename.lower().endswith(".pdf") else "text",
            storage_path=storage_path,
            text_content=text_content[:200_000],  # cap storage
        )
        db.add(doc)
        db.commit()
        db.refresh(doc)
        return doc
    finally:
        db.close()