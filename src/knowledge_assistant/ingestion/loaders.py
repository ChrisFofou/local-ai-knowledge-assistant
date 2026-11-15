import hashlib
import re
from pathlib import Path

from pypdf import PdfReader

from knowledge_assistant.ingestion.models import IngestedDocument

SUPPORTED_EXTENSIONS = {".pdf", ".md", ".txt"}


def calculate_document_id(path: Path) -> str:
    """Generate a deterministic SHA-256 identifier from file content."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize_text(text: str) -> str:
    """Normalize whitespace while preserving paragraph boundaries."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\x00", "")

    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)

    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)

    return text.strip()


def load_pdf(path: Path) -> IngestedDocument:
    reader = PdfReader(str(path))

    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""

        if page_text.strip():
            pages.append(f"[Page {page_number}]\n{page_text}")

    text = normalize_text("\n\n".join(pages))

    return IngestedDocument(
        document_id=calculate_document_id(path),
        source=path.name,
        file_type="pdf",
        text=text,
        metadata={
            "page_count": len(reader.pages),
            "file_size": path.stat().st_size,
        },
    )


def load_text_file(path: Path) -> IngestedDocument:
    text = path.read_text(encoding="utf-8")

    return IngestedDocument(
        document_id=calculate_document_id(path),
        source=path.name,
        file_type=path.suffix.lower().lstrip("."),
        text=normalize_text(text),
        metadata={"file_size": path.stat().st_size},
    )


def load_document(file_path: str | Path) -> IngestedDocument:
    """Load a PDF, Markdown, or plain-text document."""
    path = Path(file_path).expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(f"Document not found: {path}")

    if not path.is_file():
        raise ValueError(f"Expected a file, received: {path}")

    extension = path.suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file format: {extension}")

    document = load_pdf(path) if extension == ".pdf" else load_text_file(path)

    if not document.text:
        raise ValueError(f"No extractable text found in: {path.name}")

    return document
