
from pathlib import Path
from pypdf import PdfReader


def load_pdf(file_path):
    """Extract text from all pages of a PDF document."""

    file_path = Path(file_path)
    reader = PdfReader(file_path)

    pages_text = []

    for page in reader.pages:
        page_text = page.extract_text() or ""
        if page_text.strip():
            pages_text.append(page_text)

    return "\n\n".join(pages_text)
