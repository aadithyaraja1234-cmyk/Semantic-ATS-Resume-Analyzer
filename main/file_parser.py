"""Extract plain text from an uploaded resume file (PDF, DOCX, or TXT)."""


def extract_text(uploaded_file):
    name = uploaded_file.name.lower()

    if name.endswith(".pdf"):
        return _extract_pdf(uploaded_file)
    elif name.endswith(".docx"):
        return _extract_docx(uploaded_file)
    elif name.endswith(".txt"):
        return uploaded_file.read().decode("utf-8", errors="ignore").strip()
    else:
        raise ValueError(f"Unsupported file type: {uploaded_file.name}")


def _extract_pdf(uploaded_file):
    from pypdf import PdfReader

    reader = PdfReader(uploaded_file)
    text = "\n".join(page.extract_text() or "" for page in reader.pages)

    if not text.strip():
        raise ValueError(
            "No selectable text found in this PDF. It may be a scanned image -- "
            "try pasting the resume text instead."
        )

    return text.strip()


def _extract_docx(uploaded_file):
    from docx import Document

    document = Document(uploaded_file)
    text = "\n".join(paragraph.text for paragraph in document.paragraphs)
    return text.strip()
