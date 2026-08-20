import io
import os
import sys

import pytest
from docx import Document
from fpdf import FPDF

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from file_parser import extract_text


class FakeUploadedFile(io.BytesIO):
    """Mimics Streamlit's UploadedFile enough for file_parser: BytesIO + .name."""

    def __init__(self, data, name):
        super().__init__(data)
        self.name = name


def make_pdf_bytes(text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    pdf.multi_cell(0, 10, text)
    return bytes(pdf.output())


def make_docx_bytes(paragraphs):
    buf = io.BytesIO()
    doc = Document()
    for para in paragraphs:
        doc.add_paragraph(para)
    doc.save(buf)
    return buf.getvalue()


def test_extract_text_from_pdf():
    pdf_bytes = make_pdf_bytes("Python AWS Docker, 5 years of experience.")
    uploaded = FakeUploadedFile(pdf_bytes, "resume.pdf")

    text = extract_text(uploaded)

    assert "Python" in text
    assert "5 years" in text


def test_extract_text_from_blank_pdf_raises_helpful_error():
    pdf = FPDF()
    pdf.add_page()  # no text added -- simulates a scanned/image-only PDF
    blank_pdf_bytes = bytes(pdf.output())
    uploaded = FakeUploadedFile(blank_pdf_bytes, "scanned.pdf")

    with pytest.raises(ValueError, match="No selectable text"):
        extract_text(uploaded)


def test_extract_text_from_docx():
    docx_bytes = make_docx_bytes([
        "Jane Doe - Backend Engineer",
        "6 years of experience in Python, AWS, and Docker."
    ])
    uploaded = FakeUploadedFile(docx_bytes, "resume.docx")

    text = extract_text(uploaded)

    assert "Jane Doe" in text
    assert "Python, AWS, and Docker" in text


def test_extract_text_from_txt():
    uploaded = FakeUploadedFile(b"Plain text resume content.", "resume.txt")

    text = extract_text(uploaded)

    assert text == "Plain text resume content."


def test_extract_text_unsupported_extension_raises():
    uploaded = FakeUploadedFile(b"whatever", "resume.exe")

    with pytest.raises(ValueError, match="Unsupported file type"):
        extract_text(uploaded)
