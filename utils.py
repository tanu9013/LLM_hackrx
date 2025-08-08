import hashlib
import json
from pathlib import Path
from pypdf import PdfReader
import docx
import email

# -----------------
# READERS
# -----------------
def read_pdf(path: Path) -> str:
    try:
        pdf = PdfReader(str(path))
        return "\n".join([page.extract_text() or "" for page in pdf.pages])
    except Exception as e:
        print(f"Error reading PDF {path}: {e}")
        return ""

def read_docx(path: Path) -> str:
    try:
        doc = docx.Document(str(path))
        return "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        print(f"Error reading DOCX {path}: {e}")
        return ""

def read_txt(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception as e:
        print(f"Error reading TXT {path}: {e}")
        return ""

def read_eml(path: Path) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            msg = email.message_from_file(f)
        return msg.get_payload()
    except Exception as e:
        print(f"Error reading EML {path}: {e}")
        return ""

# -----------------
# CHUNKING
# -----------------
def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks
