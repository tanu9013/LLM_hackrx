import json
import hashlib
from pathlib import Path
from utils import read_pdf, read_docx, read_txt, read_eml, chunk_text

DATA_DIR = Path("data")
STORAGE_DIR = Path("storage")
STORAGE_DIR.mkdir(exist_ok=True)

JSONL_FILE = STORAGE_DIR / "chunks.jsonl"

def ingest_documents():
    metas = []
    texts = []
    with open(JSONL_FILE, "w", encoding="utf8") as fout:
        for path in DATA_DIR.glob("*"):
            suf = path.suffix.lower()
            if suf == ".pdf":
                txt = read_pdf(path)
            elif suf in (".docx", ".doc"):
                txt = read_docx(path)
            elif suf == ".eml":
                txt = read_eml(path)
            else:
                txt = read_txt(path)
            if not txt.strip():
                print(f"No text extracted from {path}")
                continue
            chunks = chunk_text(txt)
            for i, c in enumerate(chunks):
                uid = hashlib.sha1((str(path) + str(i)).encode()).hexdigest()[:12]
                meta = {"id": uid, "source": path.name, "chunk_index": i, "text": c}
                fout.write(json.dumps(meta, ensure_ascii=False) + "\n")
                metas.append({"id": uid, "source": path.name, "chunk_index": i})
                texts.append(c)
    print(f"Wrote {len(texts)} chunks to {JSONL_FILE}")

if __name__ == "__main__":
    ingest_documents()
