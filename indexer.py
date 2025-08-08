import json
import faiss
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

STORAGE_DIR = Path("storage")
JSONL_FILE = STORAGE_DIR / "chunks.jsonl"
INDEX_FILE = STORAGE_DIR / "faiss_index"
META_FILE = STORAGE_DIR / "metas.json"

def build_index():
    texts = []
    metas = []
    with open(JSONL_FILE, "r", encoding="utf8") as f:
        for line in f:
            obj = json.loads(line)
            texts.append(obj["text"])
            metas.append({"id": obj["id"], "source": obj["source"], "chunk_index": obj["chunk_index"]})

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    faiss.write_index(index, str(INDEX_FILE))
    with open(META_FILE, "w", encoding="utf8") as f:
        json.dump(metas, f, ensure_ascii=False, indent=2)

    print(f"Indexed {len(texts)} chunks. Saved to {INDEX_FILE}")

if __name__ == "__main__":
    build_index()

