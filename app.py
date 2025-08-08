import faiss
import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

STORAGE_DIR = Path("storage")
INDEX_FILE = STORAGE_DIR / "faiss_index"
META_FILE = STORAGE_DIR / "metas.json"

def load_index():
    index = faiss.read_index(str(INDEX_FILE))
    with open(META_FILE, "r", encoding="utf8") as f:
        metas = json.load(f)
    return index, metas

def retrieve(query, top_k=3):
    emb = embed_model.encode([query], convert_to_numpy=True)
    D, I = index.search(emb, top_k)
    return [metas[i] for i in I[0]], D[0]

# Load models
print("Loading FAISS index...")
index, metas = load_index()
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

print("Loading TinyLLaMA...")
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
llm_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
llm_pipe = pipeline("text-generation", model=llm_model, tokenizer=tokenizer, max_new_tokens=300)

while True:
    q = input("\nEnter your question (or 'exit'): ")
    if q.lower() == "exit":
        break
    retrieved, scores = retrieve(q)
    context = "\n\n".join([f"Source: {r['source']}, Chunk {r['chunk_index']}" for r in retrieved])
    prompt = f"Answer the following question using the provided context.\n\nContext:\n{context}\n\nQuestion: {q}\nAnswer:"
    answer = llm_pipe(prompt)[0]["generated_text"]
    print("\n--- Answer ---\n", answer)

