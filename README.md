# LLM_hackrx
llm_retrieval/
│
├── ingest.py        # Reads docs, chunks them, saves chunks.jsonl
├── indexer.py       # Creates FAISS index from chunks.jsonl
├── app.py           # Loads FAISS + TinyLLaMA for querying
├── requirements.txt # For easy install
├── data/            # Your PDFs, DOCX, emails, etc.
└── storage/         # Output: chunks.jsonl, faiss_index, metas.json
