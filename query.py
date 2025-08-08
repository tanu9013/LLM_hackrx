from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import PyPDF2

# === Configuration ===
HUGGINGFACE_TOKEN = "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"
MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"
PDF_PATH = "C:/Users/91901/OneDrive/Desktop/llm projjects/data.pdf"

# === Step 1: Load the tokenizer and model for CPU ===
print("Loading model on CPU...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    use_auth_token=HUGGINGFACE_TOKEN
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="cpu",
    torch_dtype=torch.float32,
    use_auth_token=HUGGINGFACE_TOKEN,
    use_safetensors=True,
    resume_download=True
)

# === Step 2: Extract text from PDF ===
def extract_pdf_text(pdf_path):
    with open(pdf_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
    return text

print("Extracting text from PDF...")
policy_text = extract_pdf_text(PDF_PATH)

# === Step 3: Accept and structure user query ===
user_query = "46-year-old male, knee surgery in Pune, 3-month-old insurance policy"

# Prompt template
prompt = f"""
You are an expert insurance claim analyst.

Given the following policy document and user query, return a structured JSON response with:
- "decision" (approved/rejected),
- "amount" (if applicable),
- "justification" (clause or reason).

### POLICY DOCUMENT:
{policy_text[:3000]}  # Truncate to fit into model input size

### USER QUERY:
{user_query}

### RESPONSE FORMAT:
{{"decision": "...", "amount": "...", "justification": "..."}}
"""

# === Step 4: Run inference on CPU ===
print("Running inference...")
inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=300,
        do_sample=True,
        top_k=50,
        top_p=0.9,
        temperature=0.7,
        eos_token_id=tokenizer.eos_token_id
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("\n--- Model Response ---")
print(response)
