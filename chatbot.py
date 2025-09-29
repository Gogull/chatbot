import os
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import faiss

# 1️⃣ Load OpenAI API key
load_dotenv()
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# 2️⃣ Load PDF
def load_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

# 3️⃣ Chunk text (~500 words)
def chunk_text(text, max_words=500):
    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = ""
    current_len = 0
    for p in paragraphs:
        p_len = len(p.split())
        if current_len + p_len > max_words:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = p
            current_len = p_len
        else:
            current_chunk += "\n\n" + p
            current_len += p_len
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

# 4️⃣ Load PDF and create chunks
pdf_text = load_pdf("Big Fish.pdf")
chunks = chunk_text(pdf_text)

# 5️⃣ Load lightweight embedding model
model_name = "sentence-transformers/paraphrase-MiniLM-L3-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def embed(text):
    tokens = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**tokens)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

# 6️⃣ Encode chunks
embeddings = [embed(c) for c in chunks if c.strip()]

# 7️⃣ Build FAISS index
dimension = len(embeddings[0])
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings).astype("float32"))

# 8️⃣ Query & retrieve relevant chunks
query = "Pretend you are the writer of Alien. Write a dark surreal opening."
query_emb = embed(query)
D, I = index.search(np.array([query_emb]).astype("float32"), k=5)
retrieved_chunks = [chunks[i] for i in I[0]]

# 9️⃣ Prepare prompt for GPT
prompt = f"Use these excerpts as style inspiration:\n\n{retrieved_chunks}\n\nNow: {query}"

#  🔟 Call OpenAI GPT
response = client.chat.completions.create(
    model="gpt-4.1",  # or gpt-3.5-turbo
    messages=[
        {"role": "system", "content": "You are a creative Hollywood screenwriter."},
        {"role": "user", "content": prompt}
    ]
)

# 1️⃣1️⃣ Output GPT answer
print(response.choices[0].message.content)
