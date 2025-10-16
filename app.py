import os
import pickle
import numpy as np
import faiss
from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from openai import OpenAI

# ---------------- CONFIG ----------------
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

INDEX_FILE = "docs.index"
META_FILE = "metadata.pkl"

# ---------------- FASTAPI APP ----------------
app = FastAPI(title="Embedding Retrieval API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- LOAD INDEX AT STARTUP ----------------
if not os.path.exists(INDEX_FILE) or not os.path.exists(META_FILE):
    raise FileNotFoundError("FAISS index or metadata.pkl not found. Please run /build-index first.")

index = faiss.read_index(INDEX_FILE)
with open(META_FILE, "rb") as f:
    metadata = pickle.load(f)

dimension = index.d  # should be 1536 for text-embedding-3 models

# ---------------- EMBEDDING HELPER ----------------
def embed(texts):
    response = client.embeddings.create(
        model="text-embedding-3-small",  # ✅ lightweight, fast
        input=texts
    )
    return [e.embedding for e in response.data]

# ---------------- ENDPOINT: Retrieve Chunks ----------------
@app.post("/retrieve/")
async def retrieve_relevant_chunks(
    query: str = Form(...),
    k: int = Form(5),
    min_score: float = Form(0.0)
):
    """
    Retrieve top-k relevant document chunks for a given query.
    Returns: List of {file, chunk_id, text, score}
    """
    try:
        # Embed the query
        query_emb = np.array(embed([query])[0]).astype("float32").reshape(1, -1)

        # Search FAISS
        D, I = index.search(query_emb, k)

        # Prepare results
        results = []
        for dist, idx in zip(D[0], I[0]):
            if idx == -1:
                continue
            score = float(np.exp(-dist / 100))  # convert distance → similarity-ish score
            if score < min_score:
                continue
            file, chunk_id, text = metadata[idx]
            results.append({
                "file": file,
                "chunk_id": int(chunk_id),
                "text": text,
                "score": round(score, 4)
            })

        return {"query": query, "results": results, "count": len(results)}

    except Exception as e:
        return {"error": str(e)}
