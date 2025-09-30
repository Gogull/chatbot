import os
import pickle
import numpy as np
import faiss
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

# 🔑 Load API key
load_dotenv()
#client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
INDEX_FILE = "docs.index"
META_FILE = "metadata.pkl"

# --------- Load FAISS + Metadata ---------
@st.cache_resource
def load_index_and_metadata():
    if os.path.exists(INDEX_FILE) and os.path.exists(META_FILE):
        index = faiss.read_index(INDEX_FILE)
        with open(META_FILE, "rb") as f:
            metadata = pickle.load(f)
        return index, metadata
    return None, []

index, metadata = load_index_and_metadata()
if not index:
    st.error("⚠️ No FAISS index found. Run embedding first!")
    st.stop()

# --------- Embedding Helper ---------
def embed(texts):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    return [e.embedding for e in response.data]

# --------- Streamlit UI ---------
st.title("📄 ScriptBot")
st.write("Ask about your scripts")

question = st.text_area("Enter your question:", height=120)

if st.button("Ask GPT"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Searching and thinking..."):
            # Search FAISS
            query_emb = embed([question])[0]
            D, I = index.search(np.array([query_emb]).astype("float32"), k=5)

            results = [metadata[idx] for idx in I[0]]
            context = "\n\n".join([r[2] for r in results])

            # Build prompt
            prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer creatively and in detail:"

            response = client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {"role": "system", "content": "You answer based on the provided PDF excerpts."},
                    {"role": "user", "content": prompt}
                ]
            )
            answer = response.choices[0].message.content

        st.subheader("💡 Answer")
        st.write(answer)

        st.subheader("📚 Sources")
        for r in results:
            st.markdown(f"**{r[0]}** (chunk {r[1]}): {r[2][:200]}...")
