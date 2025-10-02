import os
import numpy as np
import faiss
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
import PyPDF2
import docx
import pickle

# ----------------- CONFIG -----------------
INDEX_FILE = "docs.index"     # Permanent FAISS index file
META_FILE = "metadata.pkl"    # Permanent metadata file

# 🔑 Load API key
load_dotenv()
#client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# --------- File Reader (PDF, DOCX, TXT) ---------
def read_file(file):
    """Extracts text from uploaded file (PDF, DOCX, TXT)."""
    if file.type == "application/pdf":
        reader = PyPDF2.PdfReader(file)
        return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = docx.Document(file)
        return "\n".join([p.text for p in doc.paragraphs])
    elif file.type == "text/plain":
        return file.read().decode("utf-8")
    else:
        st.error("❌ Unsupported file type")
        return ""

# --------- Load Permanent FAISS + Metadata ---------
@st.cache_resource
def load_index_and_metadata():
    """Loads the permanent FAISS index and metadata from disk."""
    if os.path.exists(INDEX_FILE) and os.path.exists(META_FILE):
        index = faiss.read_index(INDEX_FILE)
        with open(META_FILE, "rb") as f:
            metadata = pickle.load(f)
        return index, metadata
    else:
        st.error("⚠️ No permanent FAISS index found. Run embedding first!")
        st.stop()

# Load permanent knowledge base
base_index, base_metadata = load_index_and_metadata()

# --------- Embedding Helper ---------
def embed(texts):
    """Converts a list of texts into embeddings using OpenAI."""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    return [e.embedding for e in response.data]

# --------- Streamlit UI ---------
st.title("📄 ScriptBot")
st.write("Ask me anything to improve your script")

# --------- Session State Setup ---------
# Stores conversation history
if "history" not in st.session_state:
    st.session_state.history = []

# Stores temporary FAISS index (for uploaded docs, session-only)
if "session_index" not in st.session_state:
    st.session_state.session_index = faiss.IndexFlatL2(1536)  # embedding dim

# Stores temporary metadata (parallel to session_index)
if "session_metadata" not in st.session_state:
    st.session_state.session_metadata = []

# --------- File Upload (Session Only) ---------
uploaded_file = st.file_uploader("Upload a document (temporary, only for this session)", type=["pdf", "docx", "txt"])

if uploaded_file:
    text = read_file(uploaded_file)
    if text:
        # Split into chunks (small pieces for embeddings)
        chunks = [text[i:i+500] for i in range(0, len(text), 500)]

        # Embed and add to session index
        vectors = embed(chunks)
        st.session_state.session_index.add(np.array(vectors).astype("float32"))
        st.session_state.session_metadata.extend([(uploaded_file.name, i, chunk) for i, chunk in enumerate(chunks)])

        st.success(f"✅ {uploaded_file.name} added for this session! (not saved permanently)")

# --------- Q&A Section ---------
question = st.text_area("Enter your question:", height=120)

if st.button("Ask GPT"):
    if not question.strip():
        st.warning("⚠️ Please enter a question.")
    else:
        with st.spinner("🔎 Searching knowledge base..."):
            # Create embedding for user query
            query_emb = embed([question])[0]

            # --- Search permanent knowledge base ---
            D1, I1 = base_index.search(np.array([query_emb]).astype("float32"), k=3)
            results_base = [base_metadata[idx] for idx in I1[0]]

            # --- Search session-only knowledge base ---
            results_session = []
            if st.session_state.session_metadata and st.session_state.session_index.ntotal > 0:
                D2, I2 = st.session_state.session_index.search(np.array([query_emb]).astype("float32"), k=3)
                results_session = [st.session_state.session_metadata[idx] for idx in I2[0]]

            # --- Merge results from both sources ---
            results = results_base + results_session
            context = "\n\n".join([r[2] for r in results])

            # Build conversation history (memory + current Q)
            messages = [{"role": "system", "content": "Answer only using provided document excerpts."}]
            messages.extend(st.session_state.history)
            messages.append({"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"})

            # Ask GPT
            response = client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=messages
            )
            answer = response.choices[0].message.content

        # --- Save to session history ---
        st.session_state.history.append({"role": "user", "content": question})
        st.session_state.history.append({"role": "assistant", "content": answer})

        # --- Display results ---
        st.subheader("💡 Answer")
        st.write(answer)

        st.subheader("📚 Sources")
        for r in results:
            st.markdown(f"**{r[0]}** (chunk {r[1]}): {r[2][:200]}...")
