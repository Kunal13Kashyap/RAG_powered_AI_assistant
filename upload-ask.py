import os
import json
import uuid
import time
import faiss
import fitz
import tiktoken
import numpy as np
import requests
import traceback
from datetime import datetime
from flask import Flask, request, render_template, redirect, url_for, send_from_directory, flash
from sentence_transformers import SentenceTransformer
import re

# --------------------
# CONFIG
# --------------------
UPLOAD_DIR = "uploads"
INDEX_DIR = "indexes"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)

# Embedding model
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
embedder = SentenceTransformer(EMBED_MODEL_NAME)

# Tokenizer
enc = tiktoken.get_encoding("cl100k_base")

# Ollama config
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "mistral"

# Flask
app = Flask(__name__)
app.secret_key = "supersecret"


# --------------------
# Logging helper
# --------------------
def log(msg):
    print(f"[SERVER] {msg}", flush=True)


# --------------------
# PDF extraction
# --------------------
def extract_pages(pdf_path):
    log("Reading PDF pages...")
    doc = fitz.open(pdf_path)
    pages = []

    for i, page in enumerate(doc):
        text = page.get_text("text").strip()
        text = text.replace("\u0000", "").replace("\x00", "")
        pages.append({"page": i + 1, "text": text})

    log(f"Total pages extracted: {len(pages)}")
    return pages


# --------------------
# Chunk text
# --------------------
def chunk_text(text, max_tokens=500, overlap=50):
    tokens = enc.encode(text)
    chunks = []

    i = 0
    while i < len(tokens):
        chunk_tokens = tokens[i: i + max_tokens]
        chunk_text_decoded = enc.decode(chunk_tokens).strip()
        if chunk_text_decoded:
            chunks.append(chunk_text_decoded)
        i += max_tokens - overlap

    return chunks


# --------------------
# Build FAISS index
# --------------------
def build_faiss_for_pdf(pdf_path, base_name):
    log("Starting indexing pipeline...")
    pages = extract_pages(pdf_path)

    all_chunks = []
    for p in pages:
        if not p["text"]:
            continue
        chunks = chunk_text(p["text"])
        for c in chunks:
            all_chunks.append({"page": p["page"], "text": c})

    log(f"Total chunks created: {len(all_chunks)}")
    if not all_chunks:
        raise ValueError("No text found in PDF")

    # Embeddings
    log("Generating embeddings...")
    texts = [c["text"] for c in all_chunks]
    embeddings = embedder.encode(texts, show_progress_bar=True)
    emb_matrix = np.array(embeddings).astype("float32")

    # FAISS
    log("Building FAISS index...")
    d = emb_matrix.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(emb_matrix)

    index_file = f"{INDEX_DIR}/{base_name}_index.faiss"
    meta_file = f"{INDEX_DIR}/{base_name}_meta.json"

    faiss.write_index(index, index_file)

    metadata = [{"page": c["page"], "text": c["text"][:300]} for c in all_chunks]

    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    log("Index building completed.")
    return index_file, meta_file


# --------------------
# Retrieve Top K
# --------------------
def retrieve_top_k(index_path, meta_path, query, k=4):
    log(f"Retrieving relevant chunks from {index_path}...")
    if not os.path.exists(index_path) or not os.path.exists(meta_path):
        raise FileNotFoundError("Index or metadata not found.")

    idx = faiss.read_index(index_path)

    q_emb = embedder.encode(query).astype("float32")
    if q_emb.ndim == 1:
        q_emb = np.expand_dims(q_emb, 0)

    D, I = idx.search(q_emb, k)

    with open(meta_path, "r") as f:
        metadata = json.load(f)

    results = []
    for dist, i in zip(D[0], I[0]):
        if 0 <= i < len(metadata):
            results.append(metadata[i])

    log(f"Retrieved {len(results)} chunks.")
    return results


# --------------------
# Extract legal sections
# --------------------
SECTION_PATTERNS = [
    r"\bSection\s+\d+[A-Za-z]?\b",
    r"\bSec\.?\s*\d+[A-Za-z]?\b",
    r"\bU/s\s*\d+[A-Za-z]?\b",
    r"\b\d+\s*IPC\b",
    r"\b\d+\s*NIA\b",
    r"\b\d+\s*CrPC\b",
]


def extract_sections(text):
    found = set()
    for pat in SECTION_PATTERNS:
        matches = re.findall(pat, text, flags=re.IGNORECASE)
        for m in matches:
            found.add(m.strip())
    return list(found)


# --------------------
# FIXED OLLAMA STREAMING FUNCTION
# --------------------
def call_ollama(prompt, timeout=60):
    log("Calling Mistral via Ollama...")

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": True
    }

    try:
        r = requests.post(OLLAMA_URL, json=payload, stream=True, timeout=timeout)
    except Exception as e:
        log(f"Error contacting Ollama: {e}")
        return ""

    final_text = ""

    try:
        for raw_line in r.iter_lines():
            if not raw_line:
                continue

            # Convert bytes → string
            if isinstance(raw_line, bytes):
                line = raw_line.decode("utf-8", errors="ignore").strip()
            else:
                line = str(raw_line).strip()

            if line.startswith("data:"):
                line = line[len("data:"):].strip()

            # Try JSON parsing
            try:
                data = json.loads(line)

                if isinstance(data, dict):
                    if "response" in data:
                        final_text += data["response"]
                    elif "text" in data:
                        final_text += data["text"]
                else:
                    final_text += str(data)

            except Exception:
                final_text += line

    except Exception as e:
        log(f"Streaming error: {e}\n{traceback.format_exc()}")

    final_text = final_text.strip()
    log(f"Mistral generated response (len={len(final_text)}).")
    return final_text


# --------------------
# Routes
# --------------------
@app.route("/")
def index():
    docs = [f.replace("_index.faiss", "") for f in os.listdir(INDEX_DIR) if f.endswith("_index.faiss")]
    docs.sort(reverse=True)
    return render_template("index.html", docs=docs, answer=None, pages=None, sections=None, selected_doc=None, query=None)


@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("file")
    if not file:
        flash("No file uploaded", "danger")
        return redirect(url_for("index"))

    filename = file.filename
    base_name = os.path.splitext(filename)[0]
    unique = uuid.uuid4().hex[:8]
    safe_base = f"{base_name}_{unique}"

    save_path = f"{UPLOAD_DIR}/{safe_base}.pdf"
    file.save(save_path)

    try:
        build_faiss_for_pdf(save_path, safe_base)
        flash("Document indexed successfully!", "success")
    except Exception as e:
        log(f"Indexing error: {e}")
        flash(f"Indexing error: {e}", "danger")

    return redirect(url_for("index"))


@app.route("/ask", methods=["POST"])
def ask():
    doc = request.form.get("doc")
    query = request.form.get("query")

    if not doc:
        flash("Select a document.", "danger")
        return redirect(url_for("index"))

    index_path = f"{INDEX_DIR}/{doc}_index.faiss"
    meta_path = f"{INDEX_DIR}/{doc}_meta.json"

    try:
        retrieved = retrieve_top_k(index_path, meta_path, query, k=4)
    except Exception as e:
        flash(f"Retrieval error: {e}", "danger")
        return redirect(url_for("index"))

    pages = []
    context = ""

    for r in retrieved:
        pages.append(r.get("page"))
        context += r.get("text", "") + "\n"

    if not context.strip():
        answer = "I couldn't find relevant content. Try another document."
        sections = []
    else:
        sections = extract_sections(context)

        prompt = f"""
You are a legal assistant. Use the context to answer clearly.

CONTEXT:
{context}

QUESTION:
{query}

Answer in simple English:
"""

        answer = call_ollama(prompt)
        if not answer:
            answer = "(Received empty response from LLM.)"

    docs = [f.replace("_index.faiss", "") for f in os.listdir(INDEX_DIR) if f.endswith("_index.faiss")]
    docs.sort(reverse=True)

    return render_template(
        "index.html",
        docs=docs,
        answer=answer,
        pages=pages,
        sections=sections,
        selected_doc=doc,
        query=query
    )


if __name__ == "__main__":
    app.run(debug=True, port=5001)
