A lightweight Flask + FAISS + OCR + Mistral (Ollama) project that allows you to:

✔ Upload a PDF document
✔ Extract text (page-wise)
✔ Chunk it
✔ Embed using Sentence-Transformers
✔ Store embeddings in a FAISS vector store
✔ Query using RAG
✔ Use Mistral (Ollama locally) to generate answers
✔ Highlight legal sections found in retrieved text

🚀 1. Project Overview

This is a RAG (Retrieval-Augmented Generation) pipeline designed for legal document question answering.

When a user uploads a PDF:

The PDF is saved in /uploads.

The text is extracted (OCR handled automatically by PyMuPDF/fitz).

The text is chunked into small pieces.

Each chunk is embedded using Sentence-Transformers (MiniLM).

A FAISS vector index is created and stored in /indexes.

Metadata (chunk text + page number) is saved in JSON.

When the user asks a question:

System loads the FAISS index.

Converts the question into an embedding.

Searches top-k most relevant chunks.

Extracts legal sections using regex patterns.

Sends the RAG context + user question to Mistral (via Ollama).

Displays generated answer, relevant pages, and legal sections.

🏗 2. Folder Structure
project/
│── app.py                # Main Flask application
│── templates/
│    └── index.html       # UI
│── uploads/              # Uploaded PDFs
│── indexes/              # FAISS index + metadata
│── static/               # CSS/JS (optional)

⚙️ 3. How the Pipeline Works (Simple Explanation)
STEP 1 — Upload & Save PDF

The file is uploaded via /upload.
It gets saved as:

uploads/<filename>.pdf


A unique ID is appended to avoid collisions.

STEP 2 — Extract Text from PDF

Using PyMuPDF:

doc = fitz.open(pdf_path)
page.get_text()


Each page → stored as {page_number, text}.

STEP 3 — Chunking

Large text is broken into smaller chunks (500 tokens each):

chunk1
chunk2
chunk3
...


This helps better retrieval.

STEP 4 — Embeddings

Every chunk is converted into a 384-dim vector using:

sentence-transformers/all-MiniLM-L6-v2


These embeddings are stored in a FAISS index.

STEP 5 — Save FAISS Index + Metadata

Two files are written:

indexes/doc_index.faiss
indexes/doc_meta.json


meta.json stores mapping:

[
  {
    "page": 1,
    "text": "Chunk summary..."
  }
]

🔎 4. Querying (Ask a Question)
STEP 6 — Retrieve Relevant Chunks

When a query is asked:

Convert the query to embedding

Search top-k similar vectors in FAISS

Fetch corresponding chunk texts from metadata

Combine these into a reference context

STEP 7 — Extract Legal Sections

Regex patterns detect:

“Section 420”

“U/s 302”

“Sec 125”

“304 IPC”

etc.

STEP 8 — Generate Final Answer (LLM)

Prompt is formed:

CONTEXT:
<retrieved chunks>

QUESTION:
<user question>


This is sent to Ollama:

http://localhost:11434/api/generate


Streaming is supported.

STEP 9 — Display on UI

UI shows:

✔ Final answer
✔ Pages used
✔ Extracted legal sections
✔ Previously indexed documents
