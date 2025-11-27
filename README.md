
# 🚀 RAG-Powered AI Assistant: Flask + FAISS + OCR + Mistral (Ollama)

A lightweight, local **Retrieval-Augmented Generation (RAG)** pipeline designed as an **AI Assistant** for specialized document analysis, focusing on **legal document question answering**. This project utilizes Flask for the frontend, FAISS for vector search, and Mistral (via Ollama) for powerful, context-aware generation.

## ✔ Core Capabilities

The project provides a complete end-to-end workflow allowing users to:

  * ✔ **Upload a PDF document**
  * ✔ Extract text (page-wise) and **Chunk** it
  * ✔ Embed using **Sentence-Transformers**
  * ✔ Store embeddings in a **FAISS vector store**
  * ✔ Query using **RAG** (Retrieval-Augmented Generation)
  * ✔ Use **Mistral (Ollama locally)** to generate answers
  * ✔ **Highlight legal sections** found in retrieved text

-----

## 💡 Project Overview: The RAG Pipeline

This system ensures that all generated answers are strictly grounded in the content of the uploaded PDF, making it highly reliable for technical and sensitive documents.

### A. Indexing (When a user uploads a PDF):

  * The PDF is saved in `/uploads`.
  * The text is extracted (OCR handled automatically by **PyMuPDF/fitz**).
  * The text is chunked into small pieces.
  * Each chunk is embedded using **Sentence-Transformers (MiniLM)**.
  * A **FAISS vector index** is created and stored in `/indexes`.
  * Metadata (chunk text + page number) is saved in JSON for retrieval mapping.

### B. Querying (When the user asks a question):

  * The system loads the FAISS index.
  * Converts the question into an embedding.
  * Searches **top-k** most relevant chunks.
  * Extracts **legal sections** using regex patterns.
  * Sends the combined RAG context + user question to **Mistral (via Ollama)**.
  * Displays the generated answer, relevant pages, and detected legal sections on the UI.

-----

## 🏗 Folder Structure

The project maintains a simple, modular structure:

```
project/
├── app.py          # Main Flask application logic (upload, indexing, querying)
├── templates/
│   └── index.html  # Frontend UI for upload and query
├── uploads/        # Stores uploaded PDF documents
├── indexes/        # Stores FAISS index (.faiss) and metadata (.json)
└── static/         # CSS/JS and other static assets (optional)
```

-----

## ⚙️ How the Pipeline Works (Step-by-Step)

### I. Data Ingestion & Preparation

| Step | Description | Detail |
| :--- | :--- | :--- |
| **STEP 1** | **Upload & Save PDF** | The file is uploaded via `/upload` and saved as `uploads/<filename>.pdf`. A unique ID is appended to filenames to avoid collisions. |
| **STEP 2** | **Extract Text from PDF** | Uses PyMuPDF (`fitz.open(pdf_path)`) for extraction (`page.get_text()`). Each page's text is stored as `{page_number, text}`. |
| **STEP 3** | **Chunking** | Large text is broken into smaller chunks (e.g., **500 tokens** each) to ensure precise vector search results. |
| **STEP 4** | **Embeddings** | Every chunk is converted into a **384-dim vector** using `sentence-transformers/all-MiniLM-L6-v2`. |
| **STEP 5** | **Save FAISS Index + Metadata** | Two files are persisted: `indexes/doc_index.faiss` and `indexes/doc_meta.json`. The JSON stores the chunk-to-page mapping. |

### II. Querying & Generation

| Step | Description | Detail |
| :--- | :--- | :--- |
| **STEP 6** | **Retrieve Relevant Chunks** | The query is embedded, and a search is performed against FAISS for the top-k similar vectors. The corresponding chunk texts are fetched and combined into the **reference CONTEXT**. |
| **STEP 7** | **Extract Legal Sections** | Regex patterns scan the retrieved chunks to detect specific legal references, such as: *"Section 420"*, *"U/s 302"*, *"Sec 125"*, *"304 IPC"*, etc. |
| **STEP 8** | **Generate Final Answer (LLM)** | The complete prompt (`CONTEXT: <retrieved chunks> QUESTION: <user question>`) is sent to the local Mistral model via the **Ollama API** at `http://localhost:11434/api/generate`. Streaming is supported. |
| **STEP 9** | **Display on UI** | The Flask UI presents the final output, including: **✔ Final answer**, **✔ Pages used**, **✔ Extracted legal sections**, and a list of **✔ Previously indexed documents**. |
