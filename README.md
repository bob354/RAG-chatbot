# RAG Knowledge Assistant

A Retrieval-Augmented Generation (RAG) chatbot that answers questions based on your PDF documents. Built with **LangChain**, **FAISS**, **BM25**, **Groq (LLaMA 3.1)**, **Cross-Encoder Reranking**, and **Flask**.

---

## Architecture

```text
User Question
     │
     ▼
┌──────────┐    ┌───────────────────────────┐     ┌────────────┐
│  Flask   │───▶│  Hybrid Search (LangChain)│───▶│   Groq     │
│  Web UI  │    │  (FAISS + BM25)           │     │  LLaMA 3.1 │
└──────────┘    └───────┬───────────────────┘     └────────────┘
                        │
                 ┌──────┴──────┐
                 │  Reranker   │
                 │(CrossEncoder│
                 └──────┬──────┘
                        │
                 ┌──────┴──────┐
                 │    PDFs     │
                 │ (documents/)│
                 └─────────────┘
```

**How it works:**

1. PDFs in the `documents/` folder are loaded using `pdfplumber` and split into chunks. You can dynamically upload/delete PDFs via the Web UI, which will automatically rebuild the index.
2. Chunks are embedded using a HuggingFace sentence-transformer model and stored in a **FAISS** vector store. The model dynamically utilizes **CUDA (GPU)** if available.
3. Simultaneously, a **BM25** index is built for keyword search.
4. When a user asks a follow-up question, the system uses **Query Condensation** with the conversational history to rewrite it into a standalone query.
5. Documents are then retrieved using **Hybrid Search** (Semantic + Keyword) to capture both contextual meaning and exact phrasing.
6. Candidates are reranked using a Cross-Encoder model (`BAAI/bge-reranker-base`). The model uses **dynamic quantization** to speed up inference if running on CPU.
7. The top retrieved context, chat history, and question are sent to the Groq-hosted LLaMA 3.1 model to generate a strict, citation-focused answer.

---

## Prerequisites

- **Python 3.10+**
- **A Groq API key** — get one for free at [console.groq.com](https://console.groq.com/)
- **PDF documents** you want the chatbot to answer questions about

---

## Setup Guide

### 1. Clone the repository

```bash
git clone https://github.com/bob354/RAG-chatbot.git
cd RAG-chatbot
```

### 2. Setup Virtual Environment

#### Option 1: venv (recommended)
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux
pip install -r requirements.txt
```

#### Option 2: conda (if you have Anaconda)
```bash
conda create -n rag_chatbot python=3.10 -y
conda activate rag_chatbot
pip install -r requirements.txt
```

> **Note:** The HuggingFace embedding model and the Cross-Encoder reranker model will be downloaded automatically on first run.

### 3. Configure environment variables

Create a `.env` file in the project root (or edit the existing one):

```env
# ── Embedding Model ──
HUGGINGFACE_EMBEDDING_MODEL="doan2506/vietnamese-bi-encoder-finetuned"

# ── LLM (Groq) ──
GROQ_API_KEY=your-groq-api-key-here
GROQ_MODEL="llama-3.1-8b-instant"
```

| Variable                       | Description                                                                                          |
| ------------------------------ | ---------------------------------------------------------------------------------------------------- |
| `HUGGINGFACE_EMBEDDING_MODEL`  | The sentence-transformer model used to create embeddings. The default works well for multilingual text. |
| `GROQ_API_KEY`                 | Your Groq API key. Get one at [console.groq.com](https://console.groq.com/).                         |
| `GROQ_MODEL`                   | The LLM model hosted on Groq. `llama-3.1-8b-instant` is fast and free-tier friendly.                 |

### 4. Run the application

```bash
python app.py
```

The server will start at: **http://localhost:5002**

Open this URL in your browser to start chatting!

---

## Usage

### Web Interface

1. Open **http://localhost:5002** in your browser.
2. Upload your PDF documents via the UI or place them directly in the `documents/` folder.
3. Type a question about your documents in the input field. You can also select specific active documents to filter the search space.
4. Press **Enter** (or click the send button) to submit.
5. The chatbot will retrieve relevant passages via Hybrid Search + Reranking and generate an answer with source citations.

### Command Line

You can also use the chatbot directly from the terminal:

```bash
python chatbot.py
```

This will prompt you for a question and print the answer to the console.

### Main API Endpoints

#### Chat
Send a POST request to `/api/chat`:

```bash
curl -X POST http://localhost:5002/api/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the main topic of the document?", "history": [{"role": "user", "content": "Hello!"}]}'
```

#### Source Search Endpoint
Use `/api/search` to get ranked snippets from the indexed documents, similar to NotebookLM-style source search:

```bash
curl -X POST http://localhost:5002/api/search \
    -H "Content-Type: application/json" \
    -d '{"query": "main topic", "active_docs": ["paper.pdf"], "max_results": 8}'
```

#### Document Management
- **`GET /api/documents`**: List uploaded documents.
- **`POST /api/upload`**: Upload new PDFs (multipart/form-data).
- **`DELETE /api/documents/<filename>`**: Delete a specific PDF document.

*(Uploading or deleting documents automatically triggers an index rebuild).*

---

## Project Structure

```text
RAG-chatbot/
├── app.py              # Flask web server (routes & APIs for chat, upload,delete)
├── chatbot.py          # RAG pipeline 
├── requirements.txt    # Python dependencies
├── .env                # Environment variables (API keys, model names)
├── documents/          # Place your PDF files here (managed via UI as well)
│   └── *.pdf
├── vector_db/          # Persistent FAISS vector store
└── templates/
    └── index.html      # Chat web interface
```

---

## Configuration & Tuning

### Chunk Size & Overlap

In `chatbot.py`, you can adjust how documents are split. Currently utilizing `PDFPlumberLoader` and refined text splitting:

```python
text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ". ", " ", ""],
    chunk_size=700,      # Max characters per chunk (depends on embedding models max sequence length)
    chunk_overlap=100,    # Overlap between chunks (helps preserve context at boundaries)
    add_start_index=True,
    strip_whitespace=True
)
```

### Retrieval & Reranking Settings

The pipeline uses a combination of FAISS (semantic) and BM25 (keyword) before reranking:

```python
# Number of document chunks to retrieve per search strategy (BM25 and Semantic)
search_kwargs={"k": 20} 

# Number of top documents to keep after CrossEncoder reranking
RERANK_TOP_N = 10
```

### LLM Temperature

```python
llm = ChatGroq(
    temperature=0  # 0 = deterministic, higher = more creative
)
```

---

## Troubleshooting

| Issue                                  | Solution                                                                                     |
| -------------------------------------- | -------------------------------------------------------------------------------------------- |
| `ModuleNotFoundError`                  | Make sure your virtual environment is activated and run `pip install -r requirements.txt`     |
| `GROQ_API_KEY` errors                  | Verify your `.env` file has a valid Groq API key                                             |
| Slow first startup                     | Embedding and Reranking models are being downloaded. Subsequent runs will use the cache.     |
| `No documents found` / empty responses | Upload `.pdf` files via the UI or place them in the `documents/` folder                      |
| Port 5002 already in use               | Change the port in `app.py`: `app.run(port=5003)`                                            |

---

## License

This project is for educational purposes.
