import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from langchain_community.retrievers import BM25Retriever
from sentence_transformers import CrossEncoder

import torch

load_dotenv()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device.upper()}")

def get_vector_store():
    embedding_model_name = os.getenv("HUGGINGFACE_EMBEDDING_MODEL")
    if not embedding_model_name:
        raise ValueError("HUGGINGFACE_EMBEDDING_MODEL environment variable is missing in .env.")

    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        model_kwargs={"trust_remote_code": True, "device": device},
        encode_kwargs={"normalize_embeddings": True}
    )

    index_path = "vector_db"
    force_reindex = os.getenv("FORCE_REINDEX", "false").lower() == "true"

    if os.path.exists(index_path) and not force_reindex:
        print("Loading existing FAISS index from disk...")
        try:
            return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
        except Exception as e:
            print(f"Error loading index: {e}. Rebuilding...")

    print("Building new FAISS index...")
    documents_dir = "documents"
    if not os.path.exists(documents_dir) or not os.listdir(documents_dir):
        print(f"The '{documents_dir}' folder is empty or does not exist. Clearing vector store index...")
        import shutil
        if os.path.exists(index_path):
            try:
                shutil.rmtree(index_path)
            except Exception as e:
                print(f"Error deleting index directory: {e}")
        return None

    loader = DirectoryLoader(
        path=documents_dir, 
        glob="*.pdf",
        loader_cls=PDFPlumberLoader,
        show_progress=True,
        use_multithreading=True
    )
    
    docs = loader.load()
    if not docs:
        raise ValueError("No documents loaded. Ensure there are readable PDFs in the 'documents' folder.")

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=1200,
        chunk_overlap=200,
        add_start_index=True,
        strip_whitespace=True
    )
    
    split_docs = text_splitter.split_documents(docs)

    vector_store = FAISS.from_documents(
        documents=split_docs, 
        embedding=embeddings,
        distance_strategy=DistanceStrategy.COSINE
    )
    
    vector_store.save_local(index_path)
    print(f"Index successfully saved to {index_path}.")
    return vector_store

vector_store = get_vector_store()

# Initialize retrievers and rerankers
base_retriever = None
bm25_retriever = None

print("Loading CrossEncoder reranker model (BAAI/bge-reranker-base)...")
reranker = CrossEncoder("BAAI/bge-reranker-base", device=device)

if device == "cpu":
    print("Applying dynamic quantization to CrossEncoder for CPU speedup...")
    try:
        reranker.model = torch.quantization.quantize_dynamic(
            reranker.model,
            {torch.nn.Linear},
            dtype=torch.qint8
        )
    except Exception as e:
        print(f"Failed to quantize model: {e}")

RERANK_TOP_N = 10

def init_bm25():
    global bm25_retriever
    if vector_store is not None:
        try:
            # Extract all documents from FAISS docstore
            faiss_docs = list(vector_store.docstore._dict.values())
            if faiss_docs:
                print(f"Initializing BM25 retriever with {len(faiss_docs)} document chunks...")
                bm25_retriever = BM25Retriever.from_documents(faiss_docs)
                bm25_retriever.k = 20  # Fetch top 20 keyword matches
            else:
                bm25_retriever = None
        except Exception as e:
            print(f"Error initializing BM25: {e}")
            bm25_retriever = None
    else:
        bm25_retriever = None

if vector_store is not None:
    base_retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 20
        }
    )
    init_bm25()
else:
    base_retriever = None

def rebuild_index():
    global vector_store, base_retriever, bm25_retriever
    print("Rebuilding index dynamically...")
    os.environ["FORCE_REINDEX"] = "true"
    vector_store = get_vector_store()
    if vector_store is not None:
        base_retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": 20
            }
        )
        init_bm25()
    else:
        base_retriever = None
        bm25_retriever = None
    os.environ["FORCE_REINDEX"] = "false"
    print("Index rebuild completed.")


def _matches_active_docs(doc, active_docs: list = None) -> bool:
    if not active_docs:
        return True

    active_filenames = {os.path.basename(file_path) for file_path in active_docs}
    doc_source = doc.metadata.get("source", "")
    doc_filename = os.path.basename(doc_source)
    return doc_filename in active_filenames

def retrieve_docs(query: str, active_docs: list = None):
    """Retrieve documents using Hybrid Search (BM25 + Semantic) and filter/rerank them."""
    if vector_store is None:
        return []
    
    # 1. Fetch vector candidates
    if active_docs is not None:
        # If active_docs is specified, we retrieve and manually filter to match them
        vector_candidates = vector_store.similarity_search(query, k=30)
        active_filenames = {os.path.basename(f) for f in active_docs}
        vector_docs = [
            doc for doc in vector_candidates 
            if os.path.basename(doc.metadata.get("source", "")) in active_filenames
        ][:20]
    else:
        if base_retriever is not None:
            vector_docs = base_retriever.invoke(query)
        else:
            vector_docs = []

    # 2. Fetch BM25 candidates
    bm25_docs = []
    if active_docs is not None:
        # Filter all documents to only contain active ones, and build temporary BM25
        active_filenames = {os.path.basename(f) for f in active_docs}
        all_docs = list(vector_store.docstore._dict.values())
        filtered_all = [
            doc for doc in all_docs 
            if os.path.basename(doc.metadata.get("source", "")) in active_filenames
        ]
        if filtered_all:
            try:
                temp_bm25 = BM25Retriever.from_documents(filtered_all)
                temp_bm25.k = min(20, len(filtered_all))
                bm25_docs = temp_bm25.invoke(query)
            except Exception as e:
                print(f"Error building temporary BM25: {e}")
    else:
        if bm25_retriever is not None:
            try:
                bm25_docs = bm25_retriever.invoke(query)
            except Exception as e:
                print(f"Error running global BM25 query: {e}")

    # 3. Merge and deduplicate candidates
    seen = set()
    candidates = []
    for doc in vector_docs + bm25_docs:
        doc_id = (doc.page_content, doc.metadata.get("source", ""), doc.metadata.get("page", ""))
        if doc_id not in seen:
            seen.add(doc_id)
            candidates.append(doc)

    if not candidates:
        return []

    # 4. Rerank candidates using CrossEncoder reranker
    try:
        pairs = [[query, doc.page_content] for doc in candidates]
        scores = reranker.predict(pairs)
        scored_docs = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
        # Select top RERANK_TOP_N documents
        reranked_docs = [doc for _, doc in scored_docs[:RERANK_TOP_N]]
        return reranked_docs
    except Exception as e:
        print(f"Error during reranking: {e}. Returning raw candidates.")
        return candidates[:RERANK_TOP_N]


def search_docs(query: str, active_docs: list = None, max_results: int = 8):
    """Return ranked document snippets for notebook-style source search."""
    if vector_store is None or not query.strip():
        return []

    fetch_limit = max(max_results * 3, max_results)

    try:
        scored_docs = vector_store.similarity_search_with_relevance_scores(query, k=fetch_limit)
    except Exception:
        scored_docs = [(doc, None) for doc in vector_store.similarity_search(query, k=fetch_limit)]

    results = []
    for doc, score in scored_docs:
        if not _matches_active_docs(doc, active_docs):
            continue

        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", None)
        snippet = " ".join(doc.page_content.split())
        if len(snippet) > 320:
            snippet = snippet[:317].rstrip() + "..."

        results.append({
            "source": os.path.basename(source) or source,
            "page": page,
            "snippet": snippet,
            "score": round(float(score), 4) if score is not None else None,
        })

        if len(results) >= max_results:
            break

    return results

# ── Query Condensation ──
HISTORY_MSG_MAX_CHARS = 600

condense_template = (
    "Given the following conversation history and a follow-up question, "
    "rewrite the follow-up question into a standalone question that captures "
    "the full intent. Output ONLY the rewritten question, nothing else.\n\n"
    "Chat History:\n{chat_history}\n\n"
    "Follow-up Question: {question}\n\n"
    "Standalone Question:"
)
condense_prompt = ChatPromptTemplate.from_template(condense_template)

def _truncate(text: str, max_chars: int = HISTORY_MSG_MAX_CHARS) -> str:
    """Truncate text to max_chars, appending '...' if truncated."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."

def _format_history(history: list) -> str:
    """Format history list into a readable string, truncating each message."""
    lines = []
    for msg in history:
        role = msg.get("role", "user").capitalize()
        content = _truncate(msg.get("content", ""))
        lines.append(f"{role}: {content}")
    return "\n".join(lines)

def condense_query(question: str, history: list) -> str:
    """Rewrite follow-up question into standalone query using LLM.
    Falls back to the original question on any failure."""
    if not history:
        return question
    try:
        formatted_history = _format_history(history)
        chain = condense_prompt | llm | StrOutputParser()
        rewritten = chain.invoke({
            "chat_history": formatted_history,
            "question": question
        }).strip()
        # Fallback: if rewritten is empty or suspiciously short
        if not rewritten or len(rewritten) < 3:
            print(f"[Condense] Fallback: rewritten query too short ('{rewritten}'). Using original.")
            return question
        print(f"[Condense] '{question}' → '{rewritten}'")
        return rewritten
    except Exception as e:
        print(f"[Condense] Error: {e}. Using original query.")
        return question

# ── Main RAG Prompt ──
template = (
    "You are a strict, citation-focused assistant for a private knowledge base.\n"
    "Available Documents in Knowledge Base:\n{available_docs}\n\n"
    "RULES:\n"
    "1) Use ONLY the provided context to answer questions about document content. If the user asks about the status of the database or what documents are uploaded, you may answer using the 'Available Documents in Knowledge Base' list above.\n"
    "2) If the answer is not clearly contained in the context (or in the available documents list for database/file queries), say: "
    "\"I don't know based on the provided documents.\"\n"
    "3) Do NOT use outside knowledge, guessing, or web information.\n"
    "4) You MUST cite your sources using numbered references like [1], [2], etc. "
    "that correspond to the source numbers given in the context.\n"
    "5) Place the citation numbers inline right after the relevant sentence or claim.\n"
    "6) If there is chat history, use it for conversational context but still ground your answer in the retrieved context.\n\n"
    "{chat_history_section}"
    "Context:\n{context}\n\n"
    "Question: {question}"
)

prompt = ChatPromptTemplate.from_template(template)

groq_api_key = os.getenv("GROQ_API_KEY")
groq_model = os.getenv("GROQ_MODEL")

if not groq_api_key or not groq_model:
    raise ValueError("GROQ_API_KEY and GROQ_MODEL environment variables must be set in .env.")

llm = ChatGroq(
    model=groq_model,
    api_key=groq_api_key,
    temperature=0
)

def format_docs(docs):
    """Convert retrieved Document objects into numbered text with source citations."""
    formatted = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        formatted.append(f"[{i}] (Source: {os.path.basename(source)}, Page: {page})\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)

def get_sources_metadata(docs):
    """Extract source metadata from retrieved documents for frontend citation display."""
    sources = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", None)
        snippet = doc.page_content.strip()
        sources.append({
            "source": os.path.basename(source),
            "page": page,
            "snippet": snippet,
        })
    return sources

rag_chain = (
    {
        "context": lambda q: format_docs(retrieve_docs(q)),
        "question": RunnablePassthrough(),
        "available_docs": lambda q: ", ".join(os.listdir("documents")) if os.path.exists("documents") else "None"
    }
    | prompt
    | llm
    | StrOutputParser()
)

def chat(question: str, active_docs: list = None, history: list = None) -> dict:
    """Send a question to the RAG chain and return the answer with source citations.
    
    Args:
        question: The user's current question.
        active_docs: Optional list of active document file paths.
        history: Optional list of chat history messages [{"role": ..., "content": ...}].
                 Should be capped to last 5 turns (10 messages) by the caller.
    """
    # 1. Condense follow-up questions into standalone queries for retrieval
    retrieval_query = condense_query(question, history) if history else question
    
    # 2. Retrieve using the rewritten (standalone) query
    retrieved_docs = retrieve_docs(retrieval_query, active_docs)
    context = format_docs(retrieved_docs)
    sources = get_sources_metadata(retrieved_docs)
    
    # 3. Build available docs list
    if active_docs:
        docs_list = [os.path.basename(f) for f in active_docs]
    else:
        docs_dir = "documents"
        if os.path.exists(docs_dir):
            docs_list = [f for f in os.listdir(docs_dir) if os.path.isfile(os.path.join(docs_dir, f))]
        else:
            docs_list = []
    available_docs_str = "\n".join(f"- {name}" for name in docs_list) if docs_list else "- None"

    # 4. Build chat history section for the final prompt
    if history:
        chat_history_section = "Chat History:\n" + _format_history(history) + "\n\n"
    else:
        chat_history_section = ""

    # 5. Generate answer using the ORIGINAL question (not rewritten)
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({
        "context": context,
        "question": question,
        "available_docs": available_docs_str,
        "chat_history_section": chat_history_section
    })
    return {"answer": response, "sources": sources}

if __name__ == "__main__":
    import time
    question = input("Question: ")
    start_time = time.time()
    result = chat(question)
    print(f"Elapsed time (Sim Only): {time.time() - start_time:.2f} seconds")
    print(result["answer"])
    print(f"\nSources used: {len(result['sources'])}")
    for i, src in enumerate(result["sources"], 1):
        print(f"  [{i}] {src['source']} (Page {src['page']})")

