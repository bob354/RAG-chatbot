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

load_dotenv()

def get_vector_store():
    embedding_model_name = os.getenv("HUGGINGFACE_EMBEDDING_MODEL")
    if not embedding_model_name:
        raise ValueError("HUGGINGFACE_EMBEDDING_MODEL environment variable is missing in .env.")

    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        model_kwargs={"trust_remote_code": True},
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

if vector_store is not None:
    base_retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 10
        }
    )
else:
    base_retriever = None

def rebuild_index():
    global vector_store, base_retriever
    print("Rebuilding index dynamically...")
    os.environ["FORCE_REINDEX"] = "true"
    vector_store = get_vector_store()
    if vector_store is not None:
        base_retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": 10
            }
        )
    else:
        base_retriever = None
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
    """Retrieve documents using similarity search and filter by active documents."""
    if vector_store is None:
        return []
    
    if active_docs is not None:
        docs = vector_store.similarity_search(query, k=20)
        active_filenames = {os.path.basename(f) for f in active_docs}
        filtered = []
        for doc in docs:
            doc_source = doc.metadata.get("source", "")
            doc_filename = os.path.basename(doc_source)
            if doc_filename in active_filenames:
                filtered.append(doc)
        return filtered[:10]
    else:
        if base_retriever is not None:
            return base_retriever.invoke(query)
        return []


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

template = (
    "You are a strict, citation-focused assistant for a private knowledge base.\n"
    "RULES:\n"
    "1) Use ONLY the provided context to answer.\n"
    "2) If the answer is not clearly contained in the context, say: "
    "\"I don't know based on the provided documents.\"\n"
    "3) Do NOT use outside knowledge, guessing, or web information.\n"
    "4) You MUST cite your sources using numbered references like [1], [2], etc. "
    "that correspond to the source numbers given in the context.\n"
    "5) Place the citation numbers inline right after the relevant sentence or claim.\n\n"
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
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

def chat(question: str, active_docs: list = None) -> dict:
    """Send a question to the RAG chain and return the answer with source citations."""
    retrieved_docs = retrieve_docs(question, active_docs)
    context = format_docs(retrieved_docs)
    sources = get_sources_metadata(retrieved_docs)
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"context": context, "question": question})
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

