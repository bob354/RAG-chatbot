import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain.chains import create_retrieval_chain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from sentence_transformers import CrossEncoder

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
        return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)

    print("Building new FAISS index...")
    documents_dir = "documents"
    if not os.path.exists(documents_dir) or not os.listdir(documents_dir):
        raise ValueError(f"The '{documents_dir}' folder is empty or does not exist. Please add PDF files.")

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
        chunk_size=700,
        chunk_overlap=100,
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

base_retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 15
    }
)

reranker = CrossEncoder("BAAI/bge-reranker-base")
RERANK_TOP_N = 5


def rerank_docs(query: str):
    """Retrieve documents, then rerank with a cross-encoder and return top N."""
    docs = base_retriever.invoke(query)
    if not docs:
        return []
    pairs = [[query, doc.page_content] for doc in docs]
    scores = reranker.predict(pairs)
    scored_docs = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored_docs[:RERANK_TOP_N]]

template = (
    "You are a strict, citation-focused assistant for a private knowledge base.\n"
    "RULES:\n"
    "1) Use ONLY the provided context to answer.\n"
    "2) If the answer is not clearly contained in the context, say: "
    "\"I don't know based on the provided documents.\"\n"
    "3) Do NOT use outside knowledge, guessing, or web information.\n"
    "4) If applicable, cite sources as (source:page) using the metadata.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}"
)

prompt = ChatPromptTemplate.from_template(template)

# llm = HuggingFaceEndpoint(
#     repo_id=os.getenv("HUGGINGFACE_LLM_MODEL"),
#     huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
#     temperature=0,
#     # max_new_tokens=2048,
#     # task="text-generation",
# )

groq_api_key = os.getenv("GROQ_API_KEY")
groq_model = os.getenv("GROQ_MODEL")

if not groq_api_key or not groq_model:
    raise ValueError("GROQ_API_KEY and GROQ_MODEL environment variables must be set in .env.")

llm = ChatGroq(
    model=groq_model,
    api_key=groq_api_key,
    temperature=0
)

# Build RAG chain
def format_docs(docs):
    """Convert retrieved Document objects into readable text with source citations."""
    formatted = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        formatted.append(f"[Source: {source}, Page: {page}]\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)

rag_chain = (
    {
        "context": lambda q: format_docs(rerank_docs(q)),
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)


def chat(question: str) -> str:
    """Send a question to the RAG chain and return the answer."""
    response = rag_chain.invoke(question)
    return response


if __name__ == "__main__":
    question = input("Question: ")
    response = chat(question)
    print(response)
