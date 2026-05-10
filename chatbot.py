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

load_dotenv()


loader = DirectoryLoader(
    path = "documents", 
    glob="*.pdf",
    loader_cls=PDFPlumberLoader,
    show_progress=True,
    use_multithreading=True
    )

docs = loader.load()

MARKDOWN_SEPARATORS = [
    "\n#{1,6} ",
    "```\n",
    "\n\\*\\*\\*+\n",
    "\n---+\n",
    "\n___+\n",
    "\n\n",
    "\n",
    " ",
    "",
]

text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ". ", " ", ""],
    chunk_size=1200,
    chunk_overlap=200,
    add_start_index=True,
    strip_whitespace=True
)

split_docs = text_splitter.split_documents(docs)

# print(split_docs)

embeddings = HuggingFaceEmbeddings(
    model_name=os.getenv("HUGGINGFACE_EMBEDDING_MODEL"),
    model_kwargs={"trust_remote_code": True},
    encode_kwargs={"normalize_embeddings": True}
)

vector_store = FAISS.from_documents(
    documents=split_docs, 
    embedding=embeddings,
    distance_strategy=DistanceStrategy.COSINE
)

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 5,  # number of documents to retrieve
    }
)

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

llm = ChatGroq(
    model=os.getenv("GROQ_MODEL"),
    api_key=os.getenv("GROQ_API_KEY"),
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
        "context": retriever | format_docs,
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

