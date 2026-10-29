from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHROMA_DIR = "vector_db"
COLLECTION_NAME = "meeting_transcript"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def get_embeddings():
    """Create the embedding model (runs locally on CPU)."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        # Normalized embeddings give more reliable similarity search results
        encode_kwargs={"normalize_embeddings": True},
    )


def build_vector_store(transcript: str) -> Chroma:
    """Split the transcript into chunks, embed them and store them in Chroma."""
    print("Building vector store...")

    embeddings = get_embeddings()

    # Remove the old collection so a new transcript does not mix with the previous one
    try:
        Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=CHROMA_DIR,
        ).delete_collection()
    except Exception:
        # Nothing to delete on the first run
        pass

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(transcript)

    docs = [
        Document(page_content=chunk, metadata={"chunk_index": i})
        for i, chunk in enumerate(chunks)
    ]

    vector_store = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
    )
    print(f"Vector store ready - {len(docs)} chunk(s) stored.")
    return vector_store


def load_vector_store() -> Chroma:
    """Load a previously saved vector store from disk."""
    return Chroma(
        embedding_function=get_embeddings(),
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
    )


def get_retriever(vector_store: Chroma, k: int = 4):
    """Return a retriever that fetches the k most similar transcript chunks."""
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )