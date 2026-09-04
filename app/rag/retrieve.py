import os
import chromadb
from dotenv import load_dotenv

from app.rag.embeddings import generate_embedding

load_dotenv()

DEFAULT_CHROMA_PATH = os.getenv("CHROMA_PATH", "data/chroma")
DEFAULT_RAG_TOP_K = int(os.getenv("RAG_TOP_K", "3"))
DEFAULT_COLLECTION_NAME = "documents"


def get_chroma_collection(
    chroma_path: str = DEFAULT_CHROMA_PATH,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> chromadb.Collection:
    """Retrieve or create persistent ChromaDB collection."""
    client = chromadb.PersistentClient(path=chroma_path)
    return client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )


def retrieve_relevant_chunks(
    question: str,
    top_k: int = DEFAULT_RAG_TOP_K,
    chroma_path: str = DEFAULT_CHROMA_PATH,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> list[dict]:
    """Retrieve top-k relevant document chunks for a question using vector similarity search."""
    if not question or not question.strip():
        return []

    try:
        collection = get_chroma_collection(
            chroma_path=chroma_path,
            collection_name=collection_name,
        )
    except Exception as exc:
        raise RuntimeError(f"ChromaDB connection error: {str(exc)}") from exc

    try:
        count = collection.count()
    except Exception as exc:
        raise RuntimeError(f"Failed to query ChromaDB collection count: {str(exc)}") from exc

    if count == 0:
        return []

    # Generate embedding for user question
    query_embedding = generate_embedding(question.strip())

    try:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, count),
            include=["documents", "metadatas", "distances"],
        )
    except Exception as exc:
        raise RuntimeError(f"ChromaDB similarity search error: {str(exc)}") from exc

    chunks = []
    if results and "documents" in results and results["documents"]:
        docs = results["documents"][0]
        metas = (
            results["metadatas"][0]
            if "metadatas" in results and results["metadatas"]
            else [{}] * len(docs)
        )
        distances = (
            results["distances"][0]
            if "distances" in results and results["distances"]
            else [None] * len(docs)
        )

        for doc_text, meta, dist in zip(docs, metas, distances):
            chunks.append(
                {
                    "text": doc_text,
                    "source": meta.get("source", "unknown"),
                    "chunk_id": meta.get("chunk_id", 0),
                    "distance": dist,
                }
            )

    return chunks
