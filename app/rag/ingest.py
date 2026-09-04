import json
import os
from dotenv import load_dotenv

from app.rag.embeddings import generate_embeddings
from app.rag.retrieve import get_chroma_collection

load_dotenv()

DEFAULT_CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
DEFAULT_CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
DEFAULT_CHROMA_PATH = os.getenv("CHROMA_PATH", "data/chroma")


def load_documents(directory: str = "documents") -> list[dict]:
    """Read .txt files from the directory and return document dictionaries."""
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Documents directory '{directory}' does not exist.")

    documents = []
    for filename in sorted(os.listdir(directory)):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath) and filename.endswith(".txt"):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                if content:
                    documents.append({"text": content, "source": filename})
            except Exception as exc:
                raise IOError(f"Failed to read document '{filepath}': {str(exc)}") from exc

    return documents


def chunk_text(
    text: str,
    source: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    start_chunk_id: int = 0,
) -> list[dict]:
    """Divide document text into overlapping chunks with metadata."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0.")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be non-negative and less than chunk_size.")

    cleaned = text.strip()
    if not cleaned:
        return []

    chunks = []
    step = chunk_size - chunk_overlap
    current_id = start_chunk_id

    for i in range(0, len(cleaned), step):
        chunk_content = cleaned[i : i + chunk_size].strip()
        if chunk_content:
            chunks.append(
                {
                    "chunk_id": current_id,
                    "source": source,
                    "text": chunk_content,
                }
            )
            current_id += 1
        if i + chunk_size >= len(cleaned):
            break

    return chunks


def store_in_chromadb(
    records: list[dict],
    chroma_path: str = DEFAULT_CHROMA_PATH,
    collection_name: str = "documents",
) -> None:
    """Store or upsert chunk records into ChromaDB using deterministic IDs."""
    if not records:
        return

    collection = get_chroma_collection(
        chroma_path=chroma_path,
        collection_name=collection_name,
    )

    ids = [f"{r['source']}_{r['chunk_id']}" for r in records]
    embeddings = [r["embedding"] for r in records]
    documents = [r["text"] for r in records]
    metadatas = [{"source": r["source"], "chunk_id": r["chunk_id"]} for r in records]

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )


def run_ingestion(
    docs_dir: str = "documents",
    output_path: str = "data/embeddings.json",
    chroma_path: str = DEFAULT_CHROMA_PATH,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[dict]:
    """Execute document loading, chunking, embedding generation, ChromaDB upsert, and local persistence."""
    documents = load_documents(docs_dir)
    if not documents:
        raise ValueError(f"No non-empty .txt documents found in '{docs_dir}'.")

    chunks = []
    chunk_id = 0
    for doc in documents:
        doc_chunks = chunk_text(
            text=doc["text"],
            source=doc["source"],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            start_chunk_id=chunk_id,
        )
        chunks.extend(doc_chunks)
        chunk_id += len(doc_chunks)

    if not chunks:
        raise ValueError("No text chunks generated from documents.")

    texts = [c["text"] for c in chunks]
    embeddings = generate_embeddings(texts)

    if len(embeddings) != len(chunks):
        raise RuntimeError(
            f"Embedding mismatch: received {len(embeddings)} embeddings for {len(chunks)} chunks."
        )

    records = []
    for chunk, emb in zip(chunks, embeddings):
        records.append(
            {
                "chunk_id": chunk["chunk_id"],
                "source": chunk["source"],
                "text": chunk["text"],
                "embedding": emb,
            }
        )

    # Upsert into ChromaDB
    store_in_chromadb(records=records, chroma_path=chroma_path)

    # Save to data/embeddings.json
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    return records


if __name__ == "__main__":
    docs_dir = "documents"
    output_path = "data/embeddings.json"

    docs = load_documents(docs_dir)
    records = run_ingestion(docs_dir=docs_dir, output_path=output_path)

    print(f"Loaded: {len(docs)} document{'s' if len(docs) != 1 else ''}")
    print(f"Created: {len(records)} chunks")
    print(f"Generated: {len(records)} embeddings")
    print(f"Stored in ChromaDB: {DEFAULT_CHROMA_PATH} (collection: documents)")
    print(f"Saved: {output_path}")
