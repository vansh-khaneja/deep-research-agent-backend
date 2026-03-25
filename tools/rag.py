# ============================================
# RAG Module — Session-Scoped Document Intelligence
# - Each research session gets its own ChromaDB collection
# - Upload PDFs during plan review, used only for that session
# - Collection deleted when session ends
# ============================================

import os
import logging
import hashlib
from typing import List, Dict, Any

import chromadb
from langchain_openai import OpenAIEmbeddings

from config import settings

logger = logging.getLogger("deep-research")

# --- ChromaDB Client (shared, but collections are per-session) ---

_client = None


def _get_client():
    global _client
    if _client is None:
        os.makedirs(settings.chromadb_path, exist_ok=True)
        _client = chromadb.PersistentClient(path=settings.chromadb_path)
    return _client


def _get_collection(session_id: str):
    """Get or create a session-scoped ChromaDB collection."""
    client = _get_client()
    name = f"session_{session_id}"
    return client.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})


def _get_embeddings():
    return OpenAIEmbeddings(
        api_key=settings.openai_api_key,
        model=settings.embedding_model,
    )


# --- PDF Processing ---

def _extract_text_from_pdf(file_path: str) -> str:
    try:
        from pypdf import PdfReader
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n\n"
        return text.strip()
    except Exception as e:
        logger.error(f"PDF extraction failed for {file_path}: {e}")
        return ""


def _chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        start += chunk_size - overlap
    return chunks


def _hash_content(content: str) -> str:
    return hashlib.md5(content.encode()).hexdigest()[:12]


# --- Public API ---

def ingest_pdf(session_id: str, file_path: str, metadata: Dict[str, str] = None) -> Dict[str, Any]:
    """Ingest a PDF into a session-scoped ChromaDB collection."""
    collection = _get_collection(session_id)
    embeddings = _get_embeddings()

    text = _extract_text_from_pdf(file_path)
    if not text:
        return {"error": "Could not extract text from PDF", "chunks": 0}

    chunks = _chunk_text(text)
    if not chunks:
        return {"error": "No content after chunking", "chunks": 0}

    base_meta = {"source": os.path.basename(file_path), "type": "pdf"}
    if metadata:
        base_meta.update(metadata)

    ids = []
    documents = []
    metadatas = []

    for i, chunk in enumerate(chunks):
        doc_id = f"{_hash_content(chunk)}_{i}"
        ids.append(doc_id)
        documents.append(chunk)
        metadatas.append({**base_meta, "chunk_index": str(i)})

    # Embed in batches
    batch_size = 50
    total_added = 0

    for start in range(0, len(documents), batch_size):
        batch_docs = documents[start:start + batch_size]
        batch_ids = ids[start:start + batch_size]
        batch_meta = metadatas[start:start + batch_size]
        batch_embeddings = embeddings.embed_documents(batch_docs)

        collection.upsert(
            ids=batch_ids,
            documents=batch_docs,
            embeddings=batch_embeddings,
            metadatas=batch_meta,
        )
        total_added += len(batch_docs)

    logger.info(f"Session {session_id}: ingested {total_added} chunks from {file_path}")
    return {
        "file": os.path.basename(file_path),
        "chunks": total_added,
        "total_in_session": collection.count(),
    }


def search_documents(session_id: str, query: str, k: int = 5) -> List[Dict[str, Any]]:
    """Search a session's ChromaDB collection for relevant chunks."""
    collection = _get_collection(session_id)

    if collection.count() == 0:
        return []

    embeddings = _get_embeddings()
    query_embedding = embeddings.embed_query(query)

    try:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(k, collection.count()),
            include=["documents", "metadatas", "distances"],
        )
    except Exception as e:
        logger.error(f"ChromaDB query failed: {e}")
        return []

    hits = []
    if results and results["documents"]:
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            hits.append({
                "content": doc,
                "source": meta.get("source", "unknown"),
                "metadata": meta,
                "score": round(1 - dist, 3),
            })

    return hits


def get_session_doc_stats(session_id: str) -> Dict[str, Any]:
    """Get stats for a session's document collection."""
    collection = _get_collection(session_id)
    count = collection.count()

    sources = set()
    if count > 0:
        try:
            sample = collection.peek(limit=min(count, 100))
            for meta in sample.get("metadatas", []):
                sources.add(meta.get("source", "unknown"))
        except Exception:
            pass

    return {"total_chunks": count, "sources": list(sources)}


def cleanup_session_docs(session_id: str):
    """Delete a session's ChromaDB collection."""
    try:
        client = _get_client()
        name = f"session_{session_id}"
        client.delete_collection(name=name)
        logger.info(f"Cleaned up RAG collection for session {session_id}")
    except Exception:
        pass  # Collection may not exist
