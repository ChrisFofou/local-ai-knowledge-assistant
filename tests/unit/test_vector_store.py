from uuid import uuid4

import chromadb
import numpy as np
import pytest

from knowledge_assistant.ingestion.chunker import DocumentChunk
from knowledge_assistant.vector_store.store import ChromaVectorStore


@pytest.fixture
def store():
    client = chromadb.EphemeralClient()

    return ChromaVectorStore(
        collection_name=f"test_knowledge_{uuid4().hex}",
        client=client,
    )


def make_chunk(index, document_id="doc-123"):
    return DocumentChunk(
        chunk_id=f"chunk-{index}",
        document_id=document_id,
        source="solar_guide.md",
        text=f"Solar inverter information, section {index}.",
        chunk_index=index,
        metadata={
            "category": "technical",
            "file_type": "md",
        },
    )


@pytest.fixture
def sample_embeddings():
    vectors = np.zeros((2, 384), dtype=np.float32)
    vectors[0, 0] = 1.0
    vectors[1, 1] = 1.0
    return vectors


def test_store_initially_empty(store):
    assert store.count() == 0


def test_upsert_stores_chunks(store, sample_embeddings):
    chunks = [make_chunk(0), make_chunk(1)]

    store.upsert(chunks, sample_embeddings)

    assert store.count() == 2


def test_search_returns_matching_documents(store, sample_embeddings):
    chunks = [make_chunk(0), make_chunk(1)]
    store.upsert(chunks, sample_embeddings)

    results = store.search(sample_embeddings[0], n_results=2)

    assert len(results["ids"][0]) == 2
    assert results["ids"][0][0] == "chunk-0"
    assert "Solar inverter information" in results["documents"][0][0]


def test_search_preserves_metadata(store, sample_embeddings):
    store.upsert([make_chunk(0), make_chunk(1)], sample_embeddings)

    results = store.search(sample_embeddings[0])

    metadata = results["metadatas"][0][0]

    assert metadata["source"] == "solar_guide.md"
    assert metadata["document_id"] == "doc-123"
    assert metadata["category"] == "technical"
    assert metadata["chunk_index"] == 0


def test_upsert_updates_existing_chunks(store, sample_embeddings):
    chunk = make_chunk(0)

    store.upsert([chunk], sample_embeddings[:1])

    updated_chunk = DocumentChunk(
        chunk_id="chunk-0",
        document_id="doc-123",
        source="solar_guide.md",
        text="Updated solar inverter information.",
        chunk_index=0,
    )

    store.upsert([updated_chunk], sample_embeddings[:1])

    assert store.count() == 1

    results = store.search(sample_embeddings[0])

    assert results["documents"][0][0] == "Updated solar inverter information."


def test_delete_document(store, sample_embeddings):
    chunks = [
        make_chunk(0, "doc-123"),
        make_chunk(1, "doc-123"),
        make_chunk(2, "doc-456"),
    ]

    embeddings = np.zeros((3, 384), dtype=np.float32)
    embeddings[0, 0] = 1.0
    embeddings[1, 1] = 1.0
    embeddings[2, 2] = 1.0

    store.upsert(chunks, embeddings)
    store.delete_document("doc-123")

    assert store.count() == 1


def test_empty_search_returns_empty_results(store):
    query = np.ones(384, dtype=np.float32)

    results = store.search(query)

    assert results["ids"] == [[]]


def test_empty_chunks_are_rejected(store):
    with pytest.raises(ValueError, match="empty chunk collection"):
        store.upsert([], np.empty((0, 384)))


def test_mismatched_chunk_and_embedding_counts_are_rejected(store):
    chunks = [make_chunk(0)]
    embeddings = np.ones((2, 384), dtype=np.float32)

    with pytest.raises(ValueError, match="must match"):
        store.upsert(chunks, embeddings)


def test_invalid_query_count_is_rejected(store):
    with pytest.raises(ValueError, match="n_results"):
        store.search(np.ones(384), n_results=0)


def test_invalid_query_dimensions_are_rejected(store):
    with pytest.raises(ValueError, match="one-dimensional"):
        store.search(np.ones((1, 384)))


def test_non_finite_embeddings_are_rejected(store):
    embeddings = np.ones((1, 384), dtype=np.float32)
    embeddings[0, 0] = np.nan

    with pytest.raises(ValueError, match="finite"):
        store.upsert([make_chunk(0)], embeddings)
