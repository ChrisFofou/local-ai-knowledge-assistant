from unittest.mock import Mock

import numpy as np
import pytest

from knowledge_assistant.retrieval.service import RetrievalService


@pytest.fixture
def encoder():
    mock = Mock()
    mock.encode_query.return_value = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    return mock


@pytest.fixture
def vector_store():
    mock = Mock()
    mock.search.return_value = {
        "ids": [["chunk-1", "chunk-2"]],
        "documents": [
            [
                "Solar inverters convert DC into AC.",
                "Grid support includes reactive power control.",
            ]
        ],
        "metadatas": [
            [
                {
                    "document_id": "doc-1",
                    "source": "solar_guide.pdf",
                    "chunk_index": 0,
                    "file_type": "pdf",
                },
                {
                    "document_id": "doc-1",
                    "source": "solar_guide.pdf",
                    "chunk_index": 1,
                    "file_type": "pdf",
                },
            ]
        ],
        "distances": [[0.12, 0.25]],
    }
    return mock


@pytest.fixture
def retrieval_service(encoder, vector_store):
    return RetrievalService(
        encoder=encoder,
        vector_store=vector_store,
    )


def test_retrieve_returns_ranked_chunks(retrieval_service):
    results = retrieval_service.retrieve("How do solar inverters work?")

    assert len(results) == 2
    assert results[0].chunk_id == "chunk-1"
    assert results[1].chunk_id == "chunk-2"


def test_retrieve_preserves_chunk_content(retrieval_service):
    results = retrieval_service.retrieve("How do solar inverters work?")

    assert results[0].text == "Solar inverters convert DC into AC."
    assert results[0].source == "solar_guide.pdf"
    assert results[0].document_id == "doc-1"
    assert results[0].chunk_index == 0
    assert results[0].distance == pytest.approx(0.12)


def test_retrieve_preserves_additional_metadata(retrieval_service):
    results = retrieval_service.retrieve("How do solar inverters work?")

    assert results[0].metadata["file_type"] == "pdf"


def test_retrieve_encodes_query(encoder, retrieval_service):
    retrieval_service.retrieve("How do solar inverters work?")

    encoder.encode_query.assert_called_once_with("How do solar inverters work?")


def test_retrieve_passes_top_k_to_vector_store(
    vector_store,
    retrieval_service,
):
    retrieval_service.retrieve("Solar inverters", top_k=3)

    vector_store.search.assert_called_once()
    assert vector_store.search.call_args.kwargs["n_results"] == 3


def test_retrieve_passes_metadata_filter(
    vector_store,
    retrieval_service,
):
    metadata_filter = {"document_id": "doc-1"}

    retrieval_service.retrieve(
        "Solar inverters",
        where=metadata_filter,
    )

    assert vector_store.search.call_args.kwargs["where"] == metadata_filter


def test_retrieve_returns_empty_list_when_no_results(
    vector_store,
    retrieval_service,
):
    vector_store.search.return_value = {
        "ids": [[]],
        "documents": [[]],
        "metadatas": [[]],
        "distances": [[]],
    }

    results = retrieval_service.retrieve("Unknown topic")

    assert results == []


@pytest.mark.parametrize("top_k", [0, -1])
def test_retrieve_rejects_invalid_top_k(
    retrieval_service,
    top_k,
):
    with pytest.raises(ValueError, match="top_k"):
        retrieval_service.retrieve("Solar inverters", top_k=top_k)


def test_retrieve_rejects_empty_query(retrieval_service):
    with pytest.raises(ValueError, match="Query"):
        retrieval_service.retrieve("   ")
