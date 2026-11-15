import numpy as np
import pytest

from knowledge_assistant.embeddings.encoder import EmbeddingEncoder


class FakeSentenceTransformer:
    def __init__(self, model_name, device):
        self.model_name = model_name
        self.device = device

    def get_sentence_embedding_dimension(self):
        return 384

    def encode(self, texts, **kwargs):
        if isinstance(texts, str):
            return np.ones(384, dtype=np.float32) / np.sqrt(384)

        vectors = np.ones((len(texts), 384), dtype=np.float32)
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


@pytest.fixture
def encoder(monkeypatch):
    from knowledge_assistant.embeddings import encoder as encoder_module

    monkeypatch.setattr(
        encoder_module,
        "SentenceTransformer",
        FakeSentenceTransformer,
    )

    monkeypatch.setattr(
        encoder_module.torch.cuda,
        "is_available",
        lambda: True,
    )

    return EmbeddingEncoder(batch_size=16)


def test_encoder_selects_gpu_when_available(encoder):
    assert encoder.device == "cuda"


def test_encoder_reports_embedding_dimension(encoder):
    assert encoder.embedding_dimension == 384


def test_encode_documents_returns_correct_shape(encoder):
    embeddings = encoder.encode_documents(
        ["Solar inverter converts DC to AC.", "Volt-VAR controls reactive power."]
    )

    assert embeddings.shape == (2, 384)
    assert embeddings.dtype == np.float32


def test_encode_query_returns_correct_shape(encoder):
    embedding = encoder.encode_query("What is Volt-VAR?")

    assert embedding.shape == (384,)
    assert embedding.dtype == np.float32


def test_embeddings_are_normalized(encoder):
    embeddings = encoder.encode_documents(["Solar inverter", "Grid support"])

    norms = np.linalg.norm(embeddings, axis=1)

    assert np.allclose(norms, 1.0)


def test_empty_document_collection_is_rejected(encoder):
    with pytest.raises(ValueError, match="empty document collection"):
        encoder.encode_documents([])


def test_empty_document_text_is_rejected(encoder):
    with pytest.raises(ValueError, match="cannot be empty"):
        encoder.encode_documents(["Solar inverter", "  "])


def test_empty_query_is_rejected(encoder):
    with pytest.raises(ValueError, match="Query cannot be empty"):
        encoder.encode_query("  ")


def test_invalid_batch_size_is_rejected():
    with pytest.raises(ValueError, match="batch_size"):
        EmbeddingEncoder(batch_size=0)


def test_cpu_can_be_selected_explicitly(monkeypatch):
    from knowledge_assistant.embeddings import encoder as encoder_module

    monkeypatch.setattr(
        encoder_module,
        "SentenceTransformer",
        FakeSentenceTransformer,
    )

    encoder = EmbeddingEncoder(device="cpu")

    assert encoder.device == "cpu"
