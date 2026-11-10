import torch
import chromadb
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

print("=== Hardware ===")
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

if device == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

print("\n=== Loading embedding model ===")
model = SentenceTransformer(MODEL_NAME, device=device)
print("Model loaded:", MODEL_NAME)
print("Embedding dimensions:", model.get_sentence_embedding_dimension())

print("\n=== Connecting to ChromaDB ===")
client = chromadb.HttpClient(host="localhost", port=8000)

collection = client.create_collection(
    name="setup_validation",
    metadata={"description": "Temporary RAG setup validation"}
)

documents = [
    "Photovoltaic inverters convert DC electricity from solar panels into AC electricity.",
    "Ein Wechselrichter wandelt den Gleichstrom von Solarmodulen in Wechselstrom um.",
    "Retrieval augmented generation combines document retrieval with language models."
]

embeddings = model.encode(
    documents,
    normalize_embeddings=True,
    convert_to_numpy=True
)

collection.add(
    ids=["doc1", "doc2", "doc3"],
    documents=documents,
    embeddings=embeddings.tolist()
)

print("Documents stored:", collection.count())

print("\n=== Semantic Search ===")
query = "How does a solar inverter work?"

query_embedding = model.encode(
    query,
    normalize_embeddings=True,
    convert_to_numpy=True
)

results = collection.query(
    query_embeddings=[query_embedding.tolist()],
    n_results=2
)

for document, distance in zip(
    results["documents"][0],
    results["distances"][0]
):
    print(f"Distance: {distance:.4f}")
    print("Document:", document)
    print()

client.delete_collection("setup_validation")

print("=== SUCCESS ===")
print("GPU embeddings and ChromaDB vector search are working.")
