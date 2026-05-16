from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class DocumentChunk:
    id: int
    title: str
    text: str


def load_markdown_chunks(path: str | Path) -> list[DocumentChunk]:
    """Divide una base de conocimiento Markdown en fragmentos por encabezados H2."""
    raw_text = Path(path).read_text(encoding="utf-8")
    sections = raw_text.split("\n## ")
    chunks: list[DocumentChunk] = []

    for idx, section in enumerate(sections):
        clean = section.strip()
        if not clean:
            continue
        lines = clean.splitlines()
        title = lines[0].replace("#", "").strip() if lines else f"Sección {idx + 1}"
        body = "\n".join(lines[1:]).strip() if len(lines) > 1 else clean
        chunks.append(DocumentChunk(id=idx, title=title, text=body or clean))

    return chunks


def embed_texts(client, model: str, texts: list[str]) -> np.ndarray:
    """Genera embeddings con OpenAI y retorna una matriz numpy."""
    response = client.embeddings.create(model=model, input=texts)
    vectors = [item.embedding for item in response.data]
    return np.array(vectors, dtype=np.float32)


def cosine_similarity_matrix(query_vector: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Calcula similitud coseno entre un vector de consulta y una matriz de embeddings."""
    query_norm = np.linalg.norm(query_vector)
    matrix_norms = np.linalg.norm(matrix, axis=1)
    denominator = matrix_norms * query_norm
    denominator = np.where(denominator == 0, 1e-12, denominator)
    return matrix @ query_vector / denominator


class SemanticRetriever:
    """Retriever semántico simple usando embeddings OpenAI y similitud coseno."""

    def __init__(self, client, embedding_model: str, chunks: list[DocumentChunk]):
        self.client = client
        self.embedding_model = embedding_model
        self.chunks = chunks
        self.chunk_embeddings: np.ndarray | None = None

    def build_index(self) -> None:
        texts = [f"{chunk.title}\n{chunk.text}" for chunk in self.chunks]
        self.chunk_embeddings = embed_texts(self.client, self.embedding_model, texts)

    def search(self, query: str, top_k: int = 3) -> list[tuple[DocumentChunk, float]]:
        if self.chunk_embeddings is None:
            self.build_index()

        query_embedding = embed_texts(self.client, self.embedding_model, [query])[0]
        scores = cosine_similarity_matrix(query_embedding, self.chunk_embeddings)
        ranked_indices = np.argsort(scores)[::-1][:top_k]

        return [(self.chunks[i], float(scores[i])) for i in ranked_indices]


def format_context(results: list[tuple[DocumentChunk, float]]) -> str:
    """Convierte los fragmentos recuperados en contexto para el LLM."""
    context_blocks = []
    for chunk, score in results:
        context_blocks.append(
            f"[Fuente interna: {chunk.title} | similitud={score:.3f}]\n{chunk.text}"
        )
    return "\n\n---\n\n".join(context_blocks)
