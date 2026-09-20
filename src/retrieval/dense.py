from typing import Dict, List, Optional, Tuple

import numpy as np

from src.encoders.text_encoder import TextEncoder

class DenseRetriever:
    """Поиск документов на основе сходства dense-эмбеддингов."""

    def __init__(self, encoder: TextEncoder) -> None:
        self.encoder = encoder
        self.doc_ids: List[str] = []
        self.doc_embeddings: Optional[np.ndarray] = None

    def build_index(self, documents: List[Dict[str, str]]) -> None:
        """Закодировать тексты документов и построить индекс."""
        self.doc_ids = [document["id"] for document in documents]
        texts = [document["text"] for document in documents]

        self.doc_embeddings = self.encoder.encode_documents(texts)

    def get_scores(self, query_text: str) -> np.ndarray:
        """Вычислить косинусное сходство запроса со всеми документами."""
        if self.doc_embeddings is None:
            raise ValueError(
                "Индекс DenseRetriever еще не построен. "
                "Вызовите build_index()."
            )

        query_embedding = self.encoder.encode_queries([query_text])[0]
        cosine_scores = np.dot(self.doc_embeddings, query_embedding)

        normalized_scores = (cosine_scores + 1.0) / 2.0

        return np.clip(normalized_scores, 0.0, 1.0)

    def retrieve(
        self,
        query_text: str,
        top_k: int = 50,
    ) -> List[Tuple[str, float]]:
        """Вернуть top-k документов, отсортированных по score."""
        scores = self.get_scores(query_text)
        top_indices = np.argsort(scores)[::-1][:top_k]

        return [
            (self.doc_ids[index], float(scores[index]))
            for index in top_indices
        ]
