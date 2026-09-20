from typing import Dict, List, Optional, Tuple

import numpy as np
from rank_bm25 import BM25Okapi

from src.data.text_cleaner import TextCleaner

class SparseRetriever:
    """Разреженный поиск документов на основе алгоритма BM25."""

    def __init__(self) -> None:
        self.doc_ids: List[str] = []
        self.bm25: Optional[BM25Okapi] = None

    def fit(self, documents: List[Dict[str, str]]) -> None:
        """Построить BM25-индекс на основе переданных документов."""
        self.doc_ids = [document["id"] for document in documents]
        tokenized_corpus = [
            TextCleaner.tokenize(document["text"])
            for document in documents
        ]

        self.bm25 = BM25Okapi(tokenized_corpus)

    def get_scores(self, query_text: str) -> np.ndarray:
        """Вернуть нормализованные BM25-скоры в диапазоне [0, 1]."""
        tokens = TextCleaner.tokenize(query_text)

        if not tokens or self.bm25 is None:
            return np.zeros(len(self.doc_ids), dtype=np.float32)

        raw_scores = np.asarray(
            self.bm25.get_scores(tokens),
            dtype=np.float32,
        )

        min_score = raw_scores.min()
        max_score = raw_scores.max()
        score_range = max_score - min_score

        if score_range > 1e-6:
            return (raw_scores - min_score) / score_range

        return np.zeros_like(raw_scores)

    def retrieve(
        self,
        query_text: str,
        top_k: int = 50,
    ) -> List[Tuple[str, float]]:
        """Вернуть top-k документов, отсортированных по BM25-скору."""
        scores = self.get_scores(query_text)
        top_indices = np.argsort(scores)[::-1][:top_k]

        return [
            (self.doc_ids[index], float(scores[index]))
            for index in top_indices
        ]