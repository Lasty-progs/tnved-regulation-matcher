from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from src.data.loader import Declaration
from src.encoders.text_encoder import TextEncoder
from src.retrieval.dense import DenseRetriever
from src.retrieval.sparse import SparseRetriever

class HybridRanker:
    """Гибридный ранжировщик на основе sparse и dense поиска."""

    def __init__(
        self,
        encoder: TextEncoder,
        documents: List[Dict[str, str]],
        dense_weight: float = 0.6,
        sparse_weight: float = 0.4,
        reg_tnved_links: Optional[Dict[str, Set[str]]] = None,
    ) -> None:
        """Инициализировать dense и sparse поисковые индексы."""
        self.doc_ids = [document["id"] for document in documents]
        self.documents_map = {
            document["id"]: document["text"]
            for document in documents
        }
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.reg_tnved_links = reg_tnved_links or {}

        self.sparse_retriever = SparseRetriever()
        self.sparse_retriever.fit(documents)

        self.dense_retriever = DenseRetriever(encoder)
        self.dense_retriever.build_index(documents)

    def _detect_query_domain(self, text: str) -> Set[str]:
        """Определить вероятные группы ТН ВЭД по тексту декларации."""
        text_lower = text.lower()
        groups: Set[str] = set()

        if any(
            keyword in text_lower
            for keyword in (
                "хранения данных",
                "накопител",
                "ssd",
                "сервер",
                "процессор",
                "жесткий диск",
                "sas",
                "pci-e",
            )
        ):
            groups.update({"84", "85"})

        if any(
            keyword in text_lower
            for keyword in (
                "коммутатор",
                "маршрутизатор",
                "передающ",
                "радио",
                "кабел",
                "оптоволок",
                "антенн",
            )
        ):
            groups.update({"85", "90"})

        if "огнетушител" in text_lower:
            groups.update({"84", "38"})

        if any(
            keyword in text_lower
            for keyword in (
                "уран",
                "изотоп",
                "плутоний",
                "нептуний",
                "радиоактив",
            )
        ):
            groups.update({"28", "84"})

        if any(
            keyword in text_lower
            for keyword in (
                "мясо",
                "говядин",
                "свинин",
                "куриц",
                "цыплят",
                "рыба",
                "филе",
                "форель",
                "лосось",
            )
        ):
            groups.update({"01", "02", "03", "04", "05", "16"})

        return groups

    def _apply_domain_adjustment(
        self,
        score: float,
        query_groups: Set[str],
        document_groups: Set[str],
    ) -> float:
        """Скорректировать score с учетом групп ТН ВЭД."""
        if not query_groups or not document_groups:
            return score

        if query_groups.intersection(document_groups):
            return score + 0.20

        return score - 0.10

    def _apply_entity_boost(
        self,
        score: float,
        query_text: str,
        document_text: str,
    ) -> float:
        """Увеличить score при совпадении важных сущностей."""
        query_lower = query_text.lower()
        document_lower = document_text.lower()

        if "огнетушител" in query_lower and "огнетушител" in document_lower:
            score += 0.40

        if "уран" in query_lower and "уран" in document_lower:
            score += 0.40

        return score

    def _apply_cryptography_boost(
        self,
        score: float,
        query_text: str,
        document_text: str,
    ) -> float:
        """Увеличить score для релевантных криптографических документов."""
        cryptography_keywords = ("шифров", "криптограф")
        query_keywords = (
            "хранения данных",
            "ssd",
            "сервер",
            "коммутатор",
        )

        has_cryptography = any(
            keyword in document_text.lower()
            for keyword in cryptography_keywords
        )
        has_relevant_query = any(
            keyword in query_text.lower()
            for keyword in query_keywords
        )

        if has_cryptography and has_relevant_query:
            return score + 0.25

        return score

    def rank_declaration(
        self,
        declaration: Declaration,
        top_k: int = 10,
    ) -> List[Tuple[str, float]]:
        """Вернуть top-k наиболее релевантных регуляций."""
        query_text = declaration.get_search_query()

        sparse_scores = self.sparse_retriever.get_scores(query_text)
        dense_scores = self.dense_retriever.get_scores(query_text)

        combined_scores = (
            self.dense_weight * dense_scores
            + self.sparse_weight * sparse_scores
        )

        query_groups = self._detect_query_domain(query_text)

        for index, doc_id in enumerate(self.doc_ids):
            document_text = self.documents_map[doc_id]
            document_groups = self.reg_tnved_links.get(doc_id, set())

            score = float(combined_scores[index])
            score = self._apply_domain_adjustment(
                score,
                query_groups,
                document_groups,
            )
            score = self._apply_entity_boost(
                score,
                query_text,
                document_text,
            )
            score = self._apply_cryptography_boost(
                score,
                query_text,
                document_text,
            )

            combined_scores[index] = score

        best_indices = np.argsort(combined_scores)[::-1][:top_k]

        return [
            (self.doc_ids[index], float(combined_scores[index]))
            for index in best_indices
        ]
