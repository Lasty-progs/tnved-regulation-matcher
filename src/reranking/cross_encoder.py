from typing import List, Tuple

from sentence_transformers import CrossEncoder

from src.data.loader import Declaration, Regulation


class RegulationReranker:
    """Переранжирует найденные регуляции относительно декларации."""

    def __init__(
        self,
        model_name: str,
        batch_size: int = 16,
    ) -> None:
        """Инициализировать cross-encoder модель."""
        self.model = CrossEncoder(
            model_name,
            max_length=512,
        )
        self.batch_size = batch_size

    def rerank(
        self,
        declaration: Declaration,
        regulations: List[Regulation],
    ) -> List[Tuple[Regulation, float]]:
        """Переранжировать регуляции и вернуть их вместе с оценками."""
        if not regulations:
            return []

        declaration_text = self._build_declaration_text(declaration)

        pairs = [
            (
                declaration_text,
                regulation.get_search_document(),
            )
            for regulation in regulations
        ]

        scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )

        ranked = sorted(
            zip(regulations, scores),
            key=lambda item: float(item[1]),
            reverse=True,
        )

        return [
            (regulation, float(score))
            for regulation, score in ranked
        ]

    @staticmethod
    def _build_declaration_text(
        declaration: Declaration,
    ) -> str:
        """Сформировать текст декларации для reranker."""
        return "\n".join(
            str(part)
            for part in [
                declaration.g31_1,
                declaration.g011,
                declaration.g32,
            ]
            if part is not None
        )