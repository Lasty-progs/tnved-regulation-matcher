from typing import List

from src.validation.models import (
    DeclarationValidation,
    PredictionValidation,
    ValidationMetrics,
)


class ValidationMetricsCalculator:
    """Расчет метрик качества retrieval по оценкам LLM."""

    def calculate(
        self,
        declarations: List[DeclarationValidation],
    ) -> ValidationMetrics:
        """Рассчитать агрегированные метрики по всем декларациям."""
        if not declarations:
            return ValidationMetrics(
                declarations_count=0,
                predictions_count=0,
                mean_relevance=0.0,
                relevant_at_1=0.0,
                relevant_at_3=0.0,
                relevant_at_5=0.0,
                relevant_at_10=0.0,
                mean_reciprocal_rank=0.0,
                coverage=0.0,
            )

        predictions = [
            prediction
            for declaration in declarations
            for prediction in declaration.predictions
        ]

        return ValidationMetrics(
            declarations_count=len(declarations),
            predictions_count=len(predictions),
            mean_relevance=self._calculate_mean_relevance(
                predictions
            ),
            relevant_at_1=self._calculate_relevant_at_k(
                declarations,
                1,
            ),
            relevant_at_3=self._calculate_relevant_at_k(
                declarations,
                3,
            ),
            relevant_at_5=self._calculate_relevant_at_k(
                declarations,
                5,
            ),
            relevant_at_10=self._calculate_relevant_at_k(
                declarations,
                10,
            ),
            mean_reciprocal_rank=self._calculate_mrr(declarations),
            coverage=self._calculate_coverage(declarations),
        )

    @staticmethod
    def _calculate_mean_relevance(
        predictions: List[PredictionValidation],
    ) -> float:
        """Рассчитать среднюю оценку релевантности."""
        if not predictions:
            return 0.0

        total_relevance = sum(
            prediction.relevance
            for prediction in predictions
        )

        return total_relevance / len(predictions)

    @staticmethod
    def _calculate_relevant_at_k(
        declarations: List[DeclarationValidation],
        k: int,
    ) -> float:
        """Рассчитать долю деклараций с релевантным результатом в top-k."""
        if not declarations:
            return 0.0

        relevant_declarations = 0

        for declaration in declarations:
            top_predictions = declaration.predictions[:k]

            if any(
                prediction.relevance >= 2
                for prediction in top_predictions
            ):
                relevant_declarations += 1

        return relevant_declarations / len(declarations)

    @staticmethod
    def _calculate_mrr(
        declarations: List[DeclarationValidation],
    ) -> float:
        """Рассчитать Mean Reciprocal Rank по релевантным результатам."""
        if not declarations:
            return 0.0

        reciprocal_ranks = []

        for declaration in declarations:
            reciprocal_rank = 0.0

            for prediction in declaration.predictions:
                if prediction.relevance >= 2:
                    reciprocal_rank = 1.0 / prediction.rank
                    break

            reciprocal_ranks.append(reciprocal_rank)

        return sum(reciprocal_ranks) / len(reciprocal_ranks)

    @staticmethod
    def _calculate_coverage(
        declarations: List[DeclarationValidation],
    ) -> float:
        """Рассчитать долю деклараций с хотя бы одним результатом."""
        if not declarations:
            return 0.0

        covered_declarations = sum(
            bool(declaration.predictions)
            for declaration in declarations
        )

        return covered_declarations / len(declarations)