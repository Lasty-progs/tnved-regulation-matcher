from dataclasses import dataclass
from typing import List


@dataclass
class PredictionValidation:
    """Результат оценки одной предсказанной регуляции."""

    declaration_id: str
    regulation_id: str
    rank: int
    retrieval_score: float
    relevance: int
    confidence: float
    reason: str


@dataclass
class DeclarationValidation:
    """Результаты оценки всех предсказаний одной декларации."""

    declaration_id: str
    predictions: List[PredictionValidation]


@dataclass
class ValidationMetrics:
    """Метрики качества предсказаний на основе LLM-оценки."""

    declarations_count: int
    predictions_count: int
    mean_relevance: float
    relevant_at_1: float
    relevant_at_3: float
    relevant_at_5: float
    relevant_at_10: float
    mean_reciprocal_rank: float
    coverage: float