from typing import List

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from src.data.loader import Declaration, Regulation
from src.validation.models import PredictionValidation


class LLMValidationItem(BaseModel):
    """Оценка одной регуляции."""

    regulation_id: str = Field(
        description="Идентификатор оцениваемой регуляции."
    )
    relevance: int = Field(
        ge=0,
        le=3,
        description=(
            "Релевантность: "
            "0 — нерелевантна, "
            "1 — слабая связь, "
            "2 — потенциально применима, "
            "3 — явно релевантна."
        ),
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Уверенность модели в оценке от 0 до 1.",
    )
    reason: str = Field(
        min_length=1,
        description="Краткое обоснование оценки.",
    )


class LLMValidationBatch(BaseModel):
    """Результат оценки набора регуляций."""

    results: List[LLMValidationItem] = Field(
        description="Оценки всех переданных регуляций."
    )


class LLMValidator:
    """Оценщик релевантности регуляций через llama.cpp API."""

    def __init__(
        self,
        base_url: str = "http://localhost:8080/v1",
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> None:
        """Инициализировать клиент локального llama.cpp API."""
        self.llm = ChatOpenAI(
            base_url=base_url,
            api_key="llama-cpp",
            model="ggml-org/gemma-4-E2B-it-GGUF:Q8_0",
            temperature=temperature,
            max_tokens=max_tokens,
        )

        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
Ты эксперт по таможенному регулированию и классификации
товаров по ТН ВЭД ЕАЭС.

Твоя задача — оценить релевантность нормативных актов
конкретной таможенной декларации.

Для каждой переданной регуляции верни отдельную оценку.

Шкала оценки:
0 — нормативный акт нерелевантен;
1 — имеется слабая или косвенная связь;
2 — нормативный акт потенциально применим;
3 — нормативный акт явно и непосредственно релевантен.

Оценивай только фактическую связь между товаром и нормативным
актом.

Не учитывай rank и retrieval score при определении relevance.

Для каждой регуляции обязательно верни:

1. regulation_id
   Только исходный ID из поля ID.
   Например: "NPA0334".

2. relevance
   ЦЕЛОЕ число от 0 до 3:
   0 — нерелевантна;
   1 — слабая или косвенная связь;
   2 — потенциально применима;
   3 — явно и непосредственно релевантна.

3. confidence
   ЧИСЛО С ПЛАВАЮЩЕЙ ТОЧКОЙ от 0.0 до 1.0.
   Это НЕ relevance.
   Например: 0.25, 0.5, 0.75, 0.95.
   Значения 2 или 3 для confidence запрещены.

4. reason
   Краткое текстовое объяснение оценки.

Никогда не используй шкалу relevance для confidence.

Поле regulation_id должно содержать ТОЛЬКО значение
из поля ID соответствующей регуляции.

Например, если вход содержит:

ID: NPA0334

то результат должен содержать:

"regulation_id": "NPA0334"

Никогда не используй номер пункта документа,
номер приказа, название раздела или текст документа
в качестве regulation_id.

Не пропускай ни одну регуляцию.
""",
                ),
                (
                    "human",
                    """
Декларация:

{declaration_text}

Ниже представлены нормативные акты, которые retrieval-модель
отобрала для этой декларации.

{regulations}
""",
                ),
            ]
        )

        self.structured_llm = self.llm.with_structured_output(
            LLMValidationBatch,
            # method="function_calling",
        )

        self.chain = self.prompt | self.structured_llm

    @staticmethod
    def _build_declaration_text(
        declaration: Declaration,
    ) -> str:
        """Сформировать текст декларации для LLM."""
        return declaration.get_search_query()

    @staticmethod
    def _build_regulations_text(
        regulations: List[Regulation],
        ranks: List[int],
        retrieval_scores: List[float],
    ) -> str:
        """Сформировать текст всех регуляций для одного запроса."""
        parts = []

        for regulation, rank, retrieval_score in zip(
            regulations,
            ranks,
            retrieval_scores,
        ):
            parts.append(
    f"""
=== REGULATION START ===
ID: {regulation.regulation_id}
RANK: {rank}
RETRIEVAL_SCORE: {retrieval_score:.6f}

DOCUMENT:
{regulation.get_search_document()}

=== REGULATION END ===
""".strip()
)

        return "\n\n".join(parts)

    def validate_batch(
        self,
        declaration: Declaration,
        regulations: List[Regulation],
        ranks: List[int],
        retrieval_scores: List[float],
    ) -> List[PredictionValidation]:
        """Оценить top-k регуляций одним запросом к LLM."""
        if not (
            len(regulations)
            == len(ranks)
            == len(retrieval_scores)
        ):
            raise ValueError(
                "Количество регуляций, рангов и retrieval scores "
                "должно совпадать."
            )

        if not regulations:
            return []

        result = self.chain.invoke(
            {
                "declaration_text": self._build_declaration_text(
                    declaration
                ),
                "regulations": self._build_regulations_text(
                    regulations,
                    ranks,
                    retrieval_scores,
                ),
            }
        )

        results_by_regulation_id = {
            item.regulation_id: item
            for item in result.results
        }

        predictions = []

        for regulation, rank, retrieval_score in zip(
            regulations,
            ranks,
            retrieval_scores,
        ):
            validation = results_by_regulation_id.get(
                regulation.regulation_id
            )

            if validation is None:
                print()
                print("LLM вернула следующие regulation_id:")
                print(
                    sorted(results_by_regulation_id.keys())
                )
                print(
                    "Ожидалась регуляция: "
                    f"{regulation.regulation_id}"
                )
                print()

                raise ValueError(
                    "LLM не вернула оценку для регуляции "
                    f"{regulation.regulation_id}."
                )

            predictions.append(
                PredictionValidation(
                    declaration_id=declaration.declaration_id,
                    regulation_id=regulation.regulation_id,
                    rank=rank,
                    retrieval_score=retrieval_score,
                    relevance=validation.relevance,
                    confidence=validation.confidence,
                    reason=validation.reason,
                )
            )

        return predictions