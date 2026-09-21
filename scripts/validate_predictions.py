import json
import os
import sys
from typing import Dict, List

import hydra
import pandas as pd
import psycopg
from omegaconf import DictConfig
from tqdm import tqdm

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from src.data.loader import DataLoader, Declaration, Regulation
from src.database.connection import DatabaseConnector
from src.validation.llm_validator import LLMValidator
from src.validation.metrics import ValidationMetricsCalculator
from src.validation.models import (
    DeclarationValidation,
    PredictionValidation,
)


def load_predictions(file_path: str) -> pd.DataFrame:
    """Загрузить результаты retrieval из CSV-файла."""
    predictions = pd.read_csv(file_path)

    required_columns = {
        "declaration_id",
        "rank",
        "regulation_id",
        "score",
    }
    missing_columns = required_columns - set(predictions.columns)

    if missing_columns:
        raise ValueError(
            f"В predictions.csv отсутствуют колонки: {missing_columns}"
        )

    return predictions


def load_regulations(
    conn: psycopg.Connection,
) -> Dict[str, Regulation]:
    """Загрузить нормативные акты из PostgreSQL."""
    with conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT
                regulation_id,
                decree_number,
                npa,
                source
            FROM regulations;
            """
        )
        rows = cursor.fetchall()

    return {
        str(row[0]): Regulation(
            regulation_id=str(row[0]),
            decree_number=row[1] or "",
            npa=row[2] or "",
            source=row[3],
        )
        for row in rows
    }


def validate_predictions(
    predictions: pd.DataFrame,
    declarations: Dict[str, Declaration],
    regulations: Dict[str, Regulation],
    validator: LLMValidator,
) -> List[DeclarationValidation]:
    """Оценить top-10 регуляций каждой декларации."""
    results: List[DeclarationValidation] = []

    grouped_predictions = (
        predictions
        .sort_values(["declaration_id", "rank"])
        .groupby("declaration_id")
    )

    for declaration_id, group in tqdm(
    grouped_predictions,
    total=predictions["declaration_id"].nunique(),
    desc="LLM validation",
    unit="declaration",
    ):
        declaration_id = str(declaration_id)
        declaration = declarations.get(declaration_id)

        if declaration is None:
            print(
                f"Предупреждение: декларация "
                f"{declaration_id} отсутствует в файле деклараций."
            )
            continue

        selected_regulations = []
        ranks = []
        retrieval_scores = []

        for row in group.itertuples(index=False):
            regulation_id = str(row.regulation_id)
            regulation = regulations.get(regulation_id)

            if regulation is None:
                print(
                    f"Предупреждение: регуляция "
                    f"{regulation_id} отсутствует в БД."
                )
                continue

            selected_regulations.append(regulation)
            ranks.append(int(row.rank))
            retrieval_scores.append(float(row.score))

        if not selected_regulations:
            results.append(
                DeclarationValidation(
                    declaration_id=declaration_id,
                    predictions=[],
                )
            )
            continue


        validation_results = validator.validate_batch(
            declaration=declaration,
            regulations=selected_regulations,
            ranks=ranks,
            retrieval_scores=retrieval_scores,
        )

        results.append(
            DeclarationValidation(
                declaration_id=declaration_id,
                predictions=validation_results,
            )
        )

    return results


def save_validation_results(
    results: List[DeclarationValidation],
    file_path: str,
) -> None:
    """Сохранить результаты LLM-валидации в CSV."""
    rows = []

    for declaration in results:
        for prediction in declaration.predictions:
            rows.append(
                {
                    "declaration_id": prediction.declaration_id,
                    "regulation_id": prediction.regulation_id,
                    "rank": prediction.rank,
                    "retrieval_score": prediction.retrieval_score,
                    "relevance": prediction.relevance,
                    "confidence": prediction.confidence,
                    "reason": prediction.reason,
                }
            )

    result_dataframe = pd.DataFrame(rows)

    result_dataframe.to_csv(
        file_path,
        index=False,
        encoding="utf-8-sig",
    )


def save_metrics(
    metrics,
    file_path: str,
) -> None:
    """Сохранить рассчитанные метрики в JSON."""
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(
            {
                "declarations_count": metrics.declarations_count,
                "predictions_count": metrics.predictions_count,
                "mean_relevance": metrics.mean_relevance,
                "relevant_at_1": metrics.relevant_at_1,
                "relevant_at_3": metrics.relevant_at_3,
                "relevant_at_5": metrics.relevant_at_5,
                "relevant_at_10": metrics.relevant_at_10,
                "mean_reciprocal_rank": (
                    metrics.mean_reciprocal_rank
                ),
                "coverage": metrics.coverage,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )


def print_metrics(metrics) -> None:
    """Вывести итоговые метрики в консоль."""
    print("=" * 60)
    print("РЕЗУЛЬТАТЫ LLM-ВАЛИДАЦИИ")
    print("=" * 60)
    print(f"Деклараций:        {metrics.declarations_count}")
    print(f"Предсказаний:      {metrics.predictions_count}")
    print(f"Средняя relevance: {metrics.mean_relevance:.3f}")
    print(f"Relevant@1:        {metrics.relevant_at_1:.3f}")
    print(f"Relevant@3:        {metrics.relevant_at_3:.3f}")
    print(f"Relevant@5:        {metrics.relevant_at_5:.3f}")
    print(f"Relevant@10:       {metrics.relevant_at_10:.3f}")
    print(f"MRR:               {metrics.mean_reciprocal_rank:.3f}")
    print(f"Coverage:          {metrics.coverage:.3f}")
    print("=" * 60)



@hydra.main(
    version_base=None,
    config_path="../conf",
    config_name="config",
)
def main(cfg: DictConfig) -> None:
    """Запустить LLM-валидацию результатов retrieval."""


    predictions_path = cfg.validation.predictions_path
    results_path = cfg.validation.results_path
    metrics_path = cfg.validation.metrics_path
    declarations_path = cfg.data.declarations_path

    print("=" * 60)
    print("LLM-ВАЛИДАЦИЯ PREDICTIONS")
    print("=" * 60)

    predictions = load_predictions(predictions_path)
    declarations_list = DataLoader.load_declarations(
        declarations_path
    )

    declarations = {
        declaration.declaration_id: declaration
        for declaration in declarations_list
    }

    print(f"Загружено предсказаний: {len(predictions)}")
    print(f"Загружено деклараций: {len(declarations)}")

    validator = LLMValidator(
        base_url=cfg.validation.base_url,
        temperature=cfg.validation.temperature,
        max_tokens=cfg.validation.max_tokens,
    )

    db = DatabaseConnector(cfg.db)

    with db.get_connection() as conn:
        regulations = load_regulations(conn)

        print(f"Загружено регуляций: {len(regulations)}")

        validation_results = validate_predictions(
            predictions=predictions,
            declarations=declarations,
            regulations=regulations,
            validator=validator,
        )

    metrics_calculator = ValidationMetricsCalculator()
    metrics = metrics_calculator.calculate(validation_results)

    save_validation_results(
        validation_results,
        results_path,
    )
    save_metrics(metrics, metrics_path)

    print_metrics(metrics)

    print(f"Результаты сохранены: {results_path}")
    print(f"Метрики сохранены: {metrics_path}")


if __name__ == "__main__":
    main()