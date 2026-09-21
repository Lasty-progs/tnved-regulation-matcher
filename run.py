#!/usr/bin/env python3
"""CLI-модуль запуска конвейера сопоставления таможенных деклараций и регулирующих НПА."""

import argparse
from collections import defaultdict
import os
from typing import Any, Dict, List, Set

import hydra
from omegaconf import DictConfig
import pandas as pd

from src.utils.seed import set_seed
from src.encoders.text_encoder import TextEncoder
from src.retrieval.hybrid import HybridRanker
from src.data.loader import DataLoader, Declaration, Regulation
from src.reranking.cross_encoder import RegulationReranker
from src.utils.validator import validate_predictions


def fetch_tnved_links(db_cfg: DictConfig) -> Dict[str, Set[str]]:
    """Извлекает маппинг регуляций на 2-значные товарные группы ТН ВЭД из базы данных.

    При сбое подключения возвращает пустой словарь для работы в автономном in-memory режиме.
    """
    links: Dict[str, Set[str]] = defaultdict(set)
    try:
        from src.database.connection import DatabaseConnector

        db = DatabaseConnector(db_cfg)
        with db.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT l.regulation_id, SUBSTRING(t.clean_code, 1, 2) AS group_code
                    FROM regulation_tnved_links l
                    JOIN tnved_tree t ON l.tnved_id = t.id
                    WHERE t.clean_code IS NOT NULL;
                    """
                )
                for reg_id, group_code in cur.fetchall():
                    if group_code:
                        links[reg_id].add(group_code)
    except Exception as exc:
        print(f"Предупреждение: БД недоступна ({exc}). Работаем в in-memory режиме.")

    return links


def rank_declarations(
    ranker: HybridRanker,
    reranker: RegulationReranker | None,
    declarations: List[Declaration],
    regulations: Dict[str, Regulation],
    top_k: int,
) -> pd.DataFrame:
    """Выполняет retrieval и опциональное переранжирование НПА."""
    records: List[Dict[str, Any]] = []

    for declaration in declarations:
        top_regulations = ranker.rank_declaration(
            declaration,
            top_k=top_k,
        )

        if reranker is not None:
            candidate_regulations = [
                regulations[regulation_id]
                for regulation_id, _ in top_regulations
            ]

            reranked_regulations = reranker.rerank(
                declaration,
                candidate_regulations,
            )

            for rank, (regulation, score) in enumerate(
                reranked_regulations,
                start=1,
            ):
                records.append(
                    {
                        "declaration_id": declaration.declaration_id,
                        "rank": rank,
                        "regulation_id": regulation.regulation_id,
                        "score": round(score, 6),
                    }
                )

            continue

        for rank, (regulation_id, score) in enumerate(
            top_regulations,
            start=1,
        ):
            records.append(
                {
                    "declaration_id": declaration.declaration_id,
                    "rank": rank,
                    "regulation_id": regulation_id,
                    "score": round(score, 6),
                }
            )

    return pd.DataFrame(records)


def execute_pipeline(cfg: DictConfig, output_dir: str) -> None:
    """Координирует загрузку данных, гибридное ранжирование и сохранение результатов."""
    set_seed(cfg.seed)
    os.makedirs(output_dir, exist_ok=True)
    out_csv = os.path.join(output_dir, "predictions.csv")

    raw_regulations = DataLoader.load_regulations(cfg.data.regulations_path)
    reg_documents = [
        {"id": r.regulation_id, "text": r.get_search_document()}
        for r in raw_regulations
    ]
    regulations = {
    regulation.regulation_id: regulation
    for regulation in raw_regulations
}

    declarations = DataLoader.load_declarations(cfg.data.declarations_path)
    reg_tnved_links = fetch_tnved_links(cfg.db)

    encoder = TextEncoder(cfg.model)
    ranker = HybridRanker(
        encoder=encoder,
        documents=reg_documents,
        dense_weight=cfg.pipeline.dense_weight,
        sparse_weight=cfg.pipeline.sparse_weight,
        reg_tnved_links=reg_tnved_links,
    )
    reranker = None

    if cfg.reranker.enabled:
        reranker = RegulationReranker(
            model_name=cfg.reranker.model_name,
            batch_size=cfg.reranker.batch_size,
        )

    df_predictions = rank_declarations(
    ranker=ranker,
    reranker=reranker,
    declarations=declarations,
    regulations=regulations,
    top_k=cfg.pipeline.top_k,
)

    validate_predictions(df_predictions, expected_declarations_count=len(declarations))
    df_predictions.to_csv(out_csv, index=False)

    print(f"Успешно завершено. Результат сохранен в: {out_csv}")


def main() -> None:
    """Точка входа CLI для запуска ранжирования НПА."""
    parser = argparse.ArgumentParser(
        description="Ранжирование НПА по таможенным декларациям"
    )
    parser.add_argument(
        "--out",
        type=str,
        default="./out",
        help="Путь к директории сохранения predictions.csv",
    )
    args, hydra_overrides = parser.parse_known_args()

    with hydra.initialize(version_base=None, config_path="conf"):
        cfg = hydra.compose(
            config_name="config",
            overrides=hydra_overrides,
        )
        execute_pipeline(cfg, args.out)


if __name__ == "__main__":
    main()