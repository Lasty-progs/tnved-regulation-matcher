import os
import sys

import hydra
import psycopg
import torch
from omegaconf import DictConfig
from pgvector.psycopg import register_vector

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from src.database.connection import DatabaseConnector
from src.encoders.text_encoder import TextEncoder


def prepare_db_schema(
    conn: psycopg.Connection,
    dim: int,
) -> None:
    """Адаптировать размерность векторных колонок под модель."""
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        register_vector(conn)

        cur.execute(
            f"""
            ALTER TABLE tnved_tree
            DROP COLUMN IF EXISTS embedding;

            ALTER TABLE tnved_tree
            ADD COLUMN embedding vector({dim});

            ALTER TABLE regulations
            DROP COLUMN IF EXISTS embedding;

            ALTER TABLE regulations
            ADD COLUMN embedding vector({dim});
            """
        )

    print(f"Схема БД адаптирована под размерность: {dim}")


def embed_tnved_nodes(
    conn: psycopg.Connection,
    encoder: TextEncoder,
) -> None:
    """Вычислить и сохранить эмбеддинги групп и заголовков ТН ВЭД."""
    print("Чтение категорий ТН ВЭД...")

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, full_path
            FROM tnved_tree
            WHERE code_level IN ('heading', 'group')
            ORDER BY id;
            """
        )
        tnved_rows = cur.fetchall()

    tnved_ids = [row[0] for row in tnved_rows]
    tnved_texts = [row[1] for row in tnved_rows]

    print(
        f"Вычисление эмбеддингов для "
        f"{len(tnved_texts)} узлов ТН ВЭД..."
    )
    tnved_embeddings = encoder.encode_documents(tnved_texts)

    print("Запись эмбеддингов ТН ВЭД в базу...")

    with conn.cursor() as cur:
        for node_id, embedding in zip(tnved_ids, tnved_embeddings):
            cur.execute(
                "UPDATE tnved_tree SET embedding = %s WHERE id = %s;",
                (embedding.tolist(), node_id),
            )


def embed_regulations(
    conn: psycopg.Connection,
    encoder: TextEncoder,
) -> None:
    """Вычислить и сохранить эмбеддинги текстов регуляций."""
    print("Чтение регуляций...")

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT regulation_id, clean_text
            FROM regulations
            ORDER BY regulation_id;
            """
        )
        regulation_rows = cur.fetchall()

    regulation_ids = [row[0] for row in regulation_rows]
    regulation_texts = [row[1] for row in regulation_rows]

    print(
        f"Вычисление эмбеддингов для "
        f"{len(regulation_texts)} регуляций..."
    )
    regulation_embeddings = encoder.encode_queries(regulation_texts)

    print("Запись эмбеддингов регуляций в базу...")

    with conn.cursor() as cur:
        for regulation_id, embedding in zip(
            regulation_ids,
            regulation_embeddings,
        ):
            cur.execute(
                """
                UPDATE regulations
                SET embedding = %s
                WHERE regulation_id = %s;
                """,
                (embedding.tolist(), regulation_id),
            )


def build_semantic_links(
    conn: psycopg.Connection,
    top_k: int,
) -> None:
    """Создать семантические связи регуляций с ТН ВЭД."""
    print("Построение семантических связей через pgvector...")

    with conn.cursor() as cur:
        cur.execute("TRUNCATE TABLE regulation_tnved_links;")
        cur.execute(
            f"""
            INSERT INTO regulation_tnved_links (
                regulation_id,
                tnved_id,
                match_type,
                confidence
            )
            SELECT
                r.regulation_id,
                top_t.id AS tnved_id,
                'vector_cosine' AS match_type,
                ROUND(
                    (1 - (r.embedding <=> top_t.embedding))::numeric,
                    4
                ) AS confidence
            FROM regulations r
            CROSS JOIN LATERAL (
                SELECT t.id, t.embedding
                FROM tnved_tree t
                WHERE t.embedding IS NOT NULL
                ORDER BY r.embedding <=> t.embedding ASC
                LIMIT {top_k}
            ) top_t;
            """
        )


def print_linking_stats(
    conn: psycopg.Connection,
    model_name: str,
) -> None:
    """Вывести статистику покрытия после семантической линковки."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                COUNT(DISTINCT r.regulation_id) AS total_regulations,
                COUNT(DISTINCT l.regulation_id) AS linked_regulations,
                COUNT(l.id) AS total_links
            FROM regulations r
            LEFT JOIN regulation_tnved_links l
                ON r.regulation_id = l.regulation_id;
            """
        )
        total_regulations, linked_regulations, total_links = cur.fetchone()

    coverage = (
        linked_regulations / total_regulations * 100
        if total_regulations
        else 0.0
    )

    print("=" * 60)
    print(f"ИТОГ ЛИНКОВКИ ({model_name}):")
    print(f"Всего регуляций: {total_regulations}")
    print(
        f"Связано регуляций: {linked_regulations} "
        f"(Покрытие: {coverage:.1f}%)"
    )
    print(f"Всего связей создано: {total_links}")
    print("=" * 60)


@hydra.main(
    version_base=None,
    config_path="../conf",
    config_name="config",
)
def main(cfg: DictConfig) -> None:
    """Векторизовать данные и построить семантические связи."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Загрузка модели {cfg.model.name} на {device}...")
    encoder = TextEncoder(cfg.model, device=device)
    db = DatabaseConnector(cfg.db)

    with db.get_connection() as conn:
        prepare_db_schema(conn, cfg.model.dim)
        embed_tnved_nodes(conn, encoder)
        embed_regulations(conn, encoder)
        build_semantic_links(conn, cfg.top_k_links)
        print_linking_stats(conn, cfg.model.name)


if __name__ == "__main__":
    main()