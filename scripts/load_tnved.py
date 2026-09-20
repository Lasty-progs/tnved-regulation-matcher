import os
import sys

import hydra
import psycopg
from omegaconf import DictConfig
from pgvector.psycopg import register_vector

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from src.data.tnved_parser import TnvedNode, TnvedParser
from src.database.connection import DatabaseConnector


def init_db(
    conn: psycopg.Connection,
    embedding_dim: int,
) -> None:
    """Инициализировать расширения и пересоздать таблицу tnved_tree."""
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm;")
        register_vector(conn)

        cur.execute(
            f"""
            DROP TABLE IF EXISTS regulation_tnved_links CASCADE;
            DROP TABLE IF EXISTS tnved_tree CASCADE;

            CREATE TABLE tnved_tree (
                id SERIAL PRIMARY KEY,
                code TEXT,
                clean_code TEXT,
                code_level TEXT NOT NULL,
                title TEXT NOT NULL,
                full_name TEXT,
                full_path TEXT NOT NULL,
                indent_level INT NOT NULL,
                parent_id INT REFERENCES tnved_tree(id),
                embedding vector({embedding_dim})
            );

            CREATE INDEX idx_tnved_clean_code
                ON tnved_tree(clean_code);

            CREATE INDEX idx_tnved_code_level
                ON tnved_tree(code_level);

            CREATE INDEX idx_tnved_title_trgm
                ON tnved_tree USING gin (title gin_trgm_ops);

            CREATE INDEX idx_tnved_fullname_trgm
                ON tnved_tree USING gin (full_name gin_trgm_ops);
            """
        )

    print(
        f"Таблица tnved_tree пересоздана "
        f"(vector: {embedding_dim})."
    )


def load_to_postgres(
    nodes: list[TnvedNode],
    conn: psycopg.Connection,
) -> None:
    """Загрузить узлы ТН ВЭД в PostgreSQL с восстановлением иерархии."""
    temp_id_to_db_id: dict[int, int] = {}
    total = len(nodes)

    print(f"Загрузка {total} записей в PostgreSQL...")

    with conn.cursor() as cur:
        for index, node in enumerate(nodes, start=1):
            parent_db_id = (
                temp_id_to_db_id.get(node.parent_temp_id)
                if node.parent_temp_id is not None
                else None
            )

            cur.execute(
                """
                INSERT INTO tnved_tree (
                    code,
                    clean_code,
                    code_level,
                    title,
                    full_name,
                    full_path,
                    indent_level,
                    parent_id
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id;
                """,
                (
                    node.code,
                    node.clean_code,
                    node.code_level,
                    node.title,
                    node.full_name,
                    node.full_path,
                    node.indent_level,
                    parent_db_id,
                ),
            )

            db_id = cur.fetchone()[0]
            temp_id_to_db_id[node.temp_id] = db_id

            if index % 5000 == 0 or index == total:
                print(f"Загружено: {index}/{total} записей...")

    print(f"Успешно сохранено {total} узлов ТН ВЭД.")


@hydra.main(
    version_base=None,
    config_path="../conf",
    config_name="config",
)
def main(cfg: DictConfig) -> None:
    """Распарсить дамп ТН ВЭД и загрузить его в PostgreSQL."""
    dump_path = cfg.data.tnved_dump_path
    embedding_dim = cfg.model.dim

    print("=" * 60)
    print(
        f"ЗАГРУЗКА ТН ВЭД: '{dump_path}', "
        f"размерность векторов: {embedding_dim}"
    )
    print("=" * 60)

    parsed_nodes = TnvedParser.parse(dump_path)
    db = DatabaseConnector(cfg.db)

    with db.get_connection() as conn:
        init_db(conn, embedding_dim)
        load_to_postgres(parsed_nodes, conn)


if __name__ == "__main__":
    main()