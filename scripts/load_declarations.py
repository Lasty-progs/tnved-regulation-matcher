import json
import os
import sys

import hydra
from omegaconf import DictConfig

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from src.data.loader import DataLoader
from src.database.connection import DatabaseConnector


@hydra.main(
    version_base=None,
    config_path="../conf",
    config_name="config",
)
def main(cfg: DictConfig) -> None:
    """Загрузить декларации из JSONL-файла в PostgreSQL."""
    file_path = cfg.data.declarations_path
    embedding_dim = cfg.model.dim

    print("=" * 60)
    print(f"ЗАГРУЗКА ДЕКЛАРАЦИЙ: '{file_path}'")
    print("=" * 60)

    declarations = DataLoader.load_declarations(file_path)
    db = DatabaseConnector(cfg.db)

    with db.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS declarations (
                    declaration_id VARCHAR(50) PRIMARY KEY,
                    goods_description TEXT NOT NULL,
                    clean_description TEXT NOT NULL,
                    g011 VARCHAR(10),
                    g32 INT,
                    has_acceptance_docs INT,
                    raw_data JSONB NOT NULL,
                    embedding vector({embedding_dim})
                );

                TRUNCATE TABLE declarations;
                """
            )

            for declaration in declarations:
                cur.execute(
                    """
                    INSERT INTO declarations (
                        declaration_id,
                        goods_description,
                        clean_description,
                        g011,
                        g32,
                        has_acceptance_docs,
                        raw_data
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s);
                    """,
                    (
                        declaration.declaration_id,
                        declaration.g31_1,
                        declaration.get_search_query(),
                        declaration.g011,
                        declaration.g32,
                        declaration.has_acceptance_docs,
                        json.dumps(
                            declaration.raw,
                            ensure_ascii=False,
                        ),
                    ),
                )

        conn.commit()

    print(
        f"Успешно загружено {len(declarations)} "
        "деклараций в PostgreSQL."
    )


if __name__ == "__main__":
    main()