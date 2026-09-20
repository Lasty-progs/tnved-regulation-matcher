import os
import re
import sys
from typing import Set

import hydra
import psycopg
from omegaconf import DictConfig
from pgvector.psycopg import register_vector

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from src.data.loader import DataLoader, Regulation
from src.database.connection import DatabaseConnector


def init_regulation_tables(
    conn: psycopg.Connection,
    embedding_dim: int,
) -> None:
    """Инициализировать таблицы регуляций и связей с ТН ВЭД."""
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm;")
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        register_vector(conn)

        cur.execute(
            f"""
            DROP TABLE IF EXISTS regulation_tnved_links CASCADE;
            DROP TABLE IF EXISTS regulations CASCADE;

            CREATE TABLE regulations (
                regulation_id VARCHAR(50) PRIMARY KEY,
                decree_number TEXT,
                npa TEXT NOT NULL,
                clean_text TEXT NOT NULL,
                source TEXT,
                embedding vector({embedding_dim})
            );

            CREATE TABLE regulation_tnved_links (
                id SERIAL PRIMARY KEY,
                regulation_id VARCHAR(50)
                    REFERENCES regulations(regulation_id)
                    ON DELETE CASCADE,
                tnved_id INT
                    REFERENCES tnved_tree(id)
                    ON DELETE CASCADE,
                match_type VARCHAR(30) NOT NULL,
                confidence FLOAT DEFAULT 1.0
            );

            CREATE INDEX idx_links_reg_id
                ON regulation_tnved_links(regulation_id);

            CREATE INDEX idx_links_tnved_id
                ON regulation_tnved_links(tnved_id);

            CREATE INDEX IF NOT EXISTS idx_tnved_title_trgm
                ON tnved_tree USING gin (title gin_trgm_ops);

            CREATE INDEX IF NOT EXISTS idx_tnved_fullname_trgm
                ON tnved_tree USING gin (full_name gin_trgm_ops);
            """
        )

    print(
        f"Таблицы regulations и regulation_tnved_links созданы "
        f"(vector: {embedding_dim})."
    )


def extract_tnved_codes(text: str) -> Set[str]:
    """Извлечь из текста коды ТН ВЭД допустимой длины."""
    codes: Set[str] = set()
    pattern = r"\b(\d{2}(?:\s\*\d{2}){1,4}|\d{4,10})\b"

    for match in re.finditer(pattern, text):
        clean_code = re.sub(r"\s+", "", match.group(0))

        if len(clean_code) not in (2, 4, 6, 8, 9, 10):
            continue

        if (
            len(clean_code) == 4
            and clean_code.startswith(("19", "20"))
            and int(clean_code) in range(1990, 2030)
        ):
            continue

        codes.add(clean_code)

    return codes


def link_by_similarity(
    cur: psycopg.Cursor,
    reg_id: str,
    clean_text: str,
) -> int:
    """Создать связи с узлами ТН ВЭД по триграммному сходству."""
    target_phrase = re.split(r"[,;]", clean_text)[0].strip()

    if len(target_phrase) < 3:
        target_phrase = clean_text[:50]

    cur.execute(
        """
        SELECT id, similarity(COALESCE(full_name, title), %s) AS sim
        FROM tnved_tree
        WHERE similarity(COALESCE(full_name, title), %s) > 0.25
        ORDER BY sim DESC
        LIMIT 3;
        """,
        (target_phrase, target_phrase),
    )

    matches = cur.fetchall()

    for tnved_id, score in matches:
        cur.execute(
            """
            INSERT INTO regulation_tnved_links (
                regulation_id,
                tnved_id,
                match_type,
                confidence
            )
            VALUES (%s, %s, 'trigram_match', %s);
            """,
            (reg_id, tnved_id, float(score)),
        )

    return len(matches)


def load_regulations_data(
    regulations: list[Regulation],
    conn: psycopg.Connection,
) -> None:
    """Загрузить регуляции и создать начальные связи с ТН ВЭД."""
    total_regulations = len(regulations)
    total_links = 0

    with conn.cursor() as cur:
        cur.execute("SET pg_trgm.similarity_threshold = 0.25;")

        for regulation in regulations:
            clean_text = regulation.get_clean_text()

            cur.execute(
                """
                INSERT INTO regulations (
                    regulation_id,
                    decree_number,
                    npa,
                    clean_text,
                    source
                )
                VALUES (%s, %s, %s, %s, %s);
                """,
                (
                    regulation.regulation_id,
                    regulation.decree_number,
                    regulation.npa,
                    clean_text,
                    regulation.source,
                ),
            )

            combined_text = (
                f"{regulation.decree_number} {regulation.npa}"
            )
            found_codes = extract_tnved_codes(combined_text)
            code_linked = False

            for code in found_codes:
                cur.execute(
                    """
                    SELECT id, clean_code
                    FROM tnved_tree
                    WHERE clean_code = %s OR clean_code LIKE %s;
                    """,
                    (code, f"{code}%"),
                )

                matches = cur.fetchall()

                for tnved_id, matched_code in matches:
                    match_type = (
                        "exact_code"
                        if matched_code == code
                        else "prefix_code"
                    )

                    cur.execute(
                        """
                        INSERT INTO regulation_tnved_links (
                            regulation_id,
                            tnved_id,
                            match_type,
                            confidence
                        )
                        VALUES (%s, %s, %s, 1.0);
                        """,
                        (
                            regulation.regulation_id,
                            tnved_id,
                            match_type,
                        ),
                    )

                    total_links += 1
                    code_linked = True

            if not code_linked:
                links_count = link_by_similarity(
                    cur,
                    regulation.regulation_id,
                    clean_text,
                )
                total_links += links_count

    print(
        f"Успешно загружено {total_regulations} регуляций. "
        f"Сформировано начальных связей: {total_links}."
    )


@hydra.main(
    version_base=None,
    config_path="../conf",
    config_name="config",
)
def main(cfg: DictConfig) -> None:
    """Загрузить регуляции и построить начальные связи с ТН ВЭД."""
    regulations_path = cfg.data.regulations_path
    embedding_dim = cfg.model.dim

    print("=" * 60)
    print(
        f"ЗАГРУЗКА РЕГУЛЯЦИЙ: '{regulations_path}', "
        f"размерность: {embedding_dim}"
    )
    print("=" * 60)

    regulations = DataLoader.load_regulations(regulations_path)
    db = DatabaseConnector(cfg.db)

    with db.get_connection() as conn:
        init_regulation_tables(conn, embedding_dim)
        load_regulations_data(regulations, conn)


if __name__ == "__main__":
    main()