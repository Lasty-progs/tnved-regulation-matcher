import psycopg


def apply_schema(conn: psycopg.Connection, embedding_dim: int):
    """Инициализация расширений и таблиц под заданную размерность вектора."""
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm;")

        cur.execute(
            f"""
            -- Дерево ТН ВЭД
            CREATE TABLE IF NOT EXISTS tnved_tree (
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

            CREATE INDEX IF NOT EXISTS idx_tnved_clean_code ON tnved_tree(clean_code);
            CREATE INDEX IF NOT EXISTS idx_tnved_code_level ON tnved_tree(code_level);
            CREATE INDEX IF NOT EXISTS idx_tnved_title_trgm ON tnved_tree USING gin (title gin_trgm_ops);

            -- Таблица регуляций
            CREATE TABLE IF NOT EXISTS regulations (
                regulation_id VARCHAR(50) PRIMARY KEY,
                decree_number TEXT,
                npa TEXT NOT NULL,
                clean_text TEXT NOT NULL,
                source TEXT,
                embedding vector({embedding_dim})
            );

            -- Таблица деклараций
            CREATE TABLE IF NOT EXISTS declarations (
                declaration_id VARCHAR(50) PRIMARY KEY,
                raw_data JSONB NOT NULL,
                goods_description TEXT NOT NULL,
                clean_description TEXT NOT NULL,
                embedding vector({embedding_dim})
            );

            -- Таблица связей Регуляция <-> ТН ВЭД
            CREATE TABLE IF NOT EXISTS regulation_tnved_links (
                id SERIAL PRIMARY KEY,
                regulation_id VARCHAR(50) REFERENCES regulations(regulation_id) ON DELETE CASCADE,
                tnved_id INT REFERENCES tnved_tree(id) ON DELETE CASCADE,
                match_type VARCHAR(30) NOT NULL,
                confidence FLOAT DEFAULT 1.0
            );

            CREATE INDEX IF NOT EXISTS idx_links_reg ON regulation_tnved_links(regulation_id);
            CREATE INDEX IF NOT EXISTS idx_links_tnved ON regulation_tnved_links(tnved_id);
        """
        )
    conn.commit()