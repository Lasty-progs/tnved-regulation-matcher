from contextlib import contextmanager
from typing import Generator
from omegaconf import DictConfig
from pgvector.psycopg import register_vector
import psycopg


class DatabaseConnector:

    def __init__(self, cfg: DictConfig):
        self.conn_str = (
            f"dbname={cfg.dbname} "
            f"user={cfg.user} "
            f"password={cfg.password} "
            f"host={cfg.host} "
            f"port={cfg.port}"
        )

    @contextmanager
    def get_connection(self) -> Generator[psycopg.Connection, None, None]:
        with psycopg.connect(self.conn_str) as conn:
            register_vector(conn)
            yield conn