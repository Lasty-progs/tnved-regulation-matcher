from typing import List, Optional

import numpy as np
import torch
from omegaconf import DictConfig
from sentence_transformers import SentenceTransformer

class TextEncoder:
    """Кодировщик текстов на основе SentenceTransformer."""

    def __init__(
        self,
        cfg: DictConfig,
        device: Optional[str] = None,
    ) -> None:
        self.cfg = cfg
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = SentenceTransformer(
            cfg.name,
            device=self.device,
        )

    def encode_queries(self, texts: List[str]) -> np.ndarray:
        """Закодировать поисковые запросы с query-prefix."""
        prefixed_texts = [
            f"{self.cfg.prefix_query}{text}"
            for text in texts
        ]

        return self.model.encode(
            prefixed_texts,
            batch_size=self.cfg.batch_size,
            normalize_embeddings=self.cfg.normalize,
            show_progress_bar=False,
        )

    def encode_documents(self, texts: List[str]) -> np.ndarray:
        """Закодировать документы с document-prefix."""
        prefixed_texts = [
            f"{self.cfg.prefix_doc}{text}"
            for text in texts
        ]

        return self.model.encode(
            prefixed_texts,
            batch_size=self.cfg.batch_size,
            normalize_embeddings=self.cfg.normalize,
            show_progress_bar=False,
        )