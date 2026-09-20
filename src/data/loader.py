from dataclasses import dataclass
import json
from typing import Any, Dict, List, Optional

from src.data.text_cleaner import TextCleaner

@dataclass
class Declaration:
    """Данные таможенной декларации."""

    declaration_id: str
    g31_1: str
    g011: Optional[str] = None
    g32: Optional[int] = None
    desc_extention: Optional[str] = None
    has_acceptance_docs: Optional[int] = None
    raw: Optional[Dict[str, Any]] = None

    def get_search_query(self) -> str:
        """Сформировать очищенный поисковый запрос для декларации."""
        extension = self.desc_extention or ""
        raw_text = f"{self.g31_1} {extension}"

        return TextCleaner.clean_declaration(raw_text)

@dataclass
class Regulation:
    """Данные нормативно-правового акта."""

    regulation_id: str
    decree_number: str
    npa: str
    source: Optional[str] = None

    def get_clean_text(self) -> str:
        """Вернуть очищенный текст нормативно-правового акта."""
        return TextCleaner.clean_regulation(self.npa)

    def get_search_document(self) -> str:
        """Сформировать текст нормативного акта для индексации."""
        decree = self.decree_number or ""
        cleaned_text = self.get_clean_text()

        return f"{decree} {cleaned_text}".strip()

class DataLoader:
    """Загрузчик деклараций и нормативно-правовых актов."""

    @staticmethod
    def _load_jsonl(file_path: str) -> List[Dict[str, Any]]:
        """Загрузить записи из JSONL-файла."""
        try:
            file = open(file_path, "r", encoding="utf-8")
        except FileNotFoundError as error:
            raise FileNotFoundError(
                f"Файл не найден: {file_path}"
            ) from error

        records: List[Dict[str, Any]] = []

        with file:
            for line in file:
                line = line.strip()

                if not line:
                    continue

                records.append(json.loads(line))

        return records

    @staticmethod
    def load_declarations(file_path: str) -> List[Declaration]:
        """Загрузить декларации из JSONL-файла."""
        records = DataLoader._load_jsonl(file_path)

        return [
            Declaration(
                declaration_id=str(record.get("declaration_id")),
                g31_1=str(record.get("G31_1", "")),
                g011=record.get("G011"),
                g32=record.get("G32"),
                desc_extention=record.get("desc_extention"),
                has_acceptance_docs=record.get("has_acceptance_docs"),
                raw=record,
            )
            for record in records
        ]

    @staticmethod
    def load_regulations(file_path: str) -> List[Regulation]:
        """Загрузить нормативно-правовые акты из JSONL-файла."""
        records = DataLoader._load_jsonl(file_path)

        return [
            Regulation(
                regulation_id=str(record.get("regulation_id")),
                decree_number=str(record.get("decree_number", "")),
                npa=str(record.get("npa", "")),
                source=record.get("source"),
            )
            for record in records
        ]