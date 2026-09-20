import re
from typing import List

class TextCleaner:
    """Очистка и токенизация текстов для retrieval-задач."""

    FOOTNOTE_PATTERN = re.compile(
        r"\(см\.[^)]*\)|\([^)]*прил[^)]*\)|\([^)]*решен[^)]*\)",
        re.IGNORECASE,
    )
    SECTION_NUMBER_PATTERN = re.compile(
        r"^(?:раздел\s+\d+[\s,]*|\d+(?:\.\d+)+\.?\s*)",
        re.IGNORECASE,
    )
    WHITESPACE_PATTERN = re.compile(r"\s+")
    SERIAL_NUMBER_PATTERN = re.compile(r"\b[A-Z0-9]{12,}\b")
    TOKEN_PATTERN = re.compile(r"\b\w{2,}\b")

    @staticmethod
    def clean_regulation(text: str) -> str:
        """Очистить текст нормативного акта от ссылок и служебного шума."""
        if not text:
            return ""

        text = TextCleaner.FOOTNOTE_PATTERN.sub(" ", text)
        text = TextCleaner.SECTION_NUMBER_PATTERN.sub("", text)

        return TextCleaner.WHITESPACE_PATTERN.sub(" ", text).strip()

    @staticmethod
    def clean_declaration(text: str) -> str:
        """Очистить описание товара от служебных данных и лишнего шума."""
        if not text:
            return ""

        text = TextCleaner.SERIAL_NUMBER_PATTERN.sub(" ", text)

        return TextCleaner.WHITESPACE_PATTERN.sub(" ", text).strip()

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Разбить текст на токены для лексического поиска."""
        return TextCleaner.TOKEN_PATTERN.findall(text.lower())