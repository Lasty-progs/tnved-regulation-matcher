from dataclasses import dataclass
import re
from typing import List, Optional

@dataclass
class TnvedNode:
    """Узел иерархии ТН ВЭД."""

    temp_id: int
    code: Optional[str]
    clean_code: Optional[str]
    code_level: str
    title: str
    full_name: Optional[str]
    full_path: str
    indent_level: int
    parent_temp_id: Optional[int]

class TnvedParser:
    """Парсер иерархии ТН ВЭД из текстового дампа."""

    SECTION_PATTERN = re.compile(r"^[IVXLCDM]+\b")
    FULL_NAME_PATTERN = re.compile(r"\[(.*?)\]$")
    TITLE_PREFIX_PATTERN = re.compile(r"^[\s–—-]+")

    @staticmethod
    def determine_level(
        clean_code: str,
        is_intermediate: bool,
    ) -> str:
        """Определить уровень узла по коду."""
        if is_intermediate:
            return "intermediate"

        code_length = len(clean_code)

        if code_length == 2:
            return "group"

        if code_length == 4:
            return "heading"

        if code_length == 6:
            return "subheading"

        if code_length in (8, 9, 10):
            return "item"

        return "intermediate"

    @staticmethod
    def _parse_full_name(description: str) -> tuple[str, Optional[str]]:
        """Извлечь полное наименование из квадратных скобок."""
        match = TnvedParser.FULL_NAME_PATTERN.search(description)

        if not match:
            return description, None

        full_name = match.group(1).strip()
        title = description[: match.start()].strip()

        return title, full_name

    @staticmethod
    def _normalize_code(code: str) -> Optional[str]:
        """Удалить пробелы из кода и вернуть None для пустого значения."""
        clean_code = re.sub(r"\s+", "", code)
        return clean_code or None

    @staticmethod
    def _normalize_title(title: str) -> str:
        """Удалить служебные символы в начале наименования."""
        return TnvedParser.TITLE_PREFIX_PATTERN.sub("", title).strip()

    @staticmethod
    def _update_hierarchy(
        stack: List[tuple[int, int, str]],
        indent: int,
    ) -> Optional[int]:
        """Обновить стек иерархии и вернуть идентификатор родителя."""
        while stack and stack[-1][0] >= indent:
            stack.pop()

        return stack[-1][1] if stack else None

    @staticmethod
    def _build_full_path(
        stack: List[tuple[int, int, str]],
        title: str,
        full_name: Optional[str],
    ) -> str:
        """Сформировать полный путь узла в иерархии."""
        path_titles = [item[2] for item in stack]
        path_titles.append(full_name or title)

        return " > ".join(path_titles)

    @classmethod
    def _parse_section(
        cls,
        code: str,
        description: str,
        indent: int,
        temp_id: int,
    ) -> TnvedNode:
        """Создать узел раздела ТН ВЭД."""
        return TnvedNode(
            temp_id=temp_id,
            code=code,
            clean_code=code,
            code_level="section",
            title=description,
            full_name=None,
            full_path=f"Раздел {code}: {description}",
            indent_level=indent,
            parent_temp_id=None,
        )

    @classmethod
    def parse(cls, file_path: str) -> List[TnvedNode]:
        """Распарсить иерархию ТН ВЭД из текстового файла."""
        records: List[TnvedNode] = []
        stack: List[tuple[int, int, str]] = []
        in_hierarchy = False
        temp_id_counter = 0

        try:
            file = open(file_path, "r", encoding="utf-8")
        except FileNotFoundError as error:
            raise FileNotFoundError(
                f"Файл дампа ТН ВЭД не найден: {file_path}"
            ) from error

        with file:
            for line in file:
                raw_line = line.rstrip("\r\n").replace("\xa0", " ")
                stripped_line = raw_line.strip()

                if not stripped_line:
                    continue

                if "ИЕРАРХИЯ ТН ВЭД" in stripped_line or (
                    stripped_line.startswith("ИЕРАРХИЯ")
                ):
                    in_hierarchy = True
                    continue

                if stripped_line.startswith(
                    ("ПРИМЕЧАНИЯ", "ПОЯСНЕНИЯ", "ОБЩИЕ ПОЛОЖЕНИЯ")
                ):
                    in_hierarchy = False
                    continue

                if not in_hierarchy or "|" not in raw_line:
                    continue

                pipe_position = raw_line.find("|")
                indent = pipe_position
                code_part = raw_line[:pipe_position].strip()
                description = raw_line[pipe_position + 1 :].strip()

                if cls.SECTION_PATTERN.match(code_part):
                    temp_id_counter += 1
                    node = cls._parse_section(
                        code_part,
                        description,
                        indent,
                        temp_id_counter,
                    )
                    records.append(node)
                    stack.clear()
                    stack.append(
                        (indent, temp_id_counter, description)
                    )
                    continue

                title, full_name = cls._parse_full_name(description)
                title = cls._normalize_title(title)
                clean_code = cls._normalize_code(code_part)
                is_intermediate = clean_code is None

                level = cls.determine_level(
                    clean_code or "",
                    is_intermediate,
                )
                parent_temp_id = cls._update_hierarchy(
                    stack,
                    indent,
                )
                full_path = cls._build_full_path(
                    stack,
                    title,
                    full_name,
                )

                temp_id_counter += 1
                node = TnvedNode(
                    temp_id=temp_id_counter,
                    code=code_part or None,
                    clean_code=clean_code,
                    code_level=level,
                    title=title,
                    full_name=full_name,
                    full_path=full_path,
                    indent_level=indent,
                    parent_temp_id=parent_temp_id,
                )

                records.append(node)
                stack.append((indent, temp_id_counter, title))

        return records
