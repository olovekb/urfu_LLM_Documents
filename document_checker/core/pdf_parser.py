"""Модуль для извлечения данных из PDF файлов."""

import re
import logging
import os
from typing import Optional, Tuple, List
import fitz
from ..utils.text_normalizer import TextNormalizer

logger = logging.getLogger(__name__)


class PDFParser:
    """Класс для извлечения данных из PDF файлов."""

    def __init__(self):
        self.archive_patterns = [
            r"(?i)Архивный отдел\s*([^\n]+?)(?=\n|$)",
            r"(?i)Архив\s*([^\n]+?)(?=\n|$)",
            r"(?i)Архивный фонд\s*([^\n]+?)(?=\n|$)",
        ]
        self.end_patterns = [
            r"\n\s*[А-Я][А-Я\s]+\n",  # Новый заголовок
            r"\n\s*[0-9]+\s*$",  # Номер страницы
            r"\n\s*$",  # Конец документа
        ]
        self.debug_mode = os.environ.get("DEBUG_PDF_PARSER", "False").lower() == "true"

    def extract_data(self, pdf_path: str) -> Optional[Tuple[str, str]]:
        """Извлекает архив и организацию из PDF файла.

        Args:
            pdf_path: Путь к PDF файлу

        Returns:
            Кортеж (архив, организация) или None в случае ошибки
        """
        try:
            logger.info(f"Начало обработки PDF файла: {pdf_path}")

            if not os.path.exists(pdf_path):
                logger.error(f"Файл не найден: {pdf_path}")
                return None

            with fitz.open(pdf_path) as doc:
                logger.info(f"PDF открыт. Количество страниц: {len(doc)}")

                if len(doc) > 0:
                    # Извлекаем текст разными методами
                    extraction_methods = [
                        {"method": "text", "desc": "обычный текст"},
                        {"method": "html", "desc": "HTML структура"},
                        {"method": "dict", "desc": "блоки данных"},
                        {"method": "rawdict", "desc": "сырые блоки данных"},
                    ]

                    page_text = None
                    first_page = doc[0]

                    # Пробуем разные методы извлечения текста
                    for method_info in extraction_methods:
                        method = method_info["method"]
                        try:
                            current_text = first_page.get_text(method)

                            # Для dict/rawdict преобразуем в текст
                            if method in ["dict", "rawdict"]:
                                current_text = self._convert_blocks_to_text(
                                    current_text
                                )

                            # Для html извлекаем текст
                            if method == "html":
                                current_text = self._extract_text_from_html(
                                    current_text
                                )

                            if current_text and current_text.strip():
                                logger.info(
                                    f"Успешно извлечен текст методом '{method}'"
                                )

                                if self.debug_mode:
                                    debug_file = f"{pdf_path}_{method}_debug.txt"
                                    with open(debug_file, "w", encoding="utf-8") as f:
                                        f.write(current_text)
                                    logger.info(
                                        f"Текст (метод {method}) сохранен в {debug_file}"
                                    )

                                # Если еще не нашли текст, то используем этот
                                if page_text is None:
                                    page_text = current_text

                                # Ищем архив в тексте
                                archive_value, archive_end_pos = self._extract_archive(
                                    current_text
                                )
                                if archive_value:
                                    # Если нашли архив, то используем этот текст
                                    page_text = current_text
                                    logger.info(
                                        f"Найден архив методом {method}: {archive_value}"
                                    )
                                    break
                        except Exception as e:
                            logger.warning(
                                f"Ошибка при извлечении текста методом {method}: {e}"
                            )

                    if not page_text or not page_text.strip():
                        logger.error("Не удалось извлечь текст ни одним из методов")
                        return None

                    # Логирование первых 200 символов для отладки
                    preview = page_text[:200].replace("\n", "\\n")
                    logger.info(f"Текст начала PDF: {preview}...")

                    archive_value, archive_end_pos = self._extract_archive(page_text)
                    if not archive_value:
                        logger.error("Не найден архив в PDF. Проверяемые паттерны:")
                        for i, pattern in enumerate(self.archive_patterns):
                            logger.error(f"Паттерн {i+1}: {pattern}")
                        return None

                    logger.info(f"Найден архив: {archive_value}")
                    remaining_text = page_text[archive_end_pos:]
                    preview_remain = remaining_text[:200].replace("\n", "\\n")
                    logger.info(f"Оставшийся текст (начало): {preview_remain}...")

                    org_value = self._extract_organization(remaining_text)
                    if not org_value:
                        logger.warning(
                            "Организация не найдена в PDF - попытка извлечь из всего текста после архива"
                        )
                        # Пробуем взять несколько непустых строк после архива
                        lines = [
                            line.strip()
                            for line in remaining_text.split("\n")
                            if line.strip()
                        ]
                        if lines:
                            # Берем до 4 непустых строк для наименования организации
                            org_lines = lines[:4] if len(lines) > 4 else lines
                            org_value = " ".join(org_lines)
                            logger.info(
                                f"Использованы первые непустые строки как организация: {org_value}"
                            )
                        else:
                            return None

                    # Нормализуем значения
                    archive_value = TextNormalizer.normalize(archive_value)
                    org_value = TextNormalizer.normalize_organization(org_value)

                    logger.info(f"Извлеченный архив: {archive_value}")
                    logger.info(f"Извлеченная организация: {org_value}")

                    return archive_value, org_value
                else:
                    logger.error("PDF файл не содержит страниц")
                    return None

        except Exception as e:
            logger.error(f"Ошибка чтения PDF: {e}", exc_info=True)
            return None

    def _convert_blocks_to_text(self, blocks_data) -> str:
        """Преобразует блоки данных в текст.

        Args:
            blocks_data: Данные блоков из get_text("dict") или get_text("rawdict")

        Returns:
            Извлеченный текст
        """
        if isinstance(blocks_data, dict) and "blocks" in blocks_data:
            text_parts = []
            for block in blocks_data["blocks"]:
                if "lines" in block:
                    for line in block["lines"]:
                        if "spans" in line:
                            for span in line["spans"]:
                                if "text" in span:
                                    text_parts.append(span["text"])
            return "\n".join(text_parts)
        return ""

    def _extract_text_from_html(self, html_text: str) -> str:
        """Извлекает текст из HTML.

        Args:
            html_text: HTML текст

        Returns:
            Извлеченный текст
        """
        # Удаляем HTML теги
        text = re.sub(r"<[^>]*>", " ", html_text)
        # Заменяем множественные пробелы на один
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _extract_archive(self, text: str) -> Tuple[Optional[str], int]:
        """Извлекает значение архива из текста.

        Args:
            text: Исходный текст

        Returns:
            Кортеж (значение архива, позиция конца)
        """
        for pattern in self.archive_patterns:
            archive_match = re.search(pattern, text)
            if archive_match:
                archive_value = archive_match.group(1).strip()
                if not archive_value.lower().startswith("архивный отдел"):
                    archive_value = f"Архивный отдел {archive_value}"
                return archive_value, archive_match.end()
        return None, 0

    def _extract_organization(self, text: str) -> Optional[str]:
        """Извлекает наименование организации из текста.

        Args:
            text: Исходный текст

        Returns:
            Наименование организации или None
        """
        # Сначала пробуем найти строки до разделителей
        end_pos = len(text)
        for pattern in self.end_patterns:
            match = re.search(pattern, text)
            if match and match.start() < end_pos:
                end_pos = match.start()

        # Извлекаем наименование организации
        org_text = text[:end_pos].strip()

        # Разбиваем на строки и объединяем непустые
        org_lines = [line.strip() for line in org_text.split("\n") if line.strip()]

        # Если не удалось найти организацию с помощью разделителей, берем первые 3-4 непустые строки
        if not org_lines:
            logger.warning(
                "Организация не найдена через разделители - извлекаем первые непустые строки"
            )
            org_lines = [line.strip() for line in text.split("\n") if line.strip()]
            # Берем только первые 3-4 строки
            org_lines = org_lines[:4] if len(org_lines) > 4 else org_lines
        else:
            # Если нашли много строк через разделители, тоже ограничиваем количество
            org_lines = org_lines[:4] if len(org_lines) > 4 else org_lines

        if not org_lines:
            logger.warning(
                "Организация не найдена в PDF - текст пуст или содержит только пробельные символы"
            )
            return None

        # Объединяем выбранные строки
        result = " ".join(org_lines)
        logger.info(f"Извлечены строки организации: {result}")
        return result
