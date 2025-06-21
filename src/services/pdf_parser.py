"""Модуль для извлечения данных из PDF файлов."""

import logging
import os
from typing import Optional, Tuple, List, Dict, Any
import fitz
from src.utilities.text_normalizer import TextNormalizer

logger = logging.getLogger(__name__)


class PDFParser:
    """Класс для извлечения данных из PDF файлов."""

    def __init__(self):
        self.debug_mode = os.environ.get("DEBUG_PDF_PARSER", "False").lower() == "true"
        self.max_blocks = 5  # Увеличиваем до 5 блоков вместо 3

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
                    first_page = doc[0]
                    logger.info("Извлечение данных из первой страницы")

                    try:
                        # Извлекаем текст из блоков
                        blocks_data = first_page.get_text("dict")
                        blocks_text = self._extract_blocks_text(blocks_data)

                        if not blocks_text or len(blocks_text) == 0:
                            logger.error(
                                "Не удалось извлечь блоки текста из первой страницы"
                            )
                            return None

                        if self.debug_mode:
                            debug_file = f"{pdf_path}_blocks_debug.txt"
                            with open(debug_file, "w", encoding="utf-8") as f:
                                for i, text in enumerate(blocks_text):
                                    f.write(f"--- Блок {i+1} ---\n{text}\n\n")
                            logger.info(f"Тексты блоков сохранены в {debug_file}")

                        # Первый блок - архив, блоки 2-5 (если есть) - организация
                        if len(blocks_text) >= 1:
                            archive_value = blocks_text[0]
                            logger.info(
                                f"Первый блок (архив): {archive_value[:100]}..."
                            )
                        else:
                            logger.error("Не найден блок для архива")
                            return None

                        # Объединяем блоки 2-5 (или сколько есть) для организации
                        org_blocks = blocks_text[1:] if len(blocks_text) > 1 else []
                        if org_blocks:
                            org_value = " ".join(org_blocks)
                            logger.info(f"Блоки организации: {org_value[:100]}...")
                        else:
                            logger.warning("Не найдены блоки для организации")
                            org_value = ""

                        # Нормализуем значения
                        archive_value = TextNormalizer.normalize(archive_value)
                        org_value = TextNormalizer.normalize_organization(org_value)

                        logger.info(f"Извлеченный архив: {archive_value}")
                        logger.info(f"Извлеченная организация: {org_value}")

                        return archive_value, org_value

                    except Exception as e:
                        logger.error(
                            f"Ошибка при извлечении блоков текста: {e}", exc_info=True
                        )
                        return None
                else:
                    logger.error("PDF файл не содержит страниц")
                    return None

        except Exception as e:
            logger.error(f"Ошибка чтения PDF: {e}", exc_info=True)
            return None

    def _extract_blocks_text(self, blocks_data: Dict[str, Any]) -> List[str]:
        """Извлекает текст из блоков данных с пропуском пустых блоков.

        Args:
            blocks_data: Данные блоков из get_text("dict")

        Returns:
            Список текстов из непустых блоков
        """
        if not isinstance(blocks_data, dict) or "blocks" not in blocks_data:
            logger.error("Неверный формат данных блоков")
            return []

        blocks = blocks_data["blocks"]
        logger.info(f"Всего блоков на странице: {len(blocks)}")

        # Счетчики для отслеживания обработки блоков
        processed_blocks = 0
        valid_blocks_found = 0
        skipped_empty_blocks = 0

        result = []

        # Перебираем блоки, пока не найдем достаточное количество непустых
        for i, block in enumerate(blocks):
            processed_blocks += 1
            block_text = self._extract_text_from_block(block)

            if block_text and block_text.strip():
                valid_blocks_found += 1
                result.append(block_text)
                logger.info(f"Блок {i+1}: извлечено {len(block_text)} символов")

                # Прекращаем поиск, если нашли достаточно непустых блоков
                if valid_blocks_found >= self.max_blocks:
                    break
            else:
                skipped_empty_blocks += 1
                logger.warning(f"Блок {i+1}: текст не извлечен")

        logger.info(
            f"Обработано блоков: {processed_blocks}, найдено непустых: {valid_blocks_found}, пропущено пустых: {skipped_empty_blocks}"
        )
        return result

    def _extract_text_from_block(self, block: Dict[str, Any]) -> str:
        """Извлекает текст из одного блока данных с улучшенной обработкой.

        Args:
            block: Блок данных из get_text("dict")

        Returns:
            Извлеченный текст из блока
        """
        if not block or "lines" not in block or not block["lines"]:
            return ""

        text_parts = []
        for line in block["lines"]:
            if "spans" in line and line["spans"]:
                line_parts = []
                for span in line["spans"]:
                    if "text" in span and span["text"].strip():
                        line_parts.append(span["text"])
                if line_parts:
                    text_parts.append(" ".join(line_parts))

        result = "\n".join(text_parts)

        # Дополнительная проверка на непустоту результата
        if result and result.strip():
            return result

        # Если результат пустой, и в блоке есть другие поля с текстом,
        # можно попробовать их использовать (например, alt-текст изображений)
        if "alt" in block and block["alt"]:
            return block["alt"]

        return ""
