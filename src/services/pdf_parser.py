"""Модуль для извлечения данных из PDF файлов."""

import logging
import os
import re
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
                        # ---------- 1. Извлекаем полный текст и блоки ----------
                        full_text: str = first_page.get_text("text") or ""

                        blocks_data = first_page.get_text("dict")
                        block_lines = self._extract_blocks_text(blocks_data)

                        if self.debug_mode:
                            debug_file = f"{pdf_path}_page1_debug.txt"
                            with open(debug_file, "w", encoding="utf-8") as f:
                                f.write(full_text)
                            logger.info(
                                f"Текст первой страницы сохранён в {debug_file}"
                            )

                        # Разбиваем полный текст на предложения (простое правило)
                        sentence_split_pattern = r"(?<=[.!?])\s+(?=[A-ZА-ЯЁ])"
                        sentences = re.split(sentence_split_pattern, full_text.strip())
                        sentences = [s.strip() for s in sentences if s.strip()]

                        # ---------- 2. Формируем кандидаты в порядке появления ----------
                        candidates: List[str] = []
                        candidates.extend(block_lines)
                        for sent in sentences:
                            if sent not in candidates:
                                candidates.append(sent)

                        if not candidates:
                            logger.error("Не удалось получить кандидаты строк для анализа")
                            return None

                        # ---------- 3. Функция скоринга для архива ----------
                        def score_archive(line: str) -> int:
                            l = line.lower()
                            score = 0
                            if re.search(r"\bархив\w*", l):
                                score += 3
                            if re.search(r"(отдел|управл|муниципаль|городск)", l):
                                score += 1
                            if re.search(r"(фонд|опись|дел|год)", l):
                                score -= 1
                            if len(l) > 200:
                                score -= 1
                            return score

                        # ---------- 4. Определяем архив ----------
                        best_idx = -1
                        best_score = -999
                        for idx, line in enumerate(candidates):
                            sc = score_archive(line)
                            if sc > best_score:
                                best_score = sc
                                best_idx = idx

                        archive_value = ""
                        archive_idx = -1
                        if best_score >= 2 and best_idx != -1:
                            archive_value = candidates[best_idx]
                            archive_idx = best_idx

                        # ---------- 5. Определяем организацию ----------
                        negative_org = re.compile(r"(фонд|опись|дел|год)", re.IGNORECASE)
                        org_value = ""

                        start_search = archive_idx + 1 if archive_idx != -1 else 0
                        i_ptr = start_search
                        while i_ptr < len(candidates):
                            cand = candidates[i_ptr]
                            if len(cand) < 20:
                                i_ptr += 1
                                continue
                            if negative_org.search(cand):
                                i_ptr += 1
                                continue
                            if re.search(r"\bархив\w*", cand, re.IGNORECASE):
                                i_ptr += 1
                                continue

                            # базовая строка организации
                            org_value = cand

                            # Попробуем «прицепить» 1-2 следующие строки, если они служебные продолжения
                            look_ahead = 1
                            while look_ahead <= 2 and (i_ptr + look_ahead) < len(candidates):
                                next_cand = candidates[i_ptr + look_ahead]
                                if (
                                    len(next_cand) < 15
                                    or negative_org.search(next_cand)
                                    or re.search(r"\bархив\w*", next_cand, re.IGNORECASE)
                                ):
                                    break
                                # Добавляем, если в сумме не превышаем 250 символов
                                if len(org_value) + len(next_cand) < 250:
                                    org_value = f"{org_value} {next_cand}"
                                    look_ahead += 1
                                else:
                                    break
                            break
                        # end while loop

                        # ---------- 6. Нормализация ----------
                        archive_norm = TextNormalizer.normalize(archive_value) if archive_value else ""
                        org_norm = TextNormalizer.normalize_organization(org_value)

                        logger.info(
                            f"Извлечённый архив: {archive_norm if archive_norm else '[игнорирован]'}"
                        )
                        logger.info(f"Извлечённая организация: {org_norm}")

                        return archive_norm, org_norm

                    except Exception as e:
                        logger.error(
                            f"Ошибка при извлечении текста первой страницы: {e}", exc_info=True
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
