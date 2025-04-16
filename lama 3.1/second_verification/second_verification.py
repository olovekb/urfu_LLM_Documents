import os
import re
import logging
from typing import Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import fitz  # PyMuPDF
import ollama

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("comparison.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Конфигурация
CONFIG = {
    "model_name": "llama3.1",
    "max_tokens": 300,
    "chunksize": 50,
    "flexible_config": {"temperature": 0.3, "top_p": 0.9, "frequency_penalty": 0.1},
    "file_paths": {
        "pdf": "lama 3.1/second_verification/Копейск Фин. управление оп 2 лс.pdf",
        "csv": "lama 3.1/second_verification/FINAL_CLEANED_SP3.csv",
    },
}

PROMPT_TEMPLATE = """
ТОЧНО СРАВНИТЕ ОБА ПАРАМЕТРА ПОСЛЕДОВАТЕЛЬНО:

ШАГ 1: Сравните АРХИВНЫЕ ДАННЫЕ
- Нормализуйте оба значения: игнорируйте регистр, пунктуацию, пробелы, окончания и порядок слов
- Определите: сохраняется ли основной смысл после нормализации?

ШАГ 2: Сравните НАИМЕНОВАНИЯ
- Нормализуйте оба значения аналогично первому шагу
- Проверьте: остаются ли ключевые смысловые элементы идентичными?

ФИНАЛЬНОЕ РЕШЕНИЕ (ВЫБЕРИТЕ ТОЛЬКО ОДИН ОТВЕТ):
- "Совпадение найдено" — ЕСЛИ ОБА ПАРАМЕТРА (архив И наименование) ИМЕЮТ СМЫСЛОВОЕ ЕДИНСТВО ПОСЛЕ НОРМАЛИЗАЦИИ
- "Совпадение не найдено" — ЕСЛИ ЛЮБОЙ ИЗ ПАРАМЕТРОВ (хотя бы один) ИМЕЕТ ОТЛИЧИЯ В ОСНОВНОМ СОДЕРЖАНИИ. 

Примеры:
1. PDF Архив: "Арх. 2023-дел/45", CSV Архив: "2023 дел 45" → СОВПАДАЕТ
   PDF Наименование: "Отчет финансовый", CSV: "финансовый отчет" → СОВПАДАЕТ
   >>> ОБЩИЙ РЕЗУЛЬТАТ: Совпадение найдено

2. PDF Архив: "ЛД-2024/7", CSV Архив: "ЛД 2024/8" → НЕ СОВПАДАЕТ 
   (даже если наименования совпадают) 
   >>> ОБЩИЙ РЕЗУЛЬТАТ: Совпадение не найдено

Данные для сравнения:
PDF Архив: {pdf_archive}
PDF Наименование: {pdf_name}
CSV Архив: {csv_archive}
CSV Наименование: {csv_name}

Ответьте СТРОГО ОДНОЙ ФРАЗОЙ БЕЗ ДОПОЛНЕНИЙ:
- Совпадение найдено
- Совпадение не найдено
"""


def extract_pdf_data(pdf_path: str) -> Optional[Tuple[str, str]]:
    """Корректно разделяет архив и наименование по структурным признакам."""
    try:
        with fitz.open(pdf_path) as doc:
            full_text = " ".join(page.get_text("text") for page in doc)

            # 1. Находим ядро архивной записи
            archive_core_pattern = re.compile(
                r"(Архивный отдел\b[\s\S]*?)(?=\s{3,}|\n\s*[А-ЯЁ]|$)",
                re.IGNORECASE | re.DOTALL,
            )

            archive_match = archive_core_pattern.search(full_text)
            if not archive_match:
                return None

            # 2. Извлекаем полный текст архива
            archive_value = archive_match.group(1)
            archive_value = re.sub(r"\s+", " ", archive_value).strip()

            # Извлекаем следующий значимый текст после архива как наименование
            remaining_text = full_text[archive_match.end() :]
            name_value = re.search(r"\n\s*([^\n]+)", remaining_text)

            if name_value:
                name = re.sub(r"\s+", " ", name_value.group(1)).strip()
                return archive_value, name

        logger.warning("Не удалось извлечь данные из первой страницы PDF")
        return None

    except Exception as e:
        logger.error(f"Ошибка чтения PDF: {e}", exc_info=True)
        return None


def normalize_text(text: str) -> str:
    """Нормализует текст для сравнения."""
    if not isinstance(text, str):
        return ""

    text = re.sub(r"[^\w\s]", "", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def compare_texts_with_llama(
    pdf_archive: str, pdf_name: str, csv_archive: str, csv_name: str, model_params: dict
) -> str:
    """Сравнивает оба параметра с помощью LLM."""
    try:
        response = ollama.chat(
            model=CONFIG["model_name"],
            messages=[
                {
                    "role": "user",
                    "content": PROMPT_TEMPLATE.format(
                        pdf_archive=pdf_archive,
                        pdf_name=pdf_name,
                        csv_archive=csv_archive,
                        csv_name=csv_name,
                    ),
                }
            ],
            options={**model_params, "stop": ["\n"]},
        )
        return response["message"]["content"].strip().lower()

    except Exception as e:
        logger.error(f"Ошибка запроса к модели: {e}", exc_info=True)
        return "error"


def process_csv_chunk(chunk: pd.DataFrame, pdf_data: Tuple[str, str]) -> bool:
    """Обрабатывает чанк CSV с проверкой обоих параметров."""
    required_columns = {"Архив", "Наименование"}
    if not required_columns.issubset(chunk.columns):
        raise ValueError("Отсутствуют необходимые столбцы в CSV")

    pdf_arch, pdf_name = pdf_data

    for _, row in chunk.iterrows():
        csv_arch = str(row["Архив"]).strip()
        csv_name = str(row["Наименование"]).strip()

        if not csv_arch or not csv_name:
            continue

        # Нормализация данных
        norm = lambda x: normalize_text(x) if x else ""
        norm_pdf_arch = norm(pdf_arch)
        norm_pdf_name = norm(pdf_name)
        norm_csv_arch = norm(csv_arch)
        norm_csv_name = norm(csv_name)

        # Проверка обоих параметров
        result = compare_texts_with_llama(
            norm_pdf_arch,
            norm_pdf_name,
            norm_csv_arch,
            norm_csv_name,
            CONFIG["flexible_config"],
        )

        if "совпадение найдено" in result:
            logger.info(f"Полное совпадение в строке {_}:")
            logger.info(f"PDF Архив: {pdf_arch!r}")
            logger.info(f"PDF Наименование: {pdf_name!r}")
            logger.info(f"CSV Архив: {csv_arch!r}")
            logger.info(f"CSV Наименование: {csv_name!r}")
            return True

    return False


def find_matches(pdf_path: str, csv_path: str) -> bool:
    """Основная функция для поиска совпадений.

    Args:
        pdf_path: Путь к PDF файлу
        csv_path: Путь к CSV файлу

    Returns:
        True если найдено хотя бы одно совпадение, иначе False
    """
    try:
        # Проверка существования файлов
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF файл не найден: {pdf_path}")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV файл не найден: {csv_path}")

        # Извлечение данных из PDF
        pdf_data = extract_pdf_data(pdf_path)
        if not pdf_data:
            logger.error("Не удалось извлечь данные из PDF")
            return False

        logger.info(f"Извлеченные данные из PDF: {pdf_data}")

        # Параллельная обработка чанков CSV
        with ThreadPoolExecutor() as executor:
            chunks = pd.read_csv(
                csv_path,
                chunksize=CONFIG["chunksize"],
                dtype=str,
                on_bad_lines="warn",
            )
            futures = [
                executor.submit(process_csv_chunk, chunk, pdf_data) for chunk in chunks
            ]

            for future in as_completed(futures):
                if future.result():
                    executor.shutdown(wait=False)
                    return True

        logger.info("Совпадений не найдено")
        return False

    except Exception as e:
        logger.error(f"Критическая ошибка: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    result = find_matches(
        CONFIG["file_paths"]["pdf"],
        CONFIG["file_paths"]["csv"],
    )
    logger.info(
        f"Итоговый результат: {'Совпадение найдено' if result else 'Совпадений нет'}"
    )
