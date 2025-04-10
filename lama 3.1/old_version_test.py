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
    "flexible_config": {"temperature": 0.2, "top_p": 0.9, "frequency_penalty": 0.1},
    "file_paths": {
        "pdf": "urfu_LLM_Documents/lama 3.1/Копейск Фин. управление оп 2 лс.pdf",
        "csv": "urfu_LLM_Documents/lama 3.1/FINAL_CLEANED_SP3.csv",
    },
}

PROMPT_TEMPLATE = """
Сравните смысловое содержание двух пар текстов, учитывая ОБА параметра:
1. Архивные данные
2. Наименование

Игнорируйте:
- Регистр букв
- Пунктуацию и пробелы
- Окончания слов
- Порядок слов
- Наличие дополнительных слов

Данные из документа:
1. Архив: {pdf_archive}
2. Наименование: {pdf_name}

Данные из базы:
1. Архив: {csv_archive}
2. Наименование: {csv_name}

Ответьте ТОЛЬКО одной из фраз:
- Совпадение найдено (если совпадают ОБА параметра)
- Совпадение не найдено (если не совпадает ХОТЯ БЫ ОДИН параметр)
"""


def extract_pdf_data(pdf_path: str) -> Optional[Tuple[str, str]]:
    """Извлекает данные из первой страницы PDF после предложения с архивом."""
    try:
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF файл не существует: {pdf_path}")

        with fitz.open(pdf_path) as doc:
            # Работаем только с первой страницей
            first_page = doc[0]
            page_text = first_page.get_text("text")

            # Улучшенный паттерн для поиска архивных данных
            archive_pattern = re.compile(
                r"(?i)(Архив[^\n]*?)[:—\s]+([^\n]+?)(?=\n\s*[А-ЯA-Z]|$)",
                flags=re.DOTALL,
            )

            if match := archive_pattern.search(page_text):
                archive_value = match.group(2).strip()

                # Извлекаем следующий значимый текст после архива как наименование
                remaining_text = page_text[match.end() :]
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
