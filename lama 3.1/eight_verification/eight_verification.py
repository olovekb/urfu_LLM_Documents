import os
import re
import logging
import fitz  # PyMuPDF
import ollama
from typing import Optional, Tuple

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
    "flexible_config": {"temperature": 0.3, "top_p": 0.9, "frequency_penalty": 0.1},
    "file_paths": {
        "pdf": "lama 3.1/eight_verification/Агаповский_архив,_КСП,_ф_79,_оп_2_л_с_за_2022_год.pdf",
                },
}

# Шаблон запроса
PROMPT_TEMPLATE = """
ТОЧНО СРАВНИТЕ ОБА ПАРАМЕТРА ПОСЛЕДОВАТЕЛЬНО:

ШАГ 1: Сравните НАИМЕНОВАНИЯ
- Нормализуйте оба значения: игнорируйте регистр, пунктуацию, пробелы, окончания и порядок слов
- Определите: сохраняется ли основной смысл после нормализации?

ФИНАЛЬНОЕ РЕШЕНИЕ (ВЫБЕРИТЕ ТОЛЬКО ОДИН ОТВЕТ):
- "Совпадение найдено" — ЕСЛИ ОБА ПАРАМЕТРА ИМЕЮТ СМЫСЛОВОЕ ЕДИНСТВО ПОСЛЕ НОРМАЛИЗАЦИИ
- "Совпадение не найдено" — ЕСЛИ ЛЮБОЙ ИЗ ПАРАМЕТРОВ (хотя бы один) ИМЕЕТ ОТЛИЧИЯ В ОСНОВНОМ СОДЕРЖАНИИ.

Данные для сравнения:
PDF Наименование 1: {pdf_name_1}
PDF Наименование 2: {pdf_name_2}
"""

def extract_pdf_data(pdf_path: str) -> Optional[Tuple[str, str]]:
    """Извлекает наименования организаций из строк "УТВЕРЖДАЮ" и "СОГЛАСОВАНО"."""
    try:
        with fitz.open(pdf_path) as doc:
            # Извлекаем текст с последнего листа
            last_page_text = doc[-1].get_text("text")

            # 1. Поиск строки "УТВЕРЖДАЮ" на последнем листе и извлечение наименования
            approve_match = re.search(r"УТВЕРЖДАЮ[\s\S]*?([А-ЯЁа-яё]+(?:[-\s][А-ЯЁа-яё]+)*\s+палаты\s+[А-ЯЁа-яё]+(?:[-\s][А-ЯЁа-яё]+)*[\sА-ЯЁа-яё]+)", last_page_text)
            if not approve_match:
                logger.error("Не удалось найти строку 'УТВЕРЖДАЮ' или наименование организации")
                return None

            # Извлекаем наименование организации, игнорируя должности и лишние слова
            approve_organization = approve_match.group(1).strip()
            approve_organization = re.sub(r"(района|области|года|№\s*\d+|Фонд\s№\s*\d+|опись\s№\s*\d+)", "", approve_organization).strip()

            # 2. Поиск строки "СОГЛАСОВАНО" на последнем листе и извлечение наименования
            agree_match = re.search(r"СОГЛАСОВАНО[\s\S]*?Протокол\s+(.+)", last_page_text)
            if not agree_match:
                logger.error("Не удалось найти строку 'СОГЛАСОВАНО'")
                return None

            # Извлекаем наименование организации, игнорируя "Протокол ЭК" и другие лишние части
            agree_organization = agree_match.group(1).strip()
            agree_organization = re.sub(r"(района|области|года|№\s*\d+|Фонд\s№\s*\d+|опись\s№\s*\d+)", "", agree_organization).strip()

            return approve_organization, agree_organization

    except Exception as e:
        logger.error(f"Ошибка чтения PDF: {e}", exc_info=True)
        return None

def normalize_text(text: str) -> str:
    """Нормализует текст для сравнения (игнорируя окончания и незначительные различия)."""
    if not isinstance(text, str):
        return ""

    text = re.sub(r"[^\w\s]", "", text.lower())  # Удаляем знаки препинания и приводим к нижнему регистру
    return re.sub(r"\s+", " ", text).strip()  # Убираем лишние пробелы

def compare_texts_with_llama(pdf_name_1: str, pdf_name_2: str, model_params: dict) -> str:
    """Сравнивает два наименования с помощью LLM."""
    try:
        response = ollama.chat(
            model=CONFIG["model_name"],
            messages=[{
                "role": "user",
                "content": PROMPT_TEMPLATE.format(
                    pdf_name_1=pdf_name_1,
                    pdf_name_2=pdf_name_2,
                ),
            }],
            options={**model_params, "stop": ["\n"]},
        )
        return response["message"]["content"].strip().lower()

    except Exception as e:
        logger.error(f"Ошибка запроса к модели: {e}", exc_info=True)
        return "error"

def find_matches(pdf_path: str) -> bool:
    """Основная функция для поиска совпадений в PDF."""

    try:
        # Проверка существования файла
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF файл не найден: {pdf_path}")

        # Извлечение данных из PDF
        pdf_data = extract_pdf_data(pdf_path)
        if not pdf_data:
            logger.error("Не удалось извлечь данные из PDF")
            return False

        approve_organization, agree_organization = pdf_data

        logger.info(f"Извлеченные наименования организаций: {approve_organization}, {agree_organization}")

        # Нормализация наименований
        norm_approve_org = normalize_text(approve_organization)
        norm_agree_org = normalize_text(agree_organization)

        logger.info(f"Нормализованные наименования: '{norm_approve_org}' и '{norm_agree_org}'")

        # Шаг 1: Сравнение наименования из строки "УТВЕРЖДАЮ" и строки "СОГЛАСОВАНО"
        agree_result = compare_texts_with_llama(norm_approve_org, norm_agree_org, CONFIG["flexible_config"])

        if "совпадение найдено" in agree_result:
            logger.info(f"Совпадение найдено между наименованием из строки 'УТВЕРЖДАЮ' и строкой 'СОГЛАСОВАНО'")
            return True
        else:
            logger.info(f"Совпадение не найдено между наименованием из строки 'УТВЕРЖДАЮ' и строкой 'СОГЛАСОВАНО'")
            # Логируем причины несоответствия
            logger.info(f"Причины несоответствия: {norm_approve_org} и {norm_agree_org}")
            return False

    except Exception as e:
        logger.error(f"Критическая ошибка: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    result = find_matches(CONFIG["file_paths"]["pdf"])
    logger.info(f"Итоговый результат: {'Совпадение найдено' if result else 'Совпадений нет'}")
