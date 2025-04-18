"""Основной скрипт для демонстрации работы с DocumentChecker."""

import logging
import os
import sys
import argparse
from typing import List, Dict

# Добавляем родительскую директорию в PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from document_checker import DocumentChecker

# Настройка логирования
log_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs"
)
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "document_check.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def process_document(checker: DocumentChecker, pdf_path: str) -> List[Dict]:
    """Обрабатывает один документ и выводит результаты.

    Args:
        checker: Экземпляр DocumentChecker
        pdf_path: Путь к PDF файлу

    Returns:
        Список найденных совпадений
    """
    logger.info(f"Обработка документа: {pdf_path}")
    matches = checker.find_matches(pdf_path)

    if matches:
        logger.info("Найдены совпадения:")
        for i, match in enumerate(matches, 1):
            match_type = (
                "ТОЧНОЕ СОВПАДЕНИЕ"
                if match.get("is_exact", False)
                else "Приблизительное совпадение"
            )
            logger.info(f"\nСовпадение {i} ({match_type}):")
            logger.info(
                f"Архив: {match['archive_match']['text']} "
                f"(схожесть: {match['archive_match']['similarity']:.2f})"
            )
            logger.info(
                f"Организация: {match['organization_match']['text']} "
                f"(схожесть: {match['organization_match']['similarity']:.2f})"
            )
    else:
        logger.info("Совпадений не найдено")

    return matches


def main():
    """Основная функция для демонстрации работы."""
    try:
        # Парсинг аргументов командной строки
        parser = argparse.ArgumentParser(
            description="Проверка документов по справочнику"
        )
        parser.add_argument(
            "--reference",
            "-r",
            default=os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "data",
                "combined_SP.csv",
            ),
            help="Путь к CSV файлу со справочными данными",
        )
        parser.add_argument(
            "--pdf_dir",
            "-p",
            default=os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "pdf_documents",
            ),
            help="Директория с PDF файлами для обработки",
        )
        parser.add_argument(
            "--cache_dir",
            "-c",
            default=os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "embeddings_cache",
            ),
            help="Директория для кэширования эмбеддингов",
        )
        parser.add_argument(
            "--save_embeddings",
            "-s",
            action="store_true",
            help="Сохранить эмбеддинги после обработки",
        )
        parser.add_argument(
            "--debug_pdf",
            "-d",
            action="store_true",
            help="Включить режим отладки PDF парсера",
        )
        parser.add_argument(
            "--pdf_file", "-f", help="Обработать только один указанный PDF файл"
        )
        parser.add_argument(
            "--force_rebuild",
            "-b",
            action="store_true",
            help="Принудительно пересоздать кэш эмбеддингов, игнорируя существующие файлы",
        )
        args = parser.parse_args()

        # Устанавливаем переменную окружения для отладки PDF парсера
        if args.debug_pdf:
            os.environ["DEBUG_PDF_PARSER"] = "True"
            logger.setLevel(logging.DEBUG)
            logger.info("Режим отладки PDF парсера включен")

        # Инициализация проверяльщика с кэшированием
        checker = DocumentChecker(cache_dir=args.cache_dir)

        # Загрузка справочных данных
        reference_csv = args.reference
        if not os.path.exists(reference_csv):
            logger.error(f"Файл справочных данных не найден: {reference_csv}")
            return

        if args.force_rebuild:
            logger.info("Принудительное пересоздание кэша эмбеддингов")

        checker.load_reference_data(reference_csv, force_rebuild=args.force_rebuild)

        # Обработка PDF файлов
        if args.pdf_file:
            # Обработка одного файла
            if not os.path.exists(args.pdf_file):
                logger.error(f"PDF файл не найден: {args.pdf_file}")
                return
            process_document(checker, args.pdf_file)
        else:
            # Обработка всех файлов в директории
            pdf_dir = args.pdf_dir
            if not os.path.exists(pdf_dir):
                logger.error(f"Директория с PDF файлами не найдена: {pdf_dir}")
                return

            processed = 0
            for filename in os.listdir(pdf_dir):
                if filename.lower().endswith(".pdf"):
                    pdf_path = os.path.join(pdf_dir, filename)
                    process_document(checker, pdf_path)
                    processed += 1

            logger.info(f"Обработано PDF файлов: {processed}")

        # Сохранение эмбеддингов, если запрошено
        if args.save_embeddings:
            logger.info("Сохранение эмбеддингов...")
            checker.save_embeddings()
            logger.info("Эмбеддинги сохранены")

    except Exception as e:
        logger.error(f"Ошибка при выполнении программы: {e}", exc_info=True)


if __name__ == "__main__":
    main()
