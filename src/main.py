"""Основной скрипт для демонстрации работы с DocumentChecker."""

import logging
import os
import argparse
import json
from datetime import datetime
from typing import Dict, Any, Optional

from src.services.document_checker import DocumentChecker
from config.config import CONFIG
from src.utilities.text_normalizer import TextNormalizer

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


def process_document(
    checker: DocumentChecker,
    pdf_path: str,
    current_num: Optional[int] = None,
    total_num: Optional[int] = None,
) -> Dict[str, Any]:
    """Обрабатывает один документ и выводит результаты.

    Args:
        checker: Экземпляр DocumentChecker
        pdf_path: Путь к PDF файлу
        current_num: Текущий номер документа (для отображения прогресса)
        total_num: Общее количество документов (для отображения прогресса)

    Returns:
        Словарь с результатами обработки документа
    """
    # Вывод разделителя для лучшей читаемости при обработке нескольких документов
    if current_num is not None and total_num is not None:
        divider = "=" * 60
        logger.info(f"\n{divider}")
        logger.info(f"Документ {current_num}/{total_num}: {os.path.basename(pdf_path)}")
        logger.info(f"{divider}")
    else:
        logger.info(f"\nОбработка документа: {pdf_path}")

    # Получаем информацию о файле
    filename = os.path.basename(pdf_path)

    # Начальная структура результата
    result = {
        "filename": filename,
        "date": datetime.now().strftime("%Y-%m-%d"),
        "extracted_data": None,
        "exact_match": None,
        "matches": [],  # список кандидатов для ручного просмотра
    }

    # Сначала извлекаем данные из PDF (чтобы избежать повторной обработки)
    pdf_data = checker.pdf_parser.extract_data(pdf_path)

    # Если данные извлечены успешно, сохраняем их
    if pdf_data:
        pdf_archive, pdf_org = pdf_data
        result["extracted_data"] = {
            "archive": TextNormalizer.to_title_case(pdf_archive),
            "organization": TextNormalizer.to_title_case(pdf_org),
        }

        # Теперь ищем совпадения, передавая уже извлеченные данные
        matches = checker.find_matches(pdf_path, extracted_data=pdf_data)
    else:
        # Если данные не удалось извлечь, пытаемся искать совпадения стандартным способом
        matches = checker.find_matches(pdf_path)

    # Если найдены совпадения — сохраняем точные и потенциальные
    if matches:
        # Сначала сортируем по убыванию точности
        matches.sort(key=lambda x: x["avg_similarity"], reverse=True)

        # Выбираем только самое точное совпадение (первое)
        best_match = matches[0]

        # Проверяем, является ли оно точным
        if best_match.get("is_exact", False):
            # Сохраняем только самое точное совпадение в упрощенном формате
            result["exact_match"] = {
                "archive": best_match["archive_match"]["text"],
                "organization": best_match["organization_match"]["text"],
                "similarity": round(best_match["avg_similarity"], 2),
            }
            logger.info(
                f"Найдено точное совпадение: {best_match['archive_match']['text']}"
            )
        else:
            logger.info("Точных совпадений не найдено")

        # Сохраняем топ-N результатов для ручного анализа (включая уже найденное точное)
        top_n = CONFIG.get("top_n_results", 5)
        candidates = []
        for m in matches[:top_n]:
            candidates.append(
                {
                    "archive": m["archive_match"]["text"],
                    "organization": m["organization_match"]["text"],
                    "similarity_pair": round(m["avg_similarity"], 2),
                    "similarity_archive": None
                    if m["archive_match"].get("similarity") is None
                    else round(m["archive_match"]["similarity"], 2),
                    "similarity_org": None
                    if m["organization_match"].get("similarity") is None
                    else round(m["organization_match"]["similarity"], 2),
                    "is_exact": m.get("is_exact", False),
                    "is_potential": m.get("is_potential", False),
                }
            )

        result["matches"] = candidates
    else:
        logger.info("Совпадений не найдено")

    return result


def save_results_to_json(
    results: Dict[str, Any], output_path: str, pretty: bool = True
) -> None:
    """Сохраняет результаты в JSON-файл.

    Args:
        results: Словарь с результатами
        output_path: Путь для сохранения JSON-файла
        pretty: Флаг для форматирования JSON
    """
    try:
        # Создаем директорию, если она не существует
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        # Фильтруем результаты, чтобы оставить только нужную информацию
        simplified_results = {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "documents": [],
        }

        for result in results.get("results", []):
            doc_info = {
                "filename": result.get("filename", ""),
                "extracted_data": result.get("extracted_data", {}),
                "exact_match": result.get("exact_match", None),
                "matches": result.get("matches", []),
            }

            # Добавляем документ, если есть точное совпадение или хотя бы один кандидат
            if doc_info["exact_match"] or doc_info["matches"]:
                simplified_results["documents"].append(doc_info)

        # Форматирование JSON в зависимости от флага `pretty`
        indent_value = 4 if pretty else None

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                simplified_results,
                f,
                ensure_ascii=False,
                indent=indent_value,
            )

        logger.info(f"Результаты сохранены в файл: {output_path}")
    except Exception as e:
        logger.error(f"Ошибка при сохранении результатов в JSON: {e}")


def main():
    """Основная функция для демонстрации работы."""
    try:
        # Задаем путь для сохранения результатов по умолчанию
        default_output_json = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "results",
            f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        )

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
                "merged.csv",
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
        parser.add_argument(
            "--output-json",
            "-o",
            default=default_output_json,
            help="Путь для сохранения результатов в JSON формате",
        )
        parser.add_argument(
            "--pretty-json",
            "-j",
            action="store_true",
            help="Форматировать JSON для лучшей читаемости",
        )
        parser.add_argument(
            "--quiet",
            "-q",
            action="store_true",
            help="Отключить подробный вывод в консоль",
        )
        args = parser.parse_args()

        # Настройка уровня логирования
        if args.quiet:
            logger.setLevel(logging.WARNING)
        elif args.debug_pdf:
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

        # Подготовка структуры для результатов
        results = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "version": "1.0",
                "parameters": {
                    "reference_file": reference_csv,
                    "debug_pdf": args.debug_pdf,
                    "force_rebuild": args.force_rebuild,
                },
            },
            "results": [],
        }

        # Обработка PDF файлов
        if args.pdf_file:
            # Обработка одного файла
            if not os.path.exists(args.pdf_file):
                logger.error(f"PDF файл не найден: {args.pdf_file}")
                return
            result = process_document(checker, args.pdf_file)
            results["results"].append(result)
        else:
            # Обработка всех файлов в директории
            pdf_dir = args.pdf_dir
            if not os.path.exists(pdf_dir):
                logger.error(f"Директория с PDF файлами не найдена: {pdf_dir}")
                return

            # Собираем список всех PDF файлов
            pdf_files = [
                os.path.join(pdf_dir, filename)
                for filename in os.listdir(pdf_dir)
                if filename.lower().endswith(".pdf")
            ]

            # Обрабатываем каждый файл с отображением прогресса
            for i, pdf_path in enumerate(pdf_files, 1):
                result = process_document(checker, pdf_path, i, len(pdf_files))
                results["results"].append(result)

            logger.info(f"\nОбработано PDF файлов: {len(pdf_files)}")

        # Создаем директорию для результатов, если она не существует
        os.makedirs(os.path.dirname(args.output_json), exist_ok=True)

        # Сохранение результатов в JSON с учётом флага форматирования
        save_results_to_json(results, args.output_json, args.pretty_json)

        # Сохранение эмбеддингов, если запрошено
        if args.save_embeddings:
            logger.info("Сохранение эмбеддингов...")
            checker.save_embeddings()
            logger.info("Эмбеддинги сохранены")

    except Exception as e:
        logger.error(f"Ошибка при выполнении программы: {e}", exc_info=True)


if __name__ == "__main__":
    main()
