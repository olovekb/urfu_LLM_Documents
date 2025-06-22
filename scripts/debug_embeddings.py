import argparse
import logging
from typing import List

from src.services.document_checker import DocumentChecker
from src.utilities.text_normalizer import TextNormalizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def pretty_print_results(results: List[dict], top_k: int = 5) -> None:
    """Красиво выводит результаты поиска.

    Args:
        results: Список результатов от `DocumentChecker.find_matches()`.
        top_k: Сколько первых результатов показать.
    """
    if not results:
        logger.warning("Результаты поиска пусты")
        return

    print("\n==== TOP RESULTS ====\n")
    for i, res in enumerate(results[:top_k], 1):
        match_type = "Точное" if res.get("is_exact") else "Семантическое"
        if res.get("is_potential"):
            match_type += " (потенциальное)"

        print(f"#{i} — {match_type} совпадение (avg={res['avg_similarity']:.3f})")

        # Безопасное форматирование схожести архива
        arch_sim = res['archive_match']['similarity']
        arch_sim_str = f"{arch_sim:.3f}" if arch_sim is not None else "—"

        org_sim = res['organization_match']['similarity']
        org_sim_str = f"{org_sim:.3f}" if org_sim is not None else "—"

        print(f"  · Архив: {res['archive_match']['text']} — sim={arch_sim_str}")
        print(f"  · Организация: {res['organization_match']['text']} — sim={org_sim_str}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CLI-утилита для диагностики эмбеддингов справочника и PDF."
    )
    parser.add_argument("--csv", required=True, help="Путь к CSV файлу справочника")
    parser.add_argument("--pdf", required=True, help="Путь к PDF файлу для проверки")
    parser.add_argument(
        "--top_k", type=int, default=5, help="Количество top результатов для вывода"
    )
    parser.add_argument(
        "--force_rebuild",
        action="store_true",
        help="Принудительно пересоздать кэш эмбеддингов",
    )
    parser.add_argument(
        "--pair",
        action="store_true",
        help="Использовать парный поиск (архив+организация)",
    )

    args = parser.parse_args()

    checker = DocumentChecker(cache_dir="embeddings_cache")
    checker.load_reference_data(args.csv, force_rebuild=args.force_rebuild)

    extracted = checker.pdf_parser.extract_data(args.pdf)
    if not extracted:
        logger.error("Не удалось извлечь данные из PDF — прерываю работу")
        return

    pdf_archive, pdf_org = extracted
    print("\n==== EXTRACTED DATA ====\n")
    print(f"Архив (raw): {pdf_archive}")
    print(f"Организация (raw): {pdf_org}\n")

    archive_norm = TextNormalizer.normalize(pdf_archive)
    org_norm = TextNormalizer.normalize_organization(pdf_org)
    print("Нормализованные значения:")
    print(f"  · Архив: {archive_norm}")
    print(f"  · Организация: {org_norm}\n")

    # Схожесть архива и организации сами с собой (должна быть 1.0)
    self_sim_archive = checker.debug_similarity(archive_norm, archive_norm)
    self_sim_org = checker.debug_similarity(org_norm, org_norm)
    print("Схожесть «self» (контроль):")
    print(f"  · Archive self-similarity: {self_sim_archive:.3f}")
    print(f"  · Org self-similarity: {self_sim_org:.3f}\n")

    # Поиск совпадений
    results = checker.find_matches(args.pdf, extracted_data=extracted)
    pretty_print_results(results, top_k=args.top_k)


if __name__ == "__main__":
    main() 