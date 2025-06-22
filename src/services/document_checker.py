"""Основной модуль для проверки документов."""

import logging
import os
import re

import numpy as np
import faiss
import torch

from typing import List, Dict, Optional, Tuple
from sentence_transformers import SentenceTransformer
from src.services.pdf_parser import PDFParser
from src.services.reference_index import ReferenceIndex
from config.config import CONFIG
from src.utilities.text_normalizer import TextNormalizer

logger = logging.getLogger(__name__)


class DocumentChecker:
    """Основной класс для проверки документов."""

    def __init__(self, model_name: str = CONFIG["model_name"], cache_dir: str = None):
        """Инициализирует проверяльщик документов.

        Args:
            model_name: Название модели SentenceTransformer
            cache_dir: Директория для кэширования эмбеддингов
        """
        self.model = self._get_model(model_name)
        self.pdf_parser = PDFParser()
        self.reference_index = ReferenceIndex(self.model)
        self.cache_dir = cache_dir
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

    def _get_model(self, model_name: str) -> SentenceTransformer:
        """Получает модель SentenceTransformer.

        Args:
            model_name: Имя модели

        Returns:
            Инициализированная модель
        """
        try:
            logger.info(f"Инициализация модели: {model_name}")
            # Определяем устройство: GPU (cuda) если доступно, иначе CPU
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Выбрано устройство для модели: {device}")
            model = SentenceTransformer(model_name, device=device)
            logger.info("Модель успешно инициализирована")
            return model
        except Exception as e:
            logger.error(f"Ошибка инициализации модели: {e}")
            raise

    def load_reference_data(self, csv_path: str, force_rebuild: bool = False):
        """Загружает справочные данные.

        Args:
            csv_path: Путь к CSV файлу
            force_rebuild: Принудительно пересоздать кэш эмбеддингов
        """
        self.reference_index.build_index(csv_path, self.cache_dir, force_rebuild)

    def save_embeddings(self, directory: str = None, prefix: str = ""):
        """Сохраняет эмбеддинги в указанную директорию.

        Args:
            directory: Директория для сохранения (по умолчанию cache_dir)
            prefix: Префикс для имен файлов
        """

        self.reference_index.save_embeddings(directory, prefix)
        logger.info(f"Все эмбеддинги сохранены в директорию {directory}")

    def find_matches(
        self, pdf_path: str, extracted_data: Optional[Tuple[str, str]] = None
    ) -> List[Dict]:
        """Ищет совпадения для PDF файла в справочнике.

        Args:
            pdf_path: Путь к PDF файлу
            extracted_data: Опциональные извлеченные данные (архив, организация) для предотвращения повторной обработки

        Returns:
            Отсортированный по точности список найденных совпадений
        """
        try:
            # Извлекаем данные из PDF, если не переданы
            if extracted_data:
                pdf_archive, pdf_org = extracted_data
                logger.info(f"Используем предварительно извлеченные данные")
            else:
                # Извлекаем данные из PDF
                pdf_data = self.pdf_parser.extract_data(pdf_path)
                if not pdf_data:
                    logger.error("Не удалось извлечь данные из PDF")
                    return []
                pdf_archive, pdf_org = pdf_data
                logger.info(f"Извлеченные данные из PDF: {pdf_archive}, {pdf_org}")

            # Нормализуем название организации
            pdf_org_normalized = TextNormalizer.normalize_organization(pdf_org)
            logger.info(f"Нормализованное название организации: {pdf_org_normalized}")

            # -------------------------------------------------------------
            # Определяем, нужно ли игнорировать сравнение по архиву. Если
            # извлечённый текст архива не содержит слова, начинающегося на
            # «архив» (архив, архивный, архивного ...), считаем, что архив
            # определить не удалось и выполняем поиск ТОЛЬКО по организации.
            # -------------------------------------------------------------
            ignore_archive = not bool(re.search(r"(?i)\bархив\w*", pdf_archive))

            if ignore_archive:
                logger.info(
                    "В извлечённом значении архива отсутствует слово 'архив*' — пропускаю сравнение архива"
                )

                # Сначала пытаемся найти точное совпадение по организации
                exact_results = []
                for record in self.reference_index.reference_data:
                    org_text = TextNormalizer.normalize_organization(record["Наименование"])
                    if org_text == pdf_org_normalized:
                        exact_results.append(
                            {
                                "archive_match": {
                                    "text": record["Архив"],
                                    "similarity": None,
                                },
                                "organization_match": {
                                    "text": record["Наименование"],
                                    "similarity": 1.0,
                                },
                                "avg_similarity": 1.0,
                                "is_exact": True,
                                "archive_ignored": True,
                            }
                        )

                if exact_results:
                    logger.info("Найдено точное совпадение по организации — архив возвращён из справочника")
                    return exact_results

                # Если точного совпадения нет, используем семантический поиск только по организации
                distances, indices = self.reference_index.search(
                    pdf_org_normalized, "organization", top_k=10
                )

                logger.info("Топ-5 совпадений по организациям (archive ignored):")
                for i in range(min(5, len(indices[0]))):
                    idx = indices[0][i]
                    org_name = self.reference_index.reference_data[idx]["Наименование"]
                    similarity = distances[0][i]
                    logger.info(f"{i+1}. {org_name} - {similarity:.4f}")

                results = []
                for i, idx in enumerate(indices[0]):
                    if distances[0][i] < CONFIG["threshold"]:
                        continue
                    results.append(
                        {
                            "archive_match": {
                                "text": self.reference_index.reference_data[idx]["Архив"],
                                "similarity": None,
                            },
                            "organization_match": {
                                "text": self.reference_index.reference_data[idx]["Наименование"],
                                "similarity": float(distances[0][i]),
                            },
                            "avg_similarity": float(distances[0][i]),
                            "is_exact": False,
                            "archive_ignored": True,
                        }
                    )

                # Сортируем и ограничиваем top-N
                if results:
                    results.sort(key=lambda x: x["avg_similarity"], reverse=True)
                    desired_top_n = CONFIG.get("top_n_results", 5)
                    return results[:desired_top_n]

                # Если ничего не найдено
                logger.warning("Семантический поиск по организации не дал результатов")
                return []

            # -------------------------------------------------------------
            # Если архив присутствует, пробуем парный поиск (архив+организация)
            # -------------------------------------------------------------

            logger.info("Выполняю семантический поиск по парному индексу (архив+организация)")

            pair_query = f"{TextNormalizer.normalize(pdf_archive)}; {pdf_org_normalized}"

            pair_distances, pair_indices = self.reference_index.search(
                pair_query, "pair", top_k=10
            )

            results = []

            logger.info("Топ-5 совпадений по паре архив+организация:")
            for i in range(min(5, len(pair_indices[0]))):
                idx = pair_indices[0][i]
                rec = self.reference_index.reference_data[idx]
                sim = pair_distances[0][i]
                logger.info(
                    f"{i+1}. {rec['Архив']} | {rec['Наименование']} - {sim:.4f}"
                )

            # Фильтрация по порогу
            for i, idx in enumerate(pair_indices[0]):
                sim = pair_distances[0][i]
                if sim < CONFIG["threshold"]:
                    continue

                rec = self.reference_index.reference_data[idx]
                results.append(
                    {
                        "archive_match": {
                            "text": rec["Архив"],
                            "similarity": float(sim),
                        },
                        "organization_match": {
                            "text": rec["Наименование"],
                            "similarity": float(sim),
                        },
                        "avg_similarity": float(sim),
                        "is_exact": False,
                        "archive_ignored": False,
                        "pair_search": True,
                    }
                )

            # Если получили удовлетворительные результаты — возвращаем top-N
            if results:
                results.sort(key=lambda x: x["avg_similarity"], reverse=True)
                desired_top_n = CONFIG.get("top_n_results", 5)
                return results[:desired_top_n]

            # Если парный поиск не дал результатов, fallback на старую логику ниже.

            # Создаем временное хранилище для результатов
            results = []

            # Прямой поиск точных соответствий
            logger.info("Ищу точные соответствия в справочнике...")
            exact_match_found = False

            # Получаем все данные из справочника
            for idx, record in enumerate(self.reference_index.reference_data):
                archive_text = TextNormalizer.normalize(record["Архив"])
                org_text = TextNormalizer.normalize_organization(record["Наименование"])

                # Проверяем точное совпадение архива  (затем будет True и от 0-1)
                archive_match = False
                archive_similarity = 0.0

                # Ищем ключевые слова из архива PDF в справочнике
                pdf_archive_parts = pdf_archive.split()
                if len(pdf_archive_parts) >= 3:  # Минимум три слова для надежности
                    key_location = pdf_archive_parts[
                        -1
                    ]  # Обычно название города/района
                    if key_location in archive_text:
                        archive_match = True
                        # Расчет оценки схожести по совпадающим словам
                        common_words = sum(
                            1 for word in pdf_archive_parts if word in archive_text
                        )
                        archive_similarity = common_words / len(pdf_archive_parts)

                # Проверяем название организации
                org_match = False
                org_similarity = 0.0

                pdf_org_parts = pdf_org_normalized.split()
                if len(pdf_org_parts) >= 3:
                    # Ищем совпадение первых слов (обычно тип организации)
                    first_words_match = all(
                        word in org_text for word in pdf_org_parts[:2]
                    )

                    # Ищем название местности в тексте
                    key_locations = [
                        pdf_org_parts[-1 - i] for i in range(min(5, len(pdf_org_parts)))
                    ]
                    location_match = any(loc in org_text for loc in key_locations)

                    if first_words_match and location_match:
                        org_match = True
                        # Расчет оценки схожести
                        common_words = sum(
                            1 for word in pdf_org_parts if word in org_text
                        )
                        org_similarity = common_words / len(pdf_org_parts)

                # Если оба совпадения найдены
                if archive_match and org_match:
                    exact_match_found = True
                    # Вычисляем среднюю схожесть для сортировки
                    avg_similarity = (archive_similarity + org_similarity) / 2.0

                    result = {
                        "archive_match": {
                            "text": record["Архив"],
                            "similarity": float(archive_similarity),
                        },
                        "organization_match": {
                            "text": record["Наименование"],
                            "similarity": float(org_similarity),
                        },
                        "avg_similarity": float(avg_similarity),
                        "is_exact": True,
                    }
                    results.append(result)
                    # Не логируем здесь каждое совпадение, чтобы не захламлять вывод

            # Если не найдено точных совпадений, используем семантический поиск
            if not exact_match_found:
                logger.info(
                    "Точные совпадения не найдены, использую семантический поиск"
                )

                # Ищем ближайшие совпадения по названию организации
                org_distances, org_indices = self.reference_index.search(
                    pdf_org_normalized, "organization", top_k=10
                )

                # Выводим топ совпадений для отладки
                logger.info("Топ-5 совпадений по организациям:")
                for i in range(min(5, len(org_indices[0]))):
                    idx = org_indices[0][i]
                    org_name = self.reference_index.reference_data[idx]["Наименование"]
                    similarity = org_distances[0][i]
                    logger.info(f"{i+1}. {org_name} - {similarity:.4f}")

                # Ищем архивы
                archive_distances, archive_indices = self.reference_index.search(
                    pdf_archive, "archive", top_k=10
                )

                logger.info("Топ-5 совпадений по архивам:")
                for i in range(min(5, len(archive_indices[0]))):
                    idx = archive_indices[0][i]
                    archive_name = self.reference_index.reference_data[idx]["Архив"]
                    similarity = archive_distances[0][i]
                    logger.info(f"{i+1}. {archive_name} - {similarity:.4f}")

                # Ищем записи, где совпадают индексы архива и организации
                match_found = False
                for i, org_idx in enumerate(org_indices[0]):
                    if org_distances[0][i] < CONFIG["threshold"]:
                        continue

                    for j, archive_idx in enumerate(archive_indices[0]):
                        if archive_distances[0][j] < CONFIG["threshold"]:
                            continue

                        if org_idx == archive_idx:
                            match_found = True
                            # Вычисляем среднюю схожесть для сортировки
                            avg_similarity = (
                                archive_distances[0][j] + org_distances[0][i]
                            ) / 2.0

                            result = {
                                "archive_match": {
                                    "text": self.reference_index.reference_data[
                                        archive_idx
                                    ]["Архив"],
                                    "similarity": float(archive_distances[0][j]),
                                },
                                "organization_match": {
                                    "text": self.reference_index.reference_data[
                                        org_idx
                                    ]["Наименование"],
                                    "similarity": float(org_distances[0][i]),
                                },
                                "avg_similarity": float(avg_similarity),
                                "is_exact": False,
                            }
                            results.append(result)

                # Если совпадений по индексу не найдено, используем первое лучшее совпадение
                if (
                    not match_found
                    and len(org_indices[0]) > 0
                    and len(archive_indices[0]) > 0
                ):
                    best_org_idx = org_indices[0][0]
                    best_org_similarity = org_distances[0][0]
                    best_archive_idx = archive_indices[0][0]
                    best_archive_similarity = archive_distances[0][0]

                    if (
                        best_org_similarity > CONFIG["threshold"]
                        and best_archive_similarity > CONFIG["threshold"]
                    ):
                        # Вычисляем среднюю схожесть для сортировки
                        avg_similarity = (
                            best_archive_similarity + best_org_similarity
                        ) / 2.0

                        result = {
                            "archive_match": {
                                "text": self.reference_index.reference_data[
                                    best_archive_idx
                                ]["Архив"],
                                "similarity": float(best_archive_similarity),
                            },
                            "organization_match": {
                                "text": self.reference_index.reference_data[
                                    best_org_idx
                                ]["Наименование"],
                                "similarity": float(best_org_similarity),
                            },
                            "avg_similarity": float(avg_similarity),
                            "is_exact": False,
                        }
                        results.append(result)
                        logger.info(
                            f"Найдено лучшее совпадение: "
                            f"{self.reference_index.reference_data[best_archive_idx]['Архив']} - "
                            f"{self.reference_index.reference_data[best_org_idx]['Наименование']}"
                        )

            # ---------------------------------------------------------------------------------
            # Дополняем список результатов «почти точными» совпадениями, если их меньше, чем
            # требуется по конфигу. Берём top-N (CONFIG["top_n_results"]) пар архив/организация
            # независимо от порога и помечаем их флагом is_potential=True.
            # ---------------------------------------------------------------------------------
            desired_top_n = CONFIG.get("top_n_results", 5)

            if (
                len(results) < desired_top_n
                and "org_indices" in locals()
                and "archive_indices" in locals()
            ):
                logger.info(
                    "Недостаточно совпадений выше порога — добавляю потенциальные результаты"
                )

                # Множество уже добавленных комбинаций, чтобы не дублировать записи
                existing_pairs = {
                    (r["archive_match"]["text"], r["organization_match"]["text"])
                    for r in results
                }

                # Составим кандидатов из декартова произведения top-K результатов
                candidate_items = []
                for i in range(len(org_indices[0])):
                    org_idx = int(org_indices[0][i])
                    org_text = self.reference_index.reference_data[org_idx][
                        "Наименование"
                    ]
                    org_sim = float(org_distances[0][i])

                    for j in range(len(archive_indices[0])):
                        archive_idx = int(archive_indices[0][j])
                        archive_text = self.reference_index.reference_data[archive_idx][
                            "Архив"
                        ]
                        archive_sim = float(archive_distances[0][j])

                        pair_key = (archive_text, org_text)
                        if pair_key in existing_pairs:
                            continue

                        avg_similarity = (org_sim + archive_sim) / 2.0
                        candidate_items.append(
                            (
                                avg_similarity,
                                {
                                    "archive_match": {
                                        "text": archive_text,
                                        "similarity": archive_sim,
                                    },
                                    "organization_match": {
                                        "text": org_text,
                                        "similarity": org_sim,
                                    },
                                    "avg_similarity": avg_similarity,
                                    "is_exact": False,
                                    "is_potential": True,
                                },
                            )
                        )

                # Сортируем кандидатов по убыванию средней схожести
                candidate_items.sort(key=lambda x: x[0], reverse=True)

                for _, cand in candidate_items:
                    results.append(cand)
                    if len(results) >= desired_top_n:
                        break

                # Пересортируем результаты с учётом добавленных кандидатов
                results.sort(key=lambda x: x["avg_similarity"], reverse=True)

            # Сортируем результаты по убыванию средней схожести
            if results:
                results.sort(key=lambda x: x["avg_similarity"], reverse=True)
                logger.info(f"Результаты отсортированы по убыванию точности совпадения")

                # Выводим только топ-N результатов (из конфига)
                logger.info("Найдены совпадения:")
                top_results = results[: min(desired_top_n, len(results))]
                for i, match in enumerate(top_results, 1):
                    # Только для первого (наиболее точного) совпадения указываем "ТОЧНОЕ СОВПАДЕНИЕ"
                    if i == 1:
                        match_type = "ТОЧНОЕ СОВПАДЕНИЕ"
                        logger.info(f"\nСовпадение {i} ({match_type}):")
                    else:
                        logger.info(f"\nСовпадение {i}:")

                    logger.info(
                        f"Архив: {match['archive_match']['text']} "
                        f"(схожесть: {match['archive_match']['similarity']:.2f})"
                    )
                    logger.info(
                        f"Организация: {match['organization_match']['text']} "
                        f"(схожесть: {match['organization_match']['similarity']:.2f})"
                    )

            return results

        except Exception as e:
            logger.error(f"Ошибка при поиске совпадений: {e}", exc_info=True)
            return []

    # ------------------------------------------------------------------
    # Вспомогательные методы отладки
    # ------------------------------------------------------------------
    def debug_similarity(self, text1: str, text2: str) -> float:
        """Вычисляет косинусную схожесть между двумя строками.

        Args:
            text1: Первая строка
            text2: Вторая строка

        Returns:
            Косинусная схожесть (от 0 до 1).
        """
        if not text1 or not text2:
            logger.warning("Одна из строк для debug_similarity пуста — возвращаю 0.0")
            return 0.0

        embeddings = self.model.encode([text1, text2], convert_to_tensor=False)
        embeddings = np.asarray(embeddings, dtype=np.float32)
        faiss.normalize_L2(embeddings)
        similarity = float(np.dot(embeddings[0], embeddings[1]))
        logger.info(
            f"Debug similarity между '{text1[:50]}...' и '{text2[:50]}...' = {similarity:.4f}"
        )
        return similarity
