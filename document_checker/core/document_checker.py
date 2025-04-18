"""Основной модуль для проверки документов."""

import logging
import os
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from .pdf_parser import PDFParser
from .reference_index import ReferenceIndex
from ..config.config import CONFIG
from ..utils.text_normalizer import TextNormalizer
import numpy as np
import faiss

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
            device = "cpu"  # Используем CPU для стабильности
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
        if directory is None:
            directory = self.cache_dir
            if directory is None:
                directory = os.path.join(os.getcwd(), "cache")

        self.reference_index.save_embeddings(directory, prefix)
        logger.info(f"Все эмбеддинги сохранены в директорию {directory}")

    def find_matches(self, pdf_path: str) -> List[Dict]:
        """Ищет совпадения для PDF файла в справочнике.

        Args:
            pdf_path: Путь к PDF файлу

        Returns:
            Список найденных совпадений
        """
        try:
            # Извлекаем данные из PDF
            pdf_data = self.pdf_parser.extract_data(pdf_path)
            if not pdf_data:
                logger.error("Не удалось извлечь данные из PDF")
                return []

            pdf_archive, pdf_org = pdf_data
            logger.info(f"Извлеченные данные из PDF: {pdf_archive}, {pdf_org}")

            # Нормализуем название организации, удаляя цифры и сохраняя точки и запятые
            pdf_org_normalized = TextNormalizer.normalize_organization(pdf_org)
            logger.info(f"Нормализованное название организации: {pdf_org_normalized}")

            # Создаем временное хранилище для результатов
            results = []

            # Прямой поиск точных соответствий
            logger.info("Ищу точные соответствия в справочнике...")
            exact_match_found = False

            # Получаем все данные из справочника
            for idx, record in enumerate(self.reference_index.reference_data):
                archive_text = TextNormalizer.normalize(record["Архив"])
                org_text = TextNormalizer.normalize_organization(record["Наименование"])

                # Рассчитываем точность совпадения (сначала строгое сравнение)
                # Копейск должен искать только Копейск

                # Проверяем точное совпадение архива
                archive_match = False
                archive_similarity = 0.0

                # Ищем ключевые слова из архива PDF в справочнике
                pdf_archive_parts = pdf_archive.split()
                if (
                    len(pdf_archive_parts) >= 3
                ):  # Должно быть минимум три слова для надежности
                    key_location = pdf_archive_parts[
                        -1
                    ]  # Последнее слово обычно название города/района
                    if key_location in archive_text:
                        archive_match = True
                        # Расчет грубой оценки схожести по количеству совпадающих слов
                        common_words = sum(
                            1 for word in pdf_archive_parts if word in archive_text
                        )
                        archive_similarity = common_words / len(pdf_archive_parts)

                # Проверяем, содержит ли название организации в справочнике ключевые слова из PDF
                org_match = False
                org_similarity = 0.0

                # Ищем ключевые части названия организации
                # Обычно это первые слова (например, "Финансовое управление") и название местности
                pdf_org_parts = pdf_org_normalized.split()
                if len(pdf_org_parts) >= 3:  # Должно быть минимум три слова
                    # Ищем совпадение первых слов (обычно тип организации)
                    first_words_match = all(
                        word in org_text for word in pdf_org_parts[:2]
                    )

                    # Ищем название местности в тексте
                    key_locations = []
                    # Выделяем последние слова, которые могут быть названием местности
                    for i in range(min(5, len(pdf_org_parts))):
                        if len(pdf_org_parts) > i:
                            key_locations.append(pdf_org_parts[-1 - i])

                    location_match = any(loc in org_text for loc in key_locations)

                    if first_words_match and location_match:
                        org_match = True
                        # Расчет грубой оценки схожести
                        common_words = sum(
                            1 for word in pdf_org_parts if word in org_text
                        )
                        org_similarity = common_words / len(pdf_org_parts)

                # Если оба совпадения найдены
                if archive_match and org_match:
                    exact_match_found = True
                    result = {
                        "archive_match": {
                            "text": record["Архив"],
                            "similarity": float(archive_similarity),
                        },
                        "organization_match": {
                            "text": record["Наименование"],
                            "similarity": float(org_similarity),
                        },
                        "is_exact": True,
                    }
                    results.append(result)
                    logger.info(
                        f"Найдено точное совпадение: {record['Архив']} - {record['Наименование']}"
                    )

            # Если не найдено точных совпадений, используем семантический поиск
            if not exact_match_found:
                logger.info(
                    "Точные совпадения не найдены, использую семантический поиск"
                )

                # Получаем семантические совпадения
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
                                "is_exact": False,
                            }
                            results.append(result)

                # Если совпадений по индексу не найдено, используем первое лучшее совпадение по организации
                if not match_found and len(org_indices[0]) > 0:
                    best_org_idx = org_indices[0][0]
                    best_org_similarity = org_distances[0][0]

                    if best_org_similarity > CONFIG["threshold"]:
                        # Получаем архив для этой организации из справочника
                        archive_name = self.reference_index.reference_data[
                            best_org_idx
                        ]["Архив"]

                        # Вычисляем схожесть с найденным архивом
                        pdf_archive_embedding = self.model.encode(pdf_archive)
                        archive_embedding = self.model.encode(archive_name)

                        # Нормализуем эмбеддинги
                        pdf_archive_embedding = pdf_archive_embedding / np.linalg.norm(
                            pdf_archive_embedding
                        )
                        archive_embedding = archive_embedding / np.linalg.norm(
                            archive_embedding
                        )

                        # Вычисляем косинусное сходство
                        archive_similarity = np.dot(
                            pdf_archive_embedding, archive_embedding
                        )

                        if archive_similarity > CONFIG["threshold"]:
                            result = {
                                "archive_match": {
                                    "text": archive_name,
                                    "similarity": float(archive_similarity),
                                },
                                "organization_match": {
                                    "text": self.reference_index.reference_data[
                                        best_org_idx
                                    ]["Наименование"],
                                    "similarity": float(best_org_similarity),
                                },
                                "is_exact": False,
                            }
                            results.append(result)

            # Сортируем результаты по убыванию точности (сначала точные совпадения)
            results.sort(
                key=lambda x: (
                    not x.get("is_exact", False),
                    -(
                        x["organization_match"]["similarity"]
                        + x["archive_match"]["similarity"]
                    )
                    / 2,
                )
            )

            # Возвращаем только лучшие результаты (не более 3)
            return results[:3]

        except Exception as e:
            logger.error(f"Ошибка при поиске совпадений: {e}", exc_info=True)
            return []
