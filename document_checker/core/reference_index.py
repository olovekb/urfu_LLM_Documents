"""Модуль для работы с индексами справочных данных."""

import logging
import os
from typing import List, Tuple, Dict
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from ..config.config import CONFIG
from ..utils.text_normalizer import TextNormalizer

logger = logging.getLogger(__name__)


class ReferenceIndex:
    """Класс для работы с индексами справочных данных."""

    def __init__(self, model: SentenceTransformer):
        self.model = model
        self.reference_data = []
        self.archive_index = None
        self.organization_index = None

    def build_index(
        self, csv_path: str, cache_dir: str = None, force_rebuild: bool = False
    ):
        """Строит индексы для справочных данных.

        Args:
            csv_path: Путь к CSV файлу
            cache_dir: Директория для кэширования эмбеддингов
            force_rebuild: Принудительно пересоздать кэш эмбеддингов
        """
        try:
            logger.info(f"Загрузка справочных данных из {csv_path}")
            df = pd.read_csv(csv_path)

            # Проверяем наличие необходимых колонок
            required_columns = {"Архив", "Наименование"}
            if not required_columns.issubset(df.columns):
                raise ValueError(
                    f"CSV файл должен содержать колонки: {required_columns}"
                )

            # Сохраняем исходные данные
            self.reference_data = df.to_dict("records")

            # Определяем имена файлов кэша
            cache_files = None
            if cache_dir:
                os.makedirs(cache_dir, exist_ok=True)
                csv_name = os.path.splitext(os.path.basename(csv_path))[0]
                archive_cache = os.path.join(
                    cache_dir, f"{csv_name}_archive_embeddings.npy"
                )
                org_cache = os.path.join(
                    cache_dir, f"{csv_name}_organization_embeddings.npy"
                )
                cache_files = {"archive": archive_cache, "organization": org_cache}

            # Генерируем эмбеддинги для архивов
            archives = df["Архив"].tolist()
            logger.info("Генерация эмбеддингов для архивов...")
            archive_embeddings = self._get_or_create_embeddings(
                archives, "archive", cache_files, force_rebuild
            )
            self.archive_index = self._build_faiss_index(archive_embeddings)

            # Генерируем эмбеддинги для организаций
            organizations = df["Наименование"].tolist()
            # Нормализуем названия организаций (удаляем цифры, сохраняем точки и запятые)
            normalized_organizations = [
                TextNormalizer.normalize_organization(org) for org in organizations
            ]
            logger.info("Генерация эмбеддингов для организаций...")
            organization_embeddings = self._get_or_create_embeddings(
                normalized_organizations, "organization", cache_files, force_rebuild
            )
            self.organization_index = self._build_faiss_index(organization_embeddings)

            logger.info(f"Индексация завершена. Обработано {len(df)} записей")

        except Exception as e:
            logger.error(f"Ошибка построения индекса: {e}")
            raise

    def _get_or_create_embeddings(
        self,
        texts: List[str],
        embed_type: str,
        cache_files: Dict = None,
        force_rebuild: bool = False,
    ) -> np.ndarray:
        """Получает эмбеддинги из кэша или создает новые.

        Args:
            texts: Список текстов
            embed_type: Тип эмбеддингов ('archive' или 'organization')
            cache_files: Словарь с путями к файлам кэша
            force_rebuild: Принудительно пересоздать кэш эмбеддингов

        Returns:
            Массив эмбеддингов
        """
        # Проверяем, есть ли кэш и можно ли его использовать
        if cache_files and embed_type in cache_files:
            cache_path = cache_files[embed_type]
            if os.path.exists(cache_path) and not force_rebuild:
                try:
                    logger.info(f"Загрузка эмбеддингов из кэша: {cache_path}")
                    embeddings = np.load(cache_path)
                    # Проверка размерности кэша
                    if len(embeddings) == len(texts):
                        logger.info(f"Эмбеддинги успешно загружены из кэша")
                        return embeddings
                    else:
                        logger.warning(
                            f"Размерность кэшированных эмбеддингов не совпадает: {len(embeddings)} != {len(texts)}"
                        )
                except Exception as e:
                    logger.warning(f"Ошибка при загрузке кэша эмбеддингов: {e}")

        # Создаем новые эмбеддинги
        embeddings = self._create_embeddings(texts)

        # Сохраняем в кэш, если указана директория
        if cache_files and embed_type in cache_files:
            try:
                cache_path = cache_files[embed_type]
                logger.info(f"Сохранение эмбеддингов в кэш: {cache_path}")
                np.save(cache_path, embeddings)
                logger.info(f"Эмбеддинги успешно сохранены в кэш")
            except Exception as e:
                logger.warning(f"Ошибка при сохранении кэша эмбеддингов: {e}")

        return embeddings

    def _create_embeddings(self, texts: List[str]) -> np.ndarray:
        """Создает эмбеддинги для списка текстов.

        Args:
            texts: Список текстов

        Returns:
            Массив эмбеддингов
        """
        embeddings = self.model.encode(
            texts,
            batch_size=CONFIG["batch_size"],
            show_progress_bar=True,
            convert_to_tensor=False,
        )
        faiss.normalize_L2(embeddings)
        return embeddings

    def _build_faiss_index(self, embeddings: np.ndarray) -> faiss.IndexFlatIP:
        """Строит FAISS индекс для эмбеддингов.

        Args:
            embeddings: Массив эмбеддингов

        Returns:
            FAISS индекс
        """
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        return index

    def search(
        self, query: str, index_type: str, top_k: int = 5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Ищет ближайшие совпадения в индексе.

        Args:
            query: Текст запроса
            index_type: Тип индекса ('archive' или 'organization')
            top_k: Количество результатов

        Returns:
            Кортеж (расстояния, индексы)
        """
        index = (
            self.archive_index if index_type == "archive" else self.organization_index
        )
        query_embedding = self.model.encode(query)
        query_embedding = np.expand_dims(query_embedding, axis=0)
        faiss.normalize_L2(query_embedding)
        return index.search(query_embedding, top_k)

    def save_embeddings(self, directory: str, prefix: str = ""):
        """Сохраняет эмбеддинги в указанную директорию.

        Args:
            directory: Директория для сохранения
            prefix: Префикс для имен файлов
        """
        os.makedirs(directory, exist_ok=True)

        if self.archive_index:
            archive_path = os.path.join(directory, f"{prefix}archive_embeddings.npy")
            # Для извлечения эмбеддингов из индекса FAISS используем метод с нулевым запросом
            # Этот подход работает для IndexFlatIP, который хранит все эмбеддинги
            n_vectors = self.archive_index.ntotal
            d = self.archive_index.d
            embeddings = np.zeros((n_vectors, d), dtype=np.float32)

            # Извлекаем данные из индекса
            for i in range(n_vectors):
                # Метод reconstruct извлекает вектор по его индексу
                vector = np.zeros(d, dtype=np.float32)
                self.archive_index.reconstruct(i, vector)
                embeddings[i] = vector

            np.save(archive_path, embeddings)
            logger.info(f"Эмбеддинги архивов сохранены в {archive_path}")

        if self.organization_index:
            org_path = os.path.join(directory, f"{prefix}organization_embeddings.npy")
            # Для извлечения эмбеддингов из индекса FAISS используем метод с нулевым запросом
            n_vectors = self.organization_index.ntotal
            d = self.organization_index.d
            embeddings = np.zeros((n_vectors, d), dtype=np.float32)

            # Извлекаем данные из индекса
            for i in range(n_vectors):
                vector = np.zeros(d, dtype=np.float32)
                self.organization_index.reconstruct(i, vector)
                embeddings[i] = vector

            np.save(org_path, embeddings)
            logger.info(f"Эмбеддинги организаций сохранены в {org_path}")
