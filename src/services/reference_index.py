"""Модуль для работы с индексами справочных данных."""

import logging
import os
from typing import List, Tuple, Dict
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from config.config import CONFIG
from src.utilities.text_normalizer import TextNormalizer

logger = logging.getLogger(__name__)


class ReferenceIndex:
    """Класс для работы с индексами справочных данных."""

    def __init__(self, model: SentenceTransformer):
        self.model = model
        self.reference_data = []
        self.archive_index = None
        self.organization_index = None
        self.pair_index = None  # Новый индекс для пары архив+организация

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
                pair_cache = os.path.join(
                    cache_dir, f"{csv_name}_pair_embeddings.npy"
                )
                cache_files = {
                    "archive": archive_cache,
                    "organization": org_cache,
                    "pair": pair_cache,
                }

            # Предварительно нормализуем тексты для повышения производительности
            archives = [
                TextNormalizer.normalize(archive) for archive in df["Архив"].tolist()
            ]
            normalized_organizations = [
                TextNormalizer.normalize_organization(org)
                for org in df["Наименование"].tolist()
            ]

            # Генерируем эмбеддинги для архивов
            logger.info("Генерация эмбеддингов для архивов...")
            archive_embeddings = self._get_or_create_embeddings(
                archives, "archive", cache_files, force_rebuild
            )
            self.archive_index = self._build_faiss_index(archive_embeddings)

            # Генерируем эмбеддинги для организаций
            logger.info("Генерация эмбеддингов для организаций...")
            organization_embeddings = self._get_or_create_embeddings(
                normalized_organizations, "organization", cache_files, force_rebuild
            )
            self.organization_index = self._build_faiss_index(organization_embeddings)

            # -------------------------------------------------------------
            # Генерируем эмбеддинги для пары «архив; организация»
            # -------------------------------------------------------------
            logger.info("Генерация эмбеддингов для пар архив+организация...")
            pair_texts = [
                f"{a}; {o}" for a, o in zip(archives, normalized_organizations)
            ]
            pair_embeddings = self._get_or_create_embeddings(
                pair_texts, "pair", cache_files, force_rebuild
            )
            self.pair_index = self._build_faiss_index(pair_embeddings)

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
                    # -----------------------------------------------------------------
                    # Дополнительная валидация кэша: сверяем количество строк, размерность
                    # и dtype. Если что-то не совпадает, игнорируем кэш и создаём заново.
                    # -----------------------------------------------------------------
                    expected_dim = self.model.get_sentence_embedding_dimension()
                    if (
                        len(embeddings) == len(texts)
                        and embeddings.ndim == 2
                        and embeddings.shape[1] == expected_dim
                    ):
                        logger.info(f"Эмбеддинги успешно загружены из кэша")
                        return embeddings.astype(np.float32, copy=False)
                    else:
                        logger.warning(
                            "Кэшованные эмбеддинги не прошли валидацию: "
                            f"rows={len(embeddings)} (ожидалось {len(texts)}), "
                            f"dim={embeddings.shape[1] if embeddings.ndim==2 else 'N/A'} (ожидалось {expected_dim})"
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
        # Если список пустой, сразу возвращаем пустой массив корректной формы
        if not texts:
            logger.warning("Получен пустой список текстов — возвращаю пустой массив эмбеддингов")
            dim = self.model.get_sentence_embedding_dimension()
            return np.empty((0, dim), dtype=np.float32)

        # Используем batch_size для оптимальной производительности
        batch_size = min(CONFIG["batch_size"], len(texts))

        # Используем multiprocessing, если работаем на CPU и есть несколько ядер
        if self.model.device.type == "cpu" and (os.cpu_count() or 1) > 1:
            logger.info(
                "GPU не найден. Использую multiprocessing для ускорения кодирования"
            )
            pool = self.model.start_multi_process_pool()
            try:
                embeddings = self.model.encode_multi_process(
                    texts,
                    pool,
                    batch_size=batch_size,
                    normalize_embeddings=False,
                )
            finally:
                self.model.stop_multi_process_pool(pool)
        else:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=True,
                convert_to_tensor=False,
            )

        # Приводим к numpy.float32 для совместимости с FAISS
        embeddings = np.asarray(embeddings, dtype=np.float32)
        faiss.normalize_L2(embeddings)
        return embeddings

    def _build_faiss_index(self, embeddings: np.ndarray) -> faiss.IndexFlatIP:
        """Строит FAISS индекс для эмбеддингов.

        Args:
            embeddings: Массив эмбеддингов

        Returns:
            FAISS индекс
        """
        # Используем IndexFlatIP для точного поиска по внутреннему произведению (косинусное сходство)
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
        # Нормализуем текст запроса в зависимости от типа индекса
        if index_type == "archive":
            query = TextNormalizer.normalize(query)
        elif index_type == "organization":
            query = TextNormalizer.normalize_organization(query)
        elif index_type == "pair":
            # Для пары предполагается, что строка уже объединена и нормализована ранее
            pass

        # Выбираем нужный индекс
        if index_type == "archive":
            index = self.archive_index
        elif index_type == "organization":
            index = self.organization_index
        else:
            index = self.pair_index

        # Создаем эмбеддинг для запроса
        query_embedding = self.model.encode(query)
        query_embedding = np.expand_dims(query_embedding, axis=0)
        faiss.normalize_L2(query_embedding)

        # Выполняем поиск
        distances, indices = index.search(query_embedding, top_k)
        return distances, indices

    def save_embeddings(self, directory: str, prefix: str = ""):
        """Сохраняет эмбеддинги в указанную директорию.

        Args:
            directory: Директория для сохранения
            prefix: Префикс для имен файлов
        """
        os.makedirs(directory, exist_ok=True)

        if self.archive_index:
            archive_embeddings = faiss.vector_to_array(
                self.archive_index.get_xb()
            ).reshape(self.archive_index.ntotal, self.archive_index.d)
            archive_path = os.path.join(directory, f"{prefix}archive_embeddings.npy")
            logger.info(f"Сохранение эмбеддингов архивов в {archive_path}")
            np.save(archive_path, archive_embeddings)

        if self.organization_index:
            org_embeddings = faiss.vector_to_array(
                self.organization_index.get_xb()
            ).reshape(self.organization_index.ntotal, self.organization_index.d)
            org_path = os.path.join(directory, f"{prefix}organization_embeddings.npy")
            logger.info(f"Сохранение эмбеддингов организаций в {org_path}")
            np.save(org_path, org_embeddings)

        if self.pair_index:
            pair_embeddings = faiss.vector_to_array(
                self.pair_index.get_xb()
            ).reshape(self.pair_index.ntotal, self.pair_index.d)
            pair_path = os.path.join(directory, f"{prefix}pair_embeddings.npy")
            logger.info(f"Сохранение эмбеддингов пар в {pair_path}")
            np.save(pair_path, pair_embeddings)
