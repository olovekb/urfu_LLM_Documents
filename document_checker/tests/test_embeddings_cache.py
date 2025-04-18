"""Тест для проверки функциональности кэширования эмбеддингов."""

import os
import shutil
import tempfile
import unittest
import numpy as np
import pandas as pd
from document_checker import DocumentChecker


class TestEmbeddingsCache(unittest.TestCase):
    """Тестирование функциональности кэширования эмбеддингов."""

    def setUp(self):
        """Подготовка тестового окружения."""
        # Создаем временную директорию для кэша
        self.temp_dir = tempfile.mkdtemp()
        self.cache_dir = os.path.join(self.temp_dir, "cache")

        # Создаем тестовые данные
        self.test_csv = os.path.join(self.temp_dir, "test_reference.csv")
        df = pd.DataFrame(
            {
                "Архив": ["Архивный отдел Тест 1", "Архивный отдел Тест 2"],
                "Наименование": ["Организация 1", "Организация 2"],
            }
        )
        df.to_csv(self.test_csv, index=False)

        # Инициализируем DocumentChecker
        self.checker = DocumentChecker(cache_dir=self.cache_dir)

    def tearDown(self):
        """Очистка после тестов."""
        # Удаляем временную директорию
        shutil.rmtree(self.temp_dir)

    def test_embeddings_creation_and_caching(self):
        """Тест создания и кэширования эмбеддингов."""
        # Загружаем справочные данные (первый запуск, без кэша)
        self.checker.load_reference_data(self.test_csv)

        # Проверяем, что кэш был создан
        csv_name = os.path.splitext(os.path.basename(self.test_csv))[0]
        archive_cache = os.path.join(
            self.cache_dir, f"{csv_name}_archive_embeddings.npy"
        )
        org_cache = os.path.join(
            self.cache_dir, f"{csv_name}_organization_embeddings.npy"
        )

        self.assertTrue(
            os.path.exists(archive_cache), "Кэш эмбеддингов архивов не создан"
        )
        self.assertTrue(
            os.path.exists(org_cache), "Кэш эмбеддингов организаций не создан"
        )

        # Сохраняем размер файлов кэша для сравнения
        archive_size = os.path.getsize(archive_cache)
        org_size = os.path.getsize(org_cache)

        # Перезагружаем данные
        new_checker = DocumentChecker(cache_dir=self.cache_dir)
        new_checker.load_reference_data(self.test_csv)

        # Проверяем, что файлы кэша не изменились (использовались существующие эмбеддинги)
        self.assertEqual(
            archive_size,
            os.path.getsize(archive_cache),
            "Размер кэша архивов изменился, хотя должен был использоваться существующий",
        )
        self.assertEqual(
            org_size,
            os.path.getsize(org_cache),
            "Размер кэша организаций изменился, хотя должен был использоваться существующий",
        )

    def test_save_embeddings(self):
        """Тест функции явного сохранения эмбеддингов."""
        # Загружаем справочные данные
        self.checker.load_reference_data(self.test_csv)

        # Создаем отдельную директорию для сохранения
        save_dir = os.path.join(self.temp_dir, "saved_embeddings")
        self.checker.save_embeddings(directory=save_dir, prefix="test_")

        # Проверяем наличие файлов с сохраненными эмбеддингами
        archive_path = os.path.join(save_dir, "test_archive_embeddings.npy")
        org_path = os.path.join(save_dir, "test_organization_embeddings.npy")

        self.assertTrue(os.path.exists(archive_path), "Эмбеддинги архивов не сохранены")
        self.assertTrue(os.path.exists(org_path), "Эмбеддинги организаций не сохранены")

        # Проверяем, что файлы содержат корректные данные
        archive_emb = np.load(archive_path)
        org_emb = np.load(org_path)

        self.assertEqual(
            archive_emb.shape[0], 2, "Неверное количество эмбеддингов архивов"
        )
        self.assertEqual(
            org_emb.shape[0], 2, "Неверное количество эмбеддингов организаций"
        )


if __name__ == "__main__":
    unittest.main()
