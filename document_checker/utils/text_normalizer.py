"""Модуль для нормализации текста."""

import re
from typing import Optional


class TextNormalizer:
    """Класс для нормализации текста."""

    @staticmethod
    def normalize(text: str) -> str:
        """Нормализует текст для сравнения.

        Args:
            text: Исходный текст

        Returns:
            Нормализованный текст
        """
        if not isinstance(text, str):
            return ""

        # Приводим к нижнему регистру
        text = text.lower()

        # Удаляем специальные символы, но сохраняем точки и запятые
        text = re.sub(r"[^\w\s.,]", " ", text)

        # Заменяем множественные пробелы на один
        text = re.sub(r"\s+", " ", text)

        # Удаляем пробелы в начале и конце
        return text.strip()

    @staticmethod
    def remove_digits(text: str) -> str:
        """Удаляет цифры из текста.

        Args:
            text: Исходный текст

        Returns:
            Текст без цифр
        """
        if not isinstance(text, str):
            return ""

        # Удаляем цифры
        text = re.sub(r"\d+", "", text)

        # Заменяем множественные пробелы на один
        text = re.sub(r"\s+", " ", text)

        return text.strip()

    @staticmethod
    def normalize_organization(text: str) -> str:
        """Нормализует название организации (удаляет цифры и специальные символы кроме точек и запятых).

        Args:
            text: Исходный текст

        Returns:
            Нормализованное название организации
        """
        if not isinstance(text, str):
            return ""

        # Приводим к нижнему регистру
        text = text.lower()

        # Удаляем цифры
        text = re.sub(r"\d+", "", text)

        # Удаляем специальные символы, но сохраняем точки и запятые
        text = re.sub(r"[^\w\s.,]", " ", text)

        # Заменяем множественные пробелы на один
        text = re.sub(r"\s+", " ", text)

        return text.strip()
