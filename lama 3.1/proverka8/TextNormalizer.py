import re

class TextNormalizer:
    @staticmethod
    def normalize_organization(org_name: str) -> str:
        """Нормализует название организации для корректного сравнения."""
        # Пример нормализации: убираем лишние пробелы и цифры
        normalized_name = re.sub(r'\d+', '', org_name)  # Убираем все цифры
        normalized_name = re.sub(r'\s+', ' ', normalized_name)  # Убираем лишние пробелы
        return normalized_name.strip()  # Убираем пробелы с краев
