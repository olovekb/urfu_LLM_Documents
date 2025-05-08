import re
import fitz  # PyMuPDF для работы с PDF

class PDFParser:
    """Класс для извлечения данных из PDF файлов."""

    def __init__(self):
        # Паттерны для поиска нужных блоков
        self.archive_patterns = [
            r"(?i)(УТВЕРЖДЕНО|СОГЛАСОВАНО ЭК).*?(\w+.*)"
        ]
        self.debug_mode = False

    def extract_data(self, pdf_path: str) -> str:
        """Извлекает данные из PDF файла, ищет блоки с ключевыми словами.
        
        Args:
            pdf_path: Путь к PDF файлу

        Returns:
            Название организации из двух блоков
        """
        pdf_data = ""
        doc = fitz.open(pdf_path)
        
        # Проходим по страницам PDF
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            text = page.get_text()

            for pattern in self.archive_patterns:
                matches = re.findall(pattern, text)
                for match in matches:
                    pdf_data += match[1] + "\n"  # Добавляем найденные названия организаций

        return pdf_data
