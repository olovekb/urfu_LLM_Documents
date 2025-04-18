"""Основные компоненты для проверки документов."""

from .document_checker import DocumentChecker
from .pdf_parser import PDFParser
from .reference_index import ReferenceIndex

__all__ = ["DocumentChecker", "PDFParser", "ReferenceIndex"]
