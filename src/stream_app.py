import os
import contextlib
import io

import json
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st

from src.services.document_checker import DocumentChecker
from src.main import process_document  # повторное использование существующей логики

# Дополнительные зависимости UI
import pandas as pd  # type: ignore

# --------------------------------------------------------------------------------------
# Кэширование ресурсов и данных
# --------------------------------------------------------------------------------------


@st.cache_resource(show_spinner="⏳ Загружаю модель и строю индекс...")
def get_document_checker(
    csv_path: str, cache_dir: str, force_rebuild: bool
) -> DocumentChecker:  # noqa: D401
    """Получить и закэшировать экземпляр ``DocumentChecker``.

    Используется ``st.cache_resource`` для тяжёлых объектов: модель
    SentenceTransformer и индекс справочника.

    Parameters
    ----------
    csv_path : str
        Путь к CSV-файлу со справочником.
    cache_dir : str
        Директория для кэширования эмбеддингов.
    force_rebuild : bool
        Принудительная пересборка кэша эмбеддингов.

    Returns
    -------
    DocumentChecker
        Инициализированный и готовый к работе проверяльщик документов.
    """
    checker = DocumentChecker(cache_dir=cache_dir)
    checker.load_reference_data(csv_path, force_rebuild=force_rebuild)
    return checker


@st.cache_data(show_spinner=False)
def save_uploaded_file(uploaded_file) -> str:
    """Сохраняет загруженный через Streamlit файл во временную директорию.

    Returns
    -------
    str
        Путь к сохранённому временному PDF.
    """
    suffix = Path(uploaded_file.name).suffix or ".pdf"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        return tmp_file.name


# --------------------------------------------------------------------------------------
# Настройка приложения (выполняется один раз)
# --------------------------------------------------------------------------------------

st.set_page_config(page_title="Инспектор документов", page_icon="📑", layout="wide")

# --------------------------------------------------------------------------------------
# Пользовательский интерфейс
# --------------------------------------------------------------------------------------


def main() -> None:  # noqa: D401
    """Страница проверки N2 (Document Inspector)."""

    st.title("📄 Document Inspector (Проверка N2)")
    st.write(
        "Проверка PDF-документов по справочнику с использованием SentenceTransformer."
    )

    # ----------------------------
    # Боковая панель настроек (группировка через expander)
    # ----------------------------
    st.sidebar.header("🛠️ Управление")

    with st.sidebar.expander("📁 Справочник и кэш", expanded=True):
        # Путь к CSV со справочником
        default_csv_path = os.path.join(
            Path(__file__).resolve().parent.parent, "data", "merged.csv"
        )
        csv_path = st.text_input(
            "CSV-файл справочника", value=default_csv_path, key="csv_path"
        )

        # Директория для кэша
        default_cache_dir = os.path.join(
            Path(__file__).resolve().parent.parent, "embeddings_cache"
        )
        cache_dir = st.text_input(
            "Директория кэша эмбеддингов", value=default_cache_dir, key="cache_dir"
        )

        # Флаг пересборки
        force_rebuild = st.checkbox(
            "🔄 Принудительно пересобрать кэш эмбеддингов", key="force_rebuild"
        )

    with st.sidebar.expander("📄 Источник PDF", expanded=True):
        pdf_source = st.radio(
            "Выберите источник", ("Загрузить файлы", "Директория на диске"), key="pdf_source"
        )

        uploaded_files: List[Any] = []  # «Any» из-за абстрактного объекта UploadedFile
        pdf_dir: str = ""
        if pdf_source == "Загрузить файлы":
            uploaded_files = st.file_uploader(
                "Выберите PDF", type=["pdf"], accept_multiple_files=True, key="uploader"
            )
        else:
            pdf_dir = st.text_input(
                "Путь к директории с PDF",
                value=os.path.join(Path(__file__).resolve().parent.parent, "pdf_documents"),
                key="pdf_dir",
            )

    run_btn = st.sidebar.button("🚀 Запустить обработку", use_container_width=True)

    # ----------------------------
    # Запуск обработки
    # ----------------------------
    if run_btn:
        # Проверяем CSV
        if not os.path.isfile(csv_path):
            st.sidebar.error("📄 CSV-файл не найден. Проверьте путь.")
            st.stop()

        # Подготавливаем список путей к PDF
        pdf_paths: List[str] = []
        temp_files: List[str] = []  # для последующей очистки

        if pdf_source == "Загрузить файлы":
            if not uploaded_files:
                st.sidebar.error("Не выбраны файлы для загрузки.")
                st.stop()
            for uf in uploaded_files:
                tmp_path = save_uploaded_file(uf)
                temp_files.append(tmp_path)
                pdf_paths.append(tmp_path)
        else:
            if not os.path.isdir(pdf_dir):
                st.sidebar.error("Директория с PDF не найдена.")
                st.stop()
            pdf_paths = [
                os.path.join(pdf_dir, f)
                for f in os.listdir(pdf_dir)
                if f.lower().endswith(".pdf")
            ]
            if not pdf_paths:
                st.sidebar.error("В заданной директории нет PDF-файлов.")
                st.stop()

        # Получаем DocumentChecker из кэша/создаём
        checker = get_document_checker(csv_path, cache_dir, force_rebuild)

        # Обрабатываем документы
        results: List[Dict[str, Any]] = []
        progress_bar = st.progress(0, text="Обработка документов…")

        for idx, pdf_path in enumerate(pdf_paths, start=1):
            progress_bar.progress(
                float(idx) / len(pdf_paths), text=f"Обработка {idx}/{len(pdf_paths)}…"
            )
            result = process_document(checker, pdf_path, idx, len(pdf_paths))
            results.append(result)

        progress_bar.empty()
        st.success("✅ Обработка завершена!")

        # ----------------------------
        # Отображение результатов
        # ----------------------------
        if results:
            st.subheader("🔍 Результаты")

            # ===== 1. Метрики =====
            total_docs = len(results)
            exact_docs = sum(1 for r in results if r.get("exact_match"))
            no_match_docs = sum(
                1 for r in results if not r.get("exact_match") and not r.get("matches")
            )

            col_total, col_exact, col_empty = st.columns(3)
            col_total.metric("Всего документов", total_docs)
            col_exact.metric("Точные совпадения", exact_docs)
            col_empty.metric("Без совпадений", no_match_docs)

            # ===== 2. Обзорная таблица =====
            overview_data = []
            for r in results:
                if r.get("exact_match"):
                    similarity = r["exact_match"].get("similarity")
                elif r.get("matches"):
                    similarity = r["matches"][0].get("similarity_pair", "-")
                else:
                    similarity = "-"
                overview_data.append(
                    {
                        "Файл": r["filename"],
                        "Точное": bool(r.get("exact_match")),
                        "Similarity": similarity,
                        "Кандидаты": len(r.get("matches", [])),
                    }
                )

            overview_df = pd.DataFrame(overview_data)
            st.dataframe(overview_df, use_container_width=True, hide_index=True)

            # Детальный JSON отключён по требованию
            show_json: bool = False

            # ===== 3. Детализация по документам =====
            for res in results:
                with st.expander(f"📄 {res['filename']}"):
                    left_col, right_col = st.columns([1, 2])

                    # 3.1 Левая колонка – извлечение и точное совпадение
                    with left_col:
                        extracted = res.get("extracted_data", {})
                        if extracted:
                            st.write("**Извлечённые данные из PDF:**")
                            st.json(extracted, expanded=False)

                        exact = res.get("exact_match")
                        if exact:
                            st.success(
                                f"Точное совпадение {exact['similarity']}:\n\n**Архив:** {exact['archive']}\n\n**Организация:** {exact['organization']}"
                            )
                            # Кнопка копирования через st.code (встроенный copy-icon)
                            st.code(
                                f"{exact['archive']} | {exact['organization']}",
                                language="",
                            )
                        else:
                            st.warning("Точное совпадение не найдено.")

                    # 3.2 Правая колонка – кандидаты
                    with right_col:
                        matches = res.get("matches", [])
                        if matches:
                            st.write("**Кандидаты для ручной проверки:**")
                            st.table(matches)
                        else:
                            st.info("Кандидатов не найдено.")

                    # 3.3 (Опционально) сырой JSON
                    if show_json:
                        st.divider()
                        st.json(res, expanded=False)
        else:
            st.info("Совпадений не найдено.")

        # Очищаем временные файлы
        for f in temp_files:
            try:
                os.remove(f)
            except FileNotFoundError:
                pass


# --------------------------------------------------------------------------------------
# Страница проверки N8 (OCR + сравнение подписей)
# --------------------------------------------------------------------------------------


def page_check_n8() -> None:  # noqa: D401
    """UI-обёртка над логикой из ``proverka8.DocumentChecker`` (проверка N8)."""

    import proverka8.DocumentChecker as dc  # локальный импорт, чтобы ускорить первую загрузку

    st.title("🖼️ OCR & сравнение подписей (Проверка N8)")
    st.write(
        "Поиск блоков *Утверждено / Согласовано* в PDF и сверка наименований методом FAISS."
    )

    # ----------------------------
    # Боковая панель настроек
    # ----------------------------

    st.sidebar.header("🛠️ Управление (N8)")

    pdf_source = st.sidebar.radio(
        "Источник PDF",
        ("Загрузить файлы", "Директория на диске"),
        key="n8_pdf_source",
    )

    uploaded_files = []
    pdf_dir: str = ""
    if pdf_source == "Загрузить файлы":
        uploaded_files = st.sidebar.file_uploader(
            "Выберите PDF",
            type=["pdf"],
            accept_multiple_files=True,
            key="n8_uploader",
        )
    else:
        default_dir = os.path.join(Path(__file__).resolve().parent.parent, "pdf_documents")
        pdf_dir = st.sidebar.text_input(
            "Путь к директории с PDF",
            value=default_dir,
            key="n8_pdf_dir",
        )

    run_btn = st.sidebar.button("🚀 Запустить обработку", key="n8_run_btn", use_container_width=True)

    # ----------------------------
    # Запуск обработки
    # ----------------------------

    if run_btn:
        pdf_paths: list[str] = []
        temp_files: list[str] = []

        if pdf_source == "Загрузить файлы":
            if not uploaded_files:
                st.sidebar.error("Не выбраны файлы для загрузки.")
                st.stop()
            for uf in uploaded_files:
                tmp_path = save_uploaded_file(uf)
                temp_files.append(tmp_path)
                pdf_paths.append(tmp_path)
        else:
            if not os.path.isdir(pdf_dir):
                st.sidebar.error("Директория не найдена.")
                st.stop()
            pdf_paths = [
                os.path.join(pdf_dir, f)
                for f in os.listdir(pdf_dir)
                if f.lower().endswith(".pdf")
            ]
            if not pdf_paths:
                st.sidebar.error("В директории нет PDF-файлов.")
                st.stop()

        progress = st.progress(0, text="Обработка документов…")

        for idx, pdf_path in enumerate(pdf_paths, start=1):
            progress.progress(idx / len(pdf_paths), text=f"Обработка {idx}/{len(pdf_paths)}…")

            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                try:
                    dc.main(pdf_path)
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Ошибка при обработке {Path(pdf_path).name}: {exc}")
                    continue

            with st.expander(f"📄 {Path(pdf_path).name}"):
                st.text_area("Консольный вывод", buffer.getvalue(), height=400)

        progress.empty()

        # Очистка временных файлов
        for f in temp_files:
            try:
                os.remove(f)
            except FileNotFoundError:
                pass


# --------------------------------------------------------------------------------------
# Навигация между страницами
# --------------------------------------------------------------------------------------


pages = [
    st.Page(main, title="Проверка N2", icon=":material/fact_check:", default=True),
    st.Page(page_check_n8, title="Проверка N8", icon=":material/document_scanner:"),
]

_current_page = st.navigation(pages)
_current_page.run()
