import os
import sys
import io
import streamlit as st
from DocumentChecker import main  

# Функция для сохранения загруженного файла
def save_uploaded_file(uploaded_file):
    upload_dir = "tempDir"
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir) 

    file_path = os.path.join(upload_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return file_path

# Функция для перенаправления вывода в Streamlit
def redirect_output():
    new_stdout = io.StringIO()
    sys.stdout = new_stdout
    return new_stdout

# Основная функция Streamlit
def main_streamlit():
    st.title("PDF Обработка с использованием Tesseract и OpenCV")

    # Загрузка PDF файла
    uploaded_file = st.file_uploader("Загрузите PDF файл", type=["pdf"])

    if uploaded_file is not None:
        file_path = save_uploaded_file(uploaded_file)
        st.write(f"Файл загружен: {uploaded_file.name}")

        # Перехватываем вывод консоли
        new_stdout = redirect_output()

        # Запуск анализа PDF, только после загрузки файла
        st.write("Начинаем обработку файла...")
        main(file_path)  # Здесь будет выполнен код из DocumentChecker.py

        # Отображаем результат работы кода
        output = new_stdout.getvalue()
        st.text_area("Результаты работы кода:", output, height=300)

if __name__ == "__main__":
    main_streamlit()
