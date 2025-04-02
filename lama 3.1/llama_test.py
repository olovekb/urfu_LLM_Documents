# import fitz  # PyMuPDF
# import pandas as pd
# import requests

# # Функция для извлечения первой строки из PDF
# def extract_first_line_from_pdf(pdf_path):
#     doc = fitz.open(pdf_path)
#     first_page = doc.load_page(0)  # Берем первую страницу
#     text = first_page.get_text("text")  # Извлекаем текст
#     first_line = text.split('\n')[0]  # Первая строка
#     return first_line

# # Функция для отправки запроса к модели LLaMA
# def check_line_with_llama(first_line, csv_line):
#     url = "http://localhost:11434/api/generate"  # Адрес локального сервера Ollama
#     prompt = f"""
#     Проверь схожесть между следующими двумя строками:

#     Первая строка: "{first_line}"
#     Вторая строка: "{csv_line}"

#     Учитывая возможные изменения в окончаниях слов, перестановку слов и другие вариации, скажи, схожи ли эти строки.
#     Ответь только "Проверка пройдена" или "Проверка не пройдена".
#     """
    
#     payload = {
#         "model": "llama3.1",  
#         "prompt": prompt,
#         "stream": False
#     }

#     response = requests.post(url, json=payload)
#     result = response.json().get("response", "")
    
#     return result

# # Функция для проверки строки в CSV с использованием LLaMA
# def check_line_in_csv_with_llama(first_line, csv_path):
#     # Загружаем CSV
#     df = pd.read_csv(csv_path)

#     # Проходим по строкам CSV
#     for index, row in df.iterrows():
#         for cell in row:
#             csv_line = str(cell)
#             # Отправляем запрос к модели для сравнения строк
#             result = check_line_with_llama(first_line, csv_line)
#             if result == "Проверка пройдена":
#                 return "Проверка пройдена"
    
#     return "Проверка не пройдена"

# # Основной блок
# pdf_path = "lama 3.1/Копейск Фин. управление оп 2 лс.pdf"
# csv_path = "lama 3.1/FINAL_CLEANED_SP3.csv"

# # Извлекаем первую строку из PDF
# first_line = extract_first_line_from_pdf(pdf_path)
# print(f"Первая строка из PDF: {first_line}")

# # Проверяем строку в CSV через LLaMA
# result = check_line_in_csv_with_llama(first_line, csv_path)
# print(result)


import fitz  # PyMuPDF
import pandas as pd
import requests

# Функция для извлечения текста из PDF (всего заголовка)
def extract_title_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    first_page = doc.load_page(0)  # Берем первую страницу
    text = first_page.get_text("text")  # Извлекаем текст
    title = "\n".join(text.split('\n')[:6])  # Берем первые 5 строк как заголовок (можно настроить)
    return title

# Функция для отправки запроса к модели LLaMA
def check_title_with_llama(title, csv_line):
    url = "http://localhost:11434/api/generate"  # Адрес локального сервера Ollama
    prompt = f"""
    Проверь схожесть между следующими двумя текстами:

    Первый текст: "{title}"
    Второй текст: "{csv_line}"

    Учитывая возможные изменения в окончаниях слов, перестановку слов и другие вариации, скажи, схожи ли эти тексты.
    Ответь только "Проверка пройдена" или "Проверка не пройдена".
    """
    
    payload = {
        "model": "llama3",  # Укажи точную модель, если она отличается
        "prompt": prompt,
        "stream": False
    }

    response = requests.post(url, json=payload)
    result = response.json().get("response", "")
    
    return result

# Функция для проверки строки в CSV с использованием LLaMA
def check_title_in_csv_with_llama(title, csv_path):
    # Загружаем CSV
    df = pd.read_csv(csv_path)

    # Проходим по строкам CSV
    for index, row in df.iterrows():
        for cell in row:
            csv_line = str(cell)
            # Отправляем запрос к модели для сравнения текстов
            result = check_title_with_llama(title, csv_line)
            if result == "Проверка пройдена":
                return "Проверка пройдена"
    
    return "Проверка не пройдена"

# Основной блок
pdf_path = "lama 3.1/Копейск Фин. управление оп 2 лс.pdf"
csv_path = "lama 3.1/FINAL_CLEANED_SP3.csv"

# Извлекаем весь текст (заголовок) из PDF
title = extract_title_from_pdf(pdf_path)
print(f"Извлеченный текст из PDF: {title}")

# Проверяем заголовок в CSV через LLaMA
result = check_title_in_csv_with_llama(title, csv_path)
print(result)
