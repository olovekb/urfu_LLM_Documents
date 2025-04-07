import requests
import fitz  # PyMuPDF
import pandas as pd

# Функция для извлечения текста из PDF (всего заголовка)
def extract_title_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    first_page = doc.load_page(0)  # Берем первую страницу
    text = first_page.get_text("text")  # Извлекаем текст
    # Убираем пустые строки
    lines = [line for line in text.split('\n') if line.strip()]
    # Берем первые 6 строк (после удаления пустых строк)
    title = "\n".join(lines[:4])
    return title

# Функция для отправки запроса к модели LLaMA
def check_title_with_llama(title, csv_line):
    url = "http://localhost:11434/api/generate"  # Адрес локального сервера Ollama
    prompt = f"""
    Проверь схожесть между следующими двумя текстами:

    Первый текст: "{title}"
    Второй текст: "{csv_line}"

    Учитывая возможные изменения в окончаниях слов, перестановку слов и другие вариации, скажи, схожи ли эти тексты. Весь этот текст - это наименование архива. 
    Ты проверяешь соответствие  наименования из документа списку существующих архивов из csv.
    Ответь "Проверка пройдена" или "Проверка не пройдена". 
    Если проверка не пройдена, укажи **конкретные различия** между текстами.
    """
    
    payload = {
        "model": "llama3.1:8b",  # Укажи точную модель, если она отличается
        "prompt": prompt,
        "stream": False
    }

    try:
        response = requests.post(url, json=payload)

        # Печатаем статус ответа от сервера
        print(f"Status Code: {response.status_code}")

        if response.status_code != 200:
            return f"Ошибка от сервера: {response.status_code}, {response.text}"

        # Пытаемся распарсить JSON
        try:
            result = response.json().get("response", "")
        except requests.exceptions.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            print(f"Response content: {response.text}")
            return "Ошибка при получении ответа от модели"

        if not result:
            return "Ошибка: Пустой ответ от модели"
        
        return result
    except requests.exceptions.RequestException as e:
        print(f"Error with request: {e}")
        return "Ошибка при отправке запроса"

# Функция для проверки строки в CSV с использованием LLaMA
def check_title_in_csv_with_llama(title, csv_path):
    df = pd.read_csv(csv_path)

    for index, row in df.iterrows():
        # Объединяем все ячейки строки в одну строку
        csv_line = " ".join([str(cell).strip() for cell in row if pd.notna(cell)])

        print("\n" + "-" * 60)
        print(f"[{index}] Проверка строки из CSV:\n{csv_line}\n")

        result = check_title_with_llama(title, csv_line)

        print(f"Status Code: 200")
        print(f"Ответ модели:\n{result}")
        print("-" * 60)

        if result.strip() == "Проверка пройдена":
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
