import re
import fitz  # PyMuPDF

# Функция для извлечения текста из PDF
def extract_pdf_text(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text("text")
    return text

# Функция для извлечения наименования организации, игнорируя должность и ФИО
def extract_organization_name(name):
    if name:
        # Паттерн для поиска наименования организации в кавычках
        pattern = r"«([^\»]+)»"  # Находим текст в кавычках
        match = re.search(pattern, name)
        if match:
            return re.sub(r'\s+', ' ', match.group(1).strip())  # Убираем лишние пробелы
    return None

# Функция для извлечения наименования и ФИО из блоков СОГЛАСОВАНО и УТВЕРЖДАЮ
def extract_agreed_and_approved_names(text):
    # Паттерны для блоков СОГЛАСОВАНО и УТВЕРЖДАЮ с использованием универсального паттерна для ФИО
    pattern_agreed = r"СОГЛАСОВАНО\s*(.*?)([А-ЯЁ][a-яё]+\s[А-ЯЁ]\.[А-ЯЁ]\.)"  # ФИО: Фамилия И. И.
    pattern_approved = r"УТВЕРЖДАЮ\s*(.*?)([А-ЯЁ][a-яё]+\s[А-ЯЁ]\.[А-ЯЁ]\.)"  # ФИО: Фамилия И. И.
    
    # Поиск с использованием регулярных выражений
    согласовано_name_match = re.search(pattern_agreed, text, re.DOTALL)
    утверждаю_name_match = re.search(pattern_approved, text, re.DOTALL)
    
    # Если найдено совпадение, извлекаем наименование и ФИО
    if согласовано_name_match:
        согласовано_name = согласовано_name_match.group(1).strip() + " " + согласовано_name_match.group(2).strip()
    else:
        согласовано_name = None
    
    if утверждаю_name_match:
        утверждаю_name = утверждаю_name_match.group(1).strip() + " " + утверждаю_name_match.group(2).strip()
    else:
        утверждаю_name = None
    
    return согласовано_name, утверждаю_name

# Функция для сравнения наименований
def compare_names(согласовано_name, утверждаю_name):
    # Проверяем, что имена не пустые
    if согласовано_name and утверждаю_name:
        # Извлекаем названия организации из обоих строк
        согласовано_organization = extract_organization_name(согласовано_name)
        утверждаю_organization = extract_organization_name(утверждаю_name)
        
        # Проверяем, совпадают ли наименования
        if согласовано_organization and утверждаю_organization:
            if согласовано_organization == утверждаю_organization:
                return "Наименования совпадают"
            else:
                # Добавляем вывод различий для лучшего понимания
                return f"Наименования не совпадают: {согласовано_organization} != {утверждаю_organization}"
        else:
            return "Ошибка: Не удалось извлечь наименования"
    else:
        return "Ошибка: Не удалось извлечь наименования из блоков СОГЛАСОВАНО или УТВЕРЖДАЮ"

# Основная функция
def main(pdf_path):
    text = extract_pdf_text(pdf_path)
    согласовано_name, утверждаю_name = extract_agreed_and_approved_names(text)
    
    # Сравнение наименований
    result = compare_names(согласовано_name, утверждаю_name)
    return согласовано_name, утверждаю_name, result

# Пример использования
pdf_path = "G:\\Python\\urfu_LLM_documents\\lama 3.1\\proverka8\\data\\Касли Больница оп 1 пост хр.pdf"  
согласовано_name, утверждаю_name, result = main(pdf_path)

# Выводим результат
print(f"Согласовано name: {согласовано_name}")
print(f"Утверждаю name: {утверждаю_name}")
print(f"Результат сравнения: {result}")
