import re
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer, util

# Загрузка модели эмбеддингов
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Извлечение текста из PDF
def extract_pdf_text(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text("text")
    return text

# Предобработка текста
def preprocess_text(text):
    text = re.sub(r"_+", " ", text)
    text = re.sub(r"От\s*\d{1,2}\.\d{1,2}\.\d{4}", "", text)
    text = re.sub(r"\d{1,2}\.\d{1,2}\.\d{4}", "", text)
    return text.strip()

# Извлечение из блока СОГЛАСОВАНО
def extract_agreed_name(text):
    pattern_agreed = (
        r"СОГЛАСОВАНО\s*(?:Протокол\s*ЭК\s*)?"
        r"(.*?)"
        r"(?=\s*[А-ЯЁ]\.[А-ЯЁ]\.\s*[А-ЯЁа-яё-]+|\bОт\b|\d{1,2}[./-]\d{1,2}[./-]\d{2,4})"
    )
    match = re.search(pattern_agreed, text, re.DOTALL)
    return match.group(1).strip() if match else None

# Извлечение из блока УТВЕРЖДАЮ
def extract_approved_name(text):
    pattern_approved = (
        r"УТВЕРЖДАЮ\s*"
        r"(.*?)"
        r"(?=\s*[А-ЯЁ]\.[А-ЯЁ]\.\s*[А-ЯЁа-яё-]+|\d{4}\s*год|\d{1,2}[./-]\d{1,2}[./-]\d{2,4})"
    )
    match = re.search(pattern_approved, text, re.DOTALL)
    return match.group(1).strip() if match else None

# Сравнение с использованием эмбеддингов напрямую
def compare_names_with_embeddings(name1, name2, threshold=0.8):
    if name1 and name2:
        name1_clean = preprocess_text(name1)
        name2_clean = preprocess_text(name2)
        emb1 = model.encode(name1_clean, convert_to_tensor=True)
        emb2 = model.encode(name2_clean, convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(emb1, emb2).item()

        if similarity >= threshold:
            return f"Наименования совпадают (сходство: {similarity:.2f})"
        else:
            return f"Наименования не совпадают (сходство: {similarity:.2f})"
    else:
        return "Ошибка: Не удалось извлечь наименования"

# Основная логика
def main(pdf_path):
    text = extract_pdf_text(pdf_path)
    agreed_name = extract_agreed_name(text)
    approved_name = extract_approved_name(text)
    result = compare_names_with_embeddings(agreed_name, approved_name)
    return agreed_name, approved_name, result

# Пример
pdf_path = "G:\\Python\\urfu_LLM_documents\\lama 3.1\\proverka8\\data\\Куса.отд.архитектуры.оп1пх.2022.pdf"
agreed_name, approved_name, result = main(pdf_path)

# Вывод
print(f"Согласовано name: {agreed_name}")
print(f"Утверждаю name: {approved_name}")
print(f"Результат сравнения: {result}")
