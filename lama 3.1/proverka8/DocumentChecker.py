
import re
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer, util
import numpy as np

# Загрузка модели эмбеддингов
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def extract_pdf_text(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text("text")
    return text

def preprocess_text(text):
    text = re.sub(r"_+", " ", text)
    text = re.sub(r"От\s*\d{1,2}\.\d{1,2}\.\d{4}", "", text)
    text = re.sub(r"\d{1,2}\.\d{1,2}\.\d{4}", "", text)
    return text.strip()

def extract_agreed_name(text):
    pattern_agreed = (
        r"СОГЛАСОВАНО\s*(?:Протокол\s*ЭК\s*)?"
        r"(.*?)"
        r"(?=\s*[А-ЯЁ]\.[А-ЯЁ]\.\s*[А-ЯЁа-яё-]+|\bОт\b|\d{1,2}[./-]\d{1,2}[./-]\d{2,4})"
    )
    match = re.search(pattern_agreed, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

def extract_approved_name(text):
    pattern_approved = r"УТВЕРЖДАЮ\s*(.*?)(?:\s*[А-ЯЁ][а-яё]+\s[А-ЯЁ]\.\s*[А-ЯЁ]\.|(?:\d{4}\s*год))"
    match = re.search(pattern_approved, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

def compare_names(agreed, approved):
    if agreed and approved:
        agreed_vec = model.encode(preprocess_text(agreed), convert_to_tensor=True)
        approved_vec = model.encode(preprocess_text(approved), convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(agreed_vec, approved_vec).item()
        if similarity > 0.8:
            return "Наименования совпадают", similarity
        else:
            return "Наименования не совпадают", similarity
    else:
        return "Ошибка: Не удалось извлечь наименования", None

def main(pdf_path):
    text = extract_pdf_text(pdf_path)
    agreed = extract_agreed_name(text)
    approved = extract_approved_name(text)
    result, sim = compare_names(agreed, approved)
    return agreed, approved, result, sim

# Пример использования:
if __name__ == "__main__":
    pdf_path = "G:\\Python\\urfu_LLM_documents\\lama 3.1\\proverka8\\data\\СП_16_Магнит_отдел_Выборы_Гос_Дума_2016_оп_2_п_хр_.pdf"
    agreed, approved, result, sim = main(pdf_path)
    print(f"Согласовано: {agreed}\n")
    print(f"Утверждаю: {approved}\n")
    print(f"{result} (сходство: {sim})")
